"""Contended-queue behavior tests for the queue proxy.

These exercise the parts of `queue_proxy` that the API/helper tests only touch
indirectly: FIFO admission when a per-model semaphore is full, cancellation of a
QUEUED (not-yet-dispatched) request, the project's policy that an in-flight
request is NOT cancellable, and the exact lifecycle outcome strings emitted.

Same harness as test_queue_proxy_api.py: the upstream LiteLLM gateway is faked
with a Starlette app served through httpx.ASGITransport (streams the body
properly), `emit_event` is monkeypatched to capture events in-memory, and no
real Postgres / network is touched. Outcome strings are read straight from
queue_proxy.py — never invented here.
"""

import asyncio

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse, StreamingResponse
from starlette.routing import Route

from overlaat import queue_proxy as qp


def asgi():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=qp.app), base_url="http://test")


@pytest.fixture
def isolate_state(monkeypatch):
    """Reset the proxy's module-level queue/metrics state and capture events.

    The event writer is never started in tests (ASGITransport does not run the
    lifespan), so EVENT_Q stays None; we capture events through the same
    monkeypatch the rest of the suite uses. Returns the captured-events list.

    Scheduler OFF by default here (SCHED is None and SCHEDULER_ON forced off) so
    the existing semaphore-path tests behave exactly as before; the
    scheduler-path tests opt in via the `scheduler_on` helper.
    """
    events: list[dict] = []
    monkeypatch.setattr(qp, "SEMAPHORES", {})
    monkeypatch.setattr(qp, "emit_event", lambda ev: events.append(dict(ev)))
    monkeypatch.setattr(qp, "SCHEDULER_ON", False)
    monkeypatch.setattr(qp, "SCHED", None)
    qp.METRICS.clear()
    qp.QUEUED.clear()
    yield events
    qp.METRICS.clear()
    qp.QUEUED.clear()


def scheduler_on(monkeypatch, *, caps, budget=1.0, **kw):
    """Turn the cost scheduler ON for a test: install a fresh Scheduler bound to
    the running loop and flip SCHEDULER_ON. Returns the Scheduler instance."""
    from overlaat.scheduler import Scheduler

    sched = Scheduler(budget=budget, caps=caps, **kw)
    monkeypatch.setattr(qp, "CAPS", caps)
    monkeypatch.setattr(qp, "SCHEDULER_ON", True)
    monkeypatch.setattr(qp, "SCHED", sched)
    return sched


def gated_upstream(gate: asyncio.Event, arrivals: list[str]):
    """Fake upstream that records arrival order then blocks until `gate` is set.

    Lets a test pin N requests against a full semaphore and observe the order in
    which the proxy dispatches them to the backend.
    """
    import json

    async def handler(request):
        body = await request.body()
        try:
            arrivals.append(json.loads(body).get("tag"))
        except Exception:
            arrivals.append(None)
        await gate.wait()
        return PlainTextResponse('{"usage":{"prompt_tokens":3,"completion_tokens":5}}')

    app = Starlette(routes=[Route("/{path:path}", handler, methods=["POST", "GET"])])
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=qp.UPSTREAM)


async def _wait_for(predicate, timeout=2.0, interval=0.01):
    """Poll until predicate() is truthy or timeout — avoids fixed sleeps racing."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return predicate()


# ── a. FIFO admission order ───────────────────────────────────────────────────


async def test_fifo_admission_order(monkeypatch, isolate_state):
    """With a cap of 1, three requests staggered in arrival order are dispatched
    to the backend in that same order — the semaphore admits FIFO."""
    monkeypatch.setattr(qp, "CAPS", {"m": 1})
    gate = asyncio.Event()
    arrivals: list[str] = []
    qp.app.state.client = gated_upstream(gate, arrivals)

    async def call(tag):
        async with asgi() as c:
            r = await c.post(
                "/v1/chat/completions", json={"model": "m", "stream": False, "tag": tag}
            )
        return tag, r.status_code

    tasks = []
    for tag in ("a", "b", "c"):
        tasks.append(asyncio.ensure_future(call(tag)))
        # small stagger pins arrival order deterministically
        await asyncio.sleep(0.03)

    # one in-flight + two waiting while the upstream is gated
    assert await _wait_for(lambda: qp.METRICS["m"]["queue_depth"] == 2)
    assert qp.METRICS["m"]["in_flight"] == 1
    assert len(qp.QUEUED["m"]) == 2

    gate.set()
    results = await asyncio.gather(*tasks)

    assert [code for _, code in results] == [200, 200, 200]
    # dispatched to the backend in arrival order
    assert arrivals == ["a", "b", "c"]
    # queue fully drained, slot released
    assert qp.METRICS["m"]["queue_depth"] == 0
    assert qp.METRICS["m"]["in_flight"] == 0
    assert qp.QUEUED["m"] == {}
    await qp.app.state.client.aclose()


# ── b. cancel-while-queued vs in-flight ───────────────────────────────────────


async def test_cancel_removes_queued_request(monkeypatch, isolate_state):
    """Cancelling a QUEUED request drops it from the queue: the caller gets 499
    with the queue_proxy_cancelled error type, and it never reaches the backend.
    The already-running request is untouched and completes."""
    monkeypatch.setattr(qp, "CAPS", {"m": 1})
    gate = asyncio.Event()
    arrivals: list[str] = []
    qp.app.state.client = gated_upstream(gate, arrivals)

    async def call(tag):
        async with asgi() as c:
            r = await c.post(
                "/v1/chat/completions", json={"model": "m", "stream": False, "tag": tag}
            )
        return tag, r.status_code, r.json()

    inflight = asyncio.ensure_future(call("inflight"))
    await _wait_for(lambda: qp.METRICS["m"]["in_flight"] == 1)
    queued = asyncio.ensure_future(call("queued"))
    assert await _wait_for(lambda: len(qp.QUEUED["m"]) == 1)

    req_id = next(iter(qp.QUEUED["m"]))
    async with asgi() as c:
        cancel = await c.post(f"/__queue/cancel/{req_id}")
    assert cancel.status_code == 200
    assert cancel.json() == {"cancelled": req_id, "model": "m"}

    _, q_status, q_body = await queued
    assert q_status == 499
    assert q_body["error"]["type"] == "queue_proxy_cancelled"

    # the cancelled request never touched the backend
    assert "queued" not in arrivals
    # and it left the registry / decremented depth
    assert qp.QUEUED["m"] == {}
    assert qp.METRICS["m"]["queue_depth"] == 0

    gate.set()
    tag, code, body = await inflight
    assert (tag, code) == ("inflight", 200)
    assert arrivals == ["inflight"]
    await qp.app.state.client.aclose()


async def test_inflight_request_is_not_cancellable(monkeypatch, isolate_state):
    """Policy: only QUEUED requests are cancellable. An in-flight request is not
    registered in QUEUED, so cancel-by-id returns 404 (queue_proxy_not_found) and
    the request keeps running to completion."""
    monkeypatch.setattr(qp, "CAPS", {"m": 1})
    gate = asyncio.Event()
    arrivals: list[str] = []
    qp.app.state.client = gated_upstream(gate, arrivals)

    async def call():
        async with asgi() as c:
            return await c.post("/v1/chat/completions", json={"model": "m", "stream": False})

    inflight = asyncio.ensure_future(call())
    # wait until the slot is taken and the backend has been reached
    assert await _wait_for(lambda: qp.METRICS["m"]["in_flight"] == 1)
    assert await _wait_for(lambda: arrivals == [None] or len(arrivals) == 1)

    # nothing is queued: the only request is already running
    assert qp.QUEUED["m"] == {}

    # cancel-all must report zero cancellations for the in-flight request
    async with asgi() as c:
        cancel_all = await c.post("/__queue/cancel-all", params={"model": "m"})
    assert cancel_all.json() == {"cancelled": [], "count": 0}

    # a fabricated id resolves to 404 — there is no per-id handle for in-flight work
    async with asgi() as c:
        cancel_one = await c.post("/__queue/cancel/deadbeef0000")
    assert cancel_one.status_code == 404
    assert cancel_one.json()["error"]["type"] == "queue_proxy_not_found"

    gate.set()
    r = await inflight
    assert r.status_code == 200  # ran to completion, never cancelled
    await qp.app.state.client.aclose()


# ── c. event outcomes ─────────────────────────────────────────────────────────


async def test_outcome_cancelled_queued(monkeypatch, isolate_state):
    """A cancelled QUEUED request emits exactly one event with outcome
    `cancelled_queued` and http_status 499 (and no token counts)."""
    events = isolate_state
    monkeypatch.setattr(qp, "CAPS", {"m": 1})
    gate = asyncio.Event()
    arrivals: list[str] = []
    qp.app.state.client = gated_upstream(gate, arrivals)

    async def call(tag):
        async with asgi() as c:
            return await c.post(
                "/v1/chat/completions", json={"model": "m", "stream": False, "tag": tag}
            )

    inflight = asyncio.ensure_future(call("inflight"))
    await _wait_for(lambda: qp.METRICS["m"]["in_flight"] == 1)
    queued = asyncio.ensure_future(call("queued"))
    assert await _wait_for(lambda: len(qp.QUEUED["m"]) == 1)

    req_id = next(iter(qp.QUEUED["m"]))
    async with asgi() as c:
        await c.post(f"/__queue/cancel/{req_id}")
    await queued

    cancelled = [e for e in events if e["outcome"] == "cancelled_queued"]
    assert len(cancelled) == 1
    ev = cancelled[0]
    assert ev["http_status"] == 499
    assert ev["model_requested"] == "m"
    assert ev["prompt_tokens"] is None
    assert ev["completion_tokens"] is None

    gate.set()
    await inflight
    # the in-flight one completed
    assert any(e["outcome"] == "completed" for e in events)
    await qp.app.state.client.aclose()


async def test_outcome_completed_carries_tokens(monkeypatch, isolate_state):
    """A request that streams to natural completion emits outcome `completed`
    with the token counts parsed from the upstream usage block."""
    events = isolate_state
    monkeypatch.setattr(qp, "CAPS", {"m": 1})

    async def handler(request):
        return PlainTextResponse('{"usage":{"prompt_tokens":7,"completion_tokens":9}}')

    upstream = Starlette(routes=[Route("/{path:path}", handler, methods=["POST"])])
    qp.app.state.client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=upstream), base_url=qp.UPSTREAM
    )

    async with asgi() as c:
        r = await c.post("/v1/chat/completions", json={"model": "m", "stream": False})

    assert r.status_code == 200
    assert len(events) == 1
    ev = events[0]
    assert ev["outcome"] == "completed"
    assert ev["http_status"] == 200
    assert ev["prompt_tokens"] == 7
    assert ev["completion_tokens"] == 9
    await qp.app.state.client.aclose()


async def test_outcome_upstream_error(monkeypatch, isolate_state):
    """An upstream 4xx/5xx is forwarded verbatim and emits outcome
    `upstream_error` carrying the upstream status."""
    events = isolate_state
    monkeypatch.setattr(qp, "CAPS", {"m": 1})

    async def handler(request):
        return PlainTextResponse("rate limited", status_code=429)

    upstream = Starlette(routes=[Route("/{path:path}", handler, methods=["POST"])])
    qp.app.state.client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=upstream), base_url=qp.UPSTREAM
    )

    async with asgi() as c:
        r = await c.post("/v1/chat/completions", json={"model": "m", "stream": False})

    assert r.status_code == 429
    assert len(events) == 1
    assert events[0]["outcome"] == "upstream_error"
    assert events[0]["http_status"] == 429
    assert qp.METRICS["m"]["total_errored"] == 1
    assert qp.METRICS["m"]["total_served"] == 0
    await qp.app.state.client.aclose()


async def test_outcome_client_abandoned(monkeypatch, isolate_state):
    """A client that disconnects mid-stream (before the body is fully read) emits
    outcome `client_abandoned`. Driven through `_forward` directly: closing the
    returned StreamingResponse body iterator throws GeneratorExit into the proxy's
    `stream_body` exactly like a real client disconnect, which is the only event
    path the proxy treats as abandonment. The slot is still released."""
    from starlette.requests import Request

    events = isolate_state
    monkeypatch.setattr(qp, "CAPS", {"m": 1})

    async def gen():
        yield b'{"choices":[{"delta":{"content":"hi"}}]}\n'
        yield b"middle\n"
        yield b'{"usage":{"prompt_tokens":1,"completion_tokens":2}}'

    async def handler(request):
        return StreamingResponse(gen(), media_type="text/event-stream")

    upstream = Starlette(routes=[Route("/{path:path}", handler, methods=["POST"])])
    qp.app.state.client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=upstream), base_url=qp.UPSTREAM
    )

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [(b"authorization", b"Bearer sk-x")],
        "query_string": b"",
        "app": qp.app,
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    request = Request(scope, receive)

    # mimic the proxy state right after a slot is acquired
    sem = qp.get_semaphore("m")
    await sem.acquire()
    qp.METRICS["m"]["in_flight"] += 1
    ev = {
        "t_enqueue": 0.0,
        "t_acquire": 0.0,
        "t_first_token": None,
        "model_requested": "m",
        "key_fp": "x",
        "streamed": True,
    }

    resp = await qp._forward(
        request, "/v1/chat/completions", b'{"model":"m","stream":true}', "m", sem, ev
    )
    body = resp.body_iterator
    await body.__anext__()  # read part of the stream
    await body.aclose()  # disconnect before natural completion
    await _wait_for(lambda: bool(events))

    assert len(events) == 1
    assert events[0]["outcome"] == "client_abandoned"
    assert events[0]["http_status"] == 200  # upstream itself succeeded
    # slot released despite the abandon
    assert qp.METRICS["m"]["in_flight"] == 0
    assert not sem.locked()
    await qp.app.state.client.aclose()


# ── d. cost-weighted scheduler (OVERLAAT_SCHEDULER on) ────────────────────────


async def test_no_config_backcompat(monkeypatch, isolate_state):
    """OVERLAAT_SCHEDULER=off behaves exactly like the per-model-FIFO baseline.

    Same setup as test_fifo_admission_order, but explicitly with the scheduler
    kill-switch OFF (the isolate_state default), asserting the semaphore path is
    byte-for-byte unchanged: in-flight 1 / two queued, then FIFO dispatch and a
    fully drained queue."""
    assert qp.SCHEDULER_ON is False and qp.SCHED is None
    monkeypatch.setattr(qp, "CAPS", {"m": 1})
    gate = asyncio.Event()
    arrivals: list[str] = []
    qp.app.state.client = gated_upstream(gate, arrivals)

    async def call(tag):
        async with asgi() as c:
            r = await c.post(
                "/v1/chat/completions", json={"model": "m", "stream": False, "tag": tag}
            )
        return tag, r.status_code

    tasks = []
    for tag in ("a", "b", "c"):
        tasks.append(asyncio.ensure_future(call(tag)))
        await asyncio.sleep(0.03)

    assert await _wait_for(lambda: qp.METRICS["m"]["queue_depth"] == 2)
    assert qp.METRICS["m"]["in_flight"] == 1

    gate.set()
    results = await asyncio.gather(*tasks)
    assert [code for _, code in results] == [200, 200, 200]
    assert arrivals == ["a", "b", "c"]
    assert qp.METRICS["m"]["queue_depth"] == 0
    assert qp.METRICS["m"]["in_flight"] == 0
    await qp.app.state.client.aclose()


async def test_scheduler_on_single_model_matches_fifo(monkeypatch, isolate_state):
    """With the scheduler ON, a single model with cap=1, equal priority and no
    aging reduces to FIFO — identical observable behaviour to the semaphore.

    cost = 1/cap = 1.0, B = 1.0, so exactly one run fits the budget; the rest
    wait and are admitted in arrival order."""
    scheduler_on(monkeypatch, caps={"m": 1})
    gate = asyncio.Event()
    arrivals: list[str] = []
    qp.app.state.client = gated_upstream(gate, arrivals)

    async def call(tag):
        async with asgi() as c:
            r = await c.post(
                "/v1/chat/completions", json={"model": "m", "stream": False, "tag": tag}
            )
        return tag, r.status_code

    tasks = []
    for tag in ("a", "b", "c"):
        tasks.append(asyncio.ensure_future(call(tag)))
        await asyncio.sleep(0.03)

    assert await _wait_for(lambda: qp.METRICS["m"]["queue_depth"] == 2)
    assert qp.METRICS["m"]["in_flight"] == 1
    assert len(qp.QUEUED["m"]) == 2

    gate.set()
    results = await asyncio.gather(*tasks)
    assert [code for _, code in results] == [200, 200, 200]
    assert arrivals == ["a", "b", "c"]
    assert qp.METRICS["m"]["queue_depth"] == 0
    assert qp.METRICS["m"]["in_flight"] == 0
    assert qp.SCHED.used == 0.0
    assert qp.SCHED.queue_depth() == 0
    await qp.app.state.client.aclose()


async def test_cancel_while_queued_under_global_queue(monkeypatch, isolate_state):
    """Cancelling a request waiting in the GLOBAL scheduler queue yields the same
    499 / queue_proxy_cancelled outcome as the semaphore path, never reaches the
    backend, and is withdrawn from the scheduler (queue_depth back to 0)."""
    sched = scheduler_on(monkeypatch, caps={"m": 1})
    gate = asyncio.Event()
    arrivals: list[str] = []
    qp.app.state.client = gated_upstream(gate, arrivals)

    async def call(tag):
        async with asgi() as c:
            r = await c.post(
                "/v1/chat/completions", json={"model": "m", "stream": False, "tag": tag}
            )
        return tag, r.status_code, r.json()

    inflight = asyncio.ensure_future(call("inflight"))
    await _wait_for(lambda: qp.METRICS["m"]["in_flight"] == 1)
    queued = asyncio.ensure_future(call("queued"))
    assert await _wait_for(lambda: len(qp.QUEUED["m"]) == 1)
    assert sched.queue_depth() == 1

    req_id = next(iter(qp.QUEUED["m"]))
    async with asgi() as c:
        cancel = await c.post(f"/__queue/cancel/{req_id}")
    assert cancel.status_code == 200

    _, q_status, q_body = await queued
    assert q_status == 499
    assert q_body["error"]["type"] == "queue_proxy_cancelled"
    assert "queued" not in arrivals
    assert sched.queue_depth() == 0

    gate.set()
    tag, code, _ = await inflight
    assert (tag, code) == ("inflight", 200)
    assert arrivals == ["inflight"]
    assert sched.used == 0.0
    await qp.app.state.client.aclose()


async def test_cancel_clears_reservation(monkeypatch, isolate_state):
    """Cancelling the reserved expensive HEAD clears its budget reservation, so a
    cheaper waiter that the reservation was holding budget away from can flow in.

    Setup: cap m=4 (cost 0.25), an expensive model big with explicit cost 1.0.
    Three cheap m runs occupy used=0.75. A big request arrives at the head: it
    cannot fit (needs 1.0, only 0.25 free) and becomes reserved_for, which
    blocks the 4th cheap m run (reservable = 0). Cancelling big clears the
    reservation; the 4th cheap run is then admitted."""
    sched = scheduler_on(
        monkeypatch,
        caps={"m": 4, "big": 1},
        costs={"big": 1.0},
    )
    gate = asyncio.Event()
    arrivals: list[str] = []
    qp.app.state.client = gated_upstream(gate, arrivals)

    async def call(model, tag):
        async with asgi() as c:
            r = await c.post(
                "/v1/chat/completions", json={"model": model, "stream": False, "tag": tag}
            )
        return tag, r.status_code

    cheap = [asyncio.ensure_future(call("m", f"c{i}")) for i in range(3)]
    await _wait_for(lambda: qp.METRICS["m"]["in_flight"] == 3)
    assert abs(sched.used - 0.75) < 1e-9

    # the expensive head arrives → reserved, blocks the budget for cheap #4
    big = asyncio.ensure_future(call("big", "big"))
    assert await _wait_for(lambda: sched.reserved_for is not None)
    big_req_id = next(iter(qp.QUEUED["big"]))

    # a 4th cheap arrival cannot be admitted while big holds the reservation
    cheap4 = asyncio.ensure_future(call("m", "c4"))
    assert await _wait_for(lambda: len(qp.QUEUED["m"]) == 1)
    assert "c4" not in arrivals

    # cancel the reserved head → reservation cleared → cheap #4 flows in
    async with asgi() as c:
        await c.post(f"/__queue/cancel/{big_req_id}")
    assert await _wait_for(lambda: sched.reserved_for is None)
    assert await _wait_for(lambda: qp.METRICS["m"]["in_flight"] == 4)
    assert "c4" in arrivals

    gate.set()
    await asyncio.gather(*cheap, cheap4)
    _, big_code = await big
    assert big_code == 499
    assert sched.used == 0.0
    await qp.app.state.client.aclose()


async def test_inflight_not_cancellable(monkeypatch, isolate_state):
    """Under the scheduler, an in-flight (admitted) request is still NOT
    cancellable: cancel-all reports zero, cancel-by-id 404s, and it runs to
    completion (the no-preemption rule)."""
    scheduler_on(monkeypatch, caps={"m": 1})
    gate = asyncio.Event()
    arrivals: list[str] = []
    qp.app.state.client = gated_upstream(gate, arrivals)

    async def call():
        async with asgi() as c:
            return await c.post("/v1/chat/completions", json={"model": "m", "stream": False})

    inflight = asyncio.ensure_future(call())
    assert await _wait_for(lambda: qp.METRICS["m"]["in_flight"] == 1)
    assert await _wait_for(lambda: len(arrivals) == 1)
    assert qp.QUEUED["m"] == {}

    async with asgi() as c:
        cancel_all = await c.post("/__queue/cancel-all", params={"model": "m"})
    assert cancel_all.json() == {"cancelled": [], "count": 0}

    async with asgi() as c:
        cancel_one = await c.post("/__queue/cancel/deadbeef0000")
    assert cancel_one.status_code == 404

    gate.set()
    r = await inflight
    assert r.status_code == 200
    assert qp.SCHED.used == 0.0
    await qp.app.state.client.aclose()


async def test_client_abandon_releases_budget(monkeypatch, isolate_state):
    """A client that disconnects mid-stream releases its budget back to the
    scheduler (used returns to 0, in_flight to 0), just like the semaphore path
    releases the slot."""
    from starlette.requests import Request

    events = isolate_state
    sched = scheduler_on(monkeypatch, caps={"m": 1})

    async def gen():
        yield b'{"choices":[{"delta":{"content":"hi"}}]}\n'
        yield b"middle\n"
        yield b'{"usage":{"prompt_tokens":1,"completion_tokens":2}}'

    async def handler(request):
        return StreamingResponse(gen(), media_type="text/event-stream")

    upstream = Starlette(routes=[Route("/{path:path}", handler, methods=["POST"])])
    qp.app.state.client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=upstream), base_url=qp.UPSTREAM
    )

    # admit one run through the scheduler so it owns budget
    from overlaat.scheduler import Waiter

    w = Waiter(
        req_id="r1",
        model="m",
        cost=sched.cost("m"),
        base_priority=0,
        key_fp="x",
        enqueued_at=sched._now(),
    )
    sched.enqueue(w)
    assert w.fut.done()
    qp.METRICS["m"]["in_flight"] += 1
    assert abs(sched.used - 1.0) < 1e-9

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [(b"authorization", b"Bearer sk-x")],
        "query_string": b"",
        "app": qp.app,
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    request = Request(scope, receive)
    ev = {
        "t_enqueue": 0.0,
        "t_acquire": 0.0,
        "t_first_token": None,
        "model_requested": "m",
        "key_fp": "x",
        "streamed": True,
        "priority": 0,
        "cost": sched.cost("m"),
        "wait_reason": "none",
    }

    resp = await qp._forward(
        request,
        "/v1/chat/completions",
        b'{"model":"m","stream":true}',
        "m",
        None,
        ev,
        use_scheduler=True,
    )
    body = resp.body_iterator
    await body.__anext__()
    await body.aclose()
    await _wait_for(lambda: bool(events))

    assert events[0]["outcome"] == "client_abandoned"
    assert sched.used == 0.0  # budget released
    assert qp.METRICS["m"]["in_flight"] == 0
    await qp.app.state.client.aclose()


async def test_new_event_fields_emitted(monkeypatch, isolate_state):
    """A request served under the scheduler emits priority / cost / wait_reason on
    its lifecycle event (the new observability columns)."""
    events = isolate_state
    scheduler_on(monkeypatch, caps={"m": 2})

    async def handler(request):
        return PlainTextResponse('{"usage":{"prompt_tokens":7,"completion_tokens":9}}')

    upstream = Starlette(routes=[Route("/{path:path}", handler, methods=["POST"])])
    qp.app.state.client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=upstream), base_url=qp.UPSTREAM
    )

    async with asgi() as c:
        r = await c.post(
            "/v1/chat/completions", json={"model": "m", "stream": False, "priority": 3}
        )
    assert r.status_code == 200
    assert len(events) == 1
    ev = events[0]
    assert ev["outcome"] == "completed"
    assert ev["priority"] == 3
    assert abs(ev["cost"] - 0.5) < 1e-9  # 1/cap = 1/2
    assert ev["wait_reason"] == "none"  # admitted on first pump, never waited
    await qp.app.state.client.aclose()

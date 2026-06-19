"""FIFO queue-proxy: the front-door entry on :4000, in front of LiteLLM (:4002).

Problem: LiteLLM `max_parallel_requests` + the swap-layer `concurrencyLimit` reject
with 429 on overflow — there is no wait-queue. Bursty consumers (parallel synthesis,
parallel transcription) get 429 cascades and have to implement their own backoff.

This proxy puts a per-model `asyncio.Semaphore` around every `/v1/chat/completions`,
`/v1/completions` and `/v1/embeddings` call. Slot size comes from
`litellm-config.yaml::max_parallel_requests`. Models without a cap: pass-through
without a queue.

Streaming-compatible (SSE and plain JSON); no body buffering except the initial
parse to read `model`. Headers are forwarded 1:1 except hop-by-hop.

INSTRUMENTATION: the proxy is the ONLY component that sees the full lifecycle of
every request (including queued + client-abandoned calls — both invisible to
LiteLLM SpendLogs, which inserts only on completion). Per request it writes one
row to `request_events` (Postgres): t_enqueue / t_acquire / t_first_token /
t_done, outcome, model, key_fp, streamed, and token counts (from LiteLLM's own
`usage`). Writing is non-blocking via a background writer + bounded queue. If the
queue is full or the DB write fails, the event is dropped (and counted) — the
hot path is never slowed or failed by instrumentation.

Endpoints outside the proxy:
- GET  /__queue/health           — liveness + caps + event-writer stats
- GET  /__queue/status           — per-model {in_flight, queue_depth, served, wait p50/p95}
                                    + `queued` list {id, key_fp, age_s} per model
- POST /__queue/cancel/{req_id}   — cancel one queued (waiting) request
- POST /__queue/cancel-all?model=&key_fp= — cancel all queued requests in scope

Cancel only affects queued (not-yet-dispatched) requests — those never touched the
GPU, so they are safe to drop (caller gets 499). In-flight requests are NOT
cancellable: with a single-stream engine, a client disconnect lets the engine keep
decoding while the proxy releases its slot → the next call stalls on a busy
backend. So release-on-disconnect does NOT stop the backend; only cancelling
still-queued requests is safe.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

UPSTREAM = os.environ.get("QUEUE_PROXY_UPSTREAM", "http://127.0.0.1:4002")
LITELLM_CONFIG = Path(os.environ.get(
    "QUEUE_PROXY_LITELLM_CONFIG", "./litellm-config.yaml"
))
METRICS_DB_URL = os.environ.get("METRICS_DB_URL") or os.environ.get("DATABASE_URL") or ""
WAIT_BUFFER = 200          # per-model ring buffer for p50/p95
USAGE_TAIL_BYTES = 16384   # rolling tail window scanned for token usage
EVENT_QUEUE_MAX = 10000    # bounded; overflow → drop (instrumentation never blocks)
EVENT_BATCH = 200          # max rows per INSERT batch
PROXIED_PATHS = {"/v1/chat/completions", "/v1/completions", "/v1/embeddings",
                 "/v1/rerank", "/rerank"}
SERVICE_VERSION = "0.0.1"

_INSERT_SQL = (
    "INSERT INTO request_events "
    "(t_enqueue, t_acquire, t_first_token, t_done, model_requested, key_fp, "
    " streamed, outcome, http_status, prompt_tokens, completion_tokens) "
    "VALUES (%(t_enqueue)s, %(t_acquire)s, %(t_first_token)s, %(t_done)s, "
    "%(model_requested)s, %(key_fp)s, %(streamed)s, %(outcome)s, "
    "%(http_status)s, %(prompt_tokens)s, %(completion_tokens)s)"
)
_EVENT_COLS = ("t_enqueue", "t_acquire", "t_first_token", "t_done",
               "model_requested", "key_fp", "streamed", "outcome",
               "http_status", "prompt_tokens", "completion_tokens")

_RE_PT = re.compile(rb'"prompt_tokens"\s*:\s*(\d+)')
_RE_CT = re.compile(rb'"completion_tokens"\s*:\s*(\d+)')


def _extract_tokens(tail: bytes) -> tuple[int | None, int | None]:
    """Take the LAST prompt/completion_tokens from the tail window. Works for
    streaming (usage chunk just before [DONE]) and non-stream (usage in body).
    No JSON parse → robust against nested *_tokens_details. None = not reported."""
    pt = _RE_PT.findall(tail)
    ct = _RE_CT.findall(tail)
    return (int(pt[-1]) if pt else None, int(ct[-1]) if ct else None)


def load_caps(path: Path) -> dict[str, int]:
    """Read max_parallel_requests per model_name from the LiteLLM config."""
    if not path.exists():
        return {}
    try:
        cfg = yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}
    out: dict[str, int] = {}
    for m in cfg.get("model_list", []):
        name = m.get("model_name")
        cap = (m.get("litellm_params") or {}).get("max_parallel_requests")
        if isinstance(name, str) and isinstance(cap, int) and cap > 0:
            out[name] = cap
    return out


CAPS: dict[str, int] = load_caps(LITELLM_CONFIG)
SEMAPHORES: dict[str, asyncio.Semaphore] = {}


def _new_metrics() -> dict:
    return {
        "in_flight": 0,
        "queue_depth": 0,
        "total_served": 0,
        "total_errored": 0,
        "wait_ms_buffer": deque(maxlen=WAIT_BUFFER),
    }


METRICS: dict[str, dict] = defaultdict(_new_metrics)

# Registry of WAITING requests (still on sem.acquire(), GPU not touched yet).
QUEUED: dict[str, dict[str, dict]] = defaultdict(dict)

# Instrumentation state (populated in lifespan).
EVENT_Q: asyncio.Queue | None = None
EVENT_STATS = {"emitted": 0, "written": 0, "dropped": 0}


def _key_fp(request: Request) -> str:
    """sha256(bearer)[:8] — equals LiteLLM_VerificationToken.token[:8], so the
    usage-api can resolve it to a key_alias. Not reversible, no secret leak."""
    auth = request.headers.get("authorization", "")
    tok = auth.split()[-1] if auth else ""
    if not tok:
        return "none"
    return hashlib.sha256(tok.encode()).hexdigest()[:8]


# ── event emission (non-blocking) ────────────────────────────────────────────

def emit_event(ev: dict) -> None:
    """Enqueue a lifecycle event for the background writer. Never blocks: on a
    full queue we drop (and count it). Called from the hot path."""
    q = EVENT_Q
    if q is None:
        return
    row = {c: ev.get(c) for c in _EVENT_COLS}
    EVENT_STATS["emitted"] += 1
    try:
        q.put_nowait(row)
    except asyncio.QueueFull:
        EVENT_STATS["dropped"] += 1


async def _event_writer() -> None:
    """Background task: drain EVENT_Q and batch-insert into Postgres. Reconnect
    on error. The DB is local and expected to be up; on a write error we drop the
    batch (counted) and reconnect for the next one — no on-disk fallback. Sentinel
    None = stop."""
    import psycopg  # local: only needed in the writer

    q = EVENT_Q
    conn = None
    while True:
        first = await q.get()
        if first is None:
            break
        batch = [first]
        stop = False
        while len(batch) < EVENT_BATCH:
            try:
                nxt = q.get_nowait()
            except asyncio.QueueEmpty:
                break
            if nxt is None:
                stop = True
                break
            batch.append(nxt)
        try:
            if conn is None or conn.closed:
                conn = await psycopg.AsyncConnection.connect(
                    METRICS_DB_URL, autocommit=True)
            async with conn.cursor() as cur:
                await cur.executemany(_INSERT_SQL, batch)
            EVENT_STATS["written"] += len(batch)
        except Exception as e:  # noqa: BLE001
            EVENT_STATS["dropped"] += len(batch)
            sys.stderr.write(f"event-writer: dropped {len(batch)} ({type(e).__name__}: {e})\n")
            sys.stderr.flush()
            if conn is not None:
                try:
                    await conn.close()
                except Exception:
                    pass
                conn = None
        if stop:
            break
    if conn is not None:
        try:
            await conn.close()
        except Exception:
            pass


def get_semaphore(model: str) -> asyncio.Semaphore | None:
    cap = CAPS.get(model)
    if not cap:
        return None
    sem = SEMAPHORES.get(model)
    if sem is None:
        sem = asyncio.Semaphore(cap)
        SEMAPHORES[model] = sem
    return sem


def _cancelled_response(req_id: str, model: str) -> JSONResponse:
    return JSONResponse(
        status_code=499,
        content={"error": {
            "message": f"request {req_id} cancelled while queued for {model}",
            "type": "queue_proxy_cancelled"}},
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global EVENT_Q
    app.state.client = httpx.AsyncClient(
        base_url=UPSTREAM,
        timeout=httpx.Timeout(connect=5.0, read=1200.0, write=60.0, pool=5.0),
        limits=httpx.Limits(max_keepalive_connections=64, max_connections=128),
    )
    EVENT_Q = asyncio.Queue(maxsize=EVENT_QUEUE_MAX)
    writer = asyncio.create_task(_event_writer())
    yield
    # stop writer: sentinel + drain
    if EVENT_Q is not None:
        await EVENT_Q.put(None)
    try:
        await asyncio.wait_for(writer, timeout=10.0)
    except Exception:
        writer.cancel()
    await app.state.client.aclose()


app = FastAPI(
    title="queue-proxy",
    version=SERVICE_VERSION,
    description="FIFO queue sidecar + per-request instrumentation. Front-door entry :4000.",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)


@app.get("/__queue/health")
async def health():
    return {"status": "ok", "version": SERVICE_VERSION,
            "upstream": UPSTREAM, "caps": CAPS,
            "events": dict(EVENT_STATS),
            "event_queue_depth": EVENT_Q.qsize() if EVENT_Q else None,
            "db_configured": bool(METRICS_DB_URL)}


@app.get("/__queue/status")
async def status():
    rows = []
    now = time.time()
    for model in sorted(set(CAPS.keys()) | set(METRICS.keys())):
        m = METRICS[model]
        waits = sorted(m["wait_ms_buffer"])
        n = len(waits)
        queued = sorted(QUEUED.get(model, {}).values(), key=lambda i: i["enqueued_at"])
        rows.append({
            "model": model,
            "cap": CAPS.get(model),
            "in_flight": m["in_flight"],
            "queue_depth": m["queue_depth"],
            "total_served": m["total_served"],
            "total_errored": m["total_errored"],
            "wait_ms_p50": waits[n // 2] if n else None,
            "wait_ms_p95": waits[min(int(n * 0.95), n - 1)] if n >= 20 else None,
            "wait_ms_max_recent": waits[-1] if waits else None,
            "samples": n,
            "queued": [
                {"id": i["id"], "key_fp": i["key_fp"],
                 "age_s": round(now - i["enqueued_at"], 1)}
                for i in queued
            ],
        })
    return {
        "service": "queue-proxy",
        "version": SERVICE_VERSION,
        "by_model": rows,
        "total_in_flight": sum(m["in_flight"] for m in METRICS.values()),
        "total_queue_depth": sum(m["queue_depth"] for m in METRICS.values()),
        "total_served": sum(m["total_served"] for m in METRICS.values()),
    }


@app.post("/__queue/cancel-all")
async def cancel_all(model: str | None = None, key_fp: str | None = None):
    """Cancel all WAITING requests in scope (optionally filtered by model and/or
    key_fp). Idempotent; never touches in-flight requests."""
    cancelled = []
    for m, reqs in list(QUEUED.items()):
        if model and m != model:
            continue
        for rid, info in list(reqs.items()):
            if key_fp and info.get("key_fp") != key_fp:
                continue
            cf = info["cancel_fut"]
            if not cf.done():
                cf.set_result(True)
                cancelled.append(rid)
    return {"cancelled": cancelled, "count": len(cancelled)}


@app.post("/__queue/cancel/{req_id}")
async def cancel_one(req_id: str):
    """Cancel one waiting request by id. 404 if it is already running/done."""
    for m, reqs in list(QUEUED.items()):
        info = reqs.get(req_id)
        if info:
            cf = info["cancel_fut"]
            if not cf.done():
                cf.set_result(True)
            return {"cancelled": req_id, "model": m}
    return JSONResponse(
        status_code=404,
        content={"error": {
            "message": f"{req_id} not queued (already running, done, or unknown)",
            "type": "queue_proxy_not_found"}},
    )


_HOP_HEADERS_REQ = {"host", "content-length"}
_HOP_HEADERS_RESP = {"content-encoding", "transfer-encoding",
                     "content-length", "connection"}


async def _forward(request: Request, full_path: str, body: bytes,
                   model: str | None, sem: asyncio.Semaphore | None,
                   ev: dict | None):
    """Forward to upstream with semaphore release discipline + event emission.

    Release rule: exactly once per acquired semaphore. The event is emitted
    exactly once (in stream_body.finally, or on a send error here)."""
    client: httpx.AsyncClient = request.app.state.client
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in _HOP_HEADERS_REQ}
    upstream_req = client.build_request(
        method=request.method, url=full_path, content=body, headers=headers,
        params=dict(request.query_params),
    )
    released = {"done": False}
    emitted = {"done": False}

    def release(status_code: int):
        if released["done"] or sem is None or model is None:
            return
        released["done"] = True
        sem.release()
        m = METRICS[model]
        m["in_flight"] -= 1
        if 200 <= status_code < 400:
            m["total_served"] += 1
        else:
            m["total_errored"] += 1

    def finish(status_code: int, outcome: str, tail: bytes = b""):
        if ev is None or emitted["done"]:
            return
        emitted["done"] = True
        ev["t_done"] = time.time()
        ev["http_status"] = status_code
        ev["outcome"] = outcome
        pt, ct = _extract_tokens(tail)
        ev["prompt_tokens"] = pt
        ev["completion_tokens"] = ct
        emit_event(ev)

    try:
        upstream_resp = await client.send(upstream_req, stream=True)
    except Exception:
        release(599)
        finish(599, "upstream_error")
        raise

    base_outcome = "completed" if upstream_resp.status_code < 400 else "upstream_error"

    async def stream_body():
        tail = bytearray()
        natural = False
        try:
            async for chunk in upstream_resp.aiter_raw():
                if (ev is not None and ev.get("streamed")
                        and ev.get("t_first_token") is None
                        and b'"content"' in chunk):
                    ev["t_first_token"] = time.time()
                tail.extend(chunk)
                if len(tail) > USAGE_TAIL_BYTES:
                    del tail[:-USAGE_TAIL_BYTES]
                yield chunk
            natural = True
        finally:
            await upstream_resp.aclose()
            release(upstream_resp.status_code)
            # natural completion vs client disconnect (GeneratorExit before done)
            outcome = base_outcome if (natural or base_outcome == "upstream_error") \
                else "client_abandoned"
            finish(upstream_resp.status_code, outcome, bytes(tail))

    resp_headers = {k: v for k, v in upstream_resp.headers.items()
                    if k.lower() not in _HOP_HEADERS_RESP}
    return StreamingResponse(
        stream_body(),
        status_code=upstream_resp.status_code,
        headers=resp_headers,
        media_type=upstream_resp.headers.get("content-type"),
    )


@app.api_route("/{full_path:path}",
               methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"])
async def proxy(full_path: str, request: Request):
    path = "/" + full_path
    body = await request.body()

    model: str | None = None
    sem: asyncio.Semaphore | None = None
    streamed = False
    is_llm = path in PROXIED_PATHS

    if is_llm and body:
        try:
            payload = json.loads(body)
            m = payload.get("model")
            if isinstance(m, str):
                model = m
                sem = get_semaphore(m)
            streamed = bool(payload.get("stream"))
            # Force include_usage so the usage chunk arrives in the stream → we
            # get reliable token counts without relying on the backend default.
            if streamed and path == "/v1/chat/completions":
                so = payload.get("stream_options")
                if not isinstance(so, dict):
                    so = {}
                if not so.get("include_usage"):
                    so["include_usage"] = True
                    payload["stream_options"] = so
                    body = json.dumps(payload).encode()
        except Exception:
            pass

    # Event skeleton (only for real LLM calls that carry a model).
    ev: dict | None = None
    if is_llm and model is not None:
        ev = {
            "t_enqueue": time.time(), "t_acquire": None, "t_first_token": None,
            "model_requested": model, "key_fp": _key_fp(request),
            "streamed": streamed,
        }

    if sem is None:
        # No cap → no queue; backend starts immediately.
        if ev is not None:
            ev["t_acquire"] = ev["t_enqueue"]
        return await _forward(request, path, body, model, None, ev)

    # Wait for a slot — cancellable while still queued. Register the waiter so
    # /__queue/cancel* can drop it before it touches the GPU.
    metrics = METRICS[model]
    req_id = hashlib.sha1(f"{time.monotonic_ns()}{id(request)}".encode()).hexdigest()[:12]
    cancel_fut: asyncio.Future = asyncio.get_event_loop().create_future()
    QUEUED[model][req_id] = {
        "id": req_id, "model": model, "key_fp": _key_fp(request),
        "enqueued_at": time.time(), "cancel_fut": cancel_fut,
    }
    metrics["queue_depth"] += 1
    enqueue_ts = time.monotonic()
    acquire_task = asyncio.ensure_future(sem.acquire())
    try:
        await asyncio.wait({acquire_task, cancel_fut},
                           return_when=asyncio.FIRST_COMPLETED)
    finally:
        metrics["queue_depth"] -= 1
        QUEUED[model].pop(req_id, None)

    def _emit_cancelled():
        if ev is not None:
            ev["t_done"] = time.time()
            ev["outcome"] = "cancelled_queued"
            ev["http_status"] = 499
            ev["prompt_tokens"] = None
            ev["completion_tokens"] = None
            emit_event(ev)

    if not acquire_task.done():
        # cancel won the race → we never got the slot
        acquire_task.cancel()
        try:
            got = await acquire_task
        except asyncio.CancelledError:
            got = False
        if got:                       # acquired after all → give it back
            sem.release()
        _emit_cancelled()
        return _cancelled_response(req_id, model)
    if cancel_fut.done():             # slot + cancel at once → give the slot back
        sem.release()
        _emit_cancelled()
        return _cancelled_response(req_id, model)

    wait_ms = int((time.monotonic() - enqueue_ts) * 1000)
    metrics["wait_ms_buffer"].append(wait_ms)
    metrics["in_flight"] += 1
    if ev is not None:
        ev["t_acquire"] = time.time()

    return await _forward(request, path, body, model, sem, ev)

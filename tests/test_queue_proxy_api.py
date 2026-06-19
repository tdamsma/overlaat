"""API tests for the queue proxy.

The upstream LiteLLM gateway is faked with a tiny ASGI app served through
httpx.ASGITransport — that streams the response body properly (unlike
MockTransport, which eagerly consumes it and breaks the proxy's aiter_raw).
"""

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from overlaat import queue_proxy as qp

UPSTREAM_BODY = (
    '{"choices":[{"delta":{"content":"hi"}}]}\n{"usage":{"prompt_tokens":7,"completion_tokens":9}}'
)


def asgi():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=qp.app), base_url="http://test")


async def test_health():
    async with asgi() as c:
        r = await c.get("/__queue/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


async def test_status():
    async with asgi() as c:
        r = await c.get("/__queue/status")
    body = r.json()
    assert body["service"] == "queue-proxy"
    assert "by_model" in body


async def test_cancel_unknown_is_404():
    async with asgi() as c:
        r = await c.post("/__queue/cancel/does-not-exist")
    assert r.status_code == 404
    assert r.json()["error"]["type"] == "queue_proxy_not_found"


@pytest.fixture
def mock_upstream():
    async def handler(request):
        return PlainTextResponse(UPSTREAM_BODY)

    upstream = Starlette(routes=[Route("/{path:path}", handler, methods=["POST", "GET"])])
    qp.app.state.client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=upstream), base_url=qp.UPSTREAM
    )
    yield
    # the test closes the client after its request


async def test_forward_uncapped_emits_one_event(monkeypatch, mock_upstream):
    captured = {}
    monkeypatch.setattr(qp, "CAPS", {})  # uncapped -> straight through, no queue
    monkeypatch.setattr(qp, "emit_event", lambda ev: captured.update(ev))

    async with asgi() as c:
        r = await c.post("/v1/chat/completions", json={"model": "x", "stream": False})

    assert r.status_code == 200
    assert captured["model_requested"] == "x"
    assert captured["outcome"] == "completed"
    assert captured["prompt_tokens"] == 7
    assert captured["completion_tokens"] == 9
    # The proxy stamps every event with the running Overlaat version.
    assert captured["overlaat_version"] == qp.SERVICE_VERSION
    await qp.app.state.client.aclose()


async def test_forward_capped_acquires_slot(monkeypatch, mock_upstream):
    captured = {}
    monkeypatch.setattr(qp, "CAPS", {"x": 1})
    monkeypatch.setattr(qp, "SEMAPHORES", {})
    monkeypatch.setattr(qp, "emit_event", lambda ev: captured.update(ev))

    async with asgi() as c:
        r = await c.post("/v1/chat/completions", json={"model": "x", "stream": False})

    assert r.status_code == 200
    assert captured["model_requested"] == "x"
    assert captured["t_acquire"] is not None  # got the slot
    await qp.app.state.client.aclose()

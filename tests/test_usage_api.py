"""Endpoint tests for the read-only usage-api — metrics_db and the queue
status scrape are mocked, so no Postgres or running proxy is needed."""

import httpx

from overlaat import __version__
from overlaat import metrics_db as mdb
from overlaat import usage_api as ua


def asgi():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=ua.app), base_url="http://test")


def test_parse_window():
    assert ua.parse_window("30m") == 1800
    assert ua.parse_window("2h") == 7200
    assert ua.parse_window("1d") == 86400
    assert ua.parse_window("garbage") == 1800


def test_pick_bucket():
    assert ua.pick_bucket(1800) == 5
    assert ua.pick_bucket(3600) == 15
    assert ua.pick_bucket(7 * 86400) == 3600


class _FakeCursor:
    def execute(self, *a):
        pass

    def fetchone(self):
        return (1,)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeConn:
    def cursor(self):
        return _FakeCursor()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


async def test_healthz_ok(monkeypatch):
    monkeypatch.setattr(mdb, "_connect", lambda db: _FakeConn())
    async with asgi() as c:
        r = await c.get("/healthz")
    assert r.status_code == 200
    assert r.json()["ok"] is True


async def test_healthz_db_down(monkeypatch):
    def boom(db):
        raise RuntimeError("no db")

    monkeypatch.setattr(mdb, "_connect", boom)
    async with asgi() as c:
        r = await c.get("/healthz")
    assert r.json()["ok"] is False


async def test_now(monkeypatch):
    monkeypatch.setattr(ua, "alias_map", lambda: {})
    monkeypatch.setattr(
        ua,
        "scrape_queue_status",
        lambda: {"available": True, "by_model": [], "total_in_flight": 0, "total_queue_depth": 0},
    )
    monkeypatch.setattr(mdb, "latest_host", lambda db: None)
    monkeypatch.setattr(mdb, "fetch_events", lambda *a, **k: [])
    async with asgi() as c:
        r = await c.get("/now")
    assert r.status_code == 200
    assert "verdict" in r.json()


async def test_now_enriches_queued_waiters_per_consumer(monkeypatch):
    # alias_map resolves the waiter's key_fp -> a consumer alias.
    monkeypatch.setattr(ua, "alias_map", lambda: {"fp-team-alpha": "team-alpha"})
    # A scheduler-on /__queue/status: one queued waiter carrying the scheduler
    # fields (priority/effective_priority/pool/wait_reason) as _queued_view emits.
    monkeypatch.setattr(
        ua,
        "scrape_queue_status",
        lambda: {
            "available": True,
            "scheduler": True,
            "by_model": [
                {
                    "model": "chat-large",
                    "cap": 2,
                    "in_flight": 2,
                    "queue_depth": 1,
                    "queued": [
                        {
                            "id": "req-1",
                            "key_fp": "fp-team-alpha",
                            "age_s": 4.2,
                            "priority": 5,
                            "cost": 1.0,
                            "pool": "default",
                            "effective_priority": 5.8,
                            "wait_reason": "model_cap",
                        }
                    ],
                }
            ],
            "total_in_flight": 2,
            "total_queue_depth": 1,
        },
    )
    monkeypatch.setattr(mdb, "latest_host", lambda db: None)
    monkeypatch.setattr(mdb, "fetch_events", lambda *a, **k: [])
    async with asgi() as c:
        r = await c.get("/now")
    assert r.status_code == 200
    body = r.json()
    # Per-model queued item is enriched, with the alias resolved.
    item = body["models"][0]["queued"][0]
    assert item["key"] == "team-alpha"
    assert item["model"] == "chat-large"
    assert item["id"] == "req-1"
    assert item["age_s"] == 4.2
    assert item["priority"] == 5
    assert item["effective_priority"] == 5.8
    assert item["pool"] == "default"
    assert item["wait_reason"] == "model_cap"
    # Flat top-level `queued` carries the same enriched waiters.
    assert body["queued"] == [item]


async def test_now_queued_waiters_null_safe_scheduler_off(monkeypatch):
    # Scheduler kill-switch: _queued_view omits the scheduler fields entirely.
    monkeypatch.setattr(ua, "alias_map", lambda: {})
    monkeypatch.setattr(
        ua,
        "scrape_queue_status",
        lambda: {
            "available": True,
            "scheduler": False,
            "by_model": [
                {
                    "model": "chat-small",
                    "cap": 4,
                    "in_flight": 4,
                    "queue_depth": 1,
                    "queued": [{"id": "req-9", "key_fp": "fp-x", "age_s": 1.1}],
                }
            ],
            "total_in_flight": 4,
            "total_queue_depth": 1,
        },
    )
    monkeypatch.setattr(mdb, "latest_host", lambda db: None)
    monkeypatch.setattr(mdb, "fetch_events", lambda *a, **k: [])
    async with asgi() as c:
        r = await c.get("/now")
    item = r.json()["queued"][0]
    assert item["key"] == "fp-x"  # unresolved key_fp falls through
    assert item["model"] == "chat-small"
    assert item["age_s"] == 1.1
    assert item["priority"] is None
    assert item["effective_priority"] is None
    assert item["pool"] is None
    assert item["wait_reason"] is None


async def test_timeline(monkeypatch):
    monkeypatch.setattr(ua, "alias_map", lambda: {})
    monkeypatch.setattr(mdb, "build_timeline", lambda *a, **k: {"models": {}, "buckets": []})
    async with asgi() as c:
        r = await c.get("/timeline?last=30m")
    assert r.json()["_meta"]["kind"] == "timeline"


async def test_models(monkeypatch):
    monkeypatch.setattr(ua, "alias_map", lambda: {})
    monkeypatch.setattr(mdb, "build_models", lambda *a, **k: [])
    async with asgi() as c:
        r = await c.get("/models?last=24h")
    assert r.json()["_meta"]["kind"] == "models"


async def test_consumers(monkeypatch):
    monkeypatch.setattr(ua, "alias_map", lambda: {})
    monkeypatch.setattr(mdb, "build_consumers", lambda *a, **k: [])
    async with asgi() as c:
        r = await c.get("/consumers?last=24h")
    assert r.status_code == 200


async def test_workloads(monkeypatch):
    row = {"workload": "scout", "requests": 3, "latency_ms": {}}
    monkeypatch.setattr(mdb, "build_workloads", lambda *a, **k: [row])
    async with asgi() as c:
        r = await c.get("/workloads?last=24h")
    assert r.status_code == 200
    assert r.json()["workloads"] == [row]


async def test_requests(monkeypatch):
    monkeypatch.setattr(ua, "alias_map", lambda: {})
    row = {"t_enqueue": 100.0, "model": "m1", "consumer": "k1", "outcome": "completed"}
    monkeypatch.setattr(mdb, "build_recent_requests", lambda *a, **k: [row])
    async with asgi() as c:
        r = await c.get("/requests?limit=100")
    assert r.status_code == 200
    body = r.json()
    assert body["_meta"]["kind"] == "requests"
    assert body["limit"] == 100
    assert body["requests"] == [row]


async def test_requests_limit_clamped(monkeypatch):
    monkeypatch.setattr(ua, "alias_map", lambda: {})
    seen = {}

    def fake(db, limit, aliases):
        seen["limit"] = limit
        return []

    monkeypatch.setattr(mdb, "build_recent_requests", fake)
    async with asgi() as c:
        hi = await c.get("/requests?limit=99999")
        lo = await c.get("/requests?limit=0")
    # hard cap 500, floor 1.
    assert hi.json()["limit"] == 500
    assert lo.json()["limit"] == 1
    assert seen["limit"] == 1


async def test_dashboard_shows_version():
    async with asgi() as c:
        r = await c.get("/")
    assert r.status_code == 200
    # The running Overlaat version is rendered in the dashboard byline, sourced
    # from overlaat.__version__ (no template placeholder leaks through).
    assert f"v{__version__}" in r.text
    assert "{{OVERLAAT_VERSION}}" not in r.text


async def test_healthz_reports_version(monkeypatch):
    monkeypatch.setattr(mdb, "_connect", lambda db: _FakeConn())
    async with asgi() as c:
        r = await c.get("/healthz")
    assert r.json()["version"] == __version__

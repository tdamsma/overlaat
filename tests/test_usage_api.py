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

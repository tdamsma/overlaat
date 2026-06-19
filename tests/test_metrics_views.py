"""View builders tested with fetch_* mocked — synthetic events, no DB."""

from overlaat import metrics_db as m


def ev(
    model="m1",
    key="k1",
    enq=0.0,
    acq=1.0,
    ft=2.0,
    done=5.0,
    outcome="completed",
    pt=10,
    ct=20,
    streamed=True,
    http=200,
):
    return {
        "t_enqueue": enq,
        "t_acquire": acq,
        "t_first_token": ft,
        "t_done": done,
        "model_requested": model,
        "key_fp": key,
        "streamed": streamed,
        "outcome": outcome,
        "http_status": http,
        "prompt_tokens": pt,
        "completion_tokens": ct,
    }


def test_build_models(monkeypatch):
    events = [
        ev(),
        ev(acq=0.5, ft=1.0, done=3.0, outcome="client_abandoned", ct=None),
        ev(acq=None, ft=None, done=2.0, outcome="cancelled_queued", ct=None),
    ]
    monkeypatch.setattr(m, "fetch_events", lambda *a, **k: events)
    rows = m.build_models("db", 0, 100, {})
    assert len(rows) == 1
    r = rows[0]
    assert r["model"] == "m1"
    assert r["requests"] == 3
    assert r["completed"] == 1
    assert r["abandoned"] == 1
    assert r["cancelled_queued"] == 1
    assert r["latency_ms"]["service_p50"] is not None


def test_build_consumers(monkeypatch):
    events = [ev(), ev(outcome="client_abandoned", ct=None)]
    monkeypatch.setattr(m, "fetch_events", lambda *a, **k: events)
    rows = m.build_consumers("db", 0, 100, {"k1": "alice"})
    assert rows[0]["key"] == "alice"
    assert rows[0]["requests"] == 2
    assert rows[0]["abandoned"] == 1
    assert rows[0]["abandoned_rate"] == 0.5
    assert rows[0]["completion_tokens"] == 20  # only the completed call had a count


def test_build_timeline(monkeypatch):
    monkeypatch.setattr(m, "fetch_events", lambda *a, **k: [ev(enq=0, acq=1, done=5)])
    monkeypatch.setattr(
        m,
        "fetch_host_samples",
        lambda *a, **k: [
            {"ts": 2.0, "gpu_pct": 50, "ram_wired_gb": 4.0, "backends_json": [{"rss_gb": 1.5}]}
        ],
    )
    tl = m.build_timeline("db", 0, 10, 5, {})
    assert "m1" in tl["models"]
    assert len(tl["buckets"]) >= 1
    assert tl["host"]["gpu_pct"][0] == 50
    assert tl["host"]["backend_rss_gb"][0] == 1.5


def test_build_perf_trend(monkeypatch):
    # decode rate = 40 / (4 - 2) = 20 tok/s
    monkeypatch.setattr(m, "fetch_events", lambda *a, **k: [ev(ft=2.0, done=4.0, ct=40)])
    pt = m.build_perf_trend("db", 0, 10, 5, {})
    assert "m1" in pt["models"]
    assert any(x for x in pt["models"]["m1"]["decode_p50"] if x)

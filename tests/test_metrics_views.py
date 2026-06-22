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


def test_sanitize_label():
    assert m.sanitize_label("scout") == "scout"
    assert m.sanitize_label(None) == m.UNTAGGED
    assert m.sanitize_label("") == m.UNTAGGED
    assert m.sanitize_label("   ") == m.UNTAGGED


def test_build_workloads(monkeypatch):
    events = [
        {**ev(enq=0.0, acq=1.0, done=5.0), "workload": "scout"},  # completed, tagged
        {
            **ev(enq=0.0, acq=None, done=2.0, outcome="cancelled_queued", ct=None),
            "workload": "scout",
        },
        {**ev(enq=0.0, acq=1.0, done=3.0), "workload": None},  # untagged bucket
    ]
    monkeypatch.setattr(m, "fetch_events", lambda *a, **k: events)
    rows = m.build_workloads("db", 0, 100)
    by = {r["workload"]: r for r in rows}
    assert set(by) == {"scout", m.UNTAGGED}
    scout = by["scout"]
    assert scout["requests"] == 2
    assert scout["completed"] == 1
    assert scout["cancelled_queued"] == 1
    assert scout["completion_tokens"] == 20  # only the completed call counted
    # Latency percentiles populated from the one completed call (ms).
    assert scout["latency_ms"]["queue_wait_p50"] == 1000  # (1.0-0.0)*1000
    assert scout["latency_ms"]["total_p50"] == 5000  # (5.0-0.0)*1000
    assert by[m.UNTAGGED]["requests"] == 1


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
    # new token-throughput series are present and keyed consistently
    assert "totals" in tl and "in_tok_s" in tl["totals"] and "out_tok_s" in tl["totals"]
    assert "out_models" in tl and "by_model_out_tok_s" in tl
    # one completed call: 20 completion / 10 prompt tokens over service [1,5] (=4s)
    # → tok/s spread over the window, summed across buckets * bucket_s recovers tokens
    bucket_s = tl["bucket_s"]
    assert round(sum(tl["totals"]["out_tok_s"]) * bucket_s) == 20
    assert round(sum(tl["totals"]["in_tok_s"]) * bucket_s) == 10
    # m1's per-model completion tok/s mirrors the aggregate (single model)
    assert "m1" in tl["by_model_out_tok_s"]
    assert tl["models"]["m1"]["out_tok_s"] == tl["by_model_out_tok_s"]["m1"]


def test_build_models_folded_perf_fields(monkeypatch):
    # Two NON-overlapping calls so each runs solo (mean active concurrency 1.0).
    # decode rate = ct / (done - ft); call A: 40/(4-2)=20, call B: 60/(14-12)=30.
    events = [
        ev(enq=0, acq=1, ft=2.0, done=4.0, ct=40),
        ev(enq=10, acq=11, ft=12.0, done=14.0, ct=60),
    ]
    monkeypatch.setattr(m, "fetch_events", lambda *a, **k: events)
    rows = m.build_models("db", 0, 20, {})
    r = rows[0]
    # two solo streamed calls: decode rates 20 and 30 tok/s. _pct(p=0.5) on two
    # values picks index min(int(2*0.5),1)=1 → the upper one, 30.0.
    assert r["decode_solo_tok_s"] == 30.0
    # completion tokens 40 and 60 → _pct(p=0.5) likewise picks the upper, 60.
    assert r["out_tok_p50"] == 60


def test_bucket_weighted():
    # 200 tokens over [0,10) (a full 10s bucket) → 20 tok/s in bucket 0 only.
    assert m._bucket_weighted([(0.0, 10.0, 200)], 0.0, 10.0, 3) == [20.0, 0.0, 0.0]
    # 100 tokens over [5,15) splits 5/5 across the bucket boundary at 10.
    assert m._bucket_weighted([(5.0, 15.0, 100)], 0.0, 10.0, 3) == [5.0, 5.0, 0.0]
    # total == duration reduces to mean concurrency.
    iv = [(0.0, 10.0), (5.0, 15.0)]
    w = [(s, e, e - s) for s, e in iv]
    assert m._bucket_weighted(w, 0.0, 10.0, 3) == m._bucket_concurrency(iv, 0.0, 10.0, 3)
    # degenerate intervals skipped (zero-length, None total, zero total).
    assert m._bucket_weighted([(5.0, 5.0, 9), (0.0, 10.0, None), (0.0, 10.0, 0)], 0.0, 10.0, 3) == [
        0.0,
        0.0,
        0.0,
    ]

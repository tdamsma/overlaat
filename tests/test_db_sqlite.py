"""End-to-end SQLite backend tests — fully hermetic (no services, no network).

These exercise the opt-in `sqlite:` backend against a REAL temp-file database
(not :memory:, so the writer-connection / reader-connection multi-connection
path is realistic). They cover the round-trip from the ACTUAL proxy event-writer
and host_logger SQLite write paths through the metrics_db read layer and the
Python view builders. The default Postgres path and conftest's DATABASE_URL are
untouched — these tests build their own `sqlite:` URL per-test.
"""

import json

import pytest

from overlaat import __version__, db
from overlaat import host_logger as hl
from overlaat import metrics_db as m
from overlaat import queue_proxy as qp


def _sqlite_url(tmp_path):
    return "sqlite:///" + str(tmp_path / "t.db")


def _write_events_via_proxy_path(url, events):
    """Insert request_events through the EXACT SQLite path the proxy writer uses:
    db.connect_sqlite_write + qp._sqlite_write_batch (positional params projected
    from qp._EVENT_COLS)."""
    rows = [{c: e.get(c) for c in qp._EVENT_COLS} for e in events]
    conn = db.connect_sqlite_write(url)
    try:
        qp._sqlite_write_batch(conn, rows)
    finally:
        conn.close()


# ── 1. end-to-end round-trip on a temp-file db ──────────────────────────────────


def test_end_to_end_roundtrip(tmp_path, monkeypatch):
    url = _sqlite_url(tmp_path)
    msg = db.init_db(url)
    assert "sqlite" in msg

    # Two completed streamed calls on model m1, fully overlapping [1,5] → the
    # active concurrency over that window is 2.0; offered also 2.0 since both are
    # enqueued at t=0 and done at t=5.
    events = [
        {
            "t_enqueue": 0.0,
            "t_acquire": 1.0,
            "t_first_token": 2.0,
            "t_done": 5.0,
            "model_requested": "m1",
            "key_fp": "k1",
            "streamed": True,
            "outcome": "completed",
            "http_status": 200,
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "overlaat_version": qp.SERVICE_VERSION,
        },
        {
            "t_enqueue": 0.0,
            "t_acquire": 1.0,
            "t_first_token": 2.0,
            "t_done": 5.0,
            "model_requested": "m1",
            "key_fp": "k2",
            "streamed": True,
            "outcome": "completed",
            "http_status": 200,
            "prompt_tokens": 10,
            "completion_tokens": 40,
            "overlaat_version": qp.SERVICE_VERSION,
        },
    ]
    _write_events_via_proxy_path(url, events)

    # Host samples + a model_load via the ACTUAL host_logger SQLite write path.
    # write_sample_sqlite / write_model_load_sqlite read the module-global DB_URL.
    monkeypatch.setattr(hl, "DB_URL", url)
    hl.write_sample_sqlite(
        {
            "ts_epoch": 2.0,
            "mem": {
                "total_gb": 256.0,
                "wired_gb": 4.0,
                "active_gb": 8.0,
                "inactive_gb": 2.0,
                "compressed_gb": 1.0,
                "free_gb": 100.0,
            },
            "gpu": {"active_pct": 50.0, "freq_mhz": 1000},
            "cpu": {"load1": 1.5, "load5": 2.0},
            "backends": [{"name": "model-server-8086", "rss_gb": 1.5}],
        }
    )
    hl.write_model_load_sqlite("m1", 100.0, 103.0, "starting->ready")

    # Read back through metrics_db.
    got = m.fetch_events(url, since=0.0)
    assert len(got) == 2
    assert {e["model_requested"] for e in got} == {"m1"}
    assert sorted(e["completion_tokens"] for e in got) == [20, 40]

    # The proxy writer fills overlaat_version automatically from overlaat.__version__;
    # every round-tripped request_events row records the running version.
    with db.connect(url) as c, c.cursor() as cur:
        cur.execute("SELECT overlaat_version FROM request_events")
        versions = [r[0] for r in cur.fetchall()]
    assert versions == [__version__, __version__]

    samples = m.fetch_host_samples(url, since=0.0)
    assert len(samples) == 1
    assert samples[0]["gpu_pct"] == 50.0
    assert samples[0]["ram_wired_gb"] == 4.0

    latest = m.latest_host(url)
    assert latest is not None
    assert latest["ts"] == 2.0
    assert latest["gpu_pct"] == 50.0

    # model_loads landed on the same db.
    with db.connect(url) as c, c.cursor() as cur:
        cur.execute("SELECT model, t_start, t_ready, load_s, detail FROM model_loads")
        ml = cur.fetchall()
    assert len(ml) == 1
    assert ml[0][0] == "m1"
    assert ml[0][3] == 3.0  # load_s = t_ready - t_start

    # View builders compute correct concurrency over the read-back events.
    tl = m.build_timeline(url, since=0.0, now=10.0, bucket_s=5.0, aliases={})
    assert "m1" in tl["models"]
    # bucket 0 covers [0,5): both calls offered the whole bucket → 2.0 offered;
    # active starts at t_acquire=1, so 4s of 2-deep over the 5s bucket = 1.6.
    assert tl["models"]["m1"]["offered"][0] == 2.0
    assert tl["models"]["m1"]["active"][0] == 1.6
    # host sample at t=2 lands in bucket 0.
    assert tl["host"]["gpu_pct"][0] == 50.0
    assert tl["host"]["wired_gb"][0] == 4.0
    assert tl["host"]["backend_rss_gb"][0] == 1.5

    models = m.build_models(url, since=0.0, now=10.0, aliases={})
    assert len(models) == 1
    assert models[0]["model"] == "m1"
    assert models[0]["requests"] == 2
    assert models[0]["completed"] == 2

    consumers = m.build_consumers(url, since=0.0, now=10.0, aliases={})
    assert {c["key"] for c in consumers} == {"k1", "k2"}


# ── 1b. workload column survives the proxy → metrics round-trip (#19) ───────────


def test_workload_column_roundtrip(tmp_path):
    """Proves the workload column/param parity end to end: an event carrying a
    `workload` is INSERTed through the EXACT proxy SQLite writer (qp._EVENT_COLS /
    qp._INSERT_SQL_SQLITE) and read back, then grouped by build_workloads. A column
    list ↔ params mismatch would break this write outright."""
    url = _sqlite_url(tmp_path)
    db.init_db(url)

    base = {
        "t_enqueue": 0.0,
        "t_acquire": 1.0,
        "t_first_token": 2.0,
        "t_done": 5.0,
        "model_requested": "m1",
        "key_fp": "k1",
        "streamed": True,
        "outcome": "completed",
        "http_status": 200,
        "prompt_tokens": 10,
        "completion_tokens": 20,
    }
    _write_events_via_proxy_path(
        url,
        [
            {**base, "workload": "scout"},
            {**base, "completion_tokens": 40},  # no workload key → NULL/untagged
        ],
    )

    # The raw column is read back (fetch_events selects `workload`).
    got = m.fetch_events(url, since=0.0)
    assert sorted((e.get("workload") or "") for e in got) == ["", "scout"]

    # Grouped view: tagged + untagged buckets, tokens attributed correctly.
    rows = m.build_workloads(url, since=0.0, now=10.0)
    by = {r["workload"]: r for r in rows}
    assert set(by) == {"scout", m.UNTAGGED}
    assert by["scout"]["completion_tokens"] == 20
    assert by[m.UNTAGGED]["completion_tokens"] == 40


def test_init_db_adds_workload_to_legacy_table(tmp_path):
    """Idempotent upgrade: a request_events table created WITHOUT workload (a
    pre-#19 db) gains the column on the next init_db — the sqlite mirror of
    schema.sql's `ALTER TABLE ... ADD COLUMN IF NOT EXISTS workload TEXT`."""
    url = _sqlite_url(tmp_path)
    conn = db.connect_sqlite_write(url)
    try:
        # Minimal legacy table: no `workload` column.
        conn.execute(
            "CREATE TABLE request_events ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, t_enqueue DOUBLE PRECISION NOT NULL, "
            "t_done DOUBLE PRECISION NOT NULL, model_requested TEXT NOT NULL, "
            "key_fp TEXT NOT NULL, outcome TEXT NOT NULL)"
        )
        conn.commit()
        cols = {r[1] for r in conn.execute("PRAGMA table_info(request_events)")}
        assert "workload" not in cols
    finally:
        conn.close()

    db.init_db(url)  # idempotent upgrade

    conn = db.connect_sqlite_write(url)
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(request_events)")}
        assert "workload" in cols
        db.init_db(url)  # second run is a no-op (no duplicate-column error)
    finally:
        conn.close()


# ── 2. per-backend normalization on SQLite reads ────────────────────────────────


def test_sqlite_normalization_streamed_and_backends_json(tmp_path):
    url = _sqlite_url(tmp_path)
    db.init_db(url)

    _write_events_via_proxy_path(
        url,
        [
            {
                "t_enqueue": 0.0,
                "t_acquire": 1.0,
                "t_first_token": 2.0,
                "t_done": 5.0,
                "model_requested": "m1",
                "key_fp": "k1",
                "streamed": True,
                "outcome": "completed",
                "http_status": 200,
                "prompt_tokens": 1,
                "completion_tokens": 2,
            },
            {
                "t_enqueue": 0.0,
                "t_acquire": 1.0,
                "t_first_token": None,
                "t_done": 5.0,
                "model_requested": "m1",
                "key_fp": "k1",
                "streamed": False,
                "outcome": "completed",
                "http_status": 200,
                "prompt_tokens": 1,
                "completion_tokens": 2,
            },
        ],
    )

    got = m.fetch_events(url, since=0.0)
    # SQLite stores BOOLEAN as int 0/1; metrics_db must return real Python bools.
    streamed_vals = [e["streamed"] for e in got]
    assert True in streamed_vals and False in streamed_vals
    for e in got:
        assert isinstance(e["streamed"], bool)

    # host_samples.backends_json must come back as a parsed object, not a raw str.
    backends = [{"name": "x", "rss_gb": 1.5}, {"name": "y", "rss_gb": 0.5}]
    conn = db.connect_sqlite_write(url)
    try:
        conn.execute(
            "INSERT INTO host_samples (ts, gpu_pct, backends_json) VALUES (?, ?, ?)",
            (3.0, 10.0, json.dumps(backends)),
        )
        conn.commit()
    finally:
        conn.close()

    samples = m.fetch_host_samples(url, since=0.0)
    assert len(samples) == 1
    assert isinstance(samples[0]["backends_json"], list)
    assert samples[0]["backends_json"] == backends

    latest = m.latest_host(url)
    assert isinstance(latest["backends_json"], list)
    assert latest["backends_json"] == backends


# ── 3. dialect selection unit test ──────────────────────────────────────────────


def test_dialect_selection_sqlite_vs_postgres():
    sqlite_url = "sqlite:///./x.db"
    pg_url = "postgresql://u:p@localhost:5432/d"

    assert db.dialect_for(sqlite_url) == "sqlite"
    assert db.dialect_for(pg_url) == "postgres"
    # `postgres://` scheme also resolves to postgres.
    assert db.dialect_for("postgres://u@localhost/d") == "postgres"

    # Placeholder style must differ — guards against PG `%s` silently regressing.
    assert db.placeholder(sqlite_url) == "?"
    assert db.placeholder(pg_url) == "%s"
    assert db.placeholder(sqlite_url) != db.placeholder(pg_url)

    # Leading-substring function must differ between dialects.
    assert db.left_expr(sqlite_url, "token", 8) == "substr(token, 1, 8)"
    assert db.left_expr(pg_url, "token", 8) == "left(token, 8)"
    assert db.left_expr(sqlite_url, "token", 8) != db.left_expr(pg_url, "token", 8)


# ── 4. resolve_key_aliases on SQLite returns {} without error ────────────────────


def test_resolve_key_aliases_sqlite_empty(tmp_path):
    url = _sqlite_url(tmp_path)
    db.init_db(url)  # no LiteLLM_VerificationToken table exists
    assert m.resolve_key_aliases(url) == {}


# ── 5. SQLite pragmas: WAL + busy_timeout on every connection point ──────────────


def _busy_timeout(conn) -> int:
    return conn.execute("PRAGMA busy_timeout").fetchone()[0]


def _journal_mode(conn) -> str:
    return conn.execute("PRAGMA journal_mode").fetchone()[0].lower()


def test_read_connection_sets_busy_timeout_and_wal(tmp_path):
    url = _sqlite_url(tmp_path)
    db.init_db(url)
    with db.connect(url) as conn:
        assert _busy_timeout(conn) == db._SQLITE_BUSY_TIMEOUT_MS
        assert _journal_mode(conn) == "wal"


def test_write_connection_sets_busy_timeout_and_wal(tmp_path):
    url = _sqlite_url(tmp_path)
    db.init_db(url)
    conn = db.connect_sqlite_write(url)
    try:
        assert _busy_timeout(conn) == db._SQLITE_BUSY_TIMEOUT_MS
        assert _journal_mode(conn) == "wal"
    finally:
        conn.close()


def test_host_logger_connection_sets_busy_timeout_and_wal(tmp_path):
    url = _sqlite_url(tmp_path)
    db.init_db(url)
    conn = hl._sqlite_connect(url)
    try:
        # host_logger stays stdlib-only and self-contained, so it hardcodes 5000 ms
        # rather than importing db; assert the concrete value to catch drift.
        assert _busy_timeout(conn) == 5000
        assert _journal_mode(conn) == "wal"
    finally:
        conn.close()


# ── 6. normalize_backends_json warns (and returns None) on malformed JSON ────────


def test_normalize_backends_json_warns_on_bad_json():
    with pytest.warns(UserWarning, match="backends_json"):
        assert db.normalize_backends_json("{not valid json") is None


def test_normalize_backends_json_silent_on_good_input(recwarn):
    assert db.normalize_backends_json('[{"name": "x"}]') == [{"name": "x"}]
    assert db.normalize_backends_json(None) is None
    assert db.normalize_backends_json([{"name": "x"}]) == [{"name": "x"}]
    assert len(recwarn) == 0

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

from overlaat import __version__, db
from overlaat.scheduler import Scheduler, Waiter

UPSTREAM = os.environ.get("QUEUE_PROXY_UPSTREAM", "http://127.0.0.1:4002")
LITELLM_CONFIG = Path(os.environ.get("QUEUE_PROXY_LITELLM_CONFIG", "./litellm-config.yaml"))
METRICS_DB_URL = os.environ.get("METRICS_DB_URL") or os.environ.get("DATABASE_URL") or ""


def _env_flag(name: str, default: bool) -> bool:
    """Truthy-string env flag. Anything in {0,false,off,no} (any case) is False."""
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() not in {"0", "false", "off", "no", ""}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


# Cost-weighted scheduler config (see docs/COST-SCHEDULER.md). The scheduler is
# ON BY DEFAULT; OVERLAAT_SCHEDULER=off is a kill-switch that restores the exact
# per-model asyncio.Semaphore FIFO path. The remaining knobs only matter when ON.
SCHEDULER_ON = _env_flag("OVERLAAT_SCHEDULER", True)
BUDGET = _env_float("OVERLAAT_BUDGET", 1.0)
DEFAULT_COST = _env_float("OVERLAAT_DEFAULT_COST", 1.0)
AGING_RATE = _env_float("OVERLAAT_AGING_RATE", 0.0)
DEFAULT_PRIORITY = _env_int("OVERLAAT_DEFAULT_PRIORITY", 0)
# Reservation grace is accepted as a config knob (eager reservation = 0.0). The
# core scheduler reserves eagerly today; a non-zero grace is reserved for a
# future refinement and is parsed here so the env surface is stable.
RESERVATION_GRACE = _env_float("OVERLAAT_RESERVATION_GRACE", 0.0)
# How often the per-key priority-ceiling cache is refreshed from the LiteLLM
# verification-token table (mirrors the usage-api alias refresh cadence).
KEY_CEILING_REFRESH_S = 60.0
WAIT_BUFFER = 200  # per-model ring buffer for p50/p95
USAGE_TAIL_BYTES = 16384  # rolling tail window scanned for token usage
EVENT_QUEUE_MAX = 10000  # bounded; overflow → drop (instrumentation never blocks)
EVENT_BATCH = 200  # max rows per INSERT batch
PROXIED_PATHS = {
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
    "/v1/rerank",
    "/rerank",
}
SERVICE_VERSION = __version__

_EVENT_COLS = (
    "t_enqueue",
    "t_acquire",
    "t_first_token",
    "t_done",
    "model_requested",
    "key_fp",
    "streamed",
    "outcome",
    "http_status",
    "prompt_tokens",
    "completion_tokens",
    "overlaat_version",
    "priority",
    "cost",
    "wait_reason",
)
# Postgres uses psycopg named params (the writer feeds dict rows via executemany);
# SQLite uses positional `?` params (rows projected to a tuple in _EVENT_COLS order).
_INSERT_SQL_PG = (
    "INSERT INTO request_events "
    "(t_enqueue, t_acquire, t_first_token, t_done, model_requested, key_fp, "
    " streamed, outcome, http_status, prompt_tokens, completion_tokens, overlaat_version, "
    " priority, cost, wait_reason) "
    "VALUES (%(t_enqueue)s, %(t_acquire)s, %(t_first_token)s, %(t_done)s, "
    "%(model_requested)s, %(key_fp)s, %(streamed)s, %(outcome)s, "
    "%(http_status)s, %(prompt_tokens)s, %(completion_tokens)s, %(overlaat_version)s, "
    "%(priority)s, %(cost)s, %(wait_reason)s)"
)
_INSERT_SQL_SQLITE = (
    "INSERT INTO request_events "
    "(t_enqueue, t_acquire, t_first_token, t_done, model_requested, key_fp, "
    " streamed, outcome, http_status, prompt_tokens, completion_tokens, overlaat_version, "
    " priority, cost, wait_reason) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)
# Back-compat alias: the Postgres statement remains the module-level default.
_INSERT_SQL = _INSERT_SQL_PG

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


def load_model_info(path: Path) -> tuple[dict[str, float], dict[str, str]]:
    """Read the scheduler-specific per-model knobs from the LiteLLM config's
    ``model_info`` blocks: an explicit cost override (``overlaat_cost``) and the
    swap-slot group membership (``overlaat_slot``).

    Returns ``(costs, slot_groups)`` keyed by model_name:
      - ``costs[model]``       — explicit GPU-fraction cost override (float > 0).
      - ``slot_groups[model]`` — swap-slot group name; every member is forced to
        cost 1.0 by the scheduler so the "one big model at a time" mutex falls
        out of the budget arithmetic with no separate lock.
    Models without either key simply don't appear (the scheduler then derives
    cost from ``1/cap`` or the configured default).
    """
    if not path.exists():
        return {}, {}
    try:
        cfg = yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}, {}
    costs: dict[str, float] = {}
    slots: dict[str, str] = {}
    for m in cfg.get("model_list", []):
        name = m.get("model_name")
        if not isinstance(name, str):
            continue
        info = m.get("model_info") or {}
        c = info.get("overlaat_cost")
        if isinstance(c, (int, float)) and not isinstance(c, bool) and c > 0:
            costs[name] = float(c)
        slot = info.get("overlaat_slot")
        if isinstance(slot, str) and slot:
            slots[name] = slot
    return costs, slots


CAPS: dict[str, int] = load_caps(LITELLM_CONFIG)
COSTS, SLOT_GROUPS = load_model_info(LITELLM_CONFIG)
SEMAPHORES: dict[str, asyncio.Semaphore] = {}

# The single global scheduler instance (set in lifespan when SCHEDULER_ON). One
# per process, no locks, driven by one event loop — see the never-`--workers N`
# invariant in overlaat/scheduler.py.
SCHED: Scheduler | None = None

# Per-key priority-ceiling cache: key_fp -> max_priority, refreshed periodically
# from the LiteLLM verification-token table (mirrors the usage-api alias cache).
# None ceiling for a key (or an unreachable / SQLite source) falls back to
# OVERLAAT_DEFAULT_PRIORITY inside the scheduler.
_KEY_CEILINGS: dict[str, int] = {}


def key_ceiling(key_fp: str | None) -> int | None:
    """Per-key priority ceiling from the cached map, or None when absent.

    Hot-path safe: a pure in-memory dict lookup, never a DB read. None means
    "no ceiling for this key" and the scheduler then uses OVERLAAT_DEFAULT_PRIORITY.
    """
    if key_fp is None:
        return None
    return _KEY_CEILINGS.get(key_fp)


def load_key_ceilings(db_url: str) -> dict[str, int]:
    """key_fp (token[:8]) -> overlaat_priority, from LiteLLM_VerificationToken
    metadata. Best-effort and OFF the hot path (called only by the refresh task).

    Mirrors metrics_db.resolve_key_aliases: the table lives only in the LiteLLM
    Postgres, so a SQLite backend has none and we short-circuit to empty. Any
    failure (table absent, DB unreachable) returns what we have so far, leaving
    the scheduler to fall back to the default priority — never raises.
    """
    out: dict[str, int] = {}
    if not db_url or db.dialect_for(db_url) == "sqlite":
        return out
    try:
        with db.connect(db_url) as c, c.cursor() as cur:
            cur.execute(
                f"SELECT {db.left_expr(db_url, 'token', 8)}, metadata "
                'FROM "LiteLLM_VerificationToken"'
            )
            for fp, meta in cur.fetchall():
                if not fp:
                    continue
                md = meta
                if isinstance(md, (str, bytes, bytearray)):
                    try:
                        md = json.loads(md)
                    except (ValueError, TypeError):
                        continue
                if isinstance(md, dict):
                    p = md.get("overlaat_priority")
                    if isinstance(p, int) and not isinstance(p, bool):
                        out[fp] = p
    except Exception:
        pass
    return out


async def _key_ceiling_refresher() -> None:
    """Background task: periodically refresh the per-key priority-ceiling cache.

    Runs the (blocking) DB read in a worker thread so the event loop is never
    stalled, then atomically swaps the module-level cache. Graceful on failure:
    load_key_ceilings never raises, and an empty result just means every key
    falls back to OVERLAAT_DEFAULT_PRIORITY."""
    global _KEY_CEILINGS
    while True:
        try:
            _KEY_CEILINGS = await asyncio.to_thread(load_key_ceilings, METRICS_DB_URL)
        except Exception:  # noqa: BLE001 — never let the refresher die
            pass
        await asyncio.sleep(KEY_CEILING_REFRESH_S)


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
    """Background task: drain EVENT_Q and batch-insert lifecycle events.

    Dispatches on the configured backend: Postgres (psycopg AsyncConnection) or
    the opt-in SQLite path. Reconnect on error. The DB is local and expected to
    be up; on a write error we drop the batch (counted) and reconnect for the
    next one — no on-disk fallback. Sentinel None = stop."""
    if db.dialect_for(METRICS_DB_URL) == "sqlite":
        await _event_writer_sqlite()
        return

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
                conn = await psycopg.AsyncConnection.connect(METRICS_DB_URL, autocommit=True)
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


def _sqlite_write_batch(conn, batch: list[dict]) -> None:
    """Blocking: executemany the batch (positional params in _EVENT_COLS order)
    and commit. Runs in a worker thread via asyncio.to_thread."""
    rows = [tuple(row[c] for c in _EVENT_COLS) for row in batch]
    conn.executemany(_INSERT_SQL_SQLITE, rows)
    conn.commit()


async def _event_writer_sqlite() -> None:
    """SQLite sibling of the writer: same bounded-queue draining, EVENT_BATCH
    batching, sentinel-None stop, and drop-and-reconnect error handling as the
    Postgres path, but against a stdlib sqlite3 connection (WAL) driven off the
    event loop via asyncio.to_thread so the blocking insert never stalls it."""
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
            if conn is None:
                conn = await asyncio.to_thread(db.connect_sqlite_write, METRICS_DB_URL)
            await asyncio.to_thread(_sqlite_write_batch, conn, batch)
            EVENT_STATS["written"] += len(batch)
        except Exception as e:  # noqa: BLE001
            EVENT_STATS["dropped"] += len(batch)
            sys.stderr.write(f"event-writer: dropped {len(batch)} ({type(e).__name__}: {e})\n")
            sys.stderr.flush()
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
                conn = None
        if stop:
            break
    if conn is not None:
        try:
            conn.close()
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
        content={
            "error": {
                "message": f"request {req_id} cancelled while queued for {model}",
                "type": "queue_proxy_cancelled",
            }
        },
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global EVENT_Q, SCHED
    app.state.client = httpx.AsyncClient(
        base_url=UPSTREAM,
        timeout=httpx.Timeout(connect=5.0, read=1200.0, write=60.0, pool=5.0),
        limits=httpx.Limits(max_keepalive_connections=64, max_connections=128),
    )
    EVENT_Q = asyncio.Queue(maxsize=EVENT_QUEUE_MAX)
    writer = asyncio.create_task(_event_writer())

    ceiling_task: asyncio.Task | None = None
    if SCHEDULER_ON:
        SCHED = Scheduler(
            budget=BUDGET,
            caps=CAPS,
            costs=COSTS,
            slot_groups=SLOT_GROUPS,
            default_cost=DEFAULT_COST,
            default_priority=DEFAULT_PRIORITY,
            aging_rate=AGING_RATE,
            key_ceiling=key_ceiling,
        )
        ceiling_task = asyncio.create_task(_key_ceiling_refresher())
    yield
    # stop writer: sentinel + drain
    if EVENT_Q is not None:
        await EVENT_Q.put(None)
    try:
        await asyncio.wait_for(writer, timeout=10.0)
    except Exception:
        writer.cancel()
    if ceiling_task is not None:
        ceiling_task.cancel()
    await app.state.client.aclose()


app = FastAPI(
    title="queue-proxy",
    version=SERVICE_VERSION,
    description="FIFO queue sidecar + per-request instrumentation. Front-door entry :4000.",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)


def _scheduler_view() -> dict | None:
    """Budget snapshot for the status/health endpoints, or None when OFF.

    Adds budget_pct (used/B) on top of the core snapshot so a reader doesn't have
    to recompute the utilization. All NULL-safe: returns None if the scheduler is
    not running (kill-switch or pre-lifespan)."""
    if SCHED is None:
        return None
    snap = SCHED.snapshot()
    b = snap.get("budget") or 0.0
    used = snap.get("used") or 0.0
    snap["budget_pct"] = round(used / b * 100, 1) if b > 0 else None
    return snap


@app.get("/__queue/health")
async def health():
    return {
        "status": "ok",
        "version": SERVICE_VERSION,
        "upstream": UPSTREAM,
        "caps": CAPS,
        "scheduler": SCHEDULER_ON,
        "budget": _scheduler_view(),
        "events": dict(EVENT_STATS),
        "event_queue_depth": EVENT_Q.qsize() if EVENT_Q else None,
        "db_configured": bool(METRICS_DB_URL),
    }


@app.get("/__queue/status")
async def status():
    rows = []
    now = time.time()
    # Live Waiter lookup (scheduler ON only) so each queued entry can surface its
    # effective (aged) priority and current wait_reason. NULL-safe when OFF.
    waiters_by_id: dict[str, Waiter] = {}
    sched_now = None
    if SCHED is not None:
        sched_now = SCHED._now()
        waiters_by_id = {w.req_id: w for w in SCHED.waiters}
    for model in sorted(set(CAPS.keys()) | set(METRICS.keys())):
        m = METRICS[model]
        waits = sorted(m["wait_ms_buffer"])
        n = len(waits)
        queued = sorted(QUEUED.get(model, {}).values(), key=lambda i: i["enqueued_at"])
        rows.append(
            {
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
                "queued": [_queued_view(i, now, waiters_by_id, sched_now) for i in queued],
            }
        )
    return {
        "service": "queue-proxy",
        "version": SERVICE_VERSION,
        "scheduler": SCHEDULER_ON,
        "budget": _scheduler_view(),
        "by_model": rows,
        "total_in_flight": sum(m["in_flight"] for m in METRICS.values()),
        "total_queue_depth": sum(m["queue_depth"] for m in METRICS.values()),
        "total_served": sum(m["total_served"] for m in METRICS.values()),
    }


def _queued_view(
    i: dict, now: float, waiters_by_id: dict[str, Waiter], sched_now: float | None
) -> dict:
    """Render one queued entry for /__queue/status.

    Base fields (id/key_fp/age_s) are always present. The scheduler fields
    (priority/effective_priority/cost/wait_reason) are added when the scheduler
    is ON; effective_priority and wait_reason come from the live Waiter so the
    aging shows. All NULL-safe — absent on the kill-switch path."""
    out = {
        "id": i["id"],
        "key_fp": i["key_fp"],
        "age_s": round(now - i["enqueued_at"], 1),
    }
    if "priority" in i:
        out["priority"] = i["priority"]
        out["cost"] = i["cost"]
        w = waiters_by_id.get(i["id"])
        if w is not None and SCHED is not None and sched_now is not None:
            out["effective_priority"] = round(SCHED.effective_priority(w, sched_now), 3)
            out["wait_reason"] = w.wait_reason
        else:
            out["effective_priority"] = None
            out["wait_reason"] = None
    return out


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
        content={
            "error": {
                "message": f"{req_id} not queued (already running, done, or unknown)",
                "type": "queue_proxy_not_found",
            }
        },
    )


_HOP_HEADERS_REQ = {"host", "content-length"}
_HOP_HEADERS_RESP = {"content-encoding", "transfer-encoding", "content-length", "connection"}


async def _forward(
    request: Request,
    full_path: str,
    body: bytes,
    model: str | None,
    sem: asyncio.Semaphore | None,
    ev: dict | None,
    *,
    use_scheduler: bool = False,
):
    """Forward to upstream with slot-release discipline + event emission.

    Release rule: exactly once per acquired slot. With the cost scheduler ON the
    slot is a budget reservation (``sem is None`` and ``use_scheduler`` is True),
    released via ``SCHED.release(model)``; with the scheduler OFF it is the
    per-model semaphore (``sem.release()``). Either way the release is guarded by
    ``released`` so a double release never corrupts the budget ledger or the
    in-flight counters. The event is emitted exactly once (in
    stream_body.finally, or on a send error here)."""
    client: httpx.AsyncClient = request.app.state.client
    headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_HEADERS_REQ}
    upstream_req = client.build_request(
        method=request.method,
        url=full_path,
        content=body,
        headers=headers,
        params=dict(request.query_params),
    )
    released = {"done": False}
    emitted = {"done": False}

    def release(status_code: int):
        if released["done"] or model is None:
            return
        if sem is None and not use_scheduler:
            return
        released["done"] = True
        if use_scheduler:
            if SCHED is not None:
                SCHED.release(model)
        else:
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
                if (
                    ev is not None
                    and ev.get("streamed")
                    and ev.get("t_first_token") is None
                    and b'"content"' in chunk
                ):
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
            outcome = (
                base_outcome
                if (natural or base_outcome == "upstream_error")
                else "client_abandoned"
            )
            finish(upstream_resp.status_code, outcome, bytes(tail))

    resp_headers = {
        k: v for k, v in upstream_resp.headers.items() if k.lower() not in _HOP_HEADERS_RESP
    }
    return StreamingResponse(
        stream_body(),
        status_code=upstream_resp.status_code,
        headers=resp_headers,
        media_type=upstream_resp.headers.get("content-type"),
    )


@app.api_route(
    "/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"]
)
async def proxy(full_path: str, request: Request):
    path = "/" + full_path
    body = await request.body()

    model: str | None = None
    sem: asyncio.Semaphore | None = None
    streamed = False
    is_llm = path in PROXIED_PATHS

    requested_priority: int | None = None
    if is_llm and body:
        try:
            payload = json.loads(body)
            m = payload.get("model")
            if isinstance(m, str):
                model = m
                sem = get_semaphore(m)
            streamed = bool(payload.get("stream"))
            p = payload.get("priority")
            if isinstance(p, int) and not isinstance(p, bool):
                requested_priority = p
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

    # Event skeleton (only for real LLM calls that carry a model). The scheduler
    # columns (priority/cost/wait_reason) default to None — they stay NULL for the
    # no-cap pass-through and for the entire scheduler-OFF path.
    ev: dict | None = None
    if is_llm and model is not None:
        ev = {
            "t_enqueue": time.time(),
            "t_acquire": None,
            "t_first_token": None,
            "model_requested": model,
            "key_fp": _key_fp(request),
            "streamed": streamed,
            "overlaat_version": SERVICE_VERSION,
            "priority": None,
            "cost": None,
            "wait_reason": None,
        }

    if SCHEDULER_ON and model is not None and SCHED is not None:
        return await _admit_scheduler(request, path, body, model, ev, requested_priority)

    # ── scheduler OFF (kill-switch): the original per-model semaphore path ──────
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
        "id": req_id,
        "model": model,
        "key_fp": _key_fp(request),
        "enqueued_at": time.time(),
        "cancel_fut": cancel_fut,
    }
    metrics["queue_depth"] += 1
    enqueue_ts = time.monotonic()
    acquire_task = asyncio.ensure_future(sem.acquire())
    try:
        await asyncio.wait({acquire_task, cancel_fut}, return_when=asyncio.FIRST_COMPLETED)
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
        if got:  # acquired after all → give it back
            sem.release()
        _emit_cancelled()
        return _cancelled_response(req_id, model)
    if cancel_fut.done():  # slot + cancel at once → give the slot back
        sem.release()
        _emit_cancelled()
        return _cancelled_response(req_id, model)

    wait_ms = int((time.monotonic() - enqueue_ts) * 1000)
    metrics["wait_ms_buffer"].append(wait_ms)
    metrics["in_flight"] += 1
    if ev is not None:
        ev["t_acquire"] = time.time()

    return await _forward(request, path, body, model, sem, ev)


async def _admit_scheduler(
    request: Request,
    path: str,
    body: bytes,
    model: str,
    ev: dict | None,
    requested_priority: int | None,
):
    """Cost-weighted global admission (scheduler ON).

    Builds a Waiter, enqueues it against the single global Scheduler, and awaits
    its admission future racing the cancel future (cancel-while-queued only —
    in-flight runs are never cancellable, the no-preemption rule of §2). On
    cancel-win the waiter is withdrawn (clearing any head reservation it held)
    and the 499 / ``cancelled_queued`` outcome is preserved byte-for-byte. On
    admission the slot is a budget reservation released via ``SCHED.release`` in
    ``_forward`` (``use_scheduler=True``), under the same release-once guard.

    Every model is charged a cost when the scheduler is ON — an uncapped model
    is not a silent uncounted pass-through; it costs ``OVERLAAT_DEFAULT_COST`` so
    the shared budget stays honest about the single GPU."""
    sched = SCHED
    assert sched is not None
    key_fp = _key_fp(request)
    base_priority = DEFAULT_PRIORITY if requested_priority is None else requested_priority
    cost = sched.cost(model)

    metrics = METRICS[model]
    req_id = hashlib.sha1(f"{time.monotonic_ns()}{id(request)}".encode()).hexdigest()[:12]
    cancel_fut: asyncio.Future = asyncio.get_event_loop().create_future()
    QUEUED[model][req_id] = {
        "id": req_id,
        "model": model,
        "key_fp": key_fp,
        "enqueued_at": time.time(),
        "cancel_fut": cancel_fut,
        "priority": base_priority,
        "cost": cost,
    }
    metrics["queue_depth"] += 1
    enqueue_ts = time.monotonic()

    waiter = Waiter(
        req_id=req_id,
        model=model,
        cost=cost,
        base_priority=base_priority,
        key_fp=key_fp,
        enqueued_at=sched._now(),
    )
    sched.enqueue(waiter)
    try:
        await asyncio.wait({waiter.fut, cancel_fut}, return_when=asyncio.FIRST_COMPLETED)
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
            ev["priority"] = base_priority
            ev["cost"] = cost
            ev["wait_reason"] = waiter.wait_reason
            emit_event(ev)

    if not waiter.fut.done():
        # cancel won the race → we never got admitted; withdraw (clears any
        # reservation this waiter held) and re-pump so the freed head can flow.
        sched.withdraw(waiter)
        _emit_cancelled()
        return _cancelled_response(req_id, model)
    if cancel_fut.done():
        # Admitted and cancelled at the same instant: the admission already
        # charged the budget, so give it back via release (not withdraw, which is
        # a no-op for an admitted waiter) to keep the ledger correct.
        sched.release(model)
        _emit_cancelled()
        return _cancelled_response(req_id, model)

    wait_ms = int((time.monotonic() - enqueue_ts) * 1000)
    metrics["wait_ms_buffer"].append(wait_ms)
    metrics["in_flight"] += 1
    if ev is not None:
        ev["t_acquire"] = time.time()
        ev["priority"] = base_priority
        ev["cost"] = cost
        ev["wait_reason"] = waiter.wait_reason

    return await _forward(request, path, body, model, None, ev, use_scheduler=True)

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


def _parse_weight_tiers(spec: str) -> tuple[tuple[float, float], ...]:
    """Parse ``OVERLAAT_PROMPT_WEIGHT_TIERS`` into sorted ``(upper, multiplier)``
    pairs. Format: ``"<tok>:<mult>,<tok>:<mult>,inf:<mult>"`` — each pair is an
    inclusive upper bound on estimated prompt tokens and the cost multiplier
    applied at or below it. Falls back to the module default on any parse error
    (never raises — a bad knob must not take the proxy down)."""
    try:
        tiers: list[tuple[float, float]] = []
        for part in spec.split(","):
            tok_s, mult_s = part.split(":")
            tok_s = tok_s.strip().lower()
            upper = float("inf") if tok_s in ("inf", "*") else float(tok_s)
            tiers.append((upper, float(mult_s)))
        tiers.sort(key=lambda t: t[0])
        if tiers and all(m >= 1.0 for _, m in tiers):
            return tuple(tiers)
    except Exception:
        pass
    return _DEFAULT_WEIGHT_TIERS


# Prompt-size-weighted admission cost (#18). A request's cost is its model's base
# cost times the tier multiplier for its estimated prompt size; the scheduler
# then hard-clamps that to the pool budget (per-pool ``heavy_max``). Default tiers
# leave small prompts at 1x and price heavy prompts up so they consume more of the
# pool and fewer run at once. Override with OVERLAAT_PROMPT_WEIGHT_TIERS; set every
# multiplier to 1 (or "inf:1") to disable weighting entirely.
_DEFAULT_WEIGHT_TIERS: tuple[tuple[float, float], ...] = (
    (2000.0, 1.0),  # <= ~2k tokens: interactive, full speed
    (8000.0, 2.0),  # ~2k-8k tokens: medium, 2x
    (float("inf"), 4.0),  # > ~8k tokens: heavy, 4x
)
PROMPT_WEIGHT_TIERS = _parse_weight_tiers(os.environ.get("OVERLAAT_PROMPT_WEIGHT_TIERS", ""))
CHARS_PER_TOKEN = 4  # crude bytes->tokens estimate; coarse on purpose (#18)
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
    "pool",
    "workload",
)
# Postgres uses psycopg named params (the writer feeds dict rows via executemany);
# SQLite uses positional `?` params (rows projected to a tuple in _EVENT_COLS order).
_INSERT_SQL_PG = (
    "INSERT INTO request_events "
    "(t_enqueue, t_acquire, t_first_token, t_done, model_requested, key_fp, "
    " streamed, outcome, http_status, prompt_tokens, completion_tokens, overlaat_version, "
    " priority, cost, wait_reason, pool, workload) "
    "VALUES (%(t_enqueue)s, %(t_acquire)s, %(t_first_token)s, %(t_done)s, "
    "%(model_requested)s, %(key_fp)s, %(streamed)s, %(outcome)s, "
    "%(http_status)s, %(prompt_tokens)s, %(completion_tokens)s, %(overlaat_version)s, "
    "%(priority)s, %(cost)s, %(wait_reason)s, %(pool)s, %(workload)s)"
)
_INSERT_SQL_SQLITE = (
    "INSERT INTO request_events "
    "(t_enqueue, t_acquire, t_first_token, t_done, model_requested, key_fp, "
    " streamed, outcome, http_status, prompt_tokens, completion_tokens, overlaat_version, "
    " priority, cost, wait_reason, pool, workload) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
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


def load_model_info(path: Path) -> tuple[dict[str, float], dict[str, str], dict[str, bool]]:
    """Read the scheduler-specific per-model knobs from the LiteLLM config's
    ``model_info`` blocks: an explicit cost override (``overlaat_cost``), the
    resource-pool assignment (``overlaat_pool``), and the deprecated swap-slot
    alias (``overlaat_slot``).

    Returns ``(costs, pool_of, exclusive_seed)`` keyed by model_name / pool_name:
      - ``costs[model]``    — explicit pool-fraction cost override (float > 0).
      - ``pool_of[model]``  — the named resource pool the model is admitted
        against. A model with no ``overlaat_pool`` is in the ``default`` pool.
      - ``exclusive_seed[pool]`` — True for any pool a model implied must be
        exclusive. This carries the **legacy bridge**: a model with
        ``overlaat_slot: NAME`` and no ``overlaat_pool`` is treated as
        ``overlaat_pool: NAME`` with that pool auto-marked exclusive — so old
        swap-slot configs keep working (a cap-1 slot is byte-identical).
    Models without any of these keys simply don't appear in ``costs`` (the
    scheduler derives cost from ``1/cap`` or the default) and land in ``default``.
    """
    if not path.exists():
        return {}, {}, {}
    try:
        cfg = yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}, {}, {}
    costs: dict[str, float] = {}
    pool_of: dict[str, str] = {}
    exclusive_seed: dict[str, bool] = {}
    for m in cfg.get("model_list", []):
        name = m.get("model_name")
        if not isinstance(name, str):
            continue
        info = m.get("model_info") or {}
        c = info.get("overlaat_cost")
        if isinstance(c, (int, float)) and not isinstance(c, bool) and c > 0:
            costs[name] = float(c)
        explicit_pool = info.get("overlaat_pool")
        legacy_slot = info.get("overlaat_slot")
        if isinstance(explicit_pool, str) and explicit_pool:
            pool_of[name] = explicit_pool
        elif isinstance(legacy_slot, str) and legacy_slot:
            # Legacy bridge: overlaat_slot: NAME -> overlaat_pool: NAME, auto-exclusive.
            pool_of[name] = legacy_slot
            exclusive_seed[legacy_slot] = True
    return costs, pool_of, exclusive_seed


def load_pools(path: Path) -> tuple[dict[str, float], set[str]]:
    """Read the optional top-level ``overlaat.pools`` section of the LiteLLM
    config: per-pool budget + exclusive flag.

    Returns ``(pool_budget, pool_exclusive)``:
      - ``pool_budget[pool]``  — explicit budget ``B_pool`` (float > 0). A pool
        with no budget here uses ``OVERLAAT_BUDGET`` (the auto-pool default).
      - ``pool_exclusive``     — set of pools declared ``exclusive: true``.
    The ``default`` pool is implicit; declare it only to override its budget.
    """
    if not path.exists():
        return {}, set()
    try:
        cfg = yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}, set()
    section = (cfg.get("overlaat") or {}).get("pools") or {}
    budgets: dict[str, float] = {}
    exclusive: set[str] = set()
    if isinstance(section, dict):
        for pool, spec in section.items():
            if not isinstance(pool, str) or not isinstance(spec, dict):
                continue
            b = spec.get("budget")
            if isinstance(b, (int, float)) and not isinstance(b, bool) and b > 0:
                budgets[pool] = float(b)
            if spec.get("exclusive") is True:
                exclusive.add(pool)
    return budgets, exclusive


def load_pool_heavy_max(path: Path) -> dict[str, str]:
    """Read the optional per-pool ``heavy_max`` from ``overlaat.pools`` (#18):
    how much of a pool's budget a single prompt-size-weighted request may take.

    Returns ``{pool: "full_pool" | "leave_room"}`` for pools that declare it; a
    pool absent here defaults to ``"leave_room"`` in the scheduler. An invalid
    value is ignored (logged), so a typo never silently flips the clamp."""
    if not path.exists():
        return {}
    try:
        cfg = yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}
    section = (cfg.get("overlaat") or {}).get("pools") or {}
    out: dict[str, str] = {}
    if isinstance(section, dict):
        for pool, spec in section.items():
            if not isinstance(pool, str) or not isinstance(spec, dict):
                continue
            hm = spec.get("heavy_max")
            if hm in ("full_pool", "leave_room"):
                out[pool] = hm
            elif hm is not None:
                sys.stderr.write(
                    f"queue-proxy: ignoring invalid heavy_max '{hm}' for pool "
                    f"'{pool}' (expected 'full_pool' or 'leave_room')\n"
                )
    return out


def estimate_prompt_tokens(payload: dict) -> int:
    """Cheap prompt-size estimate (chars / CHARS_PER_TOKEN) from a request body.

    Sums the text content of a chat ``messages`` array (handling both the string
    and the multimodal list-of-parts content shapes) or a completions ``prompt``.
    Returns 0 for any body with no measurable prompt (e.g. /embeddings, /rerank),
    which the caller maps to weight 1x. Intentionally coarse — admission cost is a
    blunt knob and a real tokenizer is not worth the proxy hot-path cost (#18)."""
    chars = 0
    msgs = payload.get("messages")
    if isinstance(msgs, list):
        for m in msgs:
            content = m.get("content") if isinstance(m, dict) else None
            if isinstance(content, str):
                chars += len(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        chars += len(part["text"])
    else:
        prompt = payload.get("prompt")
        if isinstance(prompt, str):
            chars += len(prompt)
        elif isinstance(prompt, list):
            chars += sum(len(p) for p in prompt if isinstance(p, str))
    return chars // CHARS_PER_TOKEN


def prompt_weight(est_tokens: int, tiers: tuple[tuple[float, float], ...]) -> float:
    """Cost multiplier for an estimated prompt size, from the tier table. Returns
    the multiplier of the first tier whose inclusive upper bound is >= the
    estimate (tiers are sorted ascending); the last tier's bound is +inf."""
    for upper, mult in tiers:
        if est_tokens <= upper:
            return mult
    return tiers[-1][1] if tiers else 1.0


def resolve_pool_config(
    caps: dict[str, int],
    costs: dict[str, float],
    pool_of: dict[str, str],
    exclusive_seed: dict[str, bool],
    declared_budgets: dict[str, float],
    declared_exclusive: set[str],
    default_budget: float,
    *,
    log: bool = True,
) -> tuple[dict[str, float], set[str]]:
    """Merge declared pools with the pools models reference, auto-creating any
    referenced-but-undeclared pool (non-exclusive, ``default_budget``) and folding
    in the legacy-slot exclusivity seed. Logs each auto-created pool and any model
    whose cost exceeds its pool budget (a never-admit guard).

    Returns the final ``(pool_budget, pool_exclusive)`` to hand the Scheduler.
    Pool budgets are returned for every referenced/declared pool (so the snapshot
    and validation see them); the scheduler still falls back to ``default_budget``
    for any pool it is not told about.
    """
    referenced = set(pool_of.values()) | {"default"}
    pool_budget: dict[str, float] = {}
    pool_exclusive: set[str] = set(declared_exclusive) | {
        p for p, ex in exclusive_seed.items() if ex
    }

    for pool in referenced | set(declared_budgets) | pool_exclusive:
        if pool in declared_budgets:
            pool_budget[pool] = declared_budgets[pool]
        else:
            pool_budget[pool] = default_budget
            # Auto-created = referenced by a model (or seeded exclusive) but never
            # declared with a budget in the overlaat.pools section.
            if log and pool not in declared_budgets and pool != "default":
                exmark = " (exclusive)" if pool in pool_exclusive else ""
                sys.stderr.write(
                    f"queue-proxy: auto-created resource pool '{pool}'{exmark} "
                    f"with budget {default_budget} (OVERLAAT_BUDGET)\n"
                )

    # Never-admit guard: warn loudly if any model's cost exceeds its pool budget.
    if log:
        for model in set(caps) | set(costs) | set(pool_of):
            pool = pool_of.get(model, "default")
            b = pool_budget.get(pool, default_budget)
            if model in costs:
                c = costs[model]
            else:
                cap = caps.get(model)
                c = 1.0 / cap if cap and cap > 0 else None
            if c is not None and c > b + 1e-9:
                sys.stderr.write(
                    f"queue-proxy: WARNING model '{model}' cost {c:g} exceeds its pool "
                    f"'{pool}' budget {b:g} — it can NEVER be admitted\n"
                )
        sys.stderr.flush()

    return pool_budget, pool_exclusive


CAPS: dict[str, int] = load_caps(LITELLM_CONFIG)
COSTS, POOL_OF, _EXCLUSIVE_SEED = load_model_info(LITELLM_CONFIG)
_DECLARED_BUDGETS, _DECLARED_EXCLUSIVE = load_pools(LITELLM_CONFIG)
POOL_BUDGET, POOL_EXCLUSIVE = resolve_pool_config(
    CAPS,
    COSTS,
    POOL_OF,
    _EXCLUSIVE_SEED,
    _DECLARED_BUDGETS,
    _DECLARED_EXCLUSIVE,
    BUDGET,
)
POOL_HEAVY_MAX: dict[str, str] = load_pool_heavy_max(LITELLM_CONFIG)
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


# Per-request workload label (#19): a caller-chosen tag (e.g. "scout" vs
# "synthesis") that segments otherwise-identical key_fp traffic in the dashboard.
# OBSERVABILITY ONLY — it is logged and displayed, never read by the scheduler.
WORKLOAD_HEADER = "x-overlaat-workload"  # lowercased; also in the request-strip set
_WORKLOAD_MAXLEN = 64  # bound the label cardinality (keys, indexes, dashboard rows)


def sanitize_workload(value: object) -> str | None:
    """Normalize a caller-supplied workload label: a non-empty string, stripped
    and truncated to 64 chars. Anything else (missing, non-string, empty after
    strip) → None. Pure — bounds cardinality before the value ever touches a row."""
    if not isinstance(value, str):
        return None
    v = value.strip()
    if not v:
        return None
    return v[:_WORKLOAD_MAXLEN]


def resolve_workload(header_value: object, payload: dict) -> tuple[str | None, bool]:
    """Resolve the request workload (header wins, body `metadata.workload` is the
    fallback) and strip the body sub-key so it never reaches the backend. Mutates
    `payload` in place, popping only `metadata.workload` (the rest of `metadata`
    stays, an emptied `{}` stays `{}`). Returns ``(workload, body_changed)`` where
    `body_changed` is True iff the sub-key was popped → the caller must re-serialize."""
    workload = sanitize_workload(header_value)
    md = payload.get("metadata")
    if isinstance(md, dict) and "workload" in md:
        body_workload = sanitize_workload(md.pop("workload"))
        if workload is None:
            workload = body_workload
        return workload, True
    return workload, False


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
            pool_of=POOL_OF,
            pool_budget=POOL_BUDGET,
            pool_exclusive=POOL_EXCLUSIVE,
            pool_heavy_max=POOL_HEAVY_MAX,
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

    Adds the legacy top-level ``budget_pct`` (used/B for the ``default`` pool) on
    top of the core snapshot so existing readers don't have to recompute it; the
    per-pool ``pools`` map carries each pool's own ``budget_pct``. All NULL-safe:
    returns None if the scheduler is not running (kill-switch or pre-lifespan)."""
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
                "pool": SCHED.pool(model) if SCHED is not None else None,
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
        out["pool"] = i.get("pool")
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


_HOP_HEADERS_REQ = {"host", "content-length", WORKLOAD_HEADER}
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
    prompt_tokens_est = 0
    # Workload label (#19): header wins over the body `metadata.workload` fallback,
    # which is stripped before forwarding so this overlaat-private input never
    # reaches the backend. The header is dropped upstream via _HOP_HEADERS_REQ.
    # Observability only — never read by the scheduler.
    workload = sanitize_workload(request.headers.get(WORKLOAD_HEADER))
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
            prompt_tokens_est = estimate_prompt_tokens(payload)
            workload, body_changed = resolve_workload(request.headers.get(WORKLOAD_HEADER), payload)
            if body_changed:
                body = json.dumps(payload).encode()
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
    # columns (priority/cost/wait_reason/pool) default to None — they stay NULL
    # for the no-cap pass-through and for the entire scheduler-OFF path. `workload`
    # is resolved once here so it flows through every emit path (admit / off /
    # cancelled) via this shared dict.
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
            "pool": None,
            "workload": workload,
        }

    if SCHEDULER_ON and model is not None and SCHED is not None:
        return await _admit_scheduler(
            request, path, body, model, ev, requested_priority, prompt_tokens_est
        )

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
    prompt_tokens_est: int = 0,
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
    # Prompt-size-weighted cost (#18): heavier prompts cost more of the pool, so
    # fewer run at once and the interactive fast lane keeps flowing. The scheduler
    # hard-clamps the weighted cost to the pool budget (per-pool heavy_max).
    weight = prompt_weight(prompt_tokens_est, PROMPT_WEIGHT_TIERS)
    cost = sched.weighted_cost(model, weight)
    pool = sched.pool(model)

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
        "pool": pool,
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
            ev["pool"] = pool
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
        ev["pool"] = pool

    return await _forward(request, path, body, model, None, ev, use_scheduler=True)

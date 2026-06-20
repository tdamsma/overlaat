"""Query layer for the metrics pipeline.

Single source of truth = `request_events` (one row per request lifecycle,
written by the queue sidecar) + `host_samples` (5s host + per-backend RSS,
written by the host sampler). This module turns those two tables into the small,
coherent set of views the dashboard renders. It deliberately does not read the
gateway's own insert-on-completion spend log — that log structurally misses
queued and client-abandoned calls, which is the whole reason this pipeline exists.

Key definitions (each computed once, here):
  queue_wait = t_acquire - t_enqueue        time spent waiting for a slot
  ttft       = t_first_token - t_acquire    prefill/first-token latency (stream only)
  decode     = t_done - t_first_token       streamed decode phase
  service    = t_done - t_acquire           true backend-busy time (slot held)
  total      = t_done - t_enqueue           true user-perceived latency

Concurrency curves (per model), each defined once:
  offered(t) = # requests with t_enqueue <= t < t_done   (demand incl. queued)
  active(t)  = # requests with t_acquire <= t < t_done   (backend busy, slot held)
  queued(t)  = offered(t) - active(t)                     (backlog)
"""

from __future__ import annotations

import bisect
from typing import Any

from overlaat import db

MIN_SAMPLES = 5  # below this, a per-concurrency cell is "insufficient", not a trend
TOP_KEYS = 8  # stacked attribution: top N keys + <other>
MAX_PLAUSIBLE_DECODE = 500  # tok/s hardware ceiling; above this a decode rate is a
# near-zero-window artifact (e.g. a reasoning model whose
# content burst lands at stream end → done≈first_token)

_EVENT_SELECT = (
    "SELECT t_enqueue, t_acquire, t_first_token, t_done, model_requested, "
    "key_fp, streamed, outcome, http_status, prompt_tokens, completion_tokens, "
    "priority, cost, wait_reason "
    "FROM request_events"
)
_EVENT_FIELDS = (
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
    "priority",
    "cost",
    "wait_reason",
)


def _connect(db_url: str):
    return db.connect(db_url)


# ── helpers ───────────────────────────────────────────────────────────────────


def _pct(vals: list[float], p: float) -> float | None:
    xs = sorted(v for v in vals if v is not None)
    if not xs:
        return None
    k = min(int(len(xs) * p), len(xs) - 1)
    return xs[k]


def _mean_concurrency(a: float, b: float, intervals: list[tuple[float, float]]) -> float:
    """Time-weighted mean number of simultaneous intervals over [a, b], counting
    the target itself. = (1/(b-a)) * Σ overlap([a,b],[s,e]). So a call that ran
    entirely alone scores 1.0; one that fully overlapped one other scores 2.0."""
    if b <= a:
        return 0.0
    tot = 0.0
    for s, e in intervals:
        ov = min(b, e) - max(a, s)
        if ov > 0:
            tot += ov
    return tot / (b - a)


class _ConcIntegral:
    """Sweep-line accelerator for time-weighted mean concurrency queries.

    Builds, in O(N log N), the step function C(t) = # of intervals active at t
    from a set of [start, end] intervals, plus its prefix integral. Each later
    `mean(a, b)` query is then O(log N) — identical in value to summing
    `_mean_concurrency` over the same interval set, but without the O(N) rescan
    per query that makes the naive form O(N²) over a whole window."""

    def __init__(self, intervals: list[tuple[float, float]]):
        pts: list[tuple[float, int]] = []
        for s, e in intervals:
            if e > s:
                pts.append((s, 1))
                pts.append((e, -1))
        pts.sort()
        self.xs: list[float] = []  # distinct breakpoint times
        self.seg: list[int] = []  # active count over [xs[k], xs[k+1])
        self.F: list[float] = [0.0]  # F[k] = ∫ from xs[0] to xs[k] of C(t) dt
        run = 0
        i, n = 0, len(pts)
        while i < n:
            t = pts[i][0]
            while i < n and pts[i][0] == t:
                run += pts[i][1]
                i += 1
            if self.xs:
                self.F.append(self.F[-1] + self.seg[-1] * (t - self.xs[-1]))
            self.xs.append(t)
            self.seg.append(run)

    def _integ(self, x: float) -> float:
        """∫ from xs[0] to x of C(t) dt (x clamped to the breakpoint range)."""
        if not self.xs or x <= self.xs[0]:
            return 0.0
        if x >= self.xs[-1]:
            return self.F[-1]
        k = bisect.bisect_right(self.xs, x) - 1
        return self.F[k] + self.seg[k] * (x - self.xs[k])

    def mean(self, a: float, b: float) -> float:
        if b <= a or not self.xs:
            return 0.0
        return (self._integ(b) - self._integ(a)) / (b - a)


def _bucket_concurrency(
    intervals: list[tuple[float, float]], t0: float, bucket_s: float, nbuckets: int
) -> list[float]:
    """Per-bucket time-weighted average concurrency from a set of intervals."""
    secs = [0.0] * nbuckets
    for s, e in intervals:
        s = max(s, t0)
        e = min(e, t0 + nbuckets * bucket_s)
        if e <= s:
            continue
        bi = int((s - t0) // bucket_s)
        while bi < nbuckets:
            bs = t0 + bi * bucket_s
            ov = min(e, bs + bucket_s) - max(s, bs)
            if ov <= 0:
                break
            secs[bi] += ov
            bi += 1
    return [x / bucket_s for x in secs]


# ── fetch ───────────────────────────────────────────────────────────────────


def resolve_key_aliases(db_url: str) -> dict[str, str]:
    """key_fp (sha256(token)[:8]) → key_alias, from the gateway's verification-token
    table (token[:8] == our key_fp). Falls back to the fp itself if unknown.

    This is the one place that reads a gateway-owned table; it is purely cosmetic
    (turns a key fingerprint into a human label) and is best-effort — any failure
    leaves the fingerprint in place rather than breaking a view."""
    out: dict[str, str] = {}
    # LiteLLM_VerificationToken lives only in the LiteLLM Postgres; a SQLite
    # backend has no such table, so there are no aliases to resolve.
    if db.dialect_for(db_url) == "sqlite":
        return out
    try:
        with _connect(db_url) as c, c.cursor() as cur:
            cur.execute(
                f"SELECT {db.left_expr(db_url, 'token', 8)}, COALESCE(key_alias, key_name) "
                'FROM "LiteLLM_VerificationToken"'
            )
            for fp, alias in cur.fetchall():
                if fp:
                    out[fp] = alias or fp
    except Exception:
        pass
    return out


def fetch_events(db_url: str, since: float, until: float | None = None) -> list[dict]:
    """All request events overlapping [since, until]: t_done >= since (and
    t_enqueue <= until if given)."""
    ph = db.placeholder(db_url)
    sql = _EVENT_SELECT + f" WHERE t_done >= {ph}"
    params: list[Any] = [since]
    if until is not None:
        sql += f" AND t_enqueue <= {ph}"
        params.append(until)
    sql += " ORDER BY t_enqueue"
    with _connect(db_url) as c, c.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    out = [dict(zip(_EVENT_FIELDS, r, strict=False)) for r in rows]
    for e in out:
        e["streamed"] = db.normalize_streamed(e["streamed"])
    return out


def fetch_host_samples(db_url: str, since: float) -> list[dict]:
    sql = (
        "SELECT ts, gpu_pct, gpu_freq_mhz, ram_wired_gb, ram_active_gb, "
        "ram_inactive_gb, ram_compressed_gb, ram_free_gb, ram_total_gb, "
        "cpu_load1, backends_json FROM host_samples "
        f"WHERE ts >= {db.placeholder(db_url)} ORDER BY ts"
    )
    fields = (
        "ts",
        "gpu_pct",
        "gpu_freq_mhz",
        "ram_wired_gb",
        "ram_active_gb",
        "ram_inactive_gb",
        "ram_compressed_gb",
        "ram_free_gb",
        "ram_total_gb",
        "cpu_load1",
        "backends_json",
    )
    with _connect(db_url) as c, c.cursor() as cur:
        cur.execute(sql, (since,))
        rows = cur.fetchall()
    out = [dict(zip(fields, r, strict=False)) for r in rows]
    for s in out:
        s["backends_json"] = db.normalize_backends_json(s["backends_json"])
    return out


def latest_host(db_url: str) -> dict | None:
    sql = (
        "SELECT ts, gpu_pct, gpu_freq_mhz, ram_wired_gb, ram_active_gb, "
        "ram_free_gb, ram_total_gb, cpu_load1, backends_json "
        "FROM host_samples ORDER BY ts DESC LIMIT 1"
    )
    fields = (
        "ts",
        "gpu_pct",
        "gpu_freq_mhz",
        "ram_wired_gb",
        "ram_active_gb",
        "ram_free_gb",
        "ram_total_gb",
        "cpu_load1",
        "backends_json",
    )
    with _connect(db_url) as c, c.cursor() as cur:
        cur.execute(sql)
        r = cur.fetchone()
    if not r:
        return None
    out = dict(zip(fields, r, strict=False))
    out["backends_json"] = db.normalize_backends_json(out["backends_json"])
    return out


# ── views ─────────────────────────────────────────────────────────────────────


def build_timeline(
    db_url: str, since: float, now: float, bucket_s: float, aliases: dict[str, str]
) -> dict:
    """Per-bucket concurrency curves (offered/active per model + active per key)
    plus host GPU%/wired + backend RSS. Drives all charts."""
    nbuckets = max(1, int((now - since) // bucket_s) + 1)
    t0 = since
    events = fetch_events(db_url, since - 1, now)

    # per-model interval sets
    models: dict[str, dict] = {}
    keys_active: dict[str, list] = {}
    for e in events:
        m = e["model_requested"]
        md = models.setdefault(m, {"offered": [], "active": []})
        md["offered"].append((e["t_enqueue"], e["t_done"]))
        if e["t_acquire"] is not None:
            md["active"].append((e["t_acquire"], e["t_done"]))
            alias = aliases.get(e["key_fp"], e["key_fp"])
            keys_active.setdefault(alias, []).append((e["t_acquire"], e["t_done"]))

    bucket_ts = [round(t0 + i * bucket_s, 3) for i in range(nbuckets)]

    model_series = {}
    for m, md in models.items():
        model_series[m] = {
            "offered": [
                round(x, 3) for x in _bucket_concurrency(md["offered"], t0, bucket_s, nbuckets)
            ],
            "active": [
                round(x, 3) for x in _bucket_concurrency(md["active"], t0, bucket_s, nbuckets)
            ],
        }

    # top keys by total active-seconds
    key_totals = {k: sum(e - s for s, e in iv) for k, iv in keys_active.items()}
    ranked = sorted(key_totals, key=lambda k: key_totals[k], reverse=True)
    top = ranked[:TOP_KEYS]
    has_other = len(ranked) > TOP_KEYS
    key_series = {
        k: [round(x, 3) for x in _bucket_concurrency(keys_active[k], t0, bucket_s, nbuckets)]
        for k in top
    }
    if has_other:
        other_iv = [iv for k in ranked[TOP_KEYS:] for iv in keys_active[k]]
        key_series["<other>"] = [
            round(x, 3) for x in _bucket_concurrency(other_iv, t0, bucket_s, nbuckets)
        ]

    # host samples → align to buckets (last sample wins per bucket)
    hs = fetch_host_samples(db_url, since - bucket_s)
    gpu = [None] * nbuckets
    wired = [None] * nbuckets
    rss_total = [None] * nbuckets
    for s in hs:
        bi = int((s["ts"] - t0) // bucket_s)
        if 0 <= bi < nbuckets:
            gpu[bi] = s["gpu_pct"]
            wired[bi] = s["ram_wired_gb"]
            bj = s.get("backends_json") or []
            rss_total[bi] = round(sum(b.get("rss_gb", 0) for b in bj), 1) if bj else None

    return {
        "bucket_s": bucket_s,
        "buckets": bucket_ts,
        "models": model_series,
        "keys": list(key_series.keys()),
        "by_key_active": key_series,
        "host": {"gpu_pct": gpu, "wired_gb": wired, "backend_rss_gb": rss_total},
    }


def build_models(db_url: str, since: float, now: float, aliases: dict[str, str]) -> list[dict]:
    """Per-model capacity table: outcome counts, latency split, and throughput
    bucketed by the MEASURED mean concurrency each completed call experienced
    (min-sample guarded)."""
    events = fetch_events(db_url, since, now)
    by_model: dict[str, list] = {}
    for e in events:
        by_model.setdefault(e["model_requested"], []).append(e)

    out = []
    for model, evs in sorted(by_model.items()):
        active_iv = [(e["t_acquire"], e["t_done"]) for e in evs if e["t_acquire"] is not None]
        conc = _ConcIntegral(active_iv)
        outcomes: dict[str, int] = {}
        for e in evs:
            outcomes[e["outcome"]] = outcomes.get(e["outcome"], 0) + 1

        completed = [e for e in evs if e["outcome"] == "completed"]
        # latency distributions (completed only)
        qwait = [(e["t_acquire"] - e["t_enqueue"]) * 1000 for e in completed if e["t_acquire"]]
        ttft = [
            (e["t_first_token"] - e["t_acquire"]) * 1000
            for e in completed
            if e["t_first_token"] and e["t_acquire"]
        ]
        service = [(e["t_done"] - e["t_acquire"]) for e in completed if e["t_acquire"]]
        total = [(e["t_done"] - e["t_enqueue"]) for e in completed]

        # throughput by measured concurrency
        conc_cells: dict[int, dict] = {}
        for e in completed:
            if e["t_acquire"] is None:
                continue
            svc = e["t_done"] - e["t_acquire"]
            if svc <= 0:
                continue
            mc = conc.mean(e["t_acquire"], e["t_done"])
            n = max(1, round(mc))
            cell = conc_cells.setdefault(n, {"n": 0, "tok": 0, "svc": 0.0, "decrates": []})
            cell["n"] += 1
            if e["completion_tokens"]:
                cell["tok"] += e["completion_tokens"]
                cell["svc"] += svc
                if e["t_first_token"] and e["t_done"] > e["t_first_token"]:
                    cell["decrates"].append(
                        e["completion_tokens"] / (e["t_done"] - e["t_first_token"])
                    )

        throughput = []
        for n in sorted(conc_cells):
            c = conc_cells[n]
            enough = c["n"] >= MIN_SAMPLES
            throughput.append(
                {
                    "concurrency": n,
                    "calls": c["n"],
                    "aggregate_tok_s": round(c["tok"] / c["svc"], 1)
                    if (enough and c["svc"] > 0)
                    else None,
                    "decode_tok_s_p50": round(_pct(c["decrates"], 0.5), 1)
                    if (enough and c["decrates"])
                    else None,
                    "sufficient": enough,
                }
            )

        out.append(
            {
                "model": model,
                "requests": len(evs),
                "outcomes": outcomes,
                "completed": len(completed),
                "abandoned": outcomes.get("client_abandoned", 0),
                "errored": outcomes.get("upstream_error", 0),
                "cancelled_queued": outcomes.get("cancelled_queued", 0),
                "latency_ms": {
                    "queue_wait_p50": round(_pct(qwait, 0.5)) if qwait else None,
                    "queue_wait_p95": round(_pct(qwait, 0.95)) if qwait else None,
                    "ttft_p50": round(_pct(ttft, 0.5)) if ttft else None,
                    "service_p50": round(_pct([s * 1000 for s in service], 0.5))
                    if service
                    else None,
                    "service_p95": round(_pct([s * 1000 for s in service], 0.95))
                    if service
                    else None,
                    "total_p50": round(_pct([s * 1000 for s in total], 0.5)) if total else None,
                    "total_p95": round(_pct([s * 1000 for s in total], 0.95)) if total else None,
                },
                "throughput_by_concurrency": throughput,
            }
        )
    out.sort(key=lambda m: m["requests"], reverse=True)
    return out


def build_perf_trend(
    db_url: str, since: float, now: float, bucket_s: float, aliases: dict[str, str]
) -> dict:
    """Per-model decode throughput (tok/s) over time — the monitoring view for
    server-health drift (e.g. an inference server that gets slower over long
    uptime; a sustained drop on the solo line is the signal to restart it).

    decode_tok_s = completion_tokens / (t_done - t_first_token), for streamed
    completed calls only (bucketed by t_done). To separate SERVER HEALTH from
    concurrency load, each call's time-weighted mean active-concurrency is
    computed and we report both an all-calls median and a SOLO median (mean
    concurrency < 1.5). The solo line is the clean health signal: a sustained
    drop = degradation, independent of how busy the box was.

    Thinking-mode models (first token lands in reasoning_content, so
    t_first_token is NULL) have no decode window and don't appear here."""
    nbuckets = max(1, int((now - since) // bucket_s) + 1)
    t0 = since
    events = fetch_events(db_url, since, now)

    by_model: dict[str, list] = {}
    for e in events:
        by_model.setdefault(e["model_requested"], []).append(e)

    out_models: dict[str, dict] = {}
    for model, evs in by_model.items():
        active_iv = [(e["t_acquire"], e["t_done"]) for e in evs if e["t_acquire"] is not None]
        conc = _ConcIntegral(active_iv)
        all_b: list[list[float]] = [[] for _ in range(nbuckets)]
        solo_b: list[list[float]] = [[] for _ in range(nbuckets)]
        tok_b: list[list[float]] = [[] for _ in range(nbuckets)]
        for e in evs:
            if e["outcome"] != "completed":
                continue
            ft, dn, ct = e["t_first_token"], e["t_done"], e["completion_tokens"]
            bi = int((dn - t0) // bucket_s)
            if not (0 <= bi < nbuckets):
                continue
            # Output-size trend: ALL completed calls with a token count, streamed or
            # not. A step change here is a behavioural shift (thinking-mode toggle,
            # prompt change, caller change) that is invisible in the tok/s lines.
            if ct is not None:
                tok_b[bi].append(ct)
            if not e["streamed"] or ft is None or ct is None or dn <= ft:
                continue
            rate = ct / (dn - ft)
            if rate > MAX_PLAUSIBLE_DECODE:  # near-zero-window artifact, not a real rate
                continue
            all_b[bi].append(rate)
            if e["t_acquire"] is not None and conc.mean(e["t_acquire"], dn) < 1.5:
                solo_b[bi].append(rate)
        if not any(all_b) and not any(tok_b):
            continue
        out_models[model] = {
            "decode_p50": [round(_pct(x, 0.5), 1) if x else None for x in all_b],
            "decode_solo_p50": [round(_pct(x, 0.5), 1) if x else None for x in solo_b],
            "comp_tok_p50": [round(_pct(x, 0.5)) if x else None for x in tok_b],
            "n": [len(x) for x in all_b],
            "n_tok": [len(x) for x in tok_b],
        }

    return {
        "bucket_s": bucket_s,
        "buckets": [round(t0 + i * bucket_s, 3) for i in range(nbuckets)],
        "models": out_models,
    }


def build_consumers(db_url: str, since: float, now: float, aliases: dict[str, str]) -> list[dict]:
    """Per consumer (key_alias): requests by outcome, tokens, service-seconds,
    abandoned rate."""
    events = fetch_events(db_url, since, now)
    by_key: dict[str, dict] = {}
    for e in events:
        alias = aliases.get(e["key_fp"], e["key_fp"])
        d = by_key.setdefault(
            alias,
            {
                "key": alias,
                "requests": 0,
                "completed": 0,
                "abandoned": 0,
                "errored": 0,
                "cancelled_queued": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "service_s": 0.0,
                "models": {},
            },
        )
        d["requests"] += 1
        oc = e["outcome"]
        if oc == "completed":
            d["completed"] += 1
        elif oc == "client_abandoned":
            d["abandoned"] += 1
        elif oc == "upstream_error":
            d["errored"] += 1
        elif oc == "cancelled_queued":
            d["cancelled_queued"] += 1
        if e["prompt_tokens"]:
            d["prompt_tokens"] += e["prompt_tokens"]
        if e["completion_tokens"]:
            d["completion_tokens"] += e["completion_tokens"]
        if e["t_acquire"] is not None:
            d["service_s"] += max(0.0, e["t_done"] - e["t_acquire"])
        d["models"][e["model_requested"]] = d["models"].get(e["model_requested"], 0) + 1

    out = []
    for d in by_key.values():
        d["service_s"] = round(d["service_s"], 1)
        d["abandoned_rate"] = round(d["abandoned"] / d["requests"], 3) if d["requests"] else 0
        out.append(d)
    out.sort(key=lambda d: d["service_s"], reverse=True)
    return out

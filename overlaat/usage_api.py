"""Overlaat usage-API — read-only FastAPI dashboard + JSON endpoints.

Reads the metrics pipeline (request_events + host_samples) and serves a small,
coherent set of views. It does NOT read the gateway's own insert-on-completion
spend log; it reads only the lifecycle events the queue emits (which include
queued and client-abandoned calls that completion-only logging structurally
misses). Bind to a trusted interface — there is no auth here; the network ACL
is the gate.

Endpoints, each defined by the question it answers:

  GET /              dashboard (HTML)
  GET /now           live: per-model in_flight/queued, host GPU%/wired + backend RSS
  GET /timeline      time-series: host + offered/active concurrency, by-consumer load,
                     input/output tok/s, completion tok/s per model
  GET /models        capacity: outcome counts, latency split, throughput-by-concurrency,
                     solo decode tok/s + p50 output tokens (backend-health + behaviour)
  GET /consumers     per key_alias: requests, tokens, service-seconds, abandoned rate
  GET /workloads     per workload label: requests, latency p50/p95, tokens, error rate
  GET /requests      most recent N requests (flat rows) for the searchable table
  GET /healthz

Live in-flight comes from the queue's in-memory state (its /__queue/status
endpoint), because request events are written on completion. Historical views
come from the two tables. Each metric has exactly one definition (see
metrics_db.py).
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from datetime import UTC, datetime

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse

from overlaat import __version__, metrics_db

DB = os.environ.get("DATABASE_URL", "")
QUEUE_STATUS_URL = os.environ.get("QUEUE_STATUS_URL", "http://127.0.0.1:4000/__queue/status")
SERVICE_VERSION = __version__

app = FastAPI(title="overlaat-usage-api", version=SERVICE_VERSION, docs_url="/swagger")

# ── small helpers ─────────────────────────────────────────────────────────────

_UNITS = {"m": 60, "h": 3600, "d": 86400}


def parse_window(s: str) -> int:
    try:
        return int(s[:-1]) * _UNITS[s[-1]]
    except Exception:
        return 1800


def pick_bucket(win_s: int) -> int:
    if win_s <= 1800:
        return 5
    if win_s <= 3600:
        return 15
    if win_s <= 6 * 3600:
        return 60
    if win_s <= 24 * 3600:
        return 300
    if win_s <= 7 * 86400:
        return 3600
    return 86400


def _meta(kind: str, ttl: int) -> dict:
    return {
        "service": "overlaat-usage-api",
        "version": SERVICE_VERSION,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kind": kind,
        "cache_ttl_hint_s": ttl,
    }


_ALIAS_CACHE = {"ts": 0.0, "map": {}}


def alias_map() -> dict[str, str]:
    now = time.time()
    if now - _ALIAS_CACHE["ts"] > 60:
        _ALIAS_CACHE["map"] = metrics_db.resolve_key_aliases(DB)
        _ALIAS_CACHE["ts"] = now
    return _ALIAS_CACHE["map"]


def scrape_queue_status() -> dict:
    try:
        with urllib.request.urlopen(QUEUE_STATUS_URL, timeout=1.5) as r:
            return json.loads(r.read().decode())
    except Exception as e:  # noqa: BLE001
        return {"available": False, "error": f"{type(e).__name__}: {e}", "by_model": []}


# ── endpoints ───────────────────────────────────────────────────────────────


@app.get("/healthz")
def healthz():
    ok_db = False
    try:
        with metrics_db._connect(DB) as c, c.cursor() as cur:
            cur.execute("SELECT 1")
            ok_db = cur.fetchone()[0] == 1
    except Exception:
        ok_db = False
    return {"ok": ok_db, "service": "overlaat-usage-api", "version": SERVICE_VERSION, "db": ok_db}


@app.get("/now")
def now():
    """Live snapshot. In-flight/queued from the queue; host from the latest
    sample; recent 5-min completed events per key for context.

    Live proxy state — *not* DB events. The `queued` list (per model row, plus a
    flat top-level `queued` across all models) is the proxy's in-memory
    wait-queue: each waiter is a request *parked right now*, with its consumer
    alias resolved and (when the scheduler is on) `priority`,
    `effective_priority`, `pool`, and `wait_reason` (`model_cap` / `budget_full`
    / `exclusive`, or null on the scheduler kill-switch). Distinct from the
    recent-requests / event tables, which only hold *finished* requests."""
    aliases = alias_map()
    qs = scrape_queue_status()
    host = metrics_db.latest_host(DB)

    models = []
    busy = []
    all_queued = []
    for m in qs.get("by_model", []):
        if not (m.get("in_flight") or m.get("queue_depth") or m.get("cap")):
            continue
        queued = []
        for q in m.get("queued", []):
            # Enrich each waiter, null-safe: _queued_view omits the scheduler
            # fields when the scheduler kill-switch is on, so .get() them.
            item = {
                "key": aliases.get(q["key_fp"], q["key_fp"]),
                "model": m["model"],
                "id": q.get("id"),
                "age_s": q.get("age_s"),
                "priority": q.get("priority"),
                "effective_priority": q.get("effective_priority"),
                "pool": q.get("pool"),
                "wait_reason": q.get("wait_reason"),
            }
            queued.append(item)
            all_queued.append(item)
        row = {
            "model": m["model"],
            "cap": m.get("cap"),
            "in_flight": m.get("in_flight", 0),
            "queue_depth": m.get("queue_depth", 0),
            "queued": queued,
            "wait_ms_p50": m.get("wait_ms_p50"),
            "wait_ms_p95": m.get("wait_ms_p95"),
        }
        models.append(row)
        if m.get("in_flight"):
            busy.append(f"{m['model']} (x{m['in_flight']})")

    # recent 5-min completed per key
    recent = {}
    try:
        for e in metrics_db.fetch_events(DB, time.time() - 300):
            if e["outcome"] != "completed":
                continue
            a = aliases.get(e["key_fp"], e["key_fp"])
            d = recent.setdefault(a, {"key": a, "calls": 0, "completion_tokens": 0})
            d["calls"] += 1
            if e["completion_tokens"]:
                d["completion_tokens"] += e["completion_tokens"]
    except Exception:
        pass

    gpu_pct = host["gpu_pct"] if host else None
    backends = (host.get("backends_json") or []) if host else []
    verdict = (
        f"GPU {gpu_pct:.0f}% — active: {', '.join(busy)}"
        if busy and gpu_pct is not None
        else (
            f"GPU {gpu_pct:.0f}% — no slots in flight" if gpu_pct is not None else "no host sample"
        )
    )
    # Slots held but the GPU is idle = a wedged backend (some engines can freeze
    # while still holding their slot), not a load question. Surface it as a
    # first-class state instead of leaving the join of gpu_pct and in_flight to
    # the reader.
    stall = bool(busy) and gpu_pct is not None and gpu_pct < 5
    if stall:
        verdict = f"⚠ STALL? {verdict} — slots held but GPU idle (wedged backend?)"

    return {
        "_meta": _meta("now", 3),
        "verdict": verdict,
        "stall": stall,
        "queue_available": qs.get("available", True),
        "scheduler": qs.get("scheduler"),
        "budget": qs.get("budget"),
        "models": models,
        # Flat list of every queued waiter across all models (same dicts as the
        # per-model `queued`). Live in-memory proxy state — used by the
        # "queued now — by user" card; null-safe scheduler fields.
        "queued": all_queued,
        "totals": {
            "in_flight": qs.get("total_in_flight", 0),
            "queue_depth": qs.get("total_queue_depth", 0),
        },
        "host": {
            "gpu_pct": gpu_pct,
            "gpu_freq_mhz": host["gpu_freq_mhz"] if host else None,
            "wired_gb": host["ram_wired_gb"] if host else None,
            "free_gb": host["ram_free_gb"] if host else None,
            "total_gb": host["ram_total_gb"] if host else None,
            "sample_age_s": round(time.time() - host["ts"], 1) if host else None,
            "backends": sorted(backends, key=lambda b: -b.get("rss_gb", 0))[:12],
        },
        "recent_5m_by_key": sorted(recent.values(), key=lambda d: -d["calls"]),
        "caveat": "Live in-flight from the queue; tokens/latency appear once a "
        "call completes. Per-process GPU is unmeasurable on macOS for "
        "Metal/MLX workloads — memory is attributed by RSS, GPU% is "
        "host-wide.",
    }


@app.get("/timeline")
def timeline(last: str = Query("30m")):
    win = parse_window(last)
    bucket = pick_bucket(win)
    now_ts = time.time()
    series = metrics_db.build_timeline(DB, now_ts - win, now_ts, bucket, alias_map())
    return {"_meta": _meta("timeline", 5), "window": {"last": last, "bucket_s": bucket}, **series}


@app.get("/models")
def models(last: str = Query("24h")):
    win = parse_window(last)
    now_ts = time.time()
    rows = metrics_db.build_models(DB, now_ts - win, now_ts, alias_map())
    return {
        "_meta": _meta("models", 30),
        "window": {"last": last},
        "min_samples": metrics_db.MIN_SAMPLES,
        "models": rows,
        "notes": [
            "concurrency = time-weighted mean # simultaneous slot-holders a "
            "completed call experienced over [acquire, done] (1.0 = ran alone).",
            "aggregate_tok_s = Σ completion_tokens / Σ service_s within the cell; "
            f"shown only when calls >= {metrics_db.MIN_SAMPLES} (else 'sufficient'=false).",
            "latency: queue_wait/ttft/service/total split, completed calls only.",
            "solo decode tok/s = median completion_tokens / (t_done - t_first_token) "
            "over near-solo (mean concurrency < 1.5) streamed completed calls — the "
            "backend-health number; a sustained drop with no load = degradation.",
            "p50 out tok = median completion tokens per completed call (behaviour/"
            "output-size; a step change = thinking-mode toggle or prompt change).",
        ],
    }


@app.get("/consumers")
def consumers(last: str = Query("24h")):
    win = parse_window(last)
    now_ts = time.time()
    rows = metrics_db.build_consumers(DB, now_ts - win, now_ts, alias_map())
    return {"_meta": _meta("consumers", 30), "window": {"last": last}, "consumers": rows}


@app.get("/workloads")
def workloads(last: str = Query("24h")):
    win = parse_window(last)
    now_ts = time.time()
    rows = metrics_db.build_workloads(DB, now_ts - win, now_ts)
    return {"_meta": _meta("workloads", 30), "window": {"last": last}, "workloads": rows}


@app.get("/requests")
def requests(limit: int = Query(100)):
    """The most recent `limit` requests (newest first) as flat rows for the
    dashboard's searchable / sortable / filterable table. `limit` is clamped to
    [1, 500] — the table is client-side, so a few hundred rows is plenty."""
    n = max(1, min(limit, 500))
    rows = metrics_db.build_recent_requests(DB, n, alias_map())
    return {"_meta": _meta("requests", 5), "limit": n, "requests": rows}


@app.get("/", response_class=HTMLResponse)
def dashboard():
    # Single-sourced from overlaat.__version__ (the running version), substituted
    # into the static template's footer byline at request time.
    return DASHBOARD_HTML.replace("{{OVERLAAT_VERSION}}", SERVICE_VERSION)


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Overlaat metrics</title>
<style>
:root{--bg:#0d1117;--panel:#161b22;--line:#30363d;--text:#e6edf3;--dim:#8b949e;
--accent:#58a6ff;--ok:#3fb950;--warn:#d29922;--hot:#f85149;--mono:ui-monospace,SFMono-Regular,Menlo,monospace;
--sans:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--text);font-family:var(--sans);font-size:14px}
header{display:flex;align-items:center;gap:16px;padding:12px 20px;border-bottom:1px solid var(--line);flex-wrap:wrap}
header h1{margin:0;font-size:16px;font-weight:600}
header .verdict{color:var(--accent);font-family:var(--mono);font-size:13px}
header .spacer{flex:1}
select,button{background:var(--panel);color:var(--text);border:1px solid var(--line);padding:6px 10px;border-radius:6px;font:inherit;font-size:12px;cursor:pointer}
main{padding:16px 20px;display:grid;gap:16px;grid-template-columns:repeat(12,1fr)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}
.card h2{margin:0 0 10px;font-size:12px;font-weight:600;text-transform:uppercase;color:var(--dim);letter-spacing:.5px}
.span12{grid-column:span 12}.span6{grid-column:span 6}.span4{grid-column:span 4}.span8{grid-column:span 8}
.kpis{display:flex;gap:24px;flex-wrap:wrap}
.kpi .v{font-family:var(--mono);font-size:24px}.kpi .l{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.5px}
.kpi .v.hot{color:var(--hot)}.kpi .v.warn{color:var(--warn)}.kpi .v.ok{color:var(--ok)}
table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:12px}
th{text-align:left;padding:6px;border-bottom:1px solid var(--line);color:var(--dim);font-weight:500;font-size:11px;text-transform:uppercase}
td{padding:6px;border-bottom:1px solid #21262d}
td.num,th.num{text-align:right}
tr.active td{background:rgba(63,185,80,.10)}
tr.active td:first-child{box-shadow:inset 3px 0 0 var(--ok);font-weight:600}
.dim{color:var(--dim)}.hot{color:var(--hot)}.warn{color:var(--warn)}.ok{color:var(--ok)}
svg.chart{width:100%;height:150px;display:block}
.tick{stroke:#21262d;stroke-width:1}
.axis{fill:var(--dim);font-size:10px;font-family:var(--mono)}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:11px;color:var(--dim);margin-top:6px}
.legend span span{display:inline-block;width:10px;height:10px;border-radius:2px;vertical-align:middle;margin-right:4px}
.note{font-size:11px;color:var(--dim);margin-top:6px}
.bar{height:6px;background:#21262d;border-radius:3px;overflow:hidden}.bar>i{display:block;height:100%;background:var(--accent)}
.pill{display:inline-block;padding:1px 6px;border-radius:10px;font-size:10px;margin-left:4px}
.pill.ab{background:rgba(248,81,73,.15);color:var(--hot)}.pill.er{background:rgba(210,153,34,.15);color:var(--warn)}
.err{color:var(--hot);font-family:var(--mono);font-size:12px}
.quser{font-family:var(--mono);font-size:12px;margin-bottom:8px}
.quser .qhead{display:flex;justify-content:space-between;gap:8px}
.quser .qhead b{font-weight:600}
.quser ul{list-style:none;margin:3px 0 0;padding:0 0 0 10px;border-left:1px solid var(--line)}
.quser li{padding:1px 0;color:var(--dim);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.quser li .qm{color:var(--text)}
footer{color:var(--dim);font-size:11px;text-align:center;padding:14px}
.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:10px}
.controls input{background:#0d1117;color:var(--text);border:1px solid var(--line);padding:6px 10px;border-radius:6px;font:inherit;font-size:12px;min-width:200px}
.controls .count{color:var(--dim);font-size:11px;margin-left:auto}
table.sortable th{cursor:pointer;user-select:none;white-space:nowrap}
table.sortable th:hover{color:var(--text)}
table.sortable th .ind{color:var(--accent);margin-left:3px}
</style></head><body>
<header>
  <h1>Overlaat metrics</h1>
  <span class="dim" style="font-family:var(--mono);font-size:11px">v{{OVERLAAT_VERSION}}</span>
  <span class="verdict" id="verdict">…</span>
  <span class="spacer"></span>
  <label class="dim" style="font-size:11px">window
    <select id="window">
      <option value="30m">30m</option><option value="1h">1h</option>
      <option value="6h" selected>6h</option><option value="24h">24h</option>
      <option value="7d">7d</option>
    </select>
  </label>
  <button id="refresh">refresh</button>
</header>
<main>
  <div class="card span12"><div class="kpis" id="kpis"></div></div>

  <div class="card span8">
    <h2>work by customer <span class="dim">(where the load comes from — GPU-busy share by consumer)</span></h2>
    <svg id="concchart" class="chart" preserveAspectRatio="none"></svg>
    <div class="legend" id="conc-legend"></div>
    <div class="note">y = service-seconds per second (active concurrency) attributed to the consumer holding the slot, stacked. This is the share of GPU-busy time each customer is responsible for; includes abandoned calls until the slot was released. Step segments = one flat value per bucket.</div>
  </div>

  <div class="card span4">
    <h2>host — GPU% &amp; wired RAM</h2>
    <svg id="hostchart" class="chart" preserveAspectRatio="none"></svg>
    <div class="legend">
      <span><span style="background:#f85149"></span>GPU %</span>
      <span><span style="background:#58a6ff"></span>wired GB / total</span>
    </div>
  </div>

  <div class="card span6">
    <h2>throughput — input vs output <span class="dim">(tok/s, all models)</span></h2>
    <svg id="iochart" class="chart" preserveAspectRatio="none"></svg>
    <div class="legend" id="io-legend"></div>
    <div class="note">total prompt (input) and completion (output) tok/s across all models, time-weighted over each call's service window [acquire, done]. Input ≫ output is normal for prompt-heavy workloads. Step segments = one flat value per bucket.</div>
  </div>

  <div class="card span6">
    <h2>output throughput by model <span class="dim">(completion tok/s, stacked)</span></h2>
    <svg id="outchart" class="chart" preserveAspectRatio="none"></svg>
    <div class="legend" id="out-legend"></div>
    <div class="note">completion tok/s per model, time-weighted over the service window and stacked (top models + &lt;other&gt;). The combined view of where output tokens are produced. Step segments = one flat value per bucket.</div>
  </div>

  <div class="card span4">
    <h2>live now</h2>
    <table id="now-models"><thead><tr><th>model</th><th class="num">in&nbsp;flight</th><th class="num">queued</th></tr></thead><tbody></tbody></table>
    <div class="note" id="now-note"></div>
  </div>

  <div class="card span4">
    <h2>queued now — by user</h2>
    <div id="queued-users"></div>
    <div class="note">Live in-memory queue state from the proxy — requests waiting <em>right now</em>, grouped by consumer. Distinct from the recent-requests table below, which is finished requests.</div>
  </div>

  <div class="card span4">
    <h2>memory holders <span class="dim">(RSS — what fills RAM)</span></h2>
    <table id="rss"><thead><tr><th>process</th><th class="num">RSS GB</th><th>share</th></tr></thead><tbody></tbody></table>
  </div>

  <div class="card span12">
    <h2>models — outcomes, latency &amp; throughput vs measured concurrency</h2>
    <table id="models"><thead><tr>
      <th>model</th><th class="num">req</th><th class="num">ok</th><th class="num">aband</th>
      <th class="num">err</th><th class="num">canc</th>
      <th class="num">qwait p50</th><th class="num">ttft p50</th><th class="num">service p50/p95</th>
      <th class="num">solo decode tok/s</th><th class="num">p50 out tok</th>
      <th>throughput @ concurrency (tok/s)</th>
    </tr></thead><tbody></tbody></table>
    <div class="note" id="models-note"></div>
  </div>

  <div class="card span12">
    <h2>consumers</h2>
    <table id="consumers"><thead><tr>
      <th>key</th><th class="num">req</th><th class="num">ok</th><th class="num">aband</th>
      <th class="num">aband %</th><th class="num">err</th>
      <th class="num">prompt tok</th><th class="num">compl tok</th><th class="num">service s</th><th>models</th>
    </tr></thead><tbody></tbody></table>
  </div>
  <div class="card span12">
    <h2>workloads</h2>
    <table id="workloads"><thead><tr>
      <th>workload</th><th class="num">req</th><th class="num">ok</th><th class="num">aband %</th>
      <th class="num">err %</th><th class="num">qwait p50</th><th class="num">qwait p95</th>
      <th class="num">total p50</th><th class="num">total p95</th><th class="num">compl tok</th>
    </tr></thead><tbody></tbody></table>
  </div>

  <div class="card span12">
    <h2>recent requests <span class="dim">(latest 100 — search, filter, sort)</span></h2>
    <div class="controls">
      <input id="req-search" type="search" placeholder="search any column…" autocomplete="off">
      <select id="req-model"><option value="">all models</option></select>
      <select id="req-consumer"><option value="">all consumers</option></select>
      <select id="req-outcome"><option value="">all outcomes</option></select>
      <span class="count" id="req-count"></span>
    </div>
    <table id="requests" class="sortable"><thead><tr></tr></thead><tbody></tbody></table>
    <div class="note">One row per request lifecycle (newest first) — includes queued / abandoned rows whose latencies are not yet computable (shown as —). No prompt or response text is stored; only counts, timings, model, consumer and workload.</div>
  </div>
</main>
<footer id="footer"></footer>
<script>
const $=s=>document.querySelector(s);
const COLORS=['#58a6ff','#3fb950','#d29922','#a371f7','#f85149','#39c5cf','#db61a2','#e3b341','#8b949e'];
const fmt=n=>n==null?'—':(n>=1e6?(n/1e6).toFixed(2)+'M':n>=1e3?(n/1e3).toFixed(1)+'k':String(n));
const ms=n=>n==null?'—':(n>=1000?(n/1000).toFixed(1)+'s':n+'ms');
async function J(u){const r=await fetch(u);if(!r.ok)throw new Error(u+' '+r.status);return r.json();}

function svgW(el){return Math.max(Math.round(el.clientWidth)||600,200);}

// ── step builders ──────────────────────────────────────────────────────────
// All time-series render as STEP charts: each bucket value is held flat across
// the bucket's horizontal extent [x_i, x_i+sx], then steps vertically to the
// next value — NO linear interpolation, NO smoothing. `sx` is the bucket width
// in px; xAt(i)=padL+i*sx is the left edge of bucket i.

// Step polyline ("d" path) through (i, y(v_i)). Holds each y flat for one bucket
// width then steps. Returns {d, started} where d is "" if no point. Null values
// break the line into separate segments (each starting with M).
function stepLine(arr,xAt,yAt,sx){
  let d='',open=false;
  for(let i=0;i<arr.length;i++){
    const v=arr[i];
    if(v==null){open=false;continue;}
    const x0=xAt(i),x1=x0+sx,y=yAt(v);
    if(!open){d+='M'+x0.toFixed(1)+','+y.toFixed(1);open=true;}
    else{d+='L'+x0.toFixed(1)+','+y.toFixed(1);}
    d+='L'+x1.toFixed(1)+','+y.toFixed(1);
  }
  return d;
}
// Stacked step-area band between a lower cumulative and an upper cumulative.
// Top boundary = forward step polyline of `upper`; bottom = reverse step of
// `lower`. Closed into a filled polygon. Treats null as 0 (stacked context).
function stepBand(lower,upper,xAt,yAt,sx){
  const n=upper.length;let top='';
  for(let i=0;i<n;i++){const x0=xAt(i),x1=x0+sx,y=yAt(upper[i]||0);
    top+=(i?'L':'M')+x0.toFixed(1)+','+y.toFixed(1)+'L'+x1.toFixed(1)+','+y.toFixed(1);}
  let bot='';
  for(let i=n-1;i>=0;i--){const x0=xAt(i),x1=x0+sx,y=yAt(lower[i]||0);
    bot+='L'+x1.toFixed(1)+','+y.toFixed(1)+'L'+x0.toFixed(1)+','+y.toFixed(1);}
  return top+bot+'Z';
}

function renderHost(tl){
  const el=$('#hostchart'),H=150,padL=30,padR=8,padT=8,padB=16;
  const W=svgW(el);el.setAttribute('viewBox',`0 0 ${W} ${H}`);
  const b=tl.buckets||[],n=b.length;
  if(!n){el.innerHTML='<text x="50%" y="50%" text-anchor="middle" class="axis">no data</text>';return;}
  const cw=W-padL-padR,ch=H-padT-padB,sx=cw/Math.max(n,1);
  const xAt=i=>padL+i*sx;
  let h='';for(let i=0;i<=4;i++){const y=padT+ch/4*i;h+=`<line class="tick" x1="${padL}" y1="${y}" x2="${W-padR}" y2="${y}"/>`;}
  const gpu=tl.host.gpu_pct,wired=tl.host.wired_gb,total=tl.host.total_gb||256;
  const yWired=v=>padT+ch-(Math.min(v,total)/total)*ch;
  const yGpu=v=>padT+ch-(Math.min(v,100)/100)*ch;
  // wired as a step area (fill under the step line)
  const lows=new Array(n).fill(0),ups=wired.map(v=>v==null?0:v);
  const wd=stepLine(wired,xAt,yWired,sx);
  if(wd){
    // build a filled step-area: top step of wired, floor at baseline
    let top='',started=false;
    for(let i=0;i<n;i++){const v=wired[i];if(v==null)continue;const x0=xAt(i),x1=x0+sx,y=yWired(v);
      top+=(started?'L':'M')+x0.toFixed(1)+','+y.toFixed(1)+'L'+x1.toFixed(1)+','+y.toFixed(1);started=true;}
    const base=(padT+ch).toFixed(1),lx=(padL+n*sx).toFixed(1);
    h+=`<path d="${top}L${lx},${base}L${padL},${base}Z" fill="#58a6ff22" stroke="none"/>`;
  }
  if(wd)h+=`<path d="${wd}" fill="none" stroke="#58a6ff" stroke-width="1.5"/>`;
  const gd=stepLine(gpu,xAt,yGpu,sx);
  if(gd)h+=`<path d="${gd}" fill="none" stroke="#f85149" stroke-width="1.5"/>`;
  h+=`<text x="${padL-4}" y="${padT+4}" text-anchor="end" class="axis">100%</text>`;
  h+=`<text x="${padL-4}" y="${padT+ch}" text-anchor="end" class="axis">0</text>`;
  h+=axisX(b,padL,cw,W,padR,H);
  el.innerHTML=h;
}

function axisX(b,padL,cw,W,padR,H){
  const n=b.length;if(n<2)return '';
  const lbl=i=>{const d=new Date(b[i]*1000);return d.toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'});};
  return `<text x="${padL}" y="${H-4}" class="axis">${lbl(0)}</text>`+
    `<text x="${padL+cw/2}" y="${H-4}" text-anchor="middle" class="axis">${lbl(Math.floor(n/2))}</text>`+
    `<text x="${W-padR}" y="${H-4}" text-anchor="end" class="axis">${lbl(n-1)}</text>`;
}

// Shared stacked STEP-area renderer (charts 1 "work by customer" and 4 "output
// by model"). `keys` is the stack order, `byk` maps key→per-bucket value array.
// fmtY formats the y-axis max label; unit is appended to legend last-values.
function renderStack(elSel,legSel,b,keys,byk,opts){
  const o=opts||{},el=$(elSel),H=150,padL=o.padL||34,padR=8,padT=8,padB=16;
  const W=svgW(el);el.setAttribute('viewBox',`0 0 ${W} ${H}`);
  const n=b.length;
  const have=keys.some(k=>(byk[k]||[]).some(v=>v>0));
  if(!n||!have){el.innerHTML=`<text x="50%" y="50%" text-anchor="middle" class="axis">${o.empty||'no data in window'}</text>`;$(legSel).innerHTML='';return;}
  const cw=W-padL-padR,ch=H-padT-padB,sx=cw/Math.max(n,1);
  const xAt=i=>padL+i*sx;
  let maxT=o.minMax||0.001;
  for(let i=0;i<n;i++){let s=0;keys.forEach(k=>s+=(byk[k]||[])[i]||0);if(s>maxT)maxT=s;}
  if(o.round)maxT=Math.ceil(maxT/o.round)*o.round;
  const yAt=v=>padT+ch-(v/maxT)*ch;
  let h='';for(let i=0;i<=4;i++){const y=padT+ch/4*i;h+=`<line class="tick" x1="${padL}" y1="${y}" x2="${W-padR}" y2="${y}"/>`;
    h+=`<text x="${padL-4}" y="${y+3}" text-anchor="end" class="axis">${(o.fmtY||(v=>v.toFixed(1)))(maxT-maxT/4*i)}</text>`;}
  const cum=new Array(n).fill(0);
  keys.forEach((k,ki)=>{const arr=byk[k]||[];const up=cum.map((c,i)=>c+(arr[i]||0));
    const c=COLORS[ki%COLORS.length];
    h+=`<path d="${stepBand(cum,up,xAt,yAt,sx)}" fill="${c}" fill-opacity="0.55" stroke="${c}" stroke-width="1"/>`;
    for(let i=0;i<n;i++)cum[i]=up[i];});
  h+=axisX(b,padL,cw,W,padR,H);
  el.innerHTML=h;
  const unit=o.unit||'';
  $(legSel).innerHTML=keys.map((k,i)=>{const arr=byk[k]||[];let last=null;
    for(let j=arr.length-1;j>=0;j--){if(arr[j]!=null){last=arr[j];break;}}
    const lv=last!=null&&unit?' '+(o.fmtLegend?o.fmtLegend(last):last)+unit:'';
    return `<span><span style="background:${COLORS[i%COLORS.length]}"></span>${k}${lv}</span>`;}).join('');
}

// Chart 1 — HEADLINE: work by customer (GPU-busy share = active concurrency by
// consumer), stacked STEP area. Reuses /timeline's by_key_active.
function renderConc(tl){
  renderStack('#concchart','#conc-legend',tl.buckets||[],tl.keys||[],tl.by_key_active||{},
    {padL:30,empty:'no calls in window',unit:''});
}

// Chart 4 — output throughput by model (completion tok/s), stacked STEP area.
function renderOut(tl){
  renderStack('#outchart','#out-legend',tl.buckets||[],tl.out_models||[],tl.by_model_out_tok_s||{},
    {padL:40,empty:'no output tokens in window',round:50,unit:' tok/s',fmtY:fmt,fmtLegend:fmt});
}

// Chart 3 — throughput input vs output (total tok/s across all models), two STEP
// lines on one auto-scaled axis.
function renderIO(tl){
  const el=$('#iochart'),H=150,padL=40,padR=8,padT=8,padB=16;
  const W=svgW(el);el.setAttribute('viewBox',`0 0 ${W} ${H}`);
  const b=tl.buckets||[],n=b.length,tot=tl.totals||{};
  const inp=tot.in_tok_s||[],out=tot.out_tok_s||[];
  const series=[['input',inp,'#58a6ff'],['output',out,'#3fb950']];
  let maxY=0;series.forEach(([,arr])=>arr.forEach(v=>{if(v!=null&&v>maxY)maxY=v;}));
  if(!n||maxY<=0){el.innerHTML='<text x="50%" y="50%" text-anchor="middle" class="axis">no token throughput in window</text>';$('#io-legend').innerHTML='';return;}
  maxY=Math.ceil(maxY/Math.max(1,Math.pow(10,Math.floor(Math.log10(maxY)))))*Math.pow(10,Math.floor(Math.log10(maxY)));
  const cw=W-padL-padR,ch=H-padT-padB,sx=cw/Math.max(n,1);
  const xAt=i=>padL+i*sx,yAt=v=>padT+ch-(Math.min(v,maxY)/maxY)*ch;
  let h='';for(let i=0;i<=4;i++){const y=padT+ch/4*i;h+=`<line class="tick" x1="${padL}" y1="${y}" x2="${W-padR}" y2="${y}"/>`;
    h+=`<text x="${padL-4}" y="${y+3}" text-anchor="end" class="axis">${fmt(Math.round(maxY-maxY/4*i))}</text>`;}
  series.forEach(([,arr,col])=>{const d=stepLine(arr,xAt,yAt,sx);if(d)h+=`<path d="${d}" fill="none" stroke="${col}" stroke-width="1.5"/>`;});
  h+=axisX(b,padL,cw,W,padR,H);
  el.innerHTML=h;
  $('#io-legend').innerHTML=series.map(([lbl,arr,col])=>{let last=null;
    for(let j=arr.length-1;j>=0;j--){if(arr[j]!=null){last=arr[j];break;}}
    return `<span><span style="background:${col}"></span>${lbl}${last!=null?' '+fmt(Math.round(last))+' tok/s':''}</span>`;}).join('');
}

function renderNow(now){
  $('#verdict').textContent=now.verdict||'';
  $('#verdict').style.color=now.stall?'#f85149':'';
  const h=now.host||{};
  const wiredPct=h.wired_gb&&h.total_gb?h.wired_gb/h.total_gb*100:0;
  const kc=[
    ['GPU now',h.gpu_pct==null?'—':h.gpu_pct.toFixed(0)+'%',h.gpu_pct>=95?'hot':h.gpu_pct>=70?'warn':'ok'],
    ['wired RAM',h.wired_gb==null?'—':h.wired_gb.toFixed(0)+' GB',wiredPct>=88?'hot':wiredPct>=70?'warn':'ok'],
    ['free RAM',h.free_gb==null?'—':h.free_gb.toFixed(1)+' GB',h.free_gb<2?'hot':''],
    ['in flight',now.totals.in_flight,''],
    ['queued',now.totals.queue_depth,now.totals.queue_depth>0?'warn':''],
  ];
  if(now.budget){const bp=now.budget.budget_pct;
    kc.push(['budget used',bp==null?'—':bp.toFixed(0)+'%',bp>=95?'hot':bp>=70?'warn':'ok']);}
  $('#kpis').innerHTML=kc.map(([l,v,c])=>`<div class="kpi"><div class="v ${c}">${v}</div><div class="l">${l}</div></div>`).join('');
  const tb=$('#now-models tbody');
  tb.innerHTML=(now.models||[]).map(m=>{
    const active=m.in_flight>0||m.queue_depth>0;
    const q=m.queue_depth?`<span class="warn">${m.queue_depth}</span>`:'0';
    const inf=m.in_flight>0?`<span class="ok">${m.in_flight}</span>/${m.cap??'—'}`:`<span class="dim">${m.in_flight}/${m.cap??'—'}</span>`;
    return `<tr class="${active?'active':''}"><td>${m.model}</td><td class="num">${inf}</td><td class="num">${q}</td></tr>`;
  }).join('')||'<tr><td colspan="3" class="dim">idle</td></tr>';
  $('#now-note').textContent=now.host.sample_age_s!=null?`host sample ${now.host.sample_age_s}s ago`:'';
  const rt=$('#rss tbody'),bs=now.host.backends||[];
  const max=bs.length?bs[0].rss_gb:1;
  rt.innerHTML=bs.map(b=>`<tr><td>${b.name}</td><td class="num">${b.rss_gb.toFixed(1)}</td><td><div class="bar"><i style="width:${(b.rss_gb/max*100).toFixed(0)}%"></i></div></td></tr>`).join('')||'<tr><td colspan="3" class="dim">—</td></tr>';
  renderQueuedByUser(now);
}

// Live in-memory queue (waiting requests), grouped by consumer. Reads the flat
// `now.queued` list (falls back to flattening per-model `queued`). Scheduler
// fields are null-safe — absent on the kill-switch path.
const WAIT_REASON={model_cap:'backend full',budget_full:'budget',exclusive:'exclusive'};
function renderQueuedByUser(now){
  const el=$('#queued-users');if(!el)return;
  let q=now.queued;
  if(!q){q=[];(now.models||[]).forEach(m=>(m.queued||[]).forEach(w=>q.push(Object.assign({model:m.model},w))));}
  if(!q.length){el.innerHTML='<div class="dim">queue empty</div>';return;}
  const groups=new Map();
  q.forEach(w=>{const k=w.key||'—';if(!groups.has(k))groups.set(k,[]);groups.get(k).push(w);});
  const blocks=[...groups.entries()].map(([key,ws])=>{
    ws.sort((a,b)=>(b.age_s||0)-(a.age_s||0));
    const oldest=Math.max(...ws.map(w=>w.age_s||0));
    return {key,ws,n:ws.length,oldest};
  }).sort((a,b)=>b.n-a.n||b.oldest-a.oldest);
  el.innerHTML=blocks.map(g=>{
    const items=g.ws.map(w=>{
      const prio=w.effective_priority!=null?w.effective_priority:w.priority;
      const why=WAIT_REASON[w.wait_reason]||'';
      const bits=[`<span class="qm">${w.model||'—'}</span>`,`${fmt(Math.round(w.age_s||0))}s`];
      if(prio!=null)bits.push(`p${prio}`);
      if(why)bits.push(why);
      return `<li>${bits.join(' · ')}</li>`;
    }).join('');
    return `<div class="quser"><div class="qhead"><b>${g.key}</b><span class="warn">${g.n} queued · oldest ${fmt(Math.round(g.oldest))}s</span></div><ul>${items}</ul></div>`;
  }).join('');
}

function renderModels(d){
  const tb=$('#models tbody');
  tb.innerHTML=(d.models||[]).map(m=>{
    const L=m.latency_ms;
    const tp=(m.throughput_by_concurrency||[]).map(c=>{
      if(!c.sufficient)return `<span class="dim">@${c.concurrency}:n=${c.calls}?</span>`;
      return `<span style="color:var(--accent)">@${c.concurrency}: ${c.aggregate_tok_s??'—'}</span> <span class="dim">(n=${c.calls})</span>`;
    }).join(' &nbsp; ')||'<span class="dim">—</span>';
    return `<tr>
      <td>${m.model}</td><td class="num">${m.requests}</td>
      <td class="num ok">${m.completed}</td>
      <td class="num ${m.abandoned?'hot':'dim'}">${m.abandoned}</td>
      <td class="num ${m.errored?'warn':'dim'}">${m.errored}</td>
      <td class="num dim">${m.cancelled_queued}</td>
      <td class="num">${ms(L.queue_wait_p50)}</td>
      <td class="num">${ms(L.ttft_p50)}</td>
      <td class="num">${ms(L.service_p50)}/${ms(L.service_p95)}</td>
      <td class="num">${m.decode_solo_tok_s??'<span class="dim">—</span>'}</td>
      <td class="num">${m.out_tok_p50==null?'<span class="dim">—</span>':fmt(m.out_tok_p50)}</td>
      <td>${tp}</td></tr>`;
  }).join('')||'<tr><td colspan="12" class="dim">no calls</td></tr>';
  $('#models-note').textContent=(d.notes||[]).join('  ·  ');
}

function renderConsumers(d){
  const tb=$('#consumers tbody');
  tb.innerHTML=(d.consumers||[]).map(c=>{
    const mdl=Object.entries(c.models||{}).sort((a,b)=>b[1]-a[1]).map(([k,v])=>`${k.split('/').pop()}:${v}`).join(' ');
    return `<tr>
      <td>${c.key}</td><td class="num">${c.requests}</td>
      <td class="num ok">${c.completed}</td>
      <td class="num ${c.abandoned?'hot':'dim'}">${c.abandoned}</td>
      <td class="num ${c.abandoned_rate>0.1?'hot':'dim'}">${(c.abandoned_rate*100).toFixed(0)}%</td>
      <td class="num ${c.errored?'warn':'dim'}">${c.errored}</td>
      <td class="num">${fmt(c.prompt_tokens)}</td><td class="num">${fmt(c.completion_tokens)}</td>
      <td class="num">${fmt(c.service_s)}</td><td class="dim">${mdl}</td></tr>`;
  }).join('')||'<tr><td colspan="10" class="dim">no calls</td></tr>';
}

function renderWorkloads(d){
  const tb=$('#workloads tbody');
  tb.innerHTML=(d.workloads||[]).map(w=>{
    const l=w.latency_ms||{};
    return `<tr>
      <td>${w.workload}</td><td class="num">${w.requests}</td>
      <td class="num ok">${w.completed}</td>
      <td class="num ${w.abandoned_rate>0.1?'hot':'dim'}">${(w.abandoned_rate*100).toFixed(0)}%</td>
      <td class="num ${w.error_rate>0?'warn':'dim'}">${(w.error_rate*100).toFixed(0)}%</td>
      <td class="num">${ms(l.queue_wait_p50)}</td><td class="num">${ms(l.queue_wait_p95)}</td>
      <td class="num">${ms(l.total_p50)}</td><td class="num">${ms(l.total_p95)}</td>
      <td class="num">${fmt(w.completion_tokens)}</td></tr>`;
  }).join('')||'<tr><td colspan="10" class="dim">no calls</td></tr>';
}

// ── recent requests: searchable / sortable / filterable client-side table ────
// Columns: {key, label, num?} — `num` columns sort numerically and right-align.
// `time` keeps the raw epoch for sorting but renders as local HH:MM:SS.
const REQ_COLS=[
  {key:'t_enqueue',label:'time',num:true,time:true},
  {key:'model',label:'model'},
  {key:'consumer',label:'consumer'},
  {key:'workload',label:'workload'},
  {key:'outcome',label:'outcome'},
  {key:'queue_wait',label:'qwait',num:true,ms:true},
  {key:'ttft',label:'ttft',num:true,ms:true},
  {key:'service',label:'service',num:true,ms:true},
  {key:'total',label:'total',num:true,ms:true},
  {key:'prompt_tokens',label:'prompt tok',num:true,tok:true},
  {key:'completion_tokens',label:'compl tok',num:true,tok:true},
  {key:'decode_tok_s',label:'decode tok/s',num:true},
];
let REQ_ROWS=[];                       // cached fetched rows (never refetched on UI change)
let REQ_SORT={key:'t_enqueue',asc:false};  // default: time descending (newest first)
const hhmmss=e=>e==null?'—':new Date(e*1000).toLocaleTimeString([],{hour12:false});
function reqOutcomeCell(r){
  // completed=ok, errored=warn, abandoned/cancelled=hot/dim; http status appended.
  const cls={completed:'ok',upstream_error:'warn',client_abandoned:'hot',cancelled_queued:'dim'}[r.outcome]||'dim';
  const st=r.http_status!=null?` <span class="dim">${r.http_status}</span>`:'';
  return `<span class="${cls}">${r.outcome||'—'}</span>${st}`;
}
function reqCell(r,c){
  if(c.time)return hhmmss(r[c.key]);
  if(c.ms)return ms(r[c.key]);
  if(c.tok)return fmt(r[c.key]);
  if(c.key==='decode_tok_s')return r[c.key]==null?'—':r[c.key];
  const v=r[c.key];return v==null||v===''?'—':v;
}
function reqMatchesText(r,q){
  if(!q)return true;
  // case-insensitive match across every VISIBLE cell's rendered text.
  return REQ_COLS.some(c=>{
    const t=(c.key==='outcome'?r.outcome:reqCell(r,c));
    return String(t==null?'':t).toLowerCase().includes(q);
  });
}
function renderRequests(){
  const thead=$('#requests thead tr'),tb=$('#requests tbody');
  thead.innerHTML=REQ_COLS.map(c=>{
    const ind=REQ_SORT.key===c.key?`<span class="ind">${REQ_SORT.asc?'▲':'▼'}</span>`:'';
    return `<th data-k="${c.key}" class="${c.num?'num':''}">${c.label}${ind}</th>`;
  }).join('');
  const q=($('#req-search').value||'').trim().toLowerCase();
  const fm=$('#req-model').value,fc=$('#req-consumer').value,fo=$('#req-outcome').value;
  let rows=REQ_ROWS.filter(r=>
    (!fm||r.model===fm)&&(!fc||r.consumer===fc)&&(!fo||r.outcome===fo)&&reqMatchesText(r,q));
  const col=REQ_COLS.find(c=>c.key===REQ_SORT.key)||REQ_COLS[0],dir=REQ_SORT.asc?1:-1;
  rows=rows.slice().sort((a,b)=>{
    let x=a[col.key],y=b[col.key];
    if(col.num){ // nulls always sort last regardless of direction
      if(x==null&&y==null)return 0;if(x==null)return 1;if(y==null)return -1;
      return (x-y)*dir;
    }
    x=String(x==null?'':x).toLowerCase();y=String(y==null?'':y).toLowerCase();
    return x<y?-1*dir:x>y?1*dir:0;
  });
  tb.innerHTML=rows.map(r=>'<tr>'+REQ_COLS.map(c=>{
    const cell=c.key==='outcome'?reqOutcomeCell(r):reqCell(r,c);
    return `<td class="${c.num?'num':''}">${cell}</td>`;
  }).join('')+'</tr>').join('')||`<tr><td colspan="${REQ_COLS.length}" class="dim">no matching requests</td></tr>`;
  $('#req-count').textContent=`${rows.length} / ${REQ_ROWS.length} rows`;
}
function fillReqFilter(sel,vals){
  const cur=sel.value,opts=['<option value="">all '+sel.dataset.all+'</option>']
    .concat([...new Set(vals)].filter(v=>v!=null&&v!=='').sort().map(v=>`<option>${v}</option>`));
  sel.innerHTML=opts.join('');
  if([...sel.options].some(o=>o.value===cur))sel.value=cur;
}
function setRequests(d){
  REQ_ROWS=(d&&d.requests)||[];
  fillReqFilter($('#req-model'),REQ_ROWS.map(r=>r.model));
  fillReqFilter($('#req-consumer'),REQ_ROWS.map(r=>r.consumer));
  fillReqFilter($('#req-outcome'),REQ_ROWS.map(r=>r.outcome));
  renderRequests();
}
$('#req-model').dataset.all='models';$('#req-consumer').dataset.all='consumers';$('#req-outcome').dataset.all='outcomes';
['#req-search','#req-model','#req-consumer','#req-outcome'].forEach(s=>{
  $(s).addEventListener('input',renderRequests);$(s).addEventListener('change',renderRequests);});
$('#requests thead').addEventListener('click',ev=>{
  const th=ev.target.closest('th');if(!th)return;const k=th.dataset.k;if(!k)return;
  if(REQ_SORT.key===k)REQ_SORT.asc=!REQ_SORT.asc;
  else REQ_SORT={key:k,asc:false}; // new column starts descending
  renderRequests();
});

async function refresh(){
  const w=$('#window').value;
  try{
    const [now,tl,mdl,cons,wl,req]=await Promise.all([
      J('/now'),J('/timeline?last='+w),J('/models?last='+w),J('/consumers?last='+w),J('/workloads?last='+w),J('/requests?limit=100')]);
    renderNow(now);renderConc(tl);renderHost(tl);renderIO(tl);renderOut(tl);renderModels(mdl);renderConsumers(cons);renderWorkloads(wl);setRequests(req);
    $('#footer').textContent='updated '+new Date().toLocaleTimeString()+' · window '+w+' · bucket '+tl.window.bucket_s+'s';
  }catch(e){$('#footer').innerHTML='<span class="err">'+e+'</span>';}
}
async function liveTick(){try{renderNow(await J('/now'));}catch(e){}}
$('#window').onchange=refresh;$('#refresh').onclick=refresh;
refresh();setInterval(liveTick,4000);setInterval(refresh,30000);
</script></body></html>"""

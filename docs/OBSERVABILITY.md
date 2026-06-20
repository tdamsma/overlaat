# Observability — the usage-API

Overlaat's observability is built ground-up on a single principle:
**instrument the call path once, derive everything.** No reconciliation of blind
sources, no survivor bias, no conflated definitions. The queue-proxy is the only
component that sits on the full request path, so it is the only thing that emits
request-level facts; everything the dashboard shows is derived from those facts
plus a host sampler.

> Status: experimental. No support promise. Names and shapes may change.

## Architecture

```
                          ┌──────────────────────────────────────────┐
  consumers ── :4000 ──▶  │ queue-proxy (single process, on the path) │
                          │  • per-model FIFO semaphore                │
                          │  • emits ONE row per request lifecycle ──▶ │── request_events (PG)
                          └──────────────────────────────────────────┘
                                                                          ▲
  host sampler (5s) ── host GPU%/RAM + per-backend RSS ────────────────────┘ host_samples (PG)

  usage-api reads both tables ──▶ /now /timeline /models /perf /consumers + dashboard
```

The queue-proxy is the only component that sees **every** request's full
lifecycle — including queued and client-abandoned calls, both of which are
invisible to insert-on-completion spend logging (which only writes a row once a
call runs to completion). So the proxy is the single source of truth for
request-level facts. The host sampler owns host facts. The usage-API only reads;
it never writes.

This matters because the usual gateway spend log inserts a row *when a call
finishes*. That structurally drops two whole classes of request:

- **Queued calls** that are still waiting for a slot (they have no completion
  yet, so no row exists).
- **Client-abandoned calls** where the consumer disconnected mid-stream (they
  never "complete" in the gateway's sense).

Both are exactly the requests you most want to see when you are reasoning about
saturation and fairness. Emitting one row per *lifecycle* — written on terminal
state, whatever that state is — closes that blind spot.

## Tables

DDL lives in `schema.sql` (idempotent). All timestamps are epoch **seconds**
(UTC, sub-millisecond) — no timezone games. The instrumentation writes to the
same Postgres the gateway uses (point `DATABASE_URL` at it).

The same event schema is backed by either **Postgres** (the default) or, opt-in
for a single-box deployment, **SQLite** (a `sqlite:///` `DATABASE_URL`). The
shapes are identical across backends — timestamps are epoch-second doubles in
both — with only the storage type of `host_samples.backends_json` differing:
`JSONB` in Postgres, JSON-as-`TEXT` in SQLite (decoded back to an object on
read). For SQLite, initialize the file with `python -m overlaat.db init
"$DATABASE_URL"` (the same entry point works for Postgres too); `psql -f
schema.sql` remains the Postgres-native way.

SQLite runs in WAL mode, which writes `-wal` and `-shm` sidecar files next to the
`.db` (expected, not an error). The single-writer model — the queue-proxy is one
process, so never run it with `--workers N` — is exactly what SQLite wants. To back
up while the services are live, use `sqlite3 file.db ".backup backup.db"` or
`VACUUM INTO 'backup.db'` rather than copying the file by hand.

### `request_events` — one row per request

Written by the queue-proxy when a request reaches a terminal state.

| column | meaning |
|---|---|
| `t_enqueue` | request hit the proxy |
| `t_acquire` | got the semaphore slot (= backend start); `NULL` if cancelled while queued |
| `t_first_token` | first content byte (TTFT marker); `NULL` for non-stream / no tokens |
| `t_done` | last byte / connection closed (slot released) |
| `model_requested` | alias the client asked for (what hits the semaphore) |
| `key_fp` | `sha256(bearer)[:8]` — a non-reversible key fingerprint; equals the gateway's verification-token prefix → resolvable to a key alias |
| `streamed` | bool |
| `outcome` | `completed` \| `client_abandoned` \| `upstream_error` \| `cancelled_queued` |
| `http_status`, `prompt_tokens`, `completion_tokens` | tokens from the gateway's own `usage` block (`NULL` = not reported, **never zero-filled**) |
| `priority` | effective base priority used at admission (request `priority` clamped to the per-key ceiling). `NULL` when the scheduler is off or the row predates 0.0.3 |
| `cost` | pool-fraction cost charged for this run (`1/cap` by default; an `overlaat_cost` override; `1.0` for an uncapped model). `NULL` when the scheduler is off |
| `wait_reason` | why the request waited before admission: `none` (admitted on first pump) \| `reserved` (was the reserved head, admitted once the budget drained for it) \| `aged_in` (aging lifted it above equal-priority peers) \| `budget_full` (the pool budget had no room) \| `model_cap` (per-model backend cap was full) \| `exclusive` (a *different* member held the request's exclusive pool's one-distinct-member mutex). `NULL` when the scheduler is off |
| `pool` | the resource pool the request was admitted against (`default` for an unconfigured model; an `overlaat_pool` name otherwise). `NULL` when the scheduler is off or the row predates this column |

The cost-weighted scheduler (on by default since 0.0.3) populates `priority`, `cost`,
`wait_reason`, and `pool`; see [`COST-SCHEDULER.md`](COST-SCHEDULER.md) for the algorithm.
With `OVERLAAT_SCHEDULER=off` (the kill-switch restoring per-model FIFO semaphores) and
for rows written before these columns existed, they are `NULL`. `wait_reason` is the
cheapest way to see *why* a queue is deep: a run of `budget_full` means the request's
**pool** budget is the bottleneck (its models contending for that pool's share), whereas
`model_cap` means a single backend's own cap is the bottleneck (that one model is saturated
while the pool has room), and `exclusive` means a *different* member is resident in an
exclusive (swap-slot) pool and the request is waiting for the mutex to hand off. `pool`
lets you slice all of this per resource pool — a deep `budget_full` queue isolated to one
pool tells you which backend is the bottleneck without touching the others.

`key_fp` is a fingerprint, not a secret: it is the first 8 hex chars of the
SHA-256 of the bearer token. It is enough to group requests by caller and to
join to a key alias the gateway already stores, but it is not reversible and
leaks no credential.

A handful of durations are derived **once**, in the query layer
(`overlaat/metrics_db.py`), and never recomputed elsewhere:

- `queue_wait = t_acquire - t_enqueue` — time spent waiting for a slot.
- `ttft       = t_first_token - t_acquire` — time-to-first-token (backend only).
- `decode     = t_done - t_first_token` — streaming decode window.
- `service    = t_done - t_acquire` — true backend-busy (slot held).
- `total      = t_done - t_enqueue` — true user-perceived latency.

The distinction between `service` and `total` is the whole point of putting a
waiting queue in front: `total - service = queue_wait` is the cost the queue
imposes, measured directly rather than inferred.

### `host_samples` — one row per ~5s

Written by the host sampler. Holds host `gpu_pct` / `gpu_freq_mhz`, RAM
breakdown (`total/wired/active/inactive/compressed/free`), CPU load averages,
and `backends_json` = the top memory holders as
`[{name, pid, rss_gb, cpu_pct, gpu_pct}, ...]`.

### `model_loads` — cold-load log (optional)

Written when a model is loaded into a shared swap-on-demand slot — the case
where only one large model is resident at a time and loading another evicts the
current one. Polling the swap manager's running-model endpoint lets the sampler
record `t_start` (first poll the model was non-ready) and `t_ready` (poll it
became ready). This makes cold-load time **explicit** instead of hiding it
inside the first request's TTFT, so a benchmark can attribute a slow first call
to the model swap rather than to the engine.

## Concurrency — three curves, each defined once

The single most important discipline here: concurrency has exactly **three**
curves, and each is defined in exactly **one** place. They are computed from
`request_events`, at a time *t*, per model:

- **offered(t)** = requests with `t_enqueue ≤ t < t_done` — demand, *including*
  the ones still queued. This is what was asked of the system.
- **active(t)** = requests with `t_acquire ≤ t < t_done` — backend busy, slot
  held. This is bounded by the per-model cap **by definition** — that is correct,
  not a bug; it is the supply side.
- **queued(t)** = offered − active — the backlog.

Because `active(t)` is bounded by admission, `offered` is the only curve that
can exceed it, and `queued` is exactly the part of demand the queue is
absorbing. Plotting all three together is how you see a spillway doing its job:
demand spikes above the line, the queue takes the overflow, the backend stays at
its designed capacity instead of thrashing.

## Budget utilization (cost-weighted scheduler)

When the scheduler is on (the default since 0.0.3), admission is cost-weighted
against each model's **resource pool** budget — the per-model curves above sum
*into* their pool. The live budget state is exposed by the queue-proxy on
`:4000/__queue/health` and `:4000/__queue/status` under a `budget` object. It keeps
the legacy top-level fields (reflecting the `default` pool, so existing readers and
the `budget_pct` computation keep working) and adds a per-pool `pools` map:

```jsonc
"budget": {
  // Legacy top-level fields — the `default` pool, for back-compat.
  "budget": 1.0,          // B_default
  "used": 0.25,           // sum of cost(model) over in-flight runs in `default`
  "budget_pct": 25.0,     // used / B_default, as a percentage
  "queue_depth": 3,       // total waiters across ALL pools
  "in_flight": { "model-a": 1, "model-b": 2 },
  "reserved_for": null,   // req_id of the `default` pool's reserved head, or null
  // Per-pool breakdown — the unit of admission.
  "pools": {
    "default":  { "budget": 1.0, "used": 0.25, "exclusive": false,
                  "active_member": null, "reserved_for": null,
                  "in_flight": { "model-a": 1 }, "budget_pct": 25.0 },
    "fat-slot": { "budget": 1.0, "used": 1.0,  "exclusive": true,
                  "active_member": "model-b",  // the resident member (mutex holder)
                  "reserved_for": null,
                  "in_flight": { "model-b": 2 }, "budget_pct": 100.0 }
  }
}
```

`used` is the summed cost of every in-flight run **in that pool**, so a pool's
`budget_pct` near 100% means that pool is fully committed (and new arrivals to it
wait on its budget, not on any one model's cap). A pool's non-null `reserved_for`
means an expensive head is holding *that pool's* budget while cheap jobs drain —
the work-conserving-vs-fairness tradeoff (§ COST-SCHEDULER 3c), now intra-pool. For
an `exclusive` pool, `active_member` is the one resident member; a different member
waits with `wait_reason = exclusive` until it hands off. Per queued request,
`/__queue/status` also surfaces `priority`, `effective_priority` (priority after
aging), `cost`, `pool`, and the live `wait_reason`, so a deep queue can be read by
*why* each entry is waiting and *which pool* it is waiting in. With
`OVERLAAT_SCHEDULER=off`, the `budget` object is `null` and these per-entry fields
are absent.

## Throughput vs concurrency — done honestly

A naive "tokens/sec at N concurrent requests" chart lies, because a single call
experiences *different* concurrency over its own lifetime. Overlaat bucket each
completed call by the **time-weighted mean of active(t) over that call's own
`[t_acquire, t_done]` interval** — the parallelism it *actually* experienced
(1.0 = it ran alone). Per bucket:

```
aggregate_tok_s = Σ tokens / Σ service
```

i.e. total tokens produced in the bucket divided by total backend-busy seconds.

Crucially, every cell is gated by a **sufficiency check**: buckets with fewer
than `MIN_SAMPLES` (default 5) calls are returned with `sufficient: false` and
are **never** shown as a trend. A two-sample throughput number is noise; the gate
keeps the chart from drawing a confident line through noise.

## Endpoints

The usage-API is a read-only FastAPI app (`overlaat.usage_api:app`). It has no
auth of its own — bind it on a trusted network and let the network ACL be the
gate.

| route | answers |
|---|---|
| `GET /` | dashboard (HTML) |
| `GET /now` | live: per-model in-flight / queued (scraped from the queue-proxy's `:4000/__queue/status`), host GPU% / wired RAM + backend RSS (latest sample), recent-5m completed per key. Live in-flight comes from the proxy because `request_events` rows are written on completion — the proxy is the only thing that knows what is in flight *right now*. |
| `GET /timeline?last=` | time-series for charts: host GPU% / wired RAM + backend RSS, per-model offered / active, per-key active concurrency. Bucket width is auto-picked from the window (30m→5s, 1h→15s, 6h→60s, 24h→5m, 7d→1h). |
| `GET /models?last=` | capacity view: per-model outcome counts, latency split (`queue_wait` / `ttft` / `service` / `total`, p50 + p95), and throughput-by-measured-concurrency (min-sample guarded). |
| `GET /perf?last=` | **backend-health monitoring**: per-model decode tok/s over time (`completion_tokens / (t_done - t_first_token)`, streamed completed calls only). Reports an all-calls median and a **solo** median (calls at mean concurrency < 1.5) that isolates backend health from load — a sustained drop in the solo line is degradation, not contention (some engines slow down over long uptime; the fix is to restart the backend service). Rates above a hardware ceiling are dropped as near-zero-window artifacts; thinking-mode models that emit no `t_first_token` do not appear here. |
| `GET /consumers?last=` | per key alias: requests by outcome, tokens, service-seconds, abandoned-rate, models used. |
| `GET /healthz` | liveness + DB check. |

`window` (`last=`) accepts `30m | 1h | 6h | 24h | 7d`.

## The only remaining caveats — each stated once

These are real limits of the measurement, written down so nobody re-discovers
them as bugs:

1. **GPU% is host-wide, not per-model.** Per-process GPU utilization is not
   reliably measurable on every platform — notably Metal/MLX workloads on macOS
   report 0 ms/s for per-process GPU. So GPU% stays host-wide. Memory, by
   contrast, *is* attributed per-process via RSS (`backends_json` in
   `host_samples`). The honest answer is: "this host is at X% GPU, and these
   processes hold this much memory" — not a fabricated per-model GPU split.

2. **Token counts are `NULL` when a backend reports no `usage`.** A `NULL` is
   never counted as 0; it is excluded. The proxy injects
   `stream_options.include_usage = true` on streaming chat to minimize this, but
   a backend that simply does not report usage leaves the counts unknown, and
   the metrics say unknown rather than zero.

3. **Engine-tail after client-abandon.** On client disconnect the slot is
   released at `t_done`, but a single-stream engine may keep decoding briefly
   after the slot is freed (a release-vs-engine desync). `service` measures
   slot-occupancy, not literal GPU-busy time past release. This means cancelling
   a request that is already *in flight* does not necessarily stop the backend;
   only cancelling a still-*queued* request is clean. Treat `service` as the
   accounting of slot time, which is what fairness is actually computed against.

## Source layout

| component | module | role |
|---|---|---|
| schema | `schema.sql` | idempotent DDL for `request_events`, `host_samples`, `model_loads` |
| emitter | `overlaat/queue_proxy.py` | FIFO entry on `:4000`; emits one lifecycle row per request |
| query layer | `overlaat/metrics_db.py` | all derived durations + concurrency curves, defined once |
| reader | `overlaat.usage_api:app` | read-only dashboard / API |
| host sampler | `overlaat/host_logger.py` | 5s host GPU%/RAM + per-backend RSS → `host_samples` |
| run scripts | `examples/run-queue-proxy.sh`, `examples/run-usage-api.sh` | launch wrappers |
| config | `examples/litellm-config.example.yaml`, `examples/overlaat.env.example` | gateway model list + environment |

Configuration is environment-driven (see `examples/overlaat.env.example`):

- `DATABASE_URL` — the Postgres both services use (proxy writes, usage-API reads).
- `QUEUE_PROXY_UPSTREAM` — the gateway the proxy forwards to (default
  `http://127.0.0.1:4002`; keep the gateway on loopback so the proxy is the only
  entry point and therefore the single, complete instrumentation site).
- `QUEUE_PROXY_LITELLM_CONFIG` — path to the gateway model list the proxy parses
  to size one semaphore per model (default `./litellm-config.yaml`). Each model's
  concurrency cap is *derived* from the backend config, not tuned here.

## Operations

- **Restart safety.** Restarting the usage-API or the host sampler is safe at any
  time — they only read / sample. The **queue-proxy is different**: restart it
  only when the queue is empty (`total_in_flight == 0 && total_queue_depth == 0`,
  visible on `/now` and `:4000/__queue/status`). A restart with calls in the
  queue drops them, turning queued requests into instant failures.

- **Apply the schema** (idempotent):

  ```sh
  psql "$DATABASE_URL" -f schema.sql
  ```

- **Clean slate** (the metrics layer treats historic data as disposable — it is
  derived, not authoritative):

  ```sh
  psql "$DATABASE_URL" -c 'TRUNCATE request_events; TRUNCATE host_samples; TRUNCATE model_loads;'
  ```

  Truncating these tables is safe while serving — they are append-only
  observability data, not request state. (Do **not** restart the proxy with a
  non-empty queue, per above.)

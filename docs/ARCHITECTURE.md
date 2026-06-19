# Overlaat — architecture

> Status: **experimental**. No support promise, no stability guarantee. MIT-licensed.
> The design is in production-shaped use behind one self-hosted gateway; treat the
> APIs and the schema as movable until a tagged release says otherwise.

Overlaat is a small sidecar that sits in front of a self-hosted, multi-backend LLM
gateway ([LiteLLM](https://github.com/BerriAI/litellm)) and does two things the
gateway does not:

1. **A fair waiting queue.** A per-model FIFO semaphore in front of the gateway, so
   that overflow traffic *waits* in line instead of being rejected with HTTP 429.
2. **Honest usage accounting.** Exactly one lifecycle event per request, written to
   Postgres — *including* queued and client-abandoned calls that insert-on-completion
   logging structurally misses. A read-only dashboard / usage-API is served over
   those events.

The name is the Dutch word *overlaat*: a controlled spillway built into a dike that
sheds floodwater in a designed, deliberate way instead of letting the dike breach
catastrophically. That is exactly the posture here — when demand exceeds what the
backends can serve, traffic spills into a controlled wait-queue rather than failing
in an uncontrolled cascade.

---

## 1. Why a waiting queue in front of a gateway

A self-hosted gateway like LiteLLM (and a model-swap layer like
[llama-swap](https://github.com/mostlygeek/llama-swap) underneath it) protects its
backends with a hard concurrency cap. Both `max_parallel_requests` (LiteLLM) and a
swap-layer `concurrencyLimit` reject with **HTTP 429 on overflow** — there is no
wait-queue. The rejection is synchronous and immediate (measured: a few hundred
microseconds — pure rejection, no work done).

That cap is the right *protection* but the wrong *interface*. The failure mode it
produces:

- A bursty consumer — parallel synthesis fan-out, parallel transcription, an agent
  loop that issues N tool calls at once — sends more than the cap allows.
- It gets a 429 cascade.
- Every such consumer now has to implement its own retry/backoff, each with its own
  jitter, its own ceiling, its own idea of fairness. None of them can see the
  others. The result is a thundering herd against the cap, and no global ordering:
  the consumer that happens to retry at the right millisecond wins, not the one that
  asked first.

Overlaat replaces "reject and make the caller cope" with "**wait in line**":

- It owns the network-facing port and forwards every request to the gateway on
  loopback *after* the request has been admitted by a **per-model `asyncio.Semaphore`**.
- Requests beyond the cap **block in FIFO order** until a slot frees up. No 429, no
  retry overhead, no race for a freed slot — first in, first served.
- The semaphore size for each model is **derived from the gateway config**: the
  proxy reads `max_parallel_requests` from the LiteLLM config's `litellm_params` for
  that model. It is not a second, independently-tuned number that can drift away from
  the backend's real capacity. One source of truth for the cap, living next to the
  model it governs. Change the backend's true parallelism, change the cap in the
  gateway config, and the proxy picks it up — not the other way around.

Models that have no cap in the config are passed through without a queue.

### Why the cap is *derived*, not tuned

Each model's cap reflects the backend's actual concurrency capacity, not a number
someone optimized in isolation:

- A backend that runs **N engine processes** behind a round-robin wrapper → cap N.
- A server that advertises a parallelism flag (e.g. a `--parallel`-style or
  `NUM_PARALLEL`-style setting) → cap = that flag.
- A **single-stream engine** — one large model on one process that does not
  parallelize internally — → the honest cap is **1**: clean FIFO, no head-of-line
  surprises, because the engine truly serves one request at a time.
- A single-process GPU server with no real internal parallelism but many short
  interactive calls → a *generous* cap can be deliberate, as a fairness/admission
  knob that keeps short calls from head-of-line-blocking behind one slow large-batch
  call. In that case the cap is an admission decision, not a literal "this many run
  at once" claim — and the comment in the config should say so.

The discipline: **if you change a cap, change the backend that serves it in the same
commit**, or the two drift and the queue will either starve a capable backend or
oversubscribe a serial one.

### What the queue deliberately does *not* do

It does not load-balance traffic, it does not divide GPU-time fairly between
consumers, and it does not predict which prompt is expensive. It only prevents the
backend from being handed more concurrent work than it can serve, and it imposes a
global FIFO order on the overflow. Fairness beyond FIFO (per-key quotas, priority
classes) is out of scope for the queue itself.

---

## 2. The single-instrumentation-point principle

The reason the proxy can produce *honest* accounting is structural, not clever:

> The queue-proxy is the **only** component that sees the full lifecycle of **every**
> request.

Everything else on the path has a blind spot:

- **Insert-on-completion spend logging** (LiteLLM `SpendLogs` and similar) writes a
  row only when a call *runs to completion*. It therefore structurally cannot see:
  - **Queued calls** — a request that is waiting (or waiting and then cancelled)
    never reaches the gateway, so it is never logged. Yet a queue that is deep is the
    single most important thing to observe — it is the signal that you are at
    capacity.
  - **Client-abandoned calls** — a caller that disconnects mid-stream produces no
    completion row. These are exactly the calls you most want to count (they cost
    real backend time and they signal client-side timeouts), and they are exactly the
    ones survivor-biased logging drops.
- **Backend-level metrics** see backend-busy time but not queue wait, not the
  consumer identity, not the per-request token counts in one place.

Because the proxy is the one process on the whole path, it emits **exactly one
lifecycle row per request** into Postgres (`request_events`), capturing the four
timestamps that define the request's life:

| timestamp | meaning |
|---|---|
| `t_enqueue` | request hit the proxy (demand begins) |
| `t_acquire` | got the semaphore slot = backend start; **NULL** if cancelled while queued |
| `t_first_token` | first content byte (TTFT marker); **NULL** for non-stream / no tokens |
| `t_done` | last byte / connection closed = slot released |

…plus `model_requested` (the alias that hit the semaphore), `key_fp`
(`sha256(bearer)[:8]`, resolvable to a key alias for attribution), `streamed`,
`outcome` ∈ {`completed`, `client_abandoned`, `upstream_error`, `cancelled_queued`},
`http_status`, and `prompt_tokens` / `completion_tokens` taken from the gateway's own
`usage` block (**NULL when the backend did not report usage — never zero-filled**, so
a missing count never silently reads as zero).

Everything else is **derived once**, in `metrics_db.py`, never re-defined per
endpoint:

```
queue_wait = t_acquire     - t_enqueue        # time spent waiting in line
ttft       = t_first_token - t_acquire        # backend time to first token
decode     = t_done        - t_first_token    # streaming decode window
service    = t_done        - t_acquire        # true backend-busy span
total      = t_done        - t_enqueue        # true user-perceived latency
```

That is the guiding principle of the whole project: **instrument the call path once,
derive everything.** No reconciliation of disagreeing sources, no survivor bias, no
two endpoints that compute "latency" differently.

To make token counts arrive reliably, the proxy injects
`stream_options.include_usage=true` on streaming chat requests — without it many
backends omit the final usage block on streamed responses.

The writer is **non-blocking**: events go onto a bounded in-memory queue drained by a
background task that batch-inserts into Postgres. If that queue is full or the DB
write fails, the event is **dropped and counted** — the hot request path is never
slowed or failed by instrumentation. There is intentionally **no on-disk / JSONL
fallback**: instrumentation is best-effort and must never become a second failure
domain in front of live traffic.

---

## 3. Topology — proxy in front, gateway on loopback

```
        ┌──────────────────────────────────────────────┐
        │  consumers  (one base_url, one key each)      │
        └───────────────────────┬──────────────────────┘
                                │  network entry  :4000/v1
                ┌───────────────▼────────────────┐
                │  Overlaat queue-proxy          │
                │   • per-model FIFO semaphore   │
                │   • wait, don't 429            │
                │   • ONE lifecycle row/request ─┼──▶ request_events  (Postgres)
                │   • /__queue/* control + status│
                └───────────────┬────────────────┘
                                │  loopback  127.0.0.1:4002/v1
                ┌───────────────▼────────────────┐
                │  LiteLLM gateway               │
                │   auth · routing · UI · usage  │
                └───────────────┬────────────────┘
                ┌───────────────▼────────────────┐
                │  backends (loopback only)      │
                │   chat / embeddings / rerank / │
                │   STT / VLM / large models …   │
                └────────────────────────────────┘

  host sampler (periodic) ── host GPU%/RAM + per-backend RSS ──▶ host_samples  (Postgres)
  swap-slot model loads   ── cold-load start/ready             ──▶ model_loads   (Postgres)

  usage-api (:4100) reads request_events + host_samples + model_loads
                    ──▶ /now /timeline /models /perf /consumers + dashboard
```

The relationship is deliberate:

- **The queue-proxy is the only network-facing entry.** It binds the
  network-facing port (`:4000`). This is what makes it the single instrumentation
  site — there is no second door into the gateway.
- **The gateway binds loopback only** (`127.0.0.1:4002`). It is not reachable from
  the network directly. The only way to reach it from off-host is *through* the
  proxy, so every off-host request is queued and logged. (An on-host operator can
  still hit `127.0.0.1:4002` directly as a debug bypass — by definition that traffic
  is not instrumented, which is fine for debugging.)
- **Backends bind loopback only** as well; the gateway is the only thing that talks
  to them. Bind-hardening the backends is what closes the bypass: there is exactly
  one network entry port, `:4000`, and it is the queue.

Single uvicorn worker, on purpose: the in-memory per-model semaphores and the FIFO
ordering live in this one process. Sharding across workers would shard the queue and
the instrumentation, defeating both. Vertical scaling (one fast async worker) is the
model; horizontal scaling of the proxy itself is a non-goal at this size.

No TLS on the entry port in the base design: it assumes a network-level ACL (a
private overlay network or a trusted LAN segment) is the gate, and keys are
**attribution**, not secrets. Put a TLS-terminating reverse proxy in front if you
expose it more widely.

---

## 4. Streaming compatibility

The proxy is fully streaming-compatible and is careful not to break the properties
callers depend on:

- **SSE and plain JSON both pass through.** The proxy streams the backend's response
  body as it arrives; it does **not** buffer the response. A streamed token-by-token
  SSE response stays token-by-token through the proxy.
- **The request body is parsed once, minimally** — just enough to read the `model`
  field so the right semaphore is chosen — then forwarded. The body is not rewritten
  except for the `stream_options.include_usage=true` injection described above.
- **Headers are forwarded 1:1**, except hop-by-hop headers.
- **Non-LLM paths pass through untouched** — gateway UI, `/v1/models`, key-management
  routes, `/metrics`, etc. are forwarded without a queue.
- `t_first_token` is recorded at the first content byte of the stream, giving an
  honest TTFT even though the proxy never holds the stream.

---

## 5. Cancellation — queued only, and why in-flight is unsafe

The control endpoints can cancel requests, but **only requests that are still queued**
(waiting, not yet dispatched to the backend):

- `POST /__queue/cancel/{req_id}` — cancel one queued request.
- `POST /__queue/cancel-all?model=&key_fp=` — cancel all queued requests in scope.

A queued request never touched the backend — it is purely waiting for a semaphore
slot — so dropping it is safe and costs the backend nothing. The caller receives a
**499** (client-closed) and the event is recorded with outcome `cancelled_queued`.

**In-flight requests are deliberately NOT cancellable**, and this is the subtle part:

> With a single-stream engine, a client disconnect (or a proxy-side cancel) lets the
> proxy release its semaphore slot, **but the engine keeps decoding.**

The slot release and the engine's actual state **desync**. The proxy thinks the
backend is free and admits the next queued request — which then **stalls on a backend
that is in fact still busy** finishing the abandoned generation. So
release-on-disconnect does *not* stop the backend; it only corrupts the proxy's model
of backend occupancy. Cancelling a still-queued request is safe precisely because it
never created that coupling.

This has a direct consequence for the accounting (stated once, in the caveats):
`service = t_done - t_acquire` measures **slot occupancy**, not literal GPU-busy time
past a release. After an abandon, a single-stream engine can keep decoding briefly
beyond `t_done`.

---

## 6. The read-only usage-API

A separate FastAPI service (`:4100`) reads the three tables and **only ever reads** —
it never writes, so it can never interfere with the live path. No auth; a network ACL
is the gate.

### Concurrency — three curves, each defined exactly once

From `request_events`, at time *t*, per model:

- **offered(t)** = requests with `t_enqueue ≤ t < t_done` — total demand, *including*
  queued.
- **active(t)** = requests with `t_acquire ≤ t < t_done` — backend busy (slot held).
  Bounded by the cap **by definition** — that is correct, not a bug.
- **queued(t)** = offered − active — the backlog.

"Throughput vs concurrency" is computed honestly: each completed call is bucketed by
the **time-weighted mean of active(t) over its own `[acquire, done]` span** — the
parallelism it actually experienced (1.0 = it ran alone) — and
`aggregate_tok_s = Σtokens / Σservice` per bucket. Cells with fewer than a minimum
sample count are returned `sufficient:false` and never shown as a trend.

### Endpoints (sketch)

| route | answers |
|---|---|
| `GET /` | HTML dashboard |
| `GET /now` | live: per-model in_flight/queued (scraped from the proxy's `/__queue/status`, because events are written on completion), latest host GPU%/RAM + backend RSS, recent completed per key |
| `GET /timeline?last=` | time-series: host GPU%/RAM + backend RSS, per-model offered/active, per-key active concurrency. Bucket size auto-picked from the window. |
| `GET /models?last=` | capacity: per model outcome counts, latency split (queue_wait / ttft / service / total p50/p95), throughput-by-measured-concurrency (min-sample guarded) |
| `GET /perf?last=` | backend-health: per-model decode tok/s over time, with an all-calls median and a **solo** median (calls that ran near-alone) that isolates engine health from load — a sustained drop in the solo line means the backend is degrading, independent of how busy it is |
| `GET /consumers?last=` | per key alias: requests by outcome, tokens, service-seconds, abandoned-rate, models used |
| `GET /healthz` | liveness + DB check |

Live in-flight necessarily comes from the proxy's in-memory status endpoint, not from
the events table — events are written on completion, so a call that is *right now*
queued or streaming is not in `request_events` yet. `/now` therefore joins the
proxy's live view with the latest persisted host sample.

---

## 7. Host facts and cold-loads (`host_samples`, `model_loads`)

Two auxiliary collectors, both writing to the same Postgres, both read-only consumed
by the usage-API:

- **Host sampler** — writes one `host_samples` row every few seconds: host GPU% / GPU
  frequency, RAM breakdown, CPU load, and `backends_json` = the top memory holders
  `[{name, pid, rss_gb, cpu_pct, gpu_pct}]`. This answers "what is filling memory
  right now" by **RSS attribution**. Per-process *GPU* attribution is **not** reliably
  measurable on all platforms (notably Metal/MLX workloads on macOS report ~0), so
  GPU% is kept host-wide and only *memory* is attributed per backend.
- **Model-load log** — when backends share a swap-on-demand slot (only one large
  model resident at a time; loading another evicts the current one), each cold load is
  recorded in `model_loads` with `t_start` / `t_ready` / `load_s`. This makes
  cold-load time **explicit** instead of hiding it inside the first request's TTFT —
  so a benchmark can attribute a slow first call to the model *swap*, not to the
  engine. Granularity is the sampler interval, so `load_s` slightly underestimates; a
  NULL `t_ready` means the load was aborted or the model was evicted before it became
  ready.

---

## 8. Routing posture — no silent failover

The gateway is configured with **no cross-model fallbacks** (by design). If a
consumer asks for a specific model, it must either get that model or see a **clean
error** — never a quiet substitution to a different or simpler model. The
orchestration layer (the caller / harness) decides what to do on failure, because
only it knows whether a different model is acceptable for the task. LiteLLM-style
fallback is appropriate for *redundant equivalent deployments* (the same model behind
multiple keys/regions), which is a different situation from this one.

The router timeout and the request timeout are set generously but **bounded**
(long-context synthesis under contention can run well past the usual default), and the
gateway timeout must be ≥ the proxy's read-timeout so the proxy does not give up on a
call the backend is still legitimately serving.

---

## 9. The tables (genericized)

DDL lives in `schema.sql` (idempotent). All timestamps are **epoch seconds (UTC,
sub-millisecond)** — no timezone games. The three tables:

- **`request_events`** — one row per request, written by the queue-proxy on
  completion. The lifecycle row described in §2.
- **`host_samples`** — one row per sampling interval, written by the host sampler.
  Host GPU%/RAM + per-backend RSS.
- **`model_loads`** — one row per cold load in a shared swap slot, written by the host
  sampler from polling the swap manager's running-model endpoint.

They live in the same Postgres database the gateway uses for its own state (e.g.
`localhost:5432/<db>`). Overlaat's schema is **not** a backwards-compatible extension
of insert-on-completion spend logging — it replaces that role entirely as the source
of request-level truth.

---

## 10. Configuration surface

Everything is environment-overridable; defaults are local and relative so the project
runs out of a checkout without site-specific paths.

| env var | default | meaning |
|---|---|---|
| `DATABASE_URL` | — | Postgres the proxy/sampler write to and the usage-API reads. No on-disk fallback; if it is down the proxy fails the request cleanly. |
| `QUEUE_PROXY_UPSTREAM` | `http://127.0.0.1:4002` | the loopback LiteLLM gateway the proxy forwards to |
| `QUEUE_PROXY_LITELLM_CONFIG` | `./litellm-config.yaml` | the gateway config the proxy parses to size one semaphore per model (it reads only `max_parallel_requests`) |

Secrets (Postgres credentials, gateway master key) come from an env-file (default
`./overlaat.env`, `chmod 600`), never from the LiteLLM config YAML and never committed.

Run each service behind a process supervisor of your choice; to pick up a config or
cap change, **restart the backend service / the proxy**. One operational caveat worth
repeating: **only restart the queue-proxy when its queue is empty** (no in-flight, no
queued) — a restart drops queued calls, turning them into instant failures for
callers who were politely waiting in line. The usage-API and host sampler are safe to
restart at any time.

---

## 11. Non-goals (so the scope stays honest)

- **Not a load balancer.** No GPU-time fairness between consumers, no cost prediction,
  no request scheduling beyond per-model FIFO admission.
- **Not multi-tenant billing.** Keys are attribution fingerprints, not metered
  secrets; "spend" is informational, not authoritative.
- **Not horizontally scaled.** One async worker owns the queue and the
  instrumentation, by design.
- **Not a tracing system.** It records request lifecycles, not multi-step agent
  traces; pair it with a dedicated tracer if you need tool-call-level trees.
- **No silent model failover.** A model works or fails cleanly; the caller decides.

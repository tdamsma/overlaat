# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) — though while the
status is **experimental**, the APIs and the Postgres schema may change between
versions without a compatibility guarantee.

## [Unreleased]

### Added
- **Per-model `overlaat_abort_on_disconnect` policy** (`model_info` bool, default `true`) —
  governs what an in-flight client disconnect does to the slot. `true` (today's behaviour)
  releases the slot the instant the client disconnects, correct for abort-honouring
  continuous-batching engines. `false` holds the slot and keeps draining the upstream to its
  natural end (bounded by the read-timeout) before releasing, for single-stream engines with
  no abort path — so slot accounting stays in sync with real backend occupancy and the next
  call queues instead of stalling on a still-busy backend. #28

## [0.0.8] — 2026-06-22

Usage-dashboard redesign. Observability/UI only — no scheduler, proxy, or schema changes.

### Added
- **Recent-requests table** + `GET /requests` — a searchable / sortable / filterable
  view of the last *N* finished requests (default 100, max 500), one row each with
  consumer, model, workload, outcome, the queue-wait / ttft / service / total latency
  split, prompt & completion tokens, and decode tok/s.
- **Queued-now-by-user card** — live in-memory queue state grouped per consumer, showing
  each waiting request's model, age, priority, and *why* it is parked
  (`model_cap` / `budget_full` / `exclusive`). `/now` now surfaces the per-waiter
  consumer attribution and `wait_reason` the scheduler already computed.
- **Server-health verdict** — a synthesized OK / degraded / stalled read with the worst
  contributing signal (GPU, slots-held-but-idle stall, queue backlog, per-model
  decode-tok/s drift).
- **Shared-axis time-chart stack** — the time-series charts share one x-axis with a hover
  crosshair that reads every series at the same instant.

### Changed
- **Dashboard charts are now time-weighted and step-rendered.** Per-call quantities are
  spread across the service window `[t_acquire, t_done]` and aggregated per bucket (the
  same interval math as the concurrency curves) instead of being point-sampled at
  `t_done` — fixing the sparse-dots throughput plot and the jerky long-window
  aggregation. New charts: input-vs-output tok/s, output tok/s stacked by model, and
  work (GPU-busy share) stacked by consumer.
- **Three-column dashboard layout** — detail tables (left), stacked shared-axis time
  charts (middle), at-a-glance health + attribution (right).
- The solo decode-throughput health signal and median output-size moved from sparse trend
  charts into per-model table columns.

### Removed
- `GET /perf` and the standalone decode-throughput / output-size trend charts (superseded
  by the per-model table columns and the new throughput charts).

## [0.0.7] — 2026-06-22

### Added
- **Configurable upstream HTTP timeouts** (#24, builds on #18). The proxy's httpx
  timeout to LiteLLM was hardcoded (`connect 5 / read 1200 / write 60 / pool 5`);
  all four are now env-overridable via `OVERLAAT_UPSTREAM_{CONNECT,READ,WRITE,POOL}_TIMEOUT`
  (defaults `5 / 300 / 60 / 5`). The read default is lowered `1200 → 300` so a
  wedged stream frees its budget slot promptly. Note that httpx `read` is the
  **inter-byte** timeout, not a total-request deadline — a slow trickle of tokens
  never trips it — so it is only a wedged-connection backstop. The authoritative
  *total* per-request deadline is the inference **engine's own `--timeout`**
  (operator deployment); the proxy read-timeout should sit just above it so the
  engine's clean cancel wins.
- **Startup warning when the cost-scheduler's self-protection is inert** (#24). The
  protection against a single workload's oversized prompts needs *both* non-flat
  prompt-weight tiers *and* a pool budget that can bind, so it is inert if **either**
  is missing (`flat OR unbounded`): **flat** tiers (every multiplier `1.0`) price a
  giant prompt the same as a tiny one, and an **effectively-unbounded** pool budget
  leaves only the raw per-model cap to bind — so a few giant prefills pin every slot
  and starve the fast lane. Critically, the unbounded-budget case is flagged *even
  with the default non-flat tiers* — the live incident config (`OVERLAAT_BUDGET=9999`
  with default tiers) is unprotected, and an earlier short-circuit that exempted
  non-flat configs would have missed it. The proxy detects this at startup and writes
  a loud `stderr` warning naming the failing condition(s), the inert pool(s), and the
  remediation (set `OVERLAAT_PROMPT_WEIGHT_TIERS` to a non-flat table and/or lower
  `OVERLAAT_BUDGET`). This also confirms the **non-flat default tiers ship enabled**,
  so a fresh deploy (bounded `OVERLAAT_BUDGET=1.0`) is protected without any env var.
- **Regression reproducer for the oversized-prompt vector** (#24). New
  scheduler-unit contrast tests (protective default config vs the inert incident
  config) plus an end-to-end load harness proving a single workload's oversized
  prompts are budget-throttled one-at-a-time and the latency-sensitive fast lane is
  never starved, alongside a unit test for the inert-config detector.

### Fixed
- **Cost-scheduler budget no longer leaks the prompt-weight surplus on heavy
  releases** (#24, builds on #18). A prompt-size-weighted request was *charged* its
  weighted cost on admission but *refunded* only the base `1/cap` on release, so the
  committed budget never returned to zero after a heavy request finished and would
  drift ever more restrictive. `Scheduler.release()` now takes the exact cost the run
  was charged (`release(model, cost=…)`), threaded through from the admitting waiter,
  and refunds precisely that. Refunding the exact charged amount — rather than a
  per-model guess — is also required for correctness under heavy+light interleaving:
  a light request finishing while a later heavy is still in flight must refund only
  its own small cost, or a second heavy would over-admit and break the `leave_room`
  fast-lane guarantee. Found by the new #24 regression tests; observability/cost
  columns are unchanged.

### Notes
- **Per-consumer (throughput) fairness remains out of scope** and is tracked
  separately — this work covers the self-protection defaults, configurable timeouts,
  and the regression gate only. The operator-side engine config (chunked prefill, an
  authoritative engine `--timeout`, `--max-tokens`) lives in the engine deployment,
  not in Overlaat, and is captured here only as a documented contract
  (see `docs/COST-SCHEDULER.md`).

## [0.0.6] — 2026-06-22

### Added
- **Per-request workload label for segmented observability** (closes #19). A single
  consumer key often serves workloads with opposite SLAs (e.g. low-priority
  huge-doc summaries vs latency-critical small-prompt synthesis), but `key_fp` was
  the finest granularity logged, so they could not be reported apart. A new nullable
  `request_events.workload TEXT` column records a caller-supplied label, resolved per
  request from the `X-Overlaat-Workload` header (wins) or the body's
  `metadata.workload` field (fallback), sanitized to a non-empty string trimmed to
  64 chars (anything else → `NULL`). Both inputs are stripped before forwarding (the
  header is dropped hop-by-hop; only the `metadata.workload` sub-key is popped, the
  rest of `metadata` is left intact) so this overlaat-private input never reaches the
  backend. The usage-api gains a `GET /workloads` endpoint and a dashboard card with
  a per-workload breakdown (requests, p50/p95 queue wait + total latency, completion
  tokens, abandoned/error rate; untagged traffic groups under `(untagged)`). Existing
  databases upgrade idempotently via `ALTER TABLE request_events ADD COLUMN IF NOT
  EXISTS workload TEXT` (sqlite mirrors this by diffing `PRAGMA table_info`).
  **Observability only — the label is logged and displayed but never read by the
  scheduler; cost, priority, pool, and admission are unchanged.**
- **Prompt-size-weighted admission cost** (closes #18). Admission cost was a flat
  `1/cap` per model, so a 50-token and a 33k-token prompt cost the same — and under
  FIFO a few heavy prompts fill every slot and inflate queue wait for the small
  interactive calls behind them. Cost is now `cost(model) × weight(prompt_tokens)`,
  where the size is a cheap chars/≈4 estimate of the request body (no tokenizer in
  the hot path; non-chat bodies like `/embeddings` get `1×`) and the weight is a
  tier table (default `≤2k → 1×, 2k–8k → 2×, >8k → 4×`, override with
  `OVERLAAT_PROMPT_WEIGHT_TIERS`). The weighted cost is **hard-clamped to the pool
  budget** per a new per-pool `overlaat.pools.<pool>.heavy_max`: `leave_room`
  (default — caps at `budget − base_cost` so one light call always fits alongside
  the heaviest prompt) or `full_pool` (a giant prompt may take the whole pool and
  run alone). The charged cost is logged in `request_events.cost` as before — no
  schema change. **Default behaviour is unchanged for prompts ≤2k tokens** (`1×`);
  set every multiplier to `1` to disable weighting entirely.

### Fixed
- **`request_events.wait_reason` no longer mislabels cap/exclusion waits as
  `budget_full`** (closes #17). The reason was finalized at admission by
  recomputing `_cap_full` / `_exclusion_blocks` — but those blockers have
  necessarily cleared by the moment a waiter is admitted (that is *why* it is
  admitted), so every cap- or exclusion-bound wait fell through to the
  `budget_full` catch-all (observed: 816/858 waits on a live deployment with a
  non-binding budget and a binding per-model cap). The live cause is now latched
  onto the waiter while it is parked (`_record_wait_reasons`) and read back at
  admission; `reserved` / `aged_in` still derive from state that survives to
  admission. Observability-only — admission, caps, pool budgets, exclusivity and
  throughput were always correct.

## [0.0.5] — 2026-06-21

### Changed
- **`host_logger` backend prefixes are now env-configurable** (closes #14). The
  `BACKEND_EXE_PREFIXES` list — executable-name prefixes that get a port-suffixed
  name in the per-backend RSS breakdown — reads from the `BACKEND_EXE_PREFIXES`
  environment variable (comma-separated), falling back to the previous hardcoded
  default (`python,ollama,model-server,mlx_lm`). A deployment with a custom
  inference binary (e.g. `ds4-server`) can now get its per-engine attribution
  without patching installed package source. Matches the module's existing
  env-config style (`TOTAL_MEM_GB`, `PSQL`, `SLOT_RUNNING_URL`); default behavior
  is unchanged.

## [0.0.4] — 2026-06-20

### Added
- **Resource pools + exclusive groups** in the cost-weighted scheduler (closes #11, #12).
  Each model is now assigned to exactly one named **resource pool** (`model_info.overlaat_pool`,
  default `default`), and admission is cost-weighted against THAT pool's own budget instead of
  one global ledger. A pool blocked on budget never idles a *different* pool, so cross-backend
  concurrency is preserved (#11) — e.g. an embeddings pool is never stalled by a busy swap
  engine. Pools are declared in a new OPTIONAL top-level `overlaat.pools` section of the LiteLLM
  config (`budget` + `exclusive` per pool); a pool a model references but does not declare is
  auto-created (non-exclusive, `OVERLAAT_BUDGET`) and logged at startup. The global priority
  queue, eager head reservation (now per-pool), linear aging, and no-preemption are unchanged.
  **Default is unchanged:** with every model in the single `default` pool the scheduler
  reproduces the previous single-shared-budget behavior exactly. There is **no new env knob** —
  `OVERLAAT_BUDGET` is the budget of the `default` pool and of any auto-created pool.
- **Honest exclusive (swap-slot) pools** replacing the forced-1.0 fat-slot cost hack (closes #12).
  An `exclusive: true` pool is a HARD MUTEX over its *distinct members*: at most one member id
  may be active at a time, but that resident member's own cap + the pool budget govern how many
  of *its* streams run concurrently. So a cap-2 dual-engine member honestly costs `1/2 = 0.5`
  and runs **both** its streams while resident — instead of being forced to `cost = 1.0` and
  capped at one stream. This also fixes the `0.5 + 0.5` trap: with a cap-2 member resident at
  `used 0.5` there is 0.5 of budget headroom, yet a *different* member is still rejected
  (`wait_reason = exclusive`) until the resident fully drains and the mutex hands off.

### Changed
- New `request_events.pool` column (the resource pool a request was admitted against; `default`
  for unconfigured models, `NULL` when the scheduler is off / pre-upgrade) and a new
  `wait_reason` value `exclusive` (a *different* member held an exclusive pool's mutex). The
  `/__queue/health` and `/__queue/status` endpoints now expose a per-pool snapshot (`budget` gains
  a `pools` map of `{budget, used, exclusive, active_member, reserved_for, in_flight, budget_pct}`)
  and per-model rows + queued entries gain `pool`; the legacy top-level `budget`/`used`/`in_flight`/
  `reserved_for`/`budget_pct` fields are kept (they reflect the `default` pool) so existing readers
  keep working.
- The `model_info.overlaat_slot` key is now a **deprecated-but-bridged LEGACY ALIAS**:
  `overlaat_slot: NAME` (with no `overlaat_pool`) is treated as `overlaat_pool: NAME` with that
  pool auto-marked `exclusive: true`. Old swap-slot configs keep working unchanged — a cap-1
  fat-slot is byte-identical to before.
- **Upgrade step for existing deployments:** the new `request_events.pool` column is in
  `schema.sql` (idempotent). To add it to an **existing** Postgres table without recreating it:
  ```sql
  ALTER TABLE request_events ADD COLUMN IF NOT EXISTS pool TEXT;
  ```
  (SQLite: `ALTER TABLE request_events ADD COLUMN pool TEXT;` — run once.) Pre-upgrade rows keep
  `NULL`, which the dashboard treats as "scheduler off".

## [0.0.3] — 2026-06-20

### Added
- **Cost-weighted priority scheduler** (`overlaat/scheduler.py`), replacing the
  independent per-model semaphores with **one global priority queue + cost-weighted
  admission against a single shared GPU budget `B`** (default `1.0`). Each in-flight run
  consumes `cost = 1/cap` (a `model_info.overlaat_cost` override, or `1.0` for a
  swap-slot member or an uncapped model); a request is admitted only when both the
  per-model cap binds (`model_in_flight < cap`) and the shared budget has room
  (`used + cost ≤ B + ε`). Packing is work-conserving with an eager head reservation and
  optional linear aging so a drip of cheap jobs cannot starve an expensive one; there is
  no preemption. Ordering is by effective priority = `min(request priority, per-key
  ceiling) + aging`. The per-key ceiling is read from `LiteLLM_VerificationToken.metadata`
  (`overlaat_priority`), cached in memory and refreshed ~every 60 s — never on the hot
  path — with graceful fallback to `OVERLAAT_DEFAULT_PRIORITY`. Swap-slot ("fat-slot")
  groups are modeled as `cost = 1.0`, so the "one big model at a time" mutex falls out of
  the budget arithmetic with no separate lock. Full design: `docs/COST-SCHEDULER.md`.
- New env knobs: `OVERLAAT_SCHEDULER` (default `on`; `off` is a kill-switch restoring the
  exact per-model `asyncio.Semaphore` FIFO path), `OVERLAAT_BUDGET` (`1.0`),
  `OVERLAAT_DEFAULT_COST` (`1.0`), `OVERLAAT_DEFAULT_PRIORITY` (`0`), `OVERLAAT_AGING_RATE`
  (`0.0`, aging off), `OVERLAAT_RESERVATION_GRACE` (`0.0`, eager reservation).
- New `request_events` columns `priority`, `cost`, and `wait_reason` (`none` |
  `reserved` | `aged_in` | `budget_full` | `model_cap`), recorded per admitted/cancelled
  request when the scheduler is on (`NULL` otherwise). The queue-proxy `/__queue/health`
  and `/__queue/status` endpoints now expose a `budget` object (`used`/`budget`/
  `budget_pct`/`reserved_for`) and per-queued-entry `priority`/`effective_priority`/
  `cost`/`wait_reason`.

### Changed
- **Behavior change (scheduler on by default):** admission is now the cost-weighted
  global scheduler, **not** independent per-model semaphores. The shared budget `B` caps
  the *summed* cost across all models, so peak concurrency is **lower** than the old
  "sum of caps" (and honest about the single GPU). With nothing configured — `cost =
  1/cap`, `B = 1.0`, equal priority, aging off — a single model reduces to per-model
  FIFO, matching the old semaphore; set `OVERLAAT_SCHEDULER=off` to restore the previous
  path byte-for-byte.
- **Upgrade step for existing deployments:** the three new `request_events` columns are
  in `schema.sql` (idempotent — `python -m overlaat.db init "$DATABASE_URL"` or `psql -f
  schema.sql` adds them on a fresh table). To add them to an **existing** Postgres table
  without recreating it:
  ```sql
  ALTER TABLE request_events ADD COLUMN IF NOT EXISTS priority    INTEGER;
  ALTER TABLE request_events ADD COLUMN IF NOT EXISTS cost        DOUBLE PRECISION;
  ALTER TABLE request_events ADD COLUMN IF NOT EXISTS wait_reason TEXT;
  ```
  (SQLite: `ALTER TABLE request_events ADD COLUMN priority INTEGER;` etc. — SQLite has no
  `IF NOT EXISTS` for columns, so run each once.) Pre-upgrade rows keep `NULL` in all
  three, which the dashboard and curves treat as "scheduler off".
- The package version is now derived from the git tag via `hatch-vcs` (`[tool.hatch.version] source = "vcs"`); the hardcoded `__version__` literal and the CI tag↔version guard have been removed. Cutting a release is just pushing a `vX.Y.Z` tag — CI builds and publishes from it. At runtime `overlaat.__version__` is read back from the installed package metadata.
- SQLite connections now set `PRAGMA busy_timeout=5000` (alongside `journal_mode=WAL`) at every connection point — read path (`db.connect`), writer (`db.connect_sqlite_write`), and the host sampler's connect helper — so concurrent dashboard reads and the single writer wait briefly instead of immediately raising "database is locked".
- Documented SQLite operations (WAL `-wal`/`-shm` sidecar files, single-writer/no-`--workers` rule, schema init via `python -m overlaat.db init`, and online backup with `sqlite3 ".backup"` / `VACUUM INTO`) in the README and `docs/OBSERVABILITY.md`.

### Fixed
- `db.normalize_backends_json` now emits a one-line warning when `host_samples.backends_json` fails to parse (returning `None` as before), so silently corrupted JSON becomes visible instead of disappearing into a NULL.

## [0.0.2] — 2026-06-19

### Added
- Wheel-install smoke-test CI job: installs the built wheel into a fresh `uv` virtualenv and imports the package and its submodules, guarding against packaging regressions.
- Contributor scaffolding: `CONTRIBUTING.md` (uv-based dev setup, test/lint commands, the version/tag-guard release rule), GitHub issue templates, and a `docs/` index.
- Tests for contended queue behavior: FIFO admission under a full per-model semaphore, cancel-while-queued vs in-flight, and the lifecycle outcome strings (`cancelled_queued`, `completed`, `upstream_error`, `client_abandoned`).
- Optional SQLite storage backend, selected by the `DATABASE_URL` scheme (`sqlite:///path.db`); Postgres remains the default. Adds an `overlaat.db` dialect layer and a `python -m overlaat.db init [DATABASE_URL]` schema initializer that works for both backends. No new mandatory dependencies (SQLite uses the standard-library `sqlite3`).
- The running Overlaat version is now shown in the usage-API dashboard.
- Each request_events row records the Overlaat version that served it (new `overlaat_version` column).

### Changed
- README: added PyPI version, CI status, Python-versions, and MIT license badges; the quickstart now leads with `uv pip install overlaat` and demotes the editable install to a uv-based development note.
- Rewrote the architecture diagram (`docs/overlaat-llm-stack.excalidraw.svg`) to show
  where Overlaat fits: a custom fair-queue + honest-accounting ring (queue-proxy `:4000`,
  usage-api `:4100`, and the event store) wrapped around an off-the-shelf serving stack,
  with model loading/eviction drawn as a separate swap layer (e.g. llama-swap) rather
  than something Overlaat does itself.

## [0.0.1] — 2026-06-19

First public release. Overlaat is a sidecar that sits in *front* of a self-hosted,
multi-backend LiteLLM gateway and adds the two things the gateway does not do: a fair
waiting-queue instead of a 429-cliff, and one honest usage event per request. This
version implements:

### queue-proxy (`:4000`) — the single network entry point
- **Per-model FIFO wait-queue.** Every `/v1/chat/completions`, `/v1/completions`,
  `/v1/embeddings`, and `/rerank` call passes through a per-model `asyncio.Semaphore`;
  overflow **waits in FIFO order** rather than being rejected with HTTP 429.
- **Caps derived from the gateway config**, not tuned separately — the slot size for
  each model is read from `max_parallel_requests` in `litellm-config.yaml`. Models with
  no cap (and all non-LLM paths) pass through without a queue.
- **Streaming-compatible** (SSE and plain JSON), forwarding the body unbuffered and
  headers 1:1 except hop-by-hop. Injects `stream_options.include_usage=true` on
  streaming chat so token counts arrive reliably.
- **One lifecycle event per request** written to Postgres `request_events`, *including
  queued and client-abandoned calls* that insert-on-completion logging structurally
  misses. Captures `t_enqueue` / `t_acquire` / `t_first_token` / `t_done`, outcome,
  model, key fingerprint, and token counts (NULL, never zero, when unreported). The
  writer is non-blocking (bounded queue + background batch insert); on overflow or DB
  error the event is dropped and counted — the hot path is never slowed.
- **Control + status endpoints:** `/__queue/health`, `/__queue/status`,
  `/__queue/cancel/{req_id}`, `/__queue/cancel-all`. Cancellation affects **queued
  requests only** (in-flight calls are deliberately not cancellable).

### usage-api (`:4100`) — read-only dashboard
- FastAPI service that **only ever reads** the event/host tables, serving an HTML
  dashboard plus `/now`, `/timeline`, `/models`, `/perf`, `/consumers`, `/healthz`.
- Derives the three honest concurrency curves (**offered / active / queued**),
  throughput bucketed by time-weighted measured concurrency (min-sample guarded), and a
  **solo decode tok/s** backend-health signal that isolates engine degradation from load.

### host sampler (optional, macOS)
- `host_logger` samples GPU% / RAM and per-backend RSS into `host_samples` every few
  seconds, and logs swap-slot cold loads into `model_loads`. Memory is attributed
  per-backend by RSS; GPU% is kept host-wide (per-process GPU is not measurable for
  Metal/MLX).

### schema & packaging
- `schema.sql` — idempotent DDL for the three tables (`request_events`, `host_samples`,
  `model_loads`); all timestamps are epoch seconds (UTC).
- Pure-Python package (`hatchling`, Python ≥ 3.11) with example config, env, and run
  scripts under `examples/`.

### Not yet implemented
- **Cost-weighted admission** is design-only (see `docs/COST-SCHEDULER.md`); the queue
  is plain per-model FIFO in this version.

[Unreleased]: https://github.com/tdamsma/overlaat/compare/v0.0.8...HEAD
[0.0.8]: https://github.com/tdamsma/overlaat/releases/tag/v0.0.8
[0.0.7]: https://github.com/tdamsma/overlaat/releases/tag/v0.0.7
[0.0.6]: https://github.com/tdamsma/overlaat/releases/tag/v0.0.6
[0.0.5]: https://github.com/tdamsma/overlaat/releases/tag/v0.0.5
[0.0.4]: https://github.com/tdamsma/overlaat/releases/tag/v0.0.4
[0.0.3]: https://github.com/tdamsma/overlaat/releases/tag/v0.0.3
[0.0.2]: https://github.com/tdamsma/overlaat/releases/tag/v0.0.2
[0.0.1]: https://github.com/tdamsma/overlaat/releases/tag/v0.0.1

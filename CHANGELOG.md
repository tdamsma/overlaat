# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) — though while the
status is **experimental**, the APIs and the Postgres schema may change between
versions without a compatibility guarantee.

## [Unreleased]

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

[Unreleased]: https://github.com/tdamsma/overlaat/compare/v0.0.5...HEAD
[0.0.5]: https://github.com/tdamsma/overlaat/releases/tag/v0.0.5
[0.0.4]: https://github.com/tdamsma/overlaat/releases/tag/v0.0.4
[0.0.3]: https://github.com/tdamsma/overlaat/releases/tag/v0.0.3
[0.0.2]: https://github.com/tdamsma/overlaat/releases/tag/v0.0.2
[0.0.1]: https://github.com/tdamsma/overlaat/releases/tag/v0.0.1

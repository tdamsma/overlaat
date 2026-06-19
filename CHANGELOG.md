# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) — though while the
status is **experimental**, the APIs and the Postgres schema may change between
versions without a compatibility guarantee.

## [Unreleased]

### Added
- Wheel-install smoke-test CI job: installs the built wheel into a fresh `uv` virtualenv and imports the package and its submodules, guarding against packaging regressions.
- Contributor scaffolding: `CONTRIBUTING.md` (uv-based dev setup, test/lint commands, the version/tag-guard release rule), GitHub issue templates, and a `docs/` index.
- Tests for contended queue behavior: FIFO admission under a full per-model semaphore, cancel-while-queued vs in-flight, and the lifecycle outcome strings (`cancelled_queued`, `completed`, `upstream_error`, `client_abandoned`).
- Optional SQLite storage backend, selected by the `DATABASE_URL` scheme (`sqlite:///path.db`); Postgres remains the default. Adds an `overlaat.db` dialect layer and a `python -m overlaat.db init [DATABASE_URL]` schema initializer that works for both backends. No new mandatory dependencies (SQLite uses the standard-library `sqlite3`).

### Changed
- README: added PyPI version, CI status, Python-versions, and MIT license badges; the quickstart now leads with `uv pip install overlaat` and demotes the editable install to a uv-based development note.

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

[0.0.1]: https://github.com/tdamsma/overlaat/releases/tag/v0.0.1

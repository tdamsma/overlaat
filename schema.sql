-- Overlaat metrics pipeline schema.
-- Single source of truth = the call path (queue-proxy) + host sampler.
-- Lives in the LiteLLM Postgres (e.g. localhost:5432/litellm). No backwards
-- compat with insert-on-completion spend logging: this replaces that entirely.
--
-- Apply: psql "$DATABASE_URL" -f schema.sql   (idempotent)

-- request lifecycle events
-- One row per request, written by the queue-proxy on completion. Captures the
-- FULL lifecycle incl. queued + client-abandoned calls (invisible to
-- insert-on-completion spend logging, which only writes rows for calls that
-- run to completion).
-- All timestamps are epoch SECONDS (UTC, sub-ms) — no timezone games.
CREATE TABLE IF NOT EXISTS request_events (
  id                BIGSERIAL PRIMARY KEY,
  t_enqueue         DOUBLE PRECISION NOT NULL,   -- request hit the proxy
  t_acquire         DOUBLE PRECISION,            -- got the semaphore slot (backend start); NULL = cancelled while queued
  t_first_token     DOUBLE PRECISION,            -- first content byte (TTFT marker); NULL = non-stream / no tokens
  t_done            DOUBLE PRECISION NOT NULL,   -- last byte / connection closed (slot released)
  model_requested   TEXT NOT NULL,               -- alias the client asked for (what hits the semaphore)
  key_fp            TEXT NOT NULL,               -- sha256(bearer)[:8] == LiteLLM_VerificationToken.token[:8]
  streamed          BOOLEAN NOT NULL DEFAULT false,
  outcome           TEXT NOT NULL,               -- completed | client_abandoned | upstream_error | cancelled_queued
  http_status       INTEGER,
  prompt_tokens     INTEGER,                     -- NULL = backend did not report usage (never zero-filled)
  completion_tokens INTEGER,
  overlaat_version  TEXT,                         -- Overlaat version that served this request; NULL for pre-upgrade rows
  priority          INTEGER,                      -- effective base priority used at admission (cost-scheduler); NULL = scheduler off / pre-upgrade
  cost              DOUBLE PRECISION,             -- pool-fraction cost charged for this run (1/cap by default); NULL = scheduler off / no cap
  wait_reason       TEXT,                         -- why the request waited: none|reserved|aged_in|budget_full|model_cap|exclusive; NULL = scheduler off
  pool              TEXT,                         -- resource pool the request was admitted against (default "default"); NULL = scheduler off / pre-upgrade
  workload          TEXT                          -- caller-supplied workload label (observability only; never affects scheduling); NULL = untagged / pre-upgrade
);
-- Idempotent upgrade for tables that predate a column. CREATE TABLE IF NOT EXISTS
-- never alters an existing table, so a column added after the table first shipped
-- needs its own guarded ALTER to reach already-deployed databases.
ALTER TABLE request_events ADD COLUMN IF NOT EXISTS workload TEXT;
CREATE INDEX IF NOT EXISTS ix_re_tenq    ON request_events (t_enqueue);
CREATE INDEX IF NOT EXISTS ix_re_tdone   ON request_events (t_done);
CREATE INDEX IF NOT EXISTS ix_re_model   ON request_events (model_requested, t_enqueue);
CREATE INDEX IF NOT EXISTS ix_re_outcome ON request_events (outcome);

-- host samples
-- One row per 5s, written by the usage-logger. Host GPU%/RAM + per-backend RSS.
-- RSS is measurable per-process; GPU% per-process is NOT reliably measurable
-- on all platforms (notably Metal/MLX workloads on macOS report 0), so we
-- attribute MEMORY by RSS and leave GPU% host-wide.
CREATE TABLE IF NOT EXISTS host_samples (
  ts                DOUBLE PRECISION PRIMARY KEY,  -- epoch seconds
  gpu_pct           DOUBLE PRECISION,
  gpu_freq_mhz      DOUBLE PRECISION,
  ram_total_gb      DOUBLE PRECISION,
  ram_wired_gb      DOUBLE PRECISION,
  ram_active_gb     DOUBLE PRECISION,
  ram_inactive_gb   DOUBLE PRECISION,
  ram_compressed_gb DOUBLE PRECISION,
  ram_free_gb       DOUBLE PRECISION,
  cpu_load1         DOUBLE PRECISION,
  cpu_load5         DOUBLE PRECISION,
  backends_json     JSONB                          -- [{name,pid,rss_gb,cpu_pct,gpu_pct}, ...] top memory holders
);
CREATE INDEX IF NOT EXISTS ix_hs_ts ON host_samples (ts);

-- model loads (cold-load log)
-- One row per model load in a shared, swap-on-demand slot (where only one large
-- model is resident at a time and loading another evicts the current one),
-- written by the usage-logger from polling the swap manager's running-model
-- endpoint. Makes cold-load time EXPLICIT instead of hiding it inside the first
-- request's TTFT — so a benchmark can attribute a slow first call to the model
-- swap, not the engine. t_start = first poll the model was non-ready;
-- t_ready = poll it became ready. Granularity is the sampler interval (e.g. 5s);
-- load_s slightly underestimates (the load began up to one interval before the
-- first non-ready poll). NULL t_ready = load aborted / model evicted before ready.
CREATE TABLE IF NOT EXISTS model_loads (
  id        BIGSERIAL PRIMARY KEY,
  model     TEXT NOT NULL,                 -- swap-slot member (e.g. model-a, model-b)
  t_start   DOUBLE PRECISION NOT NULL,     -- epoch s: first non-ready poll
  t_ready   DOUBLE PRECISION,              -- epoch s: became ready; NULL = aborted/evicted
  load_s    DOUBLE PRECISION,              -- t_ready - t_start
  detail    TEXT                           -- last observed state path, e.g. "starting->ready"
);
CREATE INDEX IF NOT EXISTS ix_ml_tstart ON model_loads (t_start);
CREATE INDEX IF NOT EXISTS ix_ml_model  ON model_loads (model, t_start);

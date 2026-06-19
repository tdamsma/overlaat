#!/usr/bin/env bash
# run-queue-proxy.sh — launch the Overlaat queue-proxy (FIFO sidecar).
#
# Binds 0.0.0.0:4000 (the aggregator entry point) and forwards admitted requests
# to the LiteLLM gateway on loopback. All inbound traffic flows through here and
# is FIFO-queued per model; this is also the single instrumentation site, so it
# emits exactly one lifecycle row per request into `request_events` — including
# queued and client-abandoned calls that insert-on-completion logging misses.
#
# Run behind a process supervisor of your choice; restart the service to pick up
# config changes.

set -euo pipefail

# Resolve the directory this script lives in, so it works regardless of CWD.
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"

# Load environment (DATABASE_URL, QUEUE_PROXY_UPSTREAM, QUEUE_PROXY_LITELLM_CONFIG).
ENV_FILE="${OVERLAAT_ENV:-$APP_DIR/overlaat.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

# Sensible defaults if the env-file omitted them.
export QUEUE_PROXY_UPSTREAM="${QUEUE_PROXY_UPSTREAM:-http://127.0.0.1:4002}"
export QUEUE_PROXY_LITELLM_CONFIG="${QUEUE_PROXY_LITELLM_CONFIG:-$APP_DIR/litellm-config.yaml}"

# Single uvicorn worker on purpose: the in-memory per-model semaphores live in
# this process, so the FIFO ordering and the instrumentation must not be sharded
# across workers.
exec uvicorn overlaat.queue_proxy:app \
  --host 0.0.0.0 \
  --port 4000 \
  --no-access-log

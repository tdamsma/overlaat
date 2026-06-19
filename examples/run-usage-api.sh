#!/usr/bin/env bash
# run-usage-api.sh — launch the Overlaat usage-API (read-only dashboard).
#
# Binds 0.0.0.0:4100 and serves a browser dashboard plus JSON endpoints over the
# events the queue-proxy emits. It only ever reads from Postgres — it never
# writes — so it is safe to restart independently of the proxy.
#
# Endpoints: `/` (dashboard), `/now` (live), `/timeline`, `/models`,
# `/consumers`, `/healthz`.

set -euo pipefail

# Resolve the directory this script lives in, so it works regardless of CWD.
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"

# Load environment (DATABASE_URL is the only one this service needs).
ENV_FILE="${OVERLAAT_ENV:-$APP_DIR/overlaat.env}"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "FATAL: env-file $ENV_FILE missing — Postgres URL unavailable." >&2
  exit 1
fi
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

if [[ -z "${DATABASE_URL:-}" ]]; then
  echo "FATAL: DATABASE_URL not set in $ENV_FILE." >&2
  exit 1
fi
export DATABASE_URL

exec uvicorn overlaat.usage_api:app \
  --host 0.0.0.0 \
  --port 4100 \
  --no-access-log

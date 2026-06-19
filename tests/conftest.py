"""Test config: set dummy connection env before the app modules import it.

No test touches a real Postgres or a real LiteLLM — DB access and the upstream
gateway are mocked per-test.
"""

import os

os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test")
os.environ.setdefault("QUEUE_STATUS_URL", "http://127.0.0.1:4000/__queue/status")

"""Database dialect layer: Postgres (default) + optional SQLite.

The metrics pipeline targets Postgres — that is the default and its behaviour is
unchanged. SQLite is a purely opt-in alternative for a single-box deployment that
does not want to run a Postgres: select it by giving a `sqlite:` DATABASE_URL.

Backend selection is by URL scheme:
  postgres://… / postgresql://…   → Postgres (psycopg)
  sqlite:///abs/path.db            → SQLite, absolute path /abs/path.db
  sqlite:///./rel.db               → SQLite, relative path ./rel.db

Only two things differ per dialect at the SQL level:
  (a) the parameter placeholder style — psycopg `%s` vs sqlite3 `?`;
  (b) the leading-substring function — Postgres `left(col, 8)` vs SQLite
      `substr(col, 1, 8)`.
Both are exposed here so call sites stay one-line-per-statement clean.

Schema: Postgres reads `schema.sql` (the single source of truth). SQLite cannot
run that DDL verbatim, so a small translated copy lives here, differing from
schema.sql in exactly three ways (BIGSERIAL→INTEGER AUTOINCREMENT, JSONB→TEXT,
BOOLEAN DEFAULT false→DEFAULT 0); everything else ports unchanged.

Stdlib-only except for psycopg, which is imported lazily and only on the Postgres
path — so importing this module never requires a DB driver to be installed.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import warnings
from pathlib import Path
from typing import Any

# Shared SQLite pragmas, applied at every connection point (read + write). WAL
# lets the single writer and concurrent dashboard reads coexist; busy_timeout
# makes a reader/writer that finds the db momentarily locked wait up to this
# many milliseconds instead of immediately raising "database is locked".
_SQLITE_BUSY_TIMEOUT_MS = 5000


def _apply_sqlite_pragmas(conn: sqlite3.Connection) -> None:
    """Set WAL journaling + a busy timeout on a freshly opened SQLite connection.

    # KEEP IN SYNC with overlaat.host_logger._sqlite_connect (stdlib-only there)."""
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={_SQLITE_BUSY_TIMEOUT_MS}")


# ── dialect detection ─────────────────────────────────────────────────────────


def dialect_for(url: str) -> str:
    """'sqlite' for a `sqlite:` URL, else 'postgres' (the default)."""
    return "sqlite" if (url or "").startswith("sqlite:") else "postgres"


def sqlite_path(url: str) -> str:
    """Filesystem path from a `sqlite:` URL.

    `sqlite:///abs/path.db`  → `/abs/path.db`  (3 slashes = absolute)
    `sqlite:///./rel.db`     → `./rel.db`      (relative to cwd)
    `sqlite:///.hidden/x.db` → `/.hidden/x.db` (absolute hidden path)
    `sqlite:////abs.db`      → `/abs.db`        (extra slashes collapsed)
    Also accepts the looser `sqlite://path` / `sqlite:path` forms.

    # KEEP IN SYNC with overlaat.host_logger.sqlite_path
    """
    rest = url[len("sqlite:") :]
    rest = rest.removeprefix("//")  # `sqlite://…` → `/…` or `…`
    # After stripping `sqlite://`, an absolute URL path like `///abs` has become
    # `/abs`. Treat the result as relative ONLY when an explicit relative marker
    # follows the leading slash (`/./…` or `/../…`) — then drop that slash. Any
    # other leading-slash path is absolute; collapse a run of leading slashes
    # (e.g. `sqlite:////abs.db` → `//abs.db`) to a single `/`.
    if rest.startswith("/./") or rest.startswith("/../"):
        rest = rest[1:]
    elif rest.startswith("/"):
        rest = "/" + rest.lstrip("/")
    return rest or ":memory:"


# ── per-dialect SQL fragments ───────────────────────────────────────────────────


def placeholder(url: str) -> str:
    """Positional parameter marker for the dialect: `%s` (PG) or `?` (SQLite)."""
    return "?" if dialect_for(url) == "sqlite" else "%s"


def left_expr(url: str, col: str, n: int) -> str:
    """Leading-substring SQL: `left(col, n)` (PG) or `substr(col, 1, n)` (SQLite)."""
    if dialect_for(url) == "sqlite":
        return f"substr({col}, 1, {n})"
    return f"left({col}, {n})"


# ── connections ─────────────────────────────────────────────────────────────────


def connect(url: str):
    """Open a sync connection for the dialect the URL names.

    Postgres: psycopg.connect (imported lazily). SQLite: sqlite3.connect with
    WAL enabled. sqlite3 rows are tuples indexed positionally — the same way
    metrics_db already unpacks psycopg rows — so existing row handling works for
    both backends.
    """
    if dialect_for(url) == "sqlite":
        conn = _SqliteConn(sqlite_path(url), timeout=5.0)
        _apply_sqlite_pragmas(conn)
        return conn
    import psycopg  # lazy: only needed on the Postgres path

    return psycopg.connect(url, connect_timeout=5)


class _SqliteCursor(sqlite3.Cursor):
    """sqlite3.Cursor with the context-manager protocol psycopg cursors have
    (stdlib sqlite3 cursors lack `__enter__`/`__exit__`), so call sites can use
    `with conn.cursor() as cur:` for both backends."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class _SqliteConn(sqlite3.Connection):
    """sqlite3.Connection whose context manager CLOSES on exit, matching how a
    psycopg connection behaves under `with`. (Plain sqlite3's `with` only ends
    the transaction and leaves the connection open — which would leak fds in the
    read path's `with _connect(url) as c` call sites.) Its cursors also support
    `with`, which stdlib sqlite3 cursors do not."""

    def cursor(self, *args, **kwargs):  # type: ignore[override]
        return super().cursor(factory=_SqliteCursor)

    def __exit__(self, exc_type, exc, tb):
        super().__exit__(exc_type, exc, tb)
        self.close()
        return False


def connect_sqlite_write(url: str) -> sqlite3.Connection:
    """A SQLite connection for the write path (host_logger / queue-proxy writer):
    WAL + the default deferred isolation. With deferred isolation, an
    executemany(...) opens an implicit transaction and the caller's explicit
    commit() commits the whole batch as one transaction (the queue-proxy writer
    relies on this for atomic batched event writes)."""
    conn = sqlite3.connect(sqlite_path(url), timeout=15.0)
    _apply_sqlite_pragmas(conn)
    return conn


# ── row normalizers (read path) ─────────────────────────────────────────────────


def normalize_streamed(value: Any) -> Any:
    """SQLite stores BOOLEAN as int 0/1 → coerce to bool. Postgres already
    returns a bool, so this is a pass-through there (None stays None)."""
    if value is None or isinstance(value, bool):
        return value
    return bool(value)


def normalize_backends_json(value: Any) -> Any:
    """SQLite stores JSONB as a TEXT string → json.loads into a Python object.
    Postgres (psycopg) already returns a decoded object, so pass it through."""
    if isinstance(value, (str, bytes, bytearray)):
        try:
            return json.loads(value)
        except (ValueError, TypeError) as e:
            # Surface silent corruption: a malformed backends_json would otherwise
            # come back as NULL with no indication of why. One concise warning per
            # failing call (warnings dedupes identical messages by default anyway).
            warnings.warn(
                f"normalize_backends_json: dropping unparseable host_samples.backends_json "
                f"({type(e).__name__}: {e})",
                stacklevel=2,
            )
            return None
    return value


# ── schema ──────────────────────────────────────────────────────────────────────

# SQLite DDL: schema.sql ported with exactly three translations vs the Postgres
# source of truth — BIGSERIAL PRIMARY KEY → INTEGER PRIMARY KEY AUTOINCREMENT,
# JSONB → TEXT, BOOLEAN … DEFAULT false → DEFAULT 0. Everything else (TEXT,
# INTEGER, DOUBLE PRECISION, IF NOT EXISTS, ON CONFLICT) ports unchanged.
# `DOUBLE PRECISION` is kept verbatim from schema.sql; in SQLite it resolves to
# REAL affinity, so values round-trip as floats — kept verbatim to minimize
# divergence from schema.sql; do not "simplify" it to REAL.
_SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS request_events (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  t_enqueue         DOUBLE PRECISION NOT NULL,
  t_acquire         DOUBLE PRECISION,
  t_first_token     DOUBLE PRECISION,
  t_done            DOUBLE PRECISION NOT NULL,
  model_requested   TEXT NOT NULL,
  key_fp            TEXT NOT NULL,
  streamed          BOOLEAN NOT NULL DEFAULT 0,
  outcome           TEXT NOT NULL,
  http_status       INTEGER,
  prompt_tokens     INTEGER,
  completion_tokens INTEGER,
  overlaat_version  TEXT,
  priority          INTEGER,
  cost              DOUBLE PRECISION,
  wait_reason       TEXT,
  pool              TEXT
);
CREATE INDEX IF NOT EXISTS ix_re_tenq    ON request_events (t_enqueue);
CREATE INDEX IF NOT EXISTS ix_re_tdone   ON request_events (t_done);
CREATE INDEX IF NOT EXISTS ix_re_model   ON request_events (model_requested, t_enqueue);
CREATE INDEX IF NOT EXISTS ix_re_outcome ON request_events (outcome);

CREATE TABLE IF NOT EXISTS host_samples (
  ts                DOUBLE PRECISION PRIMARY KEY,
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
  backends_json     TEXT
);
CREATE INDEX IF NOT EXISTS ix_hs_ts ON host_samples (ts);

CREATE TABLE IF NOT EXISTS model_loads (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  model     TEXT NOT NULL,
  t_start   DOUBLE PRECISION NOT NULL,
  t_ready   DOUBLE PRECISION,
  load_s    DOUBLE PRECISION,
  detail    TEXT
);
CREATE INDEX IF NOT EXISTS ix_ml_tstart ON model_loads (t_start);
CREATE INDEX IF NOT EXISTS ix_ml_model  ON model_loads (model, t_start);
"""


def _schema_sql_path() -> Path:
    """Locate schema.sql: repo root (one up from this package), else cwd."""
    here = Path(__file__).resolve().parent.parent / "schema.sql"
    if here.exists():
        return here
    return Path("schema.sql")


def init_db(url: str) -> str:
    """Idempotently create the schema for whichever backend the URL names.
    Returns a one-line description of what was done."""
    if dialect_for(url) == "sqlite":
        path = sqlite_path(url)
        conn = connect_sqlite_write(url)
        try:
            conn.executescript(_SQLITE_DDL)
            conn.commit()
        finally:
            conn.close()
        return f"sqlite: applied schema to {path}"

    schema_path = _schema_sql_path()
    ddl = schema_path.read_text()
    with connect(url) as conn, conn.cursor() as cur:
        cur.execute(ddl)
        conn.commit()
    return f"postgres: applied {schema_path} to the configured database"


# ── CLI: python -m overlaat.db init <DATABASE_URL> ──────────────────────────────


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] != "init":
        sys.stderr.write("usage: python -m overlaat.db init [DATABASE_URL]\n")
        return 2
    url = argv[1] if len(argv) > 1 else os.environ.get("DATABASE_URL", "")
    if not url:
        sys.stderr.write("error: no DATABASE_URL given (argument or env var)\n")
        return 2
    msg = init_db(url)
    sys.stdout.write(msg + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Contributing to Overlaat

Thanks for looking. Overlaat is **experimental**, **MIT-licensed**, and shipped with
**no support promise** — read the code before you rely on it. Issues and pull requests
are welcome, but treat everything here as movable until a tagged release says otherwise.

## Dev setup

Pure-Python project, `hatchling`-built, Python ≥ 3.11. We use [`uv`](https://docs.astral.sh/uv/):

```bash
uv venv                          # create a virtualenv (.venv)
uv pip install -e '.[dev]'       # install Overlaat + dev deps (pytest, ruff, build, twine)
```

(Plain `pip install -e '.[dev]'` works too if you'd rather not use `uv`.)

## Tests, lint, format

```bash
uv run pytest                    # the test suite (pytest, asyncio_mode=auto)
uv run ruff check                # lint
uv run ruff format               # format (line-length 100)
```

CI runs the same `ruff check` and `pytest` across Python 3.11–3.13, so run them locally
before opening a PR. Config lives in `pyproject.toml` — don't override it ad hoc.

## Releasing / version rule

The version is **single-sourced** in `overlaat/__init__.py` (`__version__`), read at
build time via `[tool.hatch.version]`. To release:

1. Bump `__version__` in `overlaat/__init__.py`.
2. Tag the commit `vX.Y.Z` where `X.Y.Z` **exactly matches** `__version__`.

CI has a guard that fails the build if the `vX.Y.Z` tag and `__version__` disagree — so a
mismatched tag will not publish. Pushing a `v*` tag triggers the build and the trusted-
publishing release to PyPI.

## Conventions

This is a **public repo** — keep it generic.

- **English only**, everywhere (code, comments, docs).
- **No site-specific identifiers**: no private hostnames/IPs, no `*.ts.net`, no
  personal/product names, no `/Users/...` paths, no real secrets or keys.
- Postgres tables are the **bare names** `request_events`, `host_samples`, `model_loads`
  — no prefixes.
- **Backend-agnostic**: assume only an OpenAI-compatible LiteLLM gateway in front and a
  Postgres to write to. Don't bake in a specific engine or platform.
- Don't imply the **capacity-aware cost scheduler** exists — it's a design-only roadmap
  (`docs/COST-SCHEDULER.md`), not shipping code.

## A few things the queue-proxy cares about

- The queue-proxy runs a **single uvicorn worker on purpose** — the per-model semaphores,
  FIFO ordering, and the event writer all live in one process. Don't add `--workers N`.
- It emits **exactly one lifecycle event per request**. If you touch the call path, keep
  that invariant: one honest row per request, queued and client-abandoned calls included.

That's it. Open an issue if something is unclear, and please keep PRs small and focused.

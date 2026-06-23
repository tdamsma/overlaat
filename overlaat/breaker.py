"""Per-model health-gated admission circuit breaker for the Overlaat queue-proxy.

Overlaat admission is otherwise *open-loop*: it admits against a static per-pool
budget and has no signal that the *runtime itself* is unhealthy. When a backend
wedges (engine scheduler hung, decode ~0 tok/s, read-timeouts firing) the proxy
keeps forwarding onto the corpse, and a consumer's lane-stop + retries pile more
work onto it. This breaker is the **dynamic** complement to the per-model size
ceiling (#30): the ceiling stops *individual jobs too big to process*; the
breaker stops *feeding a runtime that is already choking* (#31).

The model, in one paragraph: each model carries an optional
``overlaat_breaker: {fails, cooldown_s}`` config. The breaker watches the
terminal outcome the proxy already records for every real upstream attempt. After
``fails`` **consecutive** ``upstream_error`` outcomes (which already cover
upstream 5xx, connection errors, and read-timeouts — all surface as
``upstream_error`` in the proxy's ``_forward``) the model trips **open**: new
requests are fast-failed at admission with ``503 + Retry-After`` for ``cooldown_s``
seconds (do NOT hold them in queue — that would pile callers onto a wedged
backend). After the cooldown the breaker goes **half-open** and lets exactly ONE
probe through; the probe's outcome closes it (success) or re-opens it with a
fresh cooldown (failure).

Deliberate signal choice: only ``completed`` (success) and ``upstream_error``
(failure) move the counters; **zero-token completions are intentionally NOT
treated as failures** — legit short/empty/embeddings responses are too noisy a
signal and would false-trip a healthy model. The "no-tokens-produced-in-T" stall
detector the issue floats needs per-request progress tracking and is out of scope
here.

Default OFF: a model with no (or malformed/partial) ``overlaat_breaker`` config is
never gated — exactly today's behaviour.

This module is deliberately self-contained and free of FastAPI / DB / config
loading: it is a pure in-memory state machine with an injectable monotonic clock,
mirroring ``overlaat/scheduler.py``. The proxy wires it to the request path
(consult before admission; record on each terminal upstream outcome).

SINGLE-PROCESS INVARIANT: one ``Breaker`` instance per process, driven by one
asyncio event loop. There are **no locks** — all mutation happens on the loop
thread — the same reason the queue-proxy must never run with ``--workers N``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class _ModelState:
    """Per-model breaker state.

    ``state`` is one of ``"closed"`` / ``"open"`` / ``"half_open"``.
    ``consecutive_fails`` counts back-to-back ``upstream_error`` outcomes while
    closed; a ``completed`` resets it to 0. ``opened_at`` stamps when the breaker
    last tripped open (basis for the cooldown). ``probe_in_flight`` /
    ``probe_started`` track the single half-open probe so a second concurrent
    request is blocked while one probe is being tried.
    """

    state: str = "closed"
    consecutive_fails: int = 0
    opened_at: float = 0.0
    probe_in_flight: bool = False
    probe_started: float = 0.0


@dataclass
class _Config:
    """Validated per-model breaker config (``fails`` consecutive errors trip it;
    ``cooldown_s`` is both the open duration AND the stale-probe self-heal window)."""

    fails: int
    cooldown_s: float


class Breaker:
    """Per-model health-gated admission circuit breaker.

    One instance per process; no locks; single event loop (see module docstring
    and the never-``--workers N`` invariant). Constructed from a parsed config
    dict (model -> ``{"fails": int, "cooldown_s": float}``) and an injectable
    monotonic clock, exactly like :class:`overlaat.scheduler.Scheduler` takes an
    injectable ``now``.
    """

    def __init__(
        self,
        config: dict[str, dict] | None = None,
        *,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        """
        :param config: model_name -> ``{"fails": <positive int>, "cooldown_s":
            <positive number>}``. Both keys are required and positive for the
            breaker to be active for that model; a malformed / partial / missing
            entry leaves the model **ungated** (breaker off). This mirrors the
            ``load_model_info`` parser, which already validates and only emits
            well-formed entries, but the constructor re-validates so the core is
            honest on its own (and unit-testable without the parser).
        :param now: monotonic clock; injectable for deterministic tests.
        """
        self._config: dict[str, _Config] = {}
        for model, spec in (config or {}).items():
            if not isinstance(model, str) or not isinstance(spec, dict):
                continue
            fails = spec.get("fails")
            cooldown = spec.get("cooldown_s")
            if (
                isinstance(fails, int)
                and not isinstance(fails, bool)
                and fails > 0
                and isinstance(cooldown, (int, float))
                and not isinstance(cooldown, bool)
                and cooldown > 0
            ):
                self._config[model] = _Config(fails=int(fails), cooldown_s=float(cooldown))
        self._states: dict[str, _ModelState] = {}
        self._now = now

    # -- internals ---------------------------------------------------------

    def _st(self, model: str) -> _ModelState:
        """The mutable per-model state, created lazily on first touch."""
        st = self._states.get(model)
        if st is None:
            st = _ModelState()
            self._states[model] = st
        return st

    # -- admission gate ----------------------------------------------------

    def allow(self, model: str) -> tuple[bool, str | None]:
        """Consulted before admission. ``(True, None)`` = may proceed;
        ``(False, reason)`` = blocked.

        - Unconfigured model (breaker off) or **closed** → allow.
        - **open**: once ``cooldown_s`` has elapsed since ``opened_at``, transition
          to **half-open** and allow exactly ONE probe; otherwise block ``"open"``.
        - **half-open**: a probe is already in flight → block ``"half_open"``,
          UNLESS that in-flight probe is *stale* (older than ``cooldown_s`` — a
          lost / cancelled probe whose outcome was never recorded), in which case
          re-arm and allow a fresh probe. This stale-probe self-heal prevents a
          permanently stuck-open breaker if a probe never reports back.
        """
        cfg = self._config.get(model)
        if cfg is None:
            return (True, None)
        st = self._st(model)
        now = self._now()

        if st.state == "closed":
            return (True, None)

        if st.state == "open":
            if now >= st.opened_at + cfg.cooldown_s:
                st.state = "half_open"
                st.probe_in_flight = True
                st.probe_started = now
                return (True, None)
            return (False, "open")

        # half_open
        if st.probe_in_flight and now - st.probe_started < cfg.cooldown_s:
            return (False, "half_open")
        # No probe in flight, or the in-flight probe is stale → arm a fresh probe.
        st.probe_in_flight = True
        st.probe_started = now
        return (True, None)

    # -- outcome feedback --------------------------------------------------

    def record(self, model: str, outcome: str) -> None:
        """Feed one terminal upstream outcome into the breaker.

        Called exactly once per real upstream attempt (admission-time rejections
        like ``rejected_oversized`` / ``rejected_unhealthy`` / ``cancelled_queued``
        never reach a real upstream, so they never call this). Outcome mapping:

        - ``"completed"``      → **success**
        - ``"upstream_error"`` → **failure** (covers upstream 5xx, connection
          errors, and read-timeouts — all surface as ``upstream_error``)
        - everything else (``"client_abandoned"``, …) → **skip** (neutral; the
          counters are untouched, since an abandonment is not a backend-health
          signal). Note: a zero-token ``completed`` is still a success — empty /
          short / embeddings responses are deliberately NOT treated as failures.
        """
        cfg = self._config.get(model)
        if cfg is None:
            return
        if outcome == "completed":
            success = True
        elif outcome == "upstream_error":
            success = False
        else:
            return  # neutral: do not touch the counters

        st = self._st(model)

        if st.state == "half_open":
            # This is the single probe's result.
            st.probe_in_flight = False
            if success:
                st.state = "closed"
                st.consecutive_fails = 0
            else:
                st.state = "open"
                st.opened_at = self._now()
            return

        if st.state == "open":
            # A late outcome from before the trip; counters are irrelevant while
            # open (the cooldown governs recovery). Ignore.
            return

        # closed
        if success:
            st.consecutive_fails = 0
        else:
            st.consecutive_fails += 1
            if st.consecutive_fails >= cfg.fails:
                st.state = "open"
                st.opened_at = self._now()

    # -- introspection (for the Retry-After header, tests, observability) --

    def state(self, model: str) -> str:
        """Current state string for ``model``: ``"closed"`` / ``"open"`` /
        ``"half_open"``. An unconfigured model is always reported ``"closed"``."""
        if model not in self._config:
            return "closed"
        return self._st(model).state

    def retry_after(self, model: str) -> int:
        """Seconds the caller should wait before retrying, for the ``Retry-After``
        header. ``ceil`` of the remaining cooldown while **open**; 0 otherwise."""
        cfg = self._config.get(model)
        if cfg is None:
            return 0
        st = self._st(model)
        if st.state != "open":
            return 0
        remaining = (st.opened_at + cfg.cooldown_s) - self._now()
        return max(0, math.ceil(remaining))

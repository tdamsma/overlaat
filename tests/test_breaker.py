"""Pure unit tests for the per-model health-gated admission circuit breaker (#31).

No FastAPI, no DB, no real sleeps: the breaker's monotonic clock is a manually
advanced fake injected into the Breaker, so every cooldown / half-open / probe
case is deterministic. Mirrors the clock-injection style of test_scheduler.py.
"""

from __future__ import annotations

from overlaat.breaker import Breaker


class Clock:
    """A manually-advanced monotonic clock for deterministic cooldown tests."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


# --------------------------------------------------------------------------
# Unconfigured / off
# --------------------------------------------------------------------------


def test_unconfigured_model_always_allowed_and_never_changes_state():
    b = Breaker({}, now=Clock())
    # Many failures: an off model is never gated and never leaves "closed".
    for _ in range(100):
        assert b.allow("m") == (True, None)
        b.record("m", "upstream_error")
    assert b.allow("m") == (True, None)
    assert b.state("m") == "closed"
    assert b.retry_after("m") == 0


def test_malformed_config_is_off():
    # Partial / wrong-typed / non-positive configs are all skipped (off).
    b = Breaker(
        {
            "a": {"fails": 2},  # missing cooldown_s
            "b": {"cooldown_s": 30},  # missing fails
            "c": {"fails": 0, "cooldown_s": 30},  # non-positive fails
            "d": {"fails": 2, "cooldown_s": 0},  # non-positive cooldown
            "e": {"fails": True, "cooldown_s": 30},  # bool fails
            "f": {"fails": 2, "cooldown_s": "30"},  # non-numeric cooldown
        },
        now=Clock(),
    )
    for model in ("a", "b", "c", "d", "e", "f"):
        for _ in range(10):
            assert b.allow(model) == (True, None)
            b.record(model, "upstream_error")
        assert b.state(model) == "closed"


# --------------------------------------------------------------------------
# closed -> open
# --------------------------------------------------------------------------


def test_trips_open_after_exactly_fails_consecutive_errors():
    clock = Clock()
    b = Breaker({"m": {"fails": 3, "cooldown_s": 30}}, now=clock)
    # 2 errors: still closed.
    b.record("m", "upstream_error")
    b.record("m", "upstream_error")
    assert b.state("m") == "closed"
    assert b.allow("m") == (True, None)
    # 3rd consecutive error trips it.
    b.record("m", "upstream_error")
    assert b.state("m") == "open"
    assert b.allow("m") == (False, "open")


def test_completed_resets_consecutive_fail_count():
    clock = Clock()
    b = Breaker({"m": {"fails": 3, "cooldown_s": 30}}, now=clock)
    # K-1 fails, then a success, then a fail does NOT trip (counter reset).
    b.record("m", "upstream_error")
    b.record("m", "upstream_error")
    b.record("m", "completed")
    assert b.state("m") == "closed"
    b.record("m", "upstream_error")
    assert b.state("m") == "closed"  # only 1 consecutive fail now


def test_client_abandoned_is_neutral():
    clock = Clock()
    b = Breaker({"m": {"fails": 2, "cooldown_s": 30}}, now=clock)
    b.record("m", "upstream_error")
    # A neutral outcome must NOT advance (or reset) the counter.
    b.record("m", "client_abandoned")
    assert b.state("m") == "closed"
    # The next error reaches the threshold (1 + 1 = 2), proving the abandon was
    # neutral (did not reset it back to 0).
    b.record("m", "upstream_error")
    assert b.state("m") == "open"


# --------------------------------------------------------------------------
# open -> half-open -> (closed | re-open)
# --------------------------------------------------------------------------


def test_open_blocks_until_cooldown_then_half_open_single_probe():
    clock = Clock()
    b = Breaker({"m": {"fails": 2, "cooldown_s": 30}}, now=clock)
    b.record("m", "upstream_error")
    b.record("m", "upstream_error")
    assert b.state("m") == "open"

    # Before cooldown: blocked.
    clock.advance(29.0)
    assert b.allow("m") == (False, "open")

    # At/after cooldown: first allow enters half-open and returns the single probe.
    clock.advance(1.0)
    assert b.allow("m") == (True, None)
    assert b.state("m") == "half_open"
    # Second allow while the probe is in flight: blocked.
    assert b.allow("m") == (False, "half_open")


def test_probe_success_closes_and_resumes():
    clock = Clock()
    b = Breaker({"m": {"fails": 2, "cooldown_s": 30}}, now=clock)
    b.record("m", "upstream_error")
    b.record("m", "upstream_error")
    clock.advance(30.0)
    assert b.allow("m") == (True, None)  # probe
    b.record("m", "completed")  # probe succeeds
    assert b.state("m") == "closed"
    # Resumes allowing normally.
    assert b.allow("m") == (True, None)
    assert b.allow("m") == (True, None)


def test_probe_failure_reopens_with_fresh_cooldown():
    clock = Clock()
    b = Breaker({"m": {"fails": 2, "cooldown_s": 30}}, now=clock)
    b.record("m", "upstream_error")
    b.record("m", "upstream_error")
    clock.advance(30.0)
    assert b.allow("m") == (True, None)  # probe at t=30
    b.record("m", "upstream_error")  # probe fails
    assert b.state("m") == "open"
    # Fresh cooldown stamped at t=30: blocked until t=60.
    clock.advance(29.0)  # t=59
    assert b.allow("m") == (False, "open")
    clock.advance(1.0)  # t=60
    assert b.allow("m") == (True, None)


def test_stale_probe_self_heal_rearms_after_cooldown():
    clock = Clock()
    b = Breaker({"m": {"fails": 2, "cooldown_s": 30}}, now=clock)
    b.record("m", "upstream_error")
    b.record("m", "upstream_error")
    clock.advance(30.0)
    assert b.allow("m") == (True, None)  # probe armed at t=30, never records back
    assert b.state("m") == "half_open"
    # Still in flight a moment later: a second request is blocked.
    clock.advance(5.0)
    assert b.allow("m") == (False, "half_open")
    # The probe is now stale (older than cooldown_s) → re-arm a fresh probe rather
    # than staying stuck-open forever.
    clock.advance(30.0)  # probe age now 35 > 30
    assert b.allow("m") == (True, None)
    assert b.state("m") == "half_open"


# --------------------------------------------------------------------------
# retry_after
# --------------------------------------------------------------------------


def test_retry_after_decreases_as_clock_advances():
    clock = Clock()
    b = Breaker({"m": {"fails": 2, "cooldown_s": 30}}, now=clock)
    b.record("m", "upstream_error")
    b.record("m", "upstream_error")  # opens at t=0
    assert b.retry_after("m") == 30
    clock.advance(10.0)
    assert b.retry_after("m") == 20
    clock.advance(19.5)  # t=29.5, 0.5 remaining → ceil = 1
    assert b.retry_after("m") == 1
    clock.advance(0.5)  # t=30, cooldown elapsed
    assert b.retry_after("m") == 0


def test_retry_after_zero_when_not_open():
    b = Breaker({"m": {"fails": 2, "cooldown_s": 30}}, now=Clock())
    assert b.retry_after("m") == 0  # closed
    assert b.retry_after("unconfigured") == 0

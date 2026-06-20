"""Cost-weighted global admission scheduler for the Overlaat queue-proxy.

This is the core of the "cost-weighted scheduler" described in
``docs/COST-SCHEDULER.md``. It replaces the N independent per-model
``asyncio.Semaphore`` pools with **one global priority queue** guarded by
cost-weighted admission against a single shared budget ``B`` (the whole GPU).

The model, in one paragraph: every in-flight run consumes a ``cost`` equal to
its fraction of the GPU (``cost = 1 / cap`` by default). Admission requires both
the per-model backend cap (``in_flight(model) < cap(model)``) *and* the shared
budget (``used + cost <= B``). Ordering is by effective priority (per-key
ceiling-clamped request priority, plus linear aging by wait time). Packing is
work-conserving — the highest-priority waiter that *fits* is admitted, not
strictly the head — but the head of the queue, if it does not fit, gets an
**eager reservation** of budget so a steady drip of cheap jobs cannot starve it.
There is **no preemption**: Metal has no GPU preemption, so admission (which
request starts, and when) is the only lever. See the design doc for the full
rationale.

This module is deliberately self-contained and free of FastAPI / DB / config:
it is a pure in-memory state machine driven by ``enqueue`` / ``withdraw`` /
``release``, with an injectable clock so aging is testable without real sleeps.
The proxy wires it to the request path in a later stage.

SINGLE-PROCESS INVARIANT: exactly one ``Scheduler`` instance per process, driven
by one asyncio event loop. There are **no locks** — all mutation happens on the
loop thread, so the admit loop is never re-entered concurrently. This is the
same reason the queue-proxy must never run with ``--workers N``: the budget
ledger, per-model in-flight counts, and the waiters list live in this one
process, and sharding them would defeat both the global budget and the ordering.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field

# Float epsilon for the budget admit test. Costs are sums of 1/cap terms, so a
# budget that is "exactly full" can drift by a few ULPs (e.g. 3 * (1/3) != 1.0).
# We admit when used + cost <= B + EPS so such exact packings are not phantom
# overflowed. EPS is far smaller than any realistic 1/cap step.
EPS = 1e-9

# Default shared budget (the whole GPU) and default cost for models with no cap.
# These mirror the OVERLAAT_BUDGET / OVERLAAT_DEFAULT_COST env defaults; the
# proxy passes the env-resolved values in, but the constructor defaults keep the
# core usable (and the unit tests honest) on its own.
DEFAULT_BUDGET = 1.0
DEFAULT_COST = 1.0
DEFAULT_PRIORITY = 0
DEFAULT_AGING_RATE = 0.0


@dataclass
class Waiter:
    """One queued request awaiting admission.

    ``fut`` is resolved with ``True`` exactly once, when the request is admitted.
    The caller awaits it; on cancel-while-queued the caller calls
    ``Scheduler.withdraw`` instead (and never resolves the future). ``cost``,
    ``base_priority`` and ``key_fp`` are resolved once at enqueue time by the
    proxy; aging is recomputed on every pump from ``enqueued_at``.

    ``wait_reason`` is set at admission to one of the values documented on
    ``Scheduler.pump`` so the proxy can log *why* a request waited.
    """

    req_id: str
    model: str
    cost: float
    base_priority: int
    key_fp: str | None
    enqueued_at: float
    fut: asyncio.Future = field(default_factory=lambda: _new_future())
    cancelled: bool = False
    wait_reason: str = "none"
    # Set True the first time this waiter is seen by pump but not admitted, so a
    # later admission can be distinguished from a first-pump admission ("none").
    _waited: bool = False
    # True while this waiter holds (or held, up to the pass that admits it) the
    # head budget-reservation, so its admission reports "reserved" even though
    # the reservation clears on the pass that finally admits it (head then fits
    # the full budget). Reset to False if a later pass displaces it as the
    # reserved head (e.g. a higher-priority arrival), so a subsequent
    # work-conserving packing admission is not mislabeled "reserved".
    _was_reserved: bool = False


def _new_future() -> asyncio.Future:
    """A future bound to the running loop (so Waiter() works inside the loop)."""
    return asyncio.get_event_loop().create_future()


class Scheduler:
    """One global, cost-weighted, priority-ordered admission scheduler.

    One instance per process; no locks; single event loop (see module docstring
    and the never-``--workers N`` invariant). All public methods mutate in-memory
    state and re-run the admit loop (:meth:`pump`) synchronously on the loop
    thread.
    """

    def __init__(
        self,
        budget: float = DEFAULT_BUDGET,
        caps: dict[str, int] | None = None,
        *,
        costs: dict[str, float] | None = None,
        slot_groups: dict[str, str] | None = None,
        default_cost: float = DEFAULT_COST,
        default_priority: int = DEFAULT_PRIORITY,
        aging_rate: float = DEFAULT_AGING_RATE,
        key_ceiling: Callable[[str | None], int | None] | None = None,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        """
        :param budget: shared budget ``B`` (the whole GPU); default 1.0.
        :param caps: per-model backend cap (``max_parallel_requests``). A model
            absent here has no declared cap.
        :param costs: optional per-model explicit cost overrides
            (``model_info.overlaat_cost``). Takes precedence over ``1/cap``.
        :param slot_groups: model -> swap-slot group name
            (``model_info.overlaat_slot``). Any member's cost is forced to 1.0
            so the budget arithmetic enforces the "one big model at a time"
            mutex with no separate lock.
        :param default_cost: cost for a model with no declared cap and no
            override (``OVERLAAT_DEFAULT_COST``); default 1.0.
        :param default_priority: priority used when neither the request nor the
            key supplies one (``OVERLAAT_DEFAULT_PRIORITY``); default 0.
        :param aging_rate: linear priority gain per second waited
            (``OVERLAAT_AGING_RATE``); default 0.0 (aging off).
        :param key_ceiling: callable ``key_fp -> max_priority | None``; the
            per-key priority ceiling resolved from LiteLLM key metadata. Returns
            ``None`` when the key has no ceiling or the source is unreachable,
            in which case ``default_priority`` is used as the ceiling.
        :param now: monotonic clock; injectable for deterministic aging tests.
        """
        self.B = float(budget)
        self.caps: dict[str, int] = dict(caps or {})
        self._costs: dict[str, float] = dict(costs or {})
        self._slot_groups: dict[str, str] = dict(slot_groups or {})
        self.default_cost = float(default_cost)
        self.default_priority = int(default_priority)
        self.aging_rate = float(aging_rate)
        self._key_ceiling = key_ceiling
        self._now = now

        self.used: float = 0.0
        self.in_flight: dict[str, int] = {}
        self.waiters: list[Waiter] = []
        # The head expensive waiter currently holding a reservation, or None.
        self.reserved_for: Waiter | None = None

    # -- cost / cap derivation --------------------------------------------

    def cap(self, model: str) -> int | None:
        """Backend concurrency cap for a model, or None if it has no declared cap."""
        return self.caps.get(model)

    def cost(self, model: str) -> float:
        """Cost (GPU fraction) charged for one in-flight run of ``model``.

        Precedence: a swap-slot member is forced to 1.0 (full budget, so it is a
        mutex); else an explicit per-model override; else ``1 / cap``; else the
        configured default cost for an uncapped model.
        """
        if model in self._slot_groups:
            return 1.0
        if model in self._costs:
            return self._costs[model]
        cap = self.caps.get(model)
        if cap and cap > 0:
            return 1.0 / cap
        return self.default_cost

    # -- priority / aging --------------------------------------------------

    def _ceiling(self, key_fp: str | None) -> float:
        """Per-key priority ceiling.

        Three cases, mirroring the maintainer's key-metadata semantics:

        - **No ceiling source configured at all** (``key_ceiling is None`` — the
          SQLite single-box case with no LiteLLM key table): there is nothing to
          clamp against, so the ceiling is ``+inf`` and the requested priority
          passes through unchanged.
        - **Source configured, key has a ceiling**: return it (the un-gameable
          clamp — a batch key cannot exceed its operator-granted ceiling).
        - **Source configured but this key has none / source unreachable**
          (resolver returns ``None``): fall back to ``OVERLAAT_DEFAULT_PRIORITY``.
        """
        if self._key_ceiling is None:
            return float("inf")
        c = self._key_ceiling(key_fp)
        if c is not None:
            return float(c)
        return float(self.default_priority)

    def effective_priority(self, waiter: Waiter, now: float) -> float:
        """min(requested, key-ceiling) + linear aging by wait time.

        The clamp to the key ceiling makes priority un-gameable by clients: a
        batch key provisioned with a low ceiling cannot impersonate an
        interactive key no matter what ``priority`` it sends. Aging then lifts
        long-waiting jobs so "reaches the head" becomes a guarantee.
        """
        clamped = min(float(waiter.base_priority), self._ceiling(waiter.key_fp))
        wait_s = max(0.0, now - waiter.enqueued_at)
        return clamped + self.aging_rate * wait_s

    # -- queue mutation ----------------------------------------------------

    def enqueue(self, waiter: Waiter) -> None:
        """Append a waiter and run the admit loop."""
        self.waiters.append(waiter)
        self.pump(self._now())

    def withdraw(self, waiter: Waiter) -> None:
        """Remove an un-admitted waiter (cancel-while-queued) and re-pump.

        Clears the reservation if the withdrawn waiter was holding it. Admitted
        waiters are no longer in ``self.waiters`` and are released via
        :meth:`release` instead; withdrawing one is a no-op.
        """
        waiter.cancelled = True
        if waiter in self.waiters:
            self.waiters.remove(waiter)
        if self.reserved_for is waiter:
            self.reserved_for = None
        self.pump(self._now())

    def release(self, model: str) -> None:
        """Account the completion of one in-flight run of ``model`` and re-pump.

        This is the only path that frees budget; it is called on every request
        completion (success, error, or in-flight client disconnect).
        """
        self.used -= self.cost(model)
        if self.used < 0.0:
            self.used = 0.0
        self.in_flight[model] = max(0, self.in_flight.get(model, 0) - 1)
        self.pump(self._now())

    # -- the admit loop ----------------------------------------------------

    def _fits(self, w: Waiter, reservable: float) -> bool:
        """True if ``w`` satisfies both the per-model cap and the budget room.

        Backend cap binds always; the budget test uses ``reservable`` (which is
        ``B`` minus any head reservation) so cheap jobs cannot eat the budget a
        reserved expensive head is draining toward.
        """
        if self._cap_full(w):
            return False
        return self.used + w.cost <= reservable + EPS

    def _cap_full(self, w: Waiter) -> bool:
        """True if ``w``'s per-model backend cap currently has no free slot."""
        cap = self.caps.get(w.model)
        return cap is not None and self.in_flight.get(w.model, 0) >= cap

    def _admit(self, w: Waiter, reason: str) -> None:
        self.used += w.cost
        self.in_flight[w.model] = self.in_flight.get(w.model, 0) + 1
        self.waiters.remove(w)
        if self.reserved_for is w:
            self.reserved_for = None
        w.wait_reason = reason
        if not w.fut.done():
            w.fut.set_result(True)

    def pump(self, now: float | None = None) -> None:
        """The admit loop: order by priority, reserve the head, pack the rest.

        Runs to a fixed point — keeps admitting until no waiter fits — on every
        arrival, withdrawal, and release. Order is by ``(-effective_priority,
        enqueued_at)`` so higher priority wins and FIFO breaks ties.

        Reservation (eager): if the highest-priority waiter (the head) is
        blocked *by budget* — its model cap has a free slot but the shared
        budget does not fit ``head.cost`` — it becomes ``reserved_for`` and the
        budget available to *other* waiters this pass is reduced to
        ``B - head.cost`` (``reservable``). Freed budget is then held for the
        head instead of handed to a fresh cheap arrival; the head is admitted
        once ``used + head.cost <= B``. In the common case the head fits and
        nothing is reserved (fully work-conserving). A head blocked only by its
        own backend cap is never reserved for (reserving budget cannot free a
        cap slot).

        Each admission records ``waiter.wait_reason`` (so the proxy can log why a
        request waited):

        - ``"none"``      — admitted on the first pump after enqueue (never waited).
        - ``"reserved"``  — this waiter *was* the reserved head and was admitted
          once the budget drained for it.
        - ``"aged_in"``   — waited, and aging lifted its effective priority above
          equal-nominal-priority peers before it was admitted.
        - ``"budget_full"`` — waited because the shared budget had no room.
        - ``"model_cap"`` — waited because the per-model backend cap was full
          (budget had room).

        NEVER preempts: admitted runs are accounted only via :meth:`release`.
        """
        if now is None:
            now = self._now()

        while True:
            if not self.waiters:
                self.reserved_for = None
                return

            ordered = sorted(
                self.waiters,
                key=lambda w: (-self.effective_priority(w, now), w.enqueued_at),
            )
            head = ordered[0]

            # Eager reservation: if the head is blocked *by budget* (its model
            # cap has room but the shared budget does not), hold budget for it
            # and pack others only into the non-reserved portion. A head blocked
            # by its own backend cap is NOT reserved for — reserving budget can
            # never free a cap slot, and idling budget there would be pointless;
            # such a head simply waits while work-conserving packing continues.
            head_budget_blocked = not self._cap_full(head) and not self._fits(head, self.B)
            if head_budget_blocked:
                # If a different waiter previously held the reservation, it has
                # been displaced as head (e.g. a higher-priority arrival). Clear
                # its _was_reserved so it does not falsely report "reserved" if
                # later admitted via work-conserving packing rather than as the
                # reserved head.
                if self.reserved_for is not None and self.reserved_for is not head:
                    self.reserved_for._was_reserved = False
                self.reserved_for = head
                head._was_reserved = True
                reservable = self.B - head.cost
            else:
                # No budget-blocked head this pass: nobody is reserved for, so a
                # prior reserved waiter that is no longer the head must not keep
                # _was_reserved (it would mislabel a later packing admission).
                if self.reserved_for is not None and self.reserved_for is not head:
                    self.reserved_for._was_reserved = False
                self.reserved_for = None
                reservable = self.B

            # Find the highest-priority waiter that fits the (reservable) budget.
            admitted = None
            for w in ordered:
                if self._fits(w, reservable):
                    admitted = w
                    break

            if admitted is None:
                # Nobody fits this pass. Mark every still-waiting waiter as having
                # been seen (sets _waited=True) with a live wait reason, so a later
                # admission can report a "waited" reason instead of "none".
                self._record_wait_reasons(now)
                return

            reason = self._reason_for(admitted, now)
            self._admit(admitted, reason)
            # Loop: a release of budget may now let another waiter in.

    # -- wait-reason bookkeeping ------------------------------------------

    def _reason_for(self, w: Waiter, now: float) -> str:
        """Classify why ``w`` waited, evaluated at admission time."""
        if not w._waited:
            return "none"
        if w._was_reserved:
            return "reserved"
        # It waited. Decide between aging, budget, or cap as the dominant cause.
        clamped = min(float(w.base_priority), self._ceiling(w.key_fp))
        aged = self.effective_priority(w, now) - clamped
        if self.aging_rate > 0.0 and aged > 0.0:
            return "aged_in"
        if self._cap_full(w):
            return "model_cap"
        return "budget_full"

    def _record_wait_reasons(self, now: float) -> None:
        """Tag every still-waiting waiter as having waited, with a live reason."""
        for w in self.waiters:
            w._waited = True
            if self.reserved_for is w:
                w.wait_reason = "reserved"
                continue
            cap = self.caps.get(w.model)
            if cap is not None and self.in_flight.get(w.model, 0) >= cap:
                w.wait_reason = "model_cap"
            else:
                w.wait_reason = "budget_full"

    # -- introspection (for /__queue/status and tests) --------------------

    def queue_depth(self) -> int:
        return len(self.waiters)

    def total_in_flight(self) -> int:
        return sum(self.in_flight.values())

    def snapshot(self) -> dict:
        """A small, JSON-friendly view of scheduler state for status endpoints."""
        return {
            "budget": self.B,
            "used": self.used,
            "queue_depth": self.queue_depth(),
            "in_flight": dict(self.in_flight),
            "reserved_for": self.reserved_for.req_id if self.reserved_for else None,
        }

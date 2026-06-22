"""Cost-weighted global admission scheduler for the Overlaat queue-proxy.

This is the core of the "cost-weighted scheduler" described in
``docs/COST-SCHEDULER.md``. It replaces the N independent per-model
``asyncio.Semaphore`` pools with **named resource pools**: each model is
assigned to exactly one pool (default ``"default"``), and admission is
cost-weighted against that pool's own budget ``B_pool``. The single-pool case
(every model in ``default``) reproduces the previous single-global-budget
behavior exactly.

The model, in one paragraph: every in-flight run consumes a ``cost`` equal to
its fraction of its pool (``cost = 1 / cap`` by default). Admission requires the
per-model backend cap (``in_flight(model) < cap(model)``), the pool budget
(``used[pool] + cost <= B_pool``), and — for an ``exclusive`` pool — that no
*different* member is already resident (the mutex). Ordering is by effective
priority (per-key ceiling-clamped request priority, plus linear aging by wait
time) over one GLOBAL waiters list. Packing is work-conserving — the
highest-priority waiter that *fits* its pool is admitted, not strictly the head
— but each pool's head, if it is blocked *by budget*, gets an **eager
reservation** of that pool's budget so a steady drip of cheap jobs in the same
pool cannot starve it. There is **no preemption**: Metal has no GPU preemption,
so admission (which request starts, and when) is the only lever. See the design
doc for the full rationale.

Resource pools (closes #11/#12): a pool is the unit of admission. A pool blocked
on budget never idles a *different* pool — cross-backend concurrency is
preserved (#11). An ``exclusive`` pool is a hard mutex on the *distinct member*
level: at most one member id may be active (``in_flight > 0``) at a time, but
that resident member's own cap + the pool budget govern how many of *its*
streams run concurrently (so a cap-2 dual-engine member runs 2 — #12). The old
``overlaat_slot`` "fat-slot" group is bridged to an exclusive pool.

This module is deliberately self-contained and free of FastAPI / DB / config:
it is a pure in-memory state machine driven by ``enqueue`` / ``withdraw`` /
``release``, with an injectable clock so aging is testable without real sleeps.
The proxy wires it to the request path in a later stage.

SINGLE-PROCESS INVARIANT: exactly one ``Scheduler`` instance per process, driven
by one asyncio event loop. There are **no locks** — all mutation happens on the
loop thread, so the admit loop is never re-entered concurrently. This is the
same reason the queue-proxy must never run with ``--workers N``: the budget
ledger, per-model in-flight counts, and the waiters list live in this one
process, and sharding them would defeat both the per-pool budget and the ordering.
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

# Default per-pool budget (one pool = the whole GPU) and default cost for models
# with no cap. These mirror the OVERLAAT_BUDGET / OVERLAAT_DEFAULT_COST env
# defaults; the proxy passes the env-resolved values in, but the constructor
# defaults keep the core usable (and the unit tests honest) on its own.
DEFAULT_BUDGET = 1.0
DEFAULT_COST = 1.0
DEFAULT_PRIORITY = 0
DEFAULT_AGING_RATE = 0.0
DEFAULT_POOL = "default"


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
    # head budget-reservation for its pool, so its admission reports "reserved"
    # even though the reservation clears on the pass that finally admits it (the
    # head then fits the full pool budget). Reset to False if a later pass
    # displaces it as the reserved head of its pool (e.g. a higher-priority
    # arrival), so a subsequent work-conserving packing admission is not
    # mislabeled "reserved".
    _was_reserved: bool = False
    # Live wait cause latched by ``_record_wait_reasons`` while this waiter is
    # parked: one of "model_cap" / "exclusive" / "budget_full". Read back at
    # admission instead of recomputing the blocker — by admission the cap slot is
    # free and/or the exclusion mutex has handed off (that is *why* the waiter is
    # being admitted), so a fresh ``_cap_full`` / ``_exclusion_blocks`` check
    # there is always False and would misattribute every cap/exclusion wait to
    # "budget_full" (#17). None until the waiter first waits.
    _wait_cause: str | None = None


def _new_future() -> asyncio.Future:
    """A future bound to the running loop (so Waiter() works inside the loop)."""
    return asyncio.get_event_loop().create_future()


class Scheduler:
    """One global, cost-weighted, priority-ordered, per-pool admission scheduler.

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
        pool_of: dict[str, str] | None = None,
        pool_budget: dict[str, float] | None = None,
        pool_exclusive: set[str] | None = None,
        pool_heavy_max: dict[str, str] | None = None,
        default_cost: float = DEFAULT_COST,
        default_priority: int = DEFAULT_PRIORITY,
        aging_rate: float = DEFAULT_AGING_RATE,
        key_ceiling: Callable[[str | None], int | None] | None = None,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        """
        :param budget: default budget for the ``default`` pool and any pool not
            named in ``pool_budget``; default 1.0 (one pool = the whole GPU).
        :param caps: per-model backend cap (``max_parallel_requests``). A model
            absent here has no declared cap.
        :param costs: optional per-model explicit cost overrides
            (``model_info.overlaat_cost``). Takes precedence over ``1/cap``.
        :param pool_of: model -> pool name (``model_info.overlaat_pool``). A
            model absent here is in the ``default`` pool.
        :param pool_budget: pool name -> budget ``B_pool``. A pool absent here
            uses ``budget`` (the OVERLAAT_BUDGET default).
        :param pool_exclusive: set of pool names that are *exclusive* — a hard
            mutex where at most one DISTINCT member id is active at a time. The
            resident member's own cap + the pool budget govern how many of its
            streams run concurrently.
        :param pool_heavy_max: pool name -> how much of the pool budget a single
            prompt-size-weighted request may consume: ``"full_pool"`` (up to the
            whole budget, so a heavy prompt can serialize behind everything) or
            ``"leave_room"`` (capped at ``budget - base_cost`` so at least one
            more base-cost run of that model always fits). A pool absent here
            defaults to ``"leave_room"``. Only consulted by :meth:`weighted_cost`.
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
        self._pool_of: dict[str, str] = dict(pool_of or {})
        self._pool_budget: dict[str, float] = {p: float(b) for p, b in (pool_budget or {}).items()}
        self._pool_exclusive: set[str] = set(pool_exclusive or set())
        self._pool_heavy_max: dict[str, str] = dict(pool_heavy_max or {})
        self.default_cost = float(default_cost)
        self.default_priority = int(default_priority)
        self.aging_rate = float(aging_rate)
        self._key_ceiling = key_ceiling
        self._now = now

        # Per-pool committed budget (``_used``) and per-pool head reservation
        # (``_reserved``: pool -> the Waiter holding it, or None). These are the
        # real per-pool state; the ``used`` / ``reserved_for`` properties below
        # expose the ``default`` pool's value as a scalar for back-compat with
        # the single-budget readers (and the existing tests). Per-pool access is
        # via :meth:`used_in` / :meth:`reserved_for_pool`.
        self._used: dict[str, float] = {}
        self.in_flight: dict[str, int] = {}
        self.waiters: list[Waiter] = []
        self._reserved: dict[str, Waiter | None] = {}

    # -- back-compat scalar views (the default pool) ----------------------

    @property
    def used(self) -> float:
        """Committed budget of the ``default`` pool (back-compat scalar view).

        Single-pool deployments and the existing single-budget readers treat
        ``used`` as one number; that number is the ``default`` pool's. Use
        :meth:`used_in` for an arbitrary pool."""
        return self._used.get(DEFAULT_POOL, 0.0)

    @property
    def reserved_for(self) -> Waiter | None:
        """Reserved head of the ``default`` pool (back-compat scalar view).

        Use :meth:`reserved_for_pool` for an arbitrary pool."""
        return self._reserved.get(DEFAULT_POOL)

    def used_in(self, pool: str) -> float:
        """Committed budget of ``pool``."""
        return self._used.get(pool, 0.0)

    def reserved_for_pool(self, pool: str) -> Waiter | None:
        """The Waiter holding ``pool``'s head budget-reservation, or None."""
        return self._reserved.get(pool)

    # -- pool / cost / cap derivation -------------------------------------

    def pool(self, model: str) -> str:
        """Pool a model is assigned to; ``"default"`` when unconfigured."""
        return self._pool_of.get(model, DEFAULT_POOL)

    def budget(self, pool: str) -> float:
        """Budget ``B_pool`` for a pool; the global default when undeclared."""
        return self._pool_budget.get(pool, self.B)

    def is_exclusive(self, pool: str) -> bool:
        """True if ``pool`` is an exclusive (one-distinct-member-at-a-time) pool."""
        return pool in self._pool_exclusive

    def active_members(self, pool: str) -> set[str]:
        """Distinct member models with ``in_flight > 0`` in ``pool``."""
        return {m for m in self._models_in_pool(pool) if self.in_flight.get(m, 0) > 0}

    def _models_in_pool(self, pool: str) -> set[str]:
        """Every model known to be assigned to ``pool`` (explicit or via caps).

        Drawn from the union of the configured pool map, the caps map, and any
        model that has actually run (``in_flight`` keys), so an exclusive pool's
        residency is computed over all its members even if a member has no cap.
        """
        out = {m for m, p in self._pool_of.items() if p == pool}
        if pool == DEFAULT_POOL:
            # The default pool also contains every model without an explicit
            # pool assignment that the scheduler has otherwise seen.
            for m in set(self.caps) | set(self.in_flight):
                if m not in self._pool_of:
                    out.add(m)
        return out

    def cap(self, model: str) -> int | None:
        """Backend concurrency cap for a model, or None if it has no declared cap."""
        return self.caps.get(model)

    def cost(self, model: str) -> float:
        """Cost (pool fraction) charged for one in-flight run of ``model``.

        Precedence: an explicit per-model override (``overlaat_cost``); else
        ``1 / cap``; else the configured default cost for an uncapped model.

        Note: there is no forced-1.0 branch for pool members. Mutual exclusion in
        an exclusive pool is a *separate* hard constraint (see
        :meth:`_exclusion_blocks`), not a cost hack — so a cap-2 member in an
        exclusive pool honestly costs ``0.5`` and runs two streams (#12).
        """
        if model in self._costs:
            return self._costs[model]
        cap = self.caps.get(model)
        if cap and cap > 0:
            return 1.0 / cap
        return self.default_cost

    def weighted_cost(self, model: str, weight: float = 1.0) -> float:
        """Prompt-size-weighted cost, hard-clamped to the model's pool budget.

        Base cost (``cost(model)``) is multiplied by ``weight`` (>= 1.0; a
        bigger prompt costs more, so heavy requests consume more of the pool and
        fewer run concurrently — protecting the fast lane, #18). The result is
        clamped so a single request can NEVER exceed what its pool can admit:
        a cost above the pool budget would make the request un-admittable and,
        as the eager-reserved head, would starve its whole pool. The clamp mode
        is per-pool (``pool_heavy_max``):

        - ``"leave_room"`` (default): cap at ``budget - base`` so at least one
          more base-cost run of ``model`` always fits alongside the heavy one.
        - ``"full_pool"``: cap at ``budget`` (minus an epsilon) so a heavy prompt
          may take the entire pool and run strictly alone (batch-job intent).

        Never returns below the base cost (``weight <= 1`` is a no-op), and never
        below it even when ``budget - base`` would be smaller (e.g. a cap-1
        model, where no room can be left regardless).
        """
        base = self.cost(model)
        if weight <= 1.0:
            return base
        p = self.pool(model)
        b = self.budget(p)
        mode = self._pool_heavy_max.get(p, "leave_room")
        ceiling = (b - EPS) if mode == "full_pool" else (b - base)
        return max(base, min(base * weight, ceiling))

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

        Clears the per-pool reservation if the withdrawn waiter was holding it.
        Admitted waiters are no longer in ``self.waiters`` and are released via
        :meth:`release` instead; withdrawing one is a no-op.
        """
        waiter.cancelled = True
        if waiter in self.waiters:
            self.waiters.remove(waiter)
        p = self.pool(waiter.model)
        if self._reserved.get(p) is waiter:
            self._reserved[p] = None
        self.pump(self._now())

    def release(self, model: str) -> None:
        """Account the completion of one in-flight run of ``model`` and re-pump.

        This is the only path that frees budget; it is called on every request
        completion (success, error, or in-flight client disconnect). When the
        last stream of the resident member of an exclusive pool drains, the pool
        becomes idle and the next pump lets a *different* member become resident
        (the mutex hand-off).
        """
        p = self.pool(model)
        self._used[p] = max(0.0, self._used.get(p, 0.0) - self.cost(model))
        self.in_flight[model] = max(0, self.in_flight.get(model, 0) - 1)
        self.pump(self._now())

    # -- the admit loop ----------------------------------------------------

    def _exclusion_blocks(self, w: Waiter) -> bool:
        """True iff ``w`` is blocked by its pool's one-distinct-member mutex.

        Blocks only when the pool is exclusive AND some *different* member is
        currently resident. If ``w.model`` is itself the resident member, more of
        its streams may admit (subject to cap + budget). If the pool is idle,
        ``w.model`` may become the resident member.
        """
        p = self.pool(w.model)
        if not self.is_exclusive(p):
            return False
        active = self.active_members(p)
        return bool(active) and w.model not in active

    def _fits(self, w: Waiter, reservable: float) -> bool:
        """True if ``w`` satisfies the cap, the exclusion mutex, and the budget.

        Backend cap binds always; an exclusive pool's distinct-member mutex binds
        always; the budget test uses ``reservable`` (the pool budget minus any
        head reservation in that pool) so cheap jobs cannot eat budget a reserved
        expensive head is draining toward.
        """
        if self._cap_full(w):
            return False
        if self._exclusion_blocks(w):
            return False
        return self._used.get(self.pool(w.model), 0.0) + w.cost <= reservable + EPS

    def _cap_full(self, w: Waiter) -> bool:
        """True if ``w``'s per-model backend cap currently has no free slot."""
        cap = self.caps.get(w.model)
        return cap is not None and self.in_flight.get(w.model, 0) >= cap

    def _admit(self, w: Waiter, reason: str) -> None:
        p = self.pool(w.model)
        self._used[p] = self._used.get(p, 0.0) + w.cost
        self.in_flight[w.model] = self.in_flight.get(w.model, 0) + 1
        self.waiters.remove(w)
        if self._reserved.get(p) is w:
            self._reserved[p] = None
        w.wait_reason = reason
        if not w.fut.done():
            w.fut.set_result(True)

    def _pool_head(self, ordered: list[Waiter], pool: str) -> Waiter | None:
        """The highest-ordered waiter in ``pool`` from a globally-ordered list."""
        for w in ordered:
            if self.pool(w.model) == pool:
                return w
        return None

    def pump(self, now: float | None = None) -> None:
        """The admit loop: order globally, reserve each pool's head, pack the rest.

        Runs to a fixed point — keeps admitting until no waiter fits — on every
        arrival, withdrawal, and release. Order is GLOBAL by ``(-effective_priority,
        enqueued_at)`` so higher priority wins and FIFO breaks ties; aging and the
        key-ceiling clamp are unchanged. The per-pool budgets then gate admission:
        a budget-blocked head in pool A never idles pool B (cross-pool isolation,
        #11).

        Reservation (eager, per pool): for each pool with waiters, its head (the
        highest-ordered waiter in that pool) is reserved-for *only* when it is
        blocked **by budget** — its cap has a free slot, no exclusion blocks it,
        but its pool budget does not fit ``head.cost``. The pool's budget
        available to *other* waiters in that pool this pass is then reduced to
        ``B_pool - head.cost`` (the pool's ``reservable``), so freed budget is
        held for the head rather than handed to a fresh cheap arrival in the same
        pool. A head blocked by its own backend cap OR by the exclusion mutex is
        NOT reserved for (reserving budget cannot free a cap slot or evict the
        resident member) — it simply waits while work-conserving packing
        continues.

        Each admission records ``waiter.wait_reason`` (so the proxy can log why a
        request waited):

        - ``"none"``      — admitted on the first pump after enqueue (never waited).
        - ``"reserved"``  — this waiter *was* its pool's reserved head and was
          admitted once that pool's budget drained for it.
        - ``"aged_in"``   — waited, and aging lifted its effective priority above
          equal-nominal-priority peers before it was admitted.
        - ``"model_cap"`` — waited because the per-model backend cap was full.
        - ``"exclusive"`` — waited because a *different* member held its
          exclusive pool's mutex.
        - ``"budget_full"`` — waited because the pool budget had no room.

        NEVER preempts: admitted runs are accounted only via :meth:`release`.
        """
        if now is None:
            now = self._now()

        while True:
            if not self.waiters:
                # No waiters anywhere: clear every pool's reservation.
                for p in list(self._reserved):
                    self._reserved[p] = None
                return

            ordered = sorted(
                self.waiters,
                key=lambda w: (-self.effective_priority(w, now), w.enqueued_at),
            )

            # Eager reservation per pool: compute each pool's reservable budget.
            pools_with_waiters = {self.pool(w.model) for w in self.waiters}
            reservable: dict[str, float] = {}
            for p in pools_with_waiters:
                head = self._pool_head(ordered, p)
                # head is non-None: p is in pools_with_waiters.
                assert head is not None
                head_budget_blocked = (
                    not self._cap_full(head)
                    and not self._exclusion_blocks(head)
                    and self._used.get(p, 0.0) + head.cost > self.budget(p) + EPS
                )
                prev = self._reserved.get(p)
                if head_budget_blocked:
                    # A different prior reserved waiter in this pool has been
                    # displaced as head (e.g. a higher-priority arrival). Clear
                    # its _was_reserved so it does not falsely report "reserved"
                    # if later admitted via work-conserving packing.
                    if prev is not None and prev is not head:
                        prev._was_reserved = False
                    self._reserved[p] = head
                    head._was_reserved = True
                    reservable[p] = self.budget(p) - head.cost
                else:
                    # No budget-blocked head in this pool: nobody is reserved for,
                    # so a prior reserved waiter that is no longer the head must
                    # not keep _was_reserved (it would mislabel a later packing
                    # admission).
                    if prev is not None and prev is not head:
                        prev._was_reserved = False
                    self._reserved[p] = None
                    reservable[p] = self.budget(p)

            # Pools with no waiters cannot hold a reservation.
            for p in list(self._reserved):
                if p not in pools_with_waiters and self._reserved[p] is not None:
                    self._reserved[p] = None

            # Work-conserving packing over the GLOBAL ordered list: admit the
            # highest-priority waiter that fits its own pool's reservable budget.
            admitted = None
            for w in ordered:
                if self._fits(w, reservable[self.pool(w.model)]):
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
        """Classify why ``w`` waited, finalized at admission time.

        Attribution order: reserved > aged_in > model_cap > exclusive > budget_full.

        ``reserved`` and ``aged_in`` are derived from state that survives to
        admission (the ``_was_reserved`` latch and the priority/clock). The cap
        and exclusion blockers do NOT survive — they have necessarily cleared by
        the time ``w`` is admitted — so this returns the cause ``w`` was last
        observed waiting on (latched in :meth:`_record_wait_reasons`) rather than
        recomputing it post-clearance (#17).
        """
        if not w._waited:
            return "none"
        if w._was_reserved:
            return "reserved"
        # It waited. Aging is still observable at admission; cap/exclusion are
        # not, so fall back to the latched live cause (defaulting to budget_full,
        # which a waited waiter always has unless cap/exclusion latched first).
        clamped = min(float(w.base_priority), self._ceiling(w.key_fp))
        aged = self.effective_priority(w, now) - clamped
        if self.aging_rate > 0.0 and aged > 0.0:
            return "aged_in"
        return w._wait_cause or "budget_full"

    def _record_wait_reasons(self, now: float) -> None:
        """Tag every still-waiting waiter as having waited, with a live reason.

        Attribution order mirrors :meth:`_reason_for`:
        reserved > model_cap > exclusive > budget_full (aging is reflected in the
        order, not labeled here — the admission classifier owns ``aged_in``).

        The cap/exclusion/budget classification is also latched onto
        ``w._wait_cause`` so :meth:`_reason_for` can recover it at admission,
        after the blocker has cleared (#17). The ``reserved`` head is latched via
        ``_was_reserved`` instead, so it is left out of ``_wait_cause`` here.
        """
        for w in self.waiters:
            w._waited = True
            p = self.pool(w.model)
            if self._reserved.get(p) is w:
                w.wait_reason = "reserved"
                continue
            if self._cap_full(w):
                w.wait_reason = w._wait_cause = "model_cap"
            elif self._exclusion_blocks(w):
                w.wait_reason = w._wait_cause = "exclusive"
            else:
                w.wait_reason = w._wait_cause = "budget_full"

    # -- introspection (for /__queue/status and tests) --------------------

    def queue_depth(self) -> int:
        return len(self.waiters)

    def total_in_flight(self) -> int:
        return sum(self.in_flight.values())

    def _pools_seen(self) -> set[str]:
        """Every pool the scheduler currently has any state or waiters for.

        Includes the default pool, all configured pools, all pools with budget or
        exclusivity declared, plus any pool that has waiters or committed budget.
        """
        pools = {DEFAULT_POOL}
        pools |= set(self._pool_of.values())
        pools |= set(self._pool_budget)
        pools |= set(self._pool_exclusive)
        pools |= set(self._used)
        pools |= {self.pool(w.model) for w in self.waiters}
        # Pools derived from caps for any uncapped/capped model with no explicit
        # pool fall into "default", already present.
        return pools

    def _pool_in_flight(self, pool: str) -> dict[str, int]:
        """Per-model in_flight counts for the members of ``pool`` (positive only)."""
        return {m: c for m, c in self.in_flight.items() if c > 0 and self.pool(m) == pool}

    def snapshot(self) -> dict:
        """A small, JSON-friendly view of scheduler state for status endpoints.

        Returns a per-pool ``pools`` map plus legacy top-level fields reflecting
        the ``default`` pool, so existing readers (and the proxy's budget_pct
        computation) keep working unchanged.
        """
        pools: dict[str, dict] = {}
        for p in sorted(self._pools_seen()):
            b = self.budget(p)
            used = self._used.get(p, 0.0)
            inflight = self._pool_in_flight(p)
            active = self.active_members(p)
            resv = self._reserved.get(p)
            pools[p] = {
                "budget": b,
                "used": used,
                "exclusive": self.is_exclusive(p),
                "active_member": next(iter(active)) if (self.is_exclusive(p) and active) else None,
                "reserved_for": resv.req_id if resv else None,
                "in_flight": inflight,
                "budget_pct": round(used / b * 100, 1) if b > 0 else None,
            }

        default_used = self._used.get(DEFAULT_POOL, 0.0)
        default_resv = self._reserved.get(DEFAULT_POOL)
        return {
            # Legacy top-level fields: the default pool, for back-compat.
            "budget": self.budget(DEFAULT_POOL),
            "used": default_used,
            "queue_depth": self.queue_depth(),
            "in_flight": dict(self.in_flight),
            "reserved_for": default_resv.req_id if default_resv else None,
            # The per-pool map.
            "pools": pools,
        }

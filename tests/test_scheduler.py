"""Pure unit tests for the cost-weighted admission scheduler.

No FastAPI, no DB, no real sleeps: aging is driven by a manually-advanced clock
injected into the Scheduler, so every timing case is deterministic. Each test
maps to a named case from the implementation spec / docs/COST-SCHEDULER.md.
"""

from __future__ import annotations

import asyncio

import pytest

from overlaat.scheduler import Scheduler, Waiter


class Clock:
    """A manually-advanced monotonic clock for deterministic aging tests."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def make_waiter(
    sched: Scheduler,
    model: str,
    *,
    req_id: str | None = None,
    priority: int = 0,
    key_fp: str | None = None,
    at: float | None = None,
) -> Waiter:
    """Build a Waiter with cost derived from the scheduler and a fresh future."""
    return Waiter(
        req_id=req_id or f"{model}-{id(object())}",
        model=model,
        cost=sched.cost(model),
        base_priority=priority,
        key_fp=key_fp,
        enqueued_at=at if at is not None else sched._now(),
        fut=asyncio.get_event_loop().create_future(),
    )


def admitted(w: Waiter) -> bool:
    return w.fut.done() and w.fut.result() is True


# --------------------------------------------------------------------------


async def test_cost_derived_from_cap():
    s = Scheduler(budget=1.0, caps={"a": 4, "b": 2, "c": 1})
    assert s.cost("a") == pytest.approx(0.25)
    assert s.cost("b") == pytest.approx(0.5)
    assert s.cost("c") == pytest.approx(1.0)


async def test_default_cost_for_uncapped_model():
    # No cap, no override -> default cost (full budget by default).
    s = Scheduler(budget=1.0, caps={}, default_cost=1.0)
    assert s.cost("nocap") == pytest.approx(1.0)
    s2 = Scheduler(budget=1.0, caps={}, default_cost=0.5)
    assert s2.cost("nocap") == pytest.approx(0.5)


async def test_admit_immediately_when_budget_free():
    s = Scheduler(budget=1.0, caps={"a": 4})
    w = make_waiter(s, "a")
    s.enqueue(w)
    assert admitted(w)
    assert w.wait_reason == "none"
    assert s.used == pytest.approx(0.25)
    assert s.in_flight["a"] == 1


async def test_budget_boundary_packs_to_exactly_B():
    # cap=9 -> cost=1/9; nine of them must pack to exactly B=1.0. The
    # *incremental* float sum used += 1/9 nine times lands at
    # 1.0000000000000002 > 1.0 -> without the EPS in the admit test the 9th
    # would phantom-overflow the budget and be refused.
    n = 9
    acc = 0.0
    for _ in range(n):
        acc += 1.0 / n
    assert acc > 1.0  # the float pathology EPS guards against

    s = Scheduler(budget=1.0, caps={"a": n})
    ws = [make_waiter(s, "a", req_id=f"a{i}") for i in range(n)]
    for w in ws:
        s.enqueue(w)
    assert all(admitted(w) for w in ws)  # all 9 admitted despite acc > 1.0
    assert s.used > 1.0  # confirm we are exercising the EPS path
    assert s.in_flight["a"] == n
    # A 10th would exceed both budget and cap -> waits.
    extra = make_waiter(s, "a", req_id="extra")
    s.enqueue(extra)
    assert not admitted(extra)


async def test_priority_ordering():
    # cap=1 cheap so only ONE cheap fits at a time -> ordering is observable:
    # the higher-priority waiter is admitted, the lower one keeps waiting.
    s = Scheduler(budget=1.0, caps={"cheap": 1})  # cost 1.0 each, one at a time
    occ = make_waiter(s, "cheap", req_id="occ", priority=0)
    s.enqueue(occ)
    assert admitted(occ)  # budget full

    lo = make_waiter(s, "cheap", req_id="lo", priority=1)
    hi = make_waiter(s, "cheap", req_id="hi", priority=5)
    s.enqueue(lo)
    s.enqueue(hi)
    assert not admitted(lo) and not admitted(hi)

    s.release("cheap")  # one slot frees; higher priority wins, lo still waits
    assert admitted(hi)
    assert not admitted(lo)

    s.release("cheap")
    assert admitted(lo)


async def test_key_max_clamps_priority():
    # A batch key with a low ceiling cannot out-rank an interactive key even by
    # requesting priority 9999.
    ceilings = {"batchfp": 0, "interfp": 10}
    # cap=1 cheap so only one fits at a time -> ordering is observable.
    s = Scheduler(
        budget=1.0,
        caps={"cheap": 1},
        key_ceiling=lambda fp: ceilings.get(fp),
    )
    occ = make_waiter(s, "cheap", req_id="occ")
    s.enqueue(occ)
    assert admitted(occ)

    # batch requests priority 9999 but is clamped to its key ceiling (0);
    # interactive requests only 1 but its ceiling is 10 -> effective 1 > 0.
    batch = make_waiter(s, "cheap", req_id="batch", priority=9999, key_fp="batchfp")
    inter = make_waiter(s, "cheap", req_id="inter", priority=1, key_fp="interfp")
    s.enqueue(batch)
    s.enqueue(inter)
    assert s.effective_priority(batch, s._now()) == pytest.approx(0)
    assert s.effective_priority(inter, s._now()) == pytest.approx(1)

    s.release("cheap")  # interactive wins the freed slot despite batch's 9999
    assert admitted(inter)
    assert not admitted(batch)


async def test_work_conserving_packing():
    # The head is the highest-priority waiter but is blocked by its own backend
    # cap (not by budget): admitting it now is impossible, yet there is free
    # budget. Rather than idle that budget, the scheduler scans down and admits
    # the highest-priority LOWER-priority job that fits — work-conserving.
    #
    # cheap: cap 1, cost 0.25 (override) -> one cheap can run at a time.
    # other: cap 4, cost 0.25 -> plenty of budget/cap room.
    s = Scheduler(
        budget=1.0,
        caps={"cheap": 1, "other": 4},
        costs={"cheap": 0.25, "other": 0.25},
    )
    # Occupy the single cheap slot so a second cheap is cap-blocked.
    running = make_waiter(s, "cheap", req_id="running", priority=0)
    s.enqueue(running)
    assert admitted(running)
    assert s.used == pytest.approx(0.25)

    # High-priority head wants 'cheap' too, but cheap cap (1) is full -> it
    # cannot be admitted no matter the budget. It is NOT reserved for (a cap
    # block cannot be cured by reserving budget).
    head = make_waiter(s, "cheap", req_id="head", priority=10)
    # Lower-priority job on a different model that fits the free budget.
    low = make_waiter(s, "other", req_id="low", priority=1)
    s.enqueue(head)
    s.enqueue(low)

    assert not admitted(head)  # cap-blocked
    assert s.reserved_for is None  # cap block is not reserved for
    assert admitted(low)  # packed into otherwise-idle budget
    assert s.used == pytest.approx(0.5)
    assert head.wait_reason == "model_cap"


async def test_reservation_drains_for_head():
    # Two cheap (0.5 each) running fills budget. An expensive head (1.0) waits;
    # newly arriving cheap jobs are NOT admitted into the reserved budget; only
    # once both running cheap drain does the head get in. cheap cap is 4 (well
    # above 2) so the intruder below is NOT cap-blocked — only the reservation
    # holds it back, which is the property under test.
    s = Scheduler(budget=1.0, caps={"cheap": 4, "big": 1}, costs={"cheap": 0.5})
    c1 = make_waiter(s, "cheap", req_id="c1")
    c2 = make_waiter(s, "cheap", req_id="c2")
    s.enqueue(c1)
    s.enqueue(c2)
    assert s.used == pytest.approx(1.0)

    big = make_waiter(s, "big", req_id="big", priority=5)
    s.enqueue(big)
    assert s.reserved_for is big

    # A fresh cheap job of EQUAL priority arrives later. Because big is the head
    # (equal priority, earlier enqueue), the reservation holds budget for big and
    # the cheap intruder must not steal the draining budget. (A strictly higher
    # priority would legitimately preempt the head — that is correct ordering,
    # not starvation; the starvation guard protects the *head*.)
    intruder = make_waiter(s, "cheap", req_id="intruder", priority=5)
    s.enqueue(intruder)
    assert not admitted(intruder)

    s.release("cheap")  # one drains; used=0.5, still not enough for big(1.0)
    assert not admitted(big)
    assert not admitted(intruder)  # still reserved for big

    s.release("cheap")  # both drained; used=0.0, big now fits
    assert admitted(big)
    assert big.wait_reason == "reserved"
    # With big now holding the whole budget, intruder still waits — and as the
    # new head it now holds the reservation in turn.
    assert not admitted(intruder)
    assert s.reserved_for is intruder


async def test_displaced_reserved_head_not_mislabeled_reserved():
    # Regression: a budget-blocked waiter that briefly holds the head
    # reservation, is then displaced as head by a higher-priority arrival, and
    # is finally admitted via *work-conserving packing* (not as the reserved
    # head) must NOT report wait_reason "reserved". Only a waiter admitted AS
    # the reserved head reports "reserved".
    #
    # Costs are overrides so the budget arithmetic is exact:
    #   occ    cost 0.85 (fills budget, leaves 0.15 free)
    #   medium cost 0.20, priority 5
    #   hi     cost 0.60, priority 10 (higher -> takes the head reservation)
    # Caps are generous so nothing is ever cap-blocked; only budget/reservation
    # gate admission.
    s = Scheduler(
        budget=1.0,
        caps={"occ": 4, "medium": 4, "hi": 4},
        costs={"occ": 0.85, "medium": 0.20, "hi": 0.60},
    )

    occ = make_waiter(s, "occ", req_id="occ", priority=0)
    s.enqueue(occ)
    assert admitted(occ)
    assert s.used == pytest.approx(0.85)  # only 0.15 free

    # medium is the budget-blocked head -> it gets the reservation.
    medium = make_waiter(s, "medium", req_id="medium", priority=5)
    s.enqueue(medium)
    assert not admitted(medium)
    assert s.reserved_for is medium
    assert medium._was_reserved is True

    # A strictly higher-priority, also-budget-blocked job arrives and displaces
    # medium as head. medium must lose its _was_reserved flag now.
    hi = make_waiter(s, "hi", req_id="hi", priority=10)
    s.enqueue(hi)
    assert s.reserved_for is hi
    assert hi._was_reserved is True
    assert medium._was_reserved is False  # displaced -> reset (the fix)
    assert not admitted(hi) and not admitted(medium)

    # Free the whole budget. hi is admitted AS the reserved head; medium then
    # packs into the remaining budget (hi 0.60 + medium 0.20 <= 1.0) WITHOUT
    # ever being the reserved head on its admitting pass.
    s.release("occ")
    assert admitted(hi)
    assert hi.wait_reason == "reserved"  # admitted as the reserved head
    assert admitted(medium)
    assert medium.wait_reason != "reserved"  # packed in, not reserved
    assert medium.wait_reason == "budget_full"


async def test_aging_prevents_starvation():
    clock = Clock(0.0)
    # A moderate job that is not the head can be leapfrogged by fresher,
    # higher-priority cheap arrivals; aging eventually lifts it above them.
    s = Scheduler(
        budget=1.0,
        caps={"cheap": 1},  # cap 1 cheap -> cost 1.0 each, only one at a time
        aging_rate=1.0,
        now=clock,
    )
    # Occupy the only slot.
    running = make_waiter(s, "cheap", req_id="run", priority=5, at=0.0)
    s.enqueue(running)
    assert admitted(running)

    old = make_waiter(s, "cheap", req_id="old", priority=1, at=0.0)
    s.enqueue(old)
    assert not admitted(old)

    # A stream of fresher, higher-priority jobs keeps arriving.
    clock.advance(10.0)
    fresh = make_waiter(s, "cheap", req_id="fresh", priority=3, at=10.0)
    s.enqueue(fresh)
    # old eff = 1 + 1.0*10 = 11 ; fresh eff = 3 + 0 = 3 -> old now outranks fresh.
    assert s.effective_priority(old, clock()) == pytest.approx(11.0)
    assert s.effective_priority(fresh, clock()) > 0
    assert s.effective_priority(old, clock()) > s.effective_priority(fresh, clock())

    s.release("cheap")  # frees the slot; aged 'old' wins over 'fresh'
    assert admitted(old)
    assert not admitted(fresh)
    assert old.wait_reason == "aged_in"


async def test_fat_slot_mutex_only_one_resident():
    # Two members of a swap-slot group both forced to cost 1.0: admitting one
    # takes used to B, so the other cannot be admitted until release.
    s = Scheduler(
        budget=1.0,
        caps={"llamaBig": 2, "qwenBig": 2},  # caps would suggest cost 0.5...
        slot_groups={"llamaBig": "fat", "qwenBig": "fat"},
    )
    assert s.cost("llamaBig") == pytest.approx(1.0)  # forced to 1.0 by slot group
    assert s.cost("qwenBig") == pytest.approx(1.0)

    a = make_waiter(s, "llamaBig", req_id="a")
    b = make_waiter(s, "qwenBig", req_id="b")
    s.enqueue(a)
    s.enqueue(b)
    assert admitted(a)
    assert not admitted(b)  # budget full at 1.0 -> mutex falls out of arithmetic
    assert s.used == pytest.approx(1.0)

    s.release("llamaBig")
    assert admitted(b)


async def test_model_cap_binds_under_budget():
    # Budget has room but the per-model cap is full -> the model's next request
    # waits even though used + cost <= B. Use a cheap model (cost small) with a
    # tiny cap so cap binds before budget.
    s = Scheduler(budget=1.0, caps={"a": 2})  # cost 0.5 each, cap 2
    a1 = make_waiter(s, "a", req_id="a1")
    a2 = make_waiter(s, "a", req_id="a2")
    s.enqueue(a1)
    s.enqueue(a2)
    assert admitted(a1) and admitted(a2)
    assert s.used == pytest.approx(1.0)  # 2*0.5; budget also full here

    # Make budget clearly have room but cap full: cap=2 cost override 0.25.
    s2 = Scheduler(budget=1.0, caps={"a": 2}, costs={"a": 0.25})
    b1 = make_waiter(s2, "a", req_id="b1")
    b2 = make_waiter(s2, "a", req_id="b2")
    b3 = make_waiter(s2, "a", req_id="b3")
    s2.enqueue(b1)
    s2.enqueue(b2)
    s2.enqueue(b3)
    assert admitted(b1) and admitted(b2)
    assert not admitted(b3)  # used=0.5 (budget has room) but cap=2 is full
    assert s2.used == pytest.approx(0.5)
    assert b3.wait_reason == "model_cap"

    s2.release("a")
    assert admitted(b3)


async def test_release_triggers_pump():
    s = Scheduler(budget=1.0, caps={"a": 1})  # cost 1.0
    a1 = make_waiter(s, "a", req_id="a1")
    a2 = make_waiter(s, "a", req_id="a2")
    s.enqueue(a1)
    s.enqueue(a2)
    assert admitted(a1)
    assert not admitted(a2)
    # No explicit pump call here: release must re-run the admit loop itself.
    s.release("a")
    assert admitted(a2)


async def test_withdraw_clears_reservation_and_repumps():
    s = Scheduler(budget=1.0, caps={"cheap": 4, "big": 1})
    c1 = make_waiter(s, "cheap", req_id="c1")
    c2 = make_waiter(s, "cheap", req_id="c2")
    s.enqueue(c1)
    s.enqueue(c2)  # used = 0.5
    big = make_waiter(s, "big", req_id="big", priority=5)
    s.enqueue(big)
    assert s.reserved_for is big  # 1.0 does not fit 0.5 used

    cheap2 = make_waiter(s, "cheap", req_id="cheap2", priority=1)
    s.enqueue(cheap2)
    assert not admitted(cheap2)  # blocked by reservation (reservable 0.0)

    s.withdraw(big)  # big cancelled-while-queued; reservation must clear
    assert s.reserved_for is None
    assert admitted(cheap2)  # now packs into freed budget

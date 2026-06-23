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


# -- prompt-size-weighted cost (#18) ----------------------------------------


async def test_weighted_cost_leave_room_default_clamps_below_pool():
    # base 0.25 (cap 4), budget 1.0. leave_room (default) caps a weighted cost at
    # budget - base = 0.75, so one more base-cost run always still fits.
    s = Scheduler(budget=1.0, caps={"a": 4})
    assert s.weighted_cost("a", 1.0) == pytest.approx(0.25)  # weight 1 -> no-op
    assert s.weighted_cost("a", 2.0) == pytest.approx(0.5)  # 0.25*2, under cap
    assert s.weighted_cost("a", 4.0) == pytest.approx(0.75)  # 0.25*4=1.0 -> clamp 0.75
    assert s.weighted_cost("a", 99.0) == pytest.approx(0.75)  # never exceeds cap


async def test_weighted_cost_full_pool_allows_whole_budget():
    # full_pool lets a heavy prompt take the entire pool (serialize alone).
    s = Scheduler(budget=1.0, caps={"a": 4}, pool_heavy_max={"default": "full_pool"})
    assert s.weighted_cost("a", 4.0) == pytest.approx(1.0)
    assert s.weighted_cost("a", 99.0) == pytest.approx(1.0)  # clamped to budget


async def test_weighted_cost_never_below_base_for_tight_pool():
    # cap 1 -> base 1.0 = whole budget; no room can be left, so the clamp must
    # not drop below base (a 0-cost request would be nonsense).
    s = Scheduler(budget=1.0, caps={"solo": 1})
    assert s.weighted_cost("solo", 4.0) == pytest.approx(1.0)


async def test_weighted_cost_leaves_room_for_one_light_call_end_to_end():
    # The leave_room property under load: a heavy request (weight 4 -> cost 0.75)
    # still admits one light base-cost call (0.25) alongside it (0.75+0.25=1.0),
    # but a second heavy one does not fit (0.75+0.75 > 1.0).
    s = Scheduler(budget=1.0, caps={"a": 4})
    heavy1 = make_waiter(s, "a", req_id="heavy1")
    heavy1.cost = s.weighted_cost("a", 4.0)  # 0.75
    s.enqueue(heavy1)
    assert admitted(heavy1)

    light = make_waiter(s, "a", req_id="light")  # base cost 0.25
    s.enqueue(light)
    assert admitted(light)  # fits in the deliberately-left room
    assert s.used == pytest.approx(1.0)

    heavy2 = make_waiter(s, "a", req_id="heavy2")
    heavy2.cost = s.weighted_cost("a", 4.0)  # 0.75
    s.enqueue(heavy2)
    assert not admitted(heavy2)  # 1.0 + 0.75 > budget -> waits


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


async def test_admitted_after_cap_wait_reports_model_cap_not_budget_full():
    # Regression (#17): a request that waited because its per-model backend cap
    # was full, then was admitted once a slot freed, must PERSIST wait_reason
    # "model_cap" — not "budget_full". The bug: _reason_for recomputed _cap_full
    # at admission, after the cap had necessarily cleared (that is why the waiter
    # is being admitted), so every cap wait fell through to the budget_full
    # catch-all. Budget here is deliberately non-binding (the live-deployment
    # repro: OVERLAAT_BUDGET=9999 with a binding cap), so budget is never the
    # constraint.
    s = Scheduler(budget=9999.0, caps={"qwen": 1}, costs={"qwen": 0.25})
    running = make_waiter(s, "qwen", req_id="running")
    s.enqueue(running)
    assert admitted(running)

    waiter = make_waiter(s, "qwen", req_id="waiter")
    s.enqueue(waiter)
    assert not admitted(waiter)  # cap=1 full; budget has ample room
    assert waiter.wait_reason == "model_cap"  # live (parked) classification
    assert s.reserved_for is None  # a cap block is not reserved-for

    s.release("qwen")  # the slot frees; the waiter is admitted this pump
    assert admitted(waiter)
    assert waiter.wait_reason == "model_cap"  # latched through admission


async def test_admitted_after_exclusion_wait_reports_exclusive_not_budget_full():
    # Regression (#17): a request that waited because a DIFFERENT member held the
    # exclusive pool mutex, then was admitted on the mutex hand-off, must persist
    # "exclusive" — not "budget_full". Budget is non-binding so the only blocker
    # is the exclusion mutex, which has cleared by the admitting pass.
    s = Scheduler(
        budget=9999.0,
        caps={"a": 2, "b": 2},
        pool_of={"a": "fat", "b": "fat"},
        pool_exclusive={"fat"},
    )
    a = make_waiter(s, "a", req_id="a")
    s.enqueue(a)
    assert admitted(a)  # resident member of the exclusive pool

    b = make_waiter(s, "b", req_id="b")
    s.enqueue(b)
    assert not admitted(b)  # different member: exclusion-blocked, budget has room
    assert b.wait_reason == "exclusive"  # live (parked) classification

    s.release("a")  # pool drains; the mutex hands off to b
    assert admitted(b)
    assert b.wait_reason == "exclusive"  # latched through admission


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
    # Spec case #4 — single-stream fat-slot unchanged, now expressed as an
    # EXCLUSIVE pool instead of the old forced-1.0 slot hack. Two cap-1 members
    # of an exclusive pool: admitting one makes it the resident member (cost 1.0
    # fills the pool budget), so the other cannot be admitted until release, with
    # the same observable behavior as the pre-change fat slot.
    s = Scheduler(
        budget=1.0,
        caps={"llamaBig": 1, "qwenBig": 1},
        pool_of={"llamaBig": "fat", "qwenBig": "fat"},
        pool_exclusive={"fat"},
    )
    assert s.cost("llamaBig") == pytest.approx(1.0)  # 1/cap = 1/1
    assert s.cost("qwenBig") == pytest.approx(1.0)

    a = make_waiter(s, "llamaBig", req_id="a")
    b = make_waiter(s, "qwenBig", req_id="b")
    s.enqueue(a)
    s.enqueue(b)
    assert admitted(a)
    assert not admitted(b)  # different member locked out: budget full AND exclusion
    assert s.used_in("fat") == pytest.approx(1.0)
    # Pre-change wait_reasons preserved: a was admitted on first pump ("none"),
    # b waited. Here budget is also full, so b reports budget_full (cap is not
    # full for b; exclusion also blocks but budget_full wins the same as before).
    assert b.wait_reason in {"budget_full", "exclusive"}

    s.release("llamaBig")
    assert admitted(b)


# --------------------------------------------------------------------------
# Resource pools + exclusive groups (closes #11 / #12).
# --------------------------------------------------------------------------


async def test_pool_isolation_cross_backend():
    # Spec case #1 (#11): an exclusive fat-slot pool (B=1, model-b cap 1, cost
    # 1.0) and a separate embeddings pool (B=1, emb cap 2, cost 0.5). Admitting
    # model-b fills the fat-slot budget, but an embeddings waiter is admitted
    # IMMEDIATELY into its own pool — the fat-slot budget never idles embeddings.
    s = Scheduler(
        budget=1.0,
        caps={"model-b": 1, "emb": 2},
        pool_of={"model-b": "fat-slot", "emb": "embeddings"},
        pool_exclusive={"fat-slot"},
    )
    b = make_waiter(s, "model-b", req_id="b")
    s.enqueue(b)
    assert admitted(b)
    assert s.used_in("fat-slot") == pytest.approx(1.0)

    e = make_waiter(s, "emb", req_id="e")
    s.enqueue(e)
    assert admitted(e)  # isolated pool: not blocked by the full fat-slot
    assert e.wait_reason == "none"
    assert s.used_in("embeddings") == pytest.approx(0.5)


async def test_exclusive_pool_cap2_member_runs_two():
    # Spec case #2 (#12): an exclusive pool whose resident member is cap-2
    # (cost 0.5). Both of its streams run concurrently — exclusivity is across
    # DISTINCT members, not within a member.
    s = Scheduler(
        budget=1.0,
        caps={"model-b": 2},
        pool_of={"model-b": "fat-slot"},
        pool_exclusive={"fat-slot"},
    )
    assert s.cost("model-b") == pytest.approx(0.5)
    b1 = make_waiter(s, "model-b", req_id="b1")
    b2 = make_waiter(s, "model-b", req_id="b2")
    s.enqueue(b1)
    s.enqueue(b2)
    assert admitted(b1) and admitted(b2)
    assert s.used_in("fat-slot") == pytest.approx(1.0)
    assert s.in_flight["model-b"] == 2


async def test_exclusive_pool_distinct_member_trap_rejected_and_handoff():
    # Spec case #3 (#12): the 0.5 + 0.5 distinct-member trap. fat-slot exclusive,
    # model-b cap 2 (0.5) + model-c cap 2 (0.5). Admit one model-b (used 0.5);
    # a model-c waiter is NOT admitted despite 0.5 budget headroom, because a
    # DIFFERENT member is resident — wait_reason 'exclusive'. Releasing both
    # model-b streams drains the pool; model-c then becomes the resident member
    # (the mutex hand-off).
    s = Scheduler(
        budget=1.0,
        caps={"model-b": 2, "model-c": 2},
        pool_of={"model-b": "fat-slot", "model-c": "fat-slot"},
        pool_exclusive={"fat-slot"},
    )
    b1 = make_waiter(s, "model-b", req_id="b1")
    s.enqueue(b1)
    assert admitted(b1)
    assert s.used_in("fat-slot") == pytest.approx(0.5)  # 0.5 headroom remains

    c = make_waiter(s, "model-c", req_id="c")
    s.enqueue(c)
    assert not admitted(c)  # rejected despite budget room — different member
    assert c.wait_reason == "exclusive"

    # A second model-b stream still fits (same resident member).
    b2 = make_waiter(s, "model-b", req_id="b2")
    s.enqueue(b2)
    assert admitted(b2)
    assert s.used_in("fat-slot") == pytest.approx(1.0)
    assert not admitted(c)

    # Drain both model-b streams → pool idle → model-c becomes resident.
    s.release("model-b")
    assert not admitted(c)  # one model-b still resident
    s.release("model-b")
    assert admitted(c)  # mutex hand-off: model-c is now the resident member
    assert s.active_members("fat-slot") == {"model-c"}


async def test_legacy_overlaat_slot_bridge_matches_explicit_pool():
    # Spec case #5: a config that declares only the legacy overlaat_slot loads to
    # an exclusive pool, behaving identically to an explicit overlaat_pool +
    # exclusive. We exercise the proxy loader's bridge directly here.
    from pathlib import Path
    from tempfile import NamedTemporaryFile

    from overlaat import queue_proxy as qp

    cfg = """
model_list:
  - model_name: model-b
    litellm_params: { model: x/large-b, max_parallel_requests: 1 }
    model_info: { overlaat_slot: fat-slot }
  - model_name: model-c
    litellm_params: { model: x/large-c, max_parallel_requests: 1 }
    model_info: { overlaat_slot: fat-slot }
"""
    with NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(cfg)
        path = Path(f.name)

    caps = qp.load_caps(path)
    costs, pool_of, exclusive_seed, _, _ = qp.load_model_info(path)
    declared_budgets, declared_exclusive = qp.load_pools(path)
    pool_budget, pool_exclusive = qp.resolve_pool_config(
        caps, costs, pool_of, exclusive_seed, declared_budgets, declared_exclusive, 1.0, log=False
    )
    # The bridge: both members assigned to pool 'fat-slot', which is exclusive.
    assert pool_of == {"model-b": "fat-slot", "model-c": "fat-slot"}
    assert "fat-slot" in pool_exclusive

    s = Scheduler(
        budget=1.0,
        caps=caps,
        costs=costs,
        pool_of=pool_of,
        pool_budget=pool_budget,
        pool_exclusive=pool_exclusive,
    )
    b = make_waiter(s, "model-b", req_id="b")
    c = make_waiter(s, "model-c", req_id="c")
    s.enqueue(b)
    s.enqueue(c)
    assert admitted(b)
    assert not admitted(c)  # exclusive: different member locked out
    s.release("model-b")
    assert admitted(c)


async def test_cross_pool_reservation_isolation():
    # Spec case #7: pool A's head is budget-blocked and reserves in A; a cheap
    # waiter in pool B is admitted because B's reservable is the full B_B —
    # A's reservation never bleeds into B.
    s = Scheduler(
        budget=1.0,
        caps={"a-big": 1, "a-cheap": 4, "b-cheap": 4},
        costs={"a-big": 1.0, "a-cheap": 0.5, "b-cheap": 0.25},
        pool_of={"a-big": "A", "a-cheap": "A", "b-cheap": "B"},
        pool_budget={"A": 1.0, "B": 1.0},
    )
    # Fill pool A to 0.5 with one a-cheap, then a-big becomes the budget-blocked head.
    occ = make_waiter(s, "a-cheap", req_id="occ", priority=0)
    s.enqueue(occ)
    assert admitted(occ)
    big = make_waiter(s, "a-big", req_id="big", priority=10)
    s.enqueue(big)
    assert not admitted(big)
    assert s.reserved_for_pool("A") is big  # reserved within pool A
    assert s.reserved_for_pool("B") is None

    # A cheap waiter in pool B is admitted into B's untouched budget.
    bc = make_waiter(s, "b-cheap", req_id="bc", priority=1)
    s.enqueue(bc)
    assert admitted(bc)
    assert s.used_in("B") == pytest.approx(0.25)


async def test_exclusion_blocked_head_not_reserved_for():
    # Spec case #8: in an exclusive pool with a resident member, a DIFFERENT
    # member at the head is NOT reserved-for (reserving budget cannot evict the
    # resident); meanwhile another stream of the resident member still packs into
    # the remaining budget.
    s = Scheduler(
        budget=1.0,
        caps={"resident": 2, "other": 2},
        pool_of={"resident": "fat", "other": "fat"},
        pool_exclusive={"fat"},
    )
    r1 = make_waiter(s, "resident", req_id="r1", priority=0)
    s.enqueue(r1)
    assert admitted(r1)  # resident member, used 0.5

    # A different member at higher priority is the pool head but exclusion-blocked.
    head_other = make_waiter(s, "other", req_id="other", priority=10)
    s.enqueue(head_other)
    assert not admitted(head_other)
    assert s.reserved_for_pool("fat") is None  # exclusion-blocked head NOT reserved
    assert head_other.wait_reason == "exclusive"

    # A second resident stream still packs into the remaining 0.5 budget.
    r2 = make_waiter(s, "resident", req_id="r2", priority=1)
    s.enqueue(r2)
    assert admitted(r2)
    assert s.used_in("fat") == pytest.approx(1.0)
    assert not admitted(head_other)


async def test_aging_within_exclusive_pool_never_bypasses_exclusion():
    # Spec case #9: aging changes ORDER (a long-waiting different-member waiter
    # can out-rank fresher peers) but never bypasses the exclusion mutex — it
    # still cannot be admitted while a different member is resident.
    clock = Clock(0.0)
    s = Scheduler(
        budget=1.0,
        caps={"resident": 1, "other": 1},
        pool_of={"resident": "fat", "other": "fat"},
        pool_exclusive={"fat"},
        aging_rate=1.0,
        now=clock,
    )
    r = make_waiter(s, "resident", req_id="r", priority=5, at=0.0)
    s.enqueue(r)
    assert admitted(r)

    other = make_waiter(s, "other", req_id="other", priority=1, at=0.0)
    s.enqueue(other)
    assert not admitted(other)

    # Age 'other' well past the resident's nominal priority.
    clock.advance(100.0)
    s.pump(clock())
    assert s.effective_priority(other, clock()) > 5  # aged above resident's priority
    assert not admitted(other)  # exclusion still blocks despite the higher rank
    assert other.wait_reason == "exclusive"

    # Only releasing the resident lets 'other' in (the mutex, not the order).
    s.release("resident")
    assert admitted(other)


async def test_snapshot_back_compat_and_pools_map():
    # Spec case #10: snapshot exposes a per-pool 'pools' map AND keeps the legacy
    # top-level default-pool fields so the proxy's budget_pct computation and
    # existing readers keep working.
    s = Scheduler(
        budget=1.0,
        caps={"d": 4, "model-b": 1},
        pool_of={"model-b": "fat-slot"},
        pool_exclusive={"fat-slot"},
    )
    d = make_waiter(s, "d", req_id="d")  # default pool
    b = make_waiter(s, "model-b", req_id="b")  # fat-slot pool
    s.enqueue(d)
    s.enqueue(b)
    assert admitted(d) and admitted(b)

    snap = s.snapshot()
    # Legacy top-level fields reflect the default pool.
    assert snap["budget"] == pytest.approx(1.0)
    assert snap["used"] == pytest.approx(0.25)  # only 'd' is in default
    assert snap["reserved_for"] is None
    assert snap["in_flight"] == {"d": 1, "model-b": 1}
    # The proxy's budget_pct computation off the top-level fields still works.
    assert round(snap["used"] / snap["budget"] * 100, 1) == 25.0

    # The pools map carries each pool's own state.
    pools = snap["pools"]
    assert pools["default"]["used"] == pytest.approx(0.25)
    assert pools["default"]["exclusive"] is False
    assert pools["fat-slot"]["used"] == pytest.approx(1.0)
    assert pools["fat-slot"]["exclusive"] is True
    assert pools["fat-slot"]["active_member"] == "model-b"
    assert pools["fat-slot"]["in_flight"] == {"model-b": 1}
    assert pools["fat-slot"]["budget_pct"] == 100.0


async def test_event_wiring_column_counts_match():
    # Spec case #11 (wiring): the event column tuple, the SQLite positional
    # placeholders, and the Postgres named params must all agree in count, or a
    # column-order drift silently corrupts SQLite rows.
    from overlaat import queue_proxy as qp

    n = len(qp._EVENT_COLS)
    assert qp._INSERT_SQL_SQLITE.count("?") == n
    # Each named param appears once as %(name)s in the PG statement.
    for col in qp._EVENT_COLS:
        assert f"%({col})s" in qp._INSERT_SQL_PG
    assert qp._INSERT_SQL_PG.count("%(") == n
    # 'workload' is the newly-added column (#19), last in the tuple; 'pool' (the
    # prior addition) is still wired.
    assert qp._EVENT_COLS[-1] == "workload"
    assert "pool" in qp._EVENT_COLS


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


# -- self-protection: oversized-prompt vector (#24) -------------------------


async def test_fast_lane_survives_under_protective_config():
    """The FIX (now the default): non-flat tiers + a bounded budget + leave_room
    keep a single workload's oversized prompts budget-throttled, so they cannot
    pin every cap slot and the latency-sensitive fast lane keeps flowing.

    base cost = 1/cap = 0.25 (cap 4, budget 1.0). A heavy prompt (weight 4×) is
    clamped to leave_room = budget - base = 0.75, so two heavies (0.75+0.75 > 1.0)
    never co-reside even though the cap has free slots — and exactly one base-cost
    fast-lane call (0.25) always fits alongside the resident heavy (0.75+0.25=1.0).

    DEVIATION FROM SPEC (#24): the spec enqueued the second heavy *before* the
    light call and expected the light call to still admit. But a budget-blocked
    head is EAGERLY RESERVED (the anti-starvation guard, §3c), so a reserved
    second heavy would hold the leave_room budget away from a later light arrival
    — the fast lane survives only for a light call that is already resident or
    arrives while no heavy is reserved ahead of it. So the light call is enqueued
    first (the proven leave_room ordering, matching
    test_weighted_cost_leaves_room_for_one_light_call_end_to_end), then the second
    heavy is shown to wait. This is the faithful encoding of the invariant given
    the scheduler's real reservation semantics.
    """
    s = Scheduler(budget=1.0, caps={"m": 4})  # base cost 0.25; default heavy_max

    heavy_cost = s.weighted_cost("m", 4.0)
    assert heavy_cost == pytest.approx(0.75)  # leave_room clamp: budget - base

    heavy1 = make_waiter(s, "m", req_id="heavy1")
    heavy1.cost = heavy_cost
    s.enqueue(heavy1)
    assert admitted(heavy1)  # the resident heavy prompt holds 0.75
    assert s.used == pytest.approx(0.75)

    light = make_waiter(s, "m", req_id="light")  # base cost 0.25
    s.enqueue(light)
    # The fast lane survives: the light call admits on its first pump into the
    # deliberately-left room (0.75 + 0.25 == 1.0), alongside the resident heavy.
    assert admitted(light)
    assert light.wait_reason == "none"
    assert s.used == pytest.approx(1.0)

    heavy2 = make_waiter(s, "m", req_id="heavy2")
    heavy2.cost = heavy_cost
    s.enqueue(heavy2)
    # The second heavy does NOT fit: 1.0 + 0.75 > 1.0 — budget-throttled, NOT
    # cap-throttled (the cap of 4 still has 2 free slots). A single workload's
    # heavy prompts are serialized by the budget. As the sole budget-blocked pool
    # head it is eagerly reserved ("reserved" = the budget-blocked-head refinement
    # of "budget_full"), and crucially NOT "model_cap" (the inert-config symptom;
    # see the next test).
    assert not admitted(heavy2)
    assert heavy2.wait_reason == "reserved"
    assert heavy2.wait_reason != "model_cap"  # the contrast that matters

    # A single heavy can never take the whole pool, however large the prompt.
    assert s.weighted_cost("m", 99.0) == pytest.approx(0.75)


async def test_inert_config_starves_fast_lane_demonstrates_incident():
    """The incident reproducer (documents WHY the protective defaults matter).

    Flat tiers + an unbounded budget ⇒ the per-model cap is the ONLY binder ⇒ a
    handful of giant prefills pin every slot ⇒ the latency-sensitive fast lane
    starves. This is the inert config from #24 (OVERLAAT_BUDGET=9999 + flat
    tiers), encoded as an asserted-bad-outcome: with a huge budget the leave_room
    clamp never bites and weighting is flat, so we model it by charging every
    request its BASE cost (weight 1×) regardless of prompt size. The protective
    defaults (non-flat tiers + bounded budget + leave_room) prevent this — see
    test_fast_lane_survives_under_protective_config above.
    """
    s = Scheduler(budget=9999.0, caps={"m": 4})  # base cost 0.25
    # Flat-equivalent: a heavy prompt is charged the base cost, not weighted up.
    assert s.cost("m") == pytest.approx(0.25)

    heavies = [make_waiter(s, "m", req_id=f"heavy{i}") for i in range(4)]
    for w in heavies:
        w.cost = s.cost("m")  # flat: every request costs the base 0.25
        s.enqueue(w)
    assert all(admitted(w) for w in heavies)  # cap=4 full; budget 9999 never binds
    assert s.in_flight["m"] == 4

    light = make_waiter(s, "m", req_id="light")
    s.enqueue(light)
    # The collapse: the light call is cap-blocked behind 4 heavy prefills, even
    # though the (unbounded) budget has acres of room. The budget cannot protect it.
    assert not admitted(light)
    assert light.wait_reason == "model_cap"

    # On the inert config, the only relief is a heavy finishing (no budget lever).
    s.release("m")
    assert admitted(light)


async def test_release_refunds_exact_charged_cost_not_a_per_model_guess():
    # Regression (#24 review): release() must refund the ACTUAL cost the completing
    # run was charged, threaded by the caller — not a per-model guess (the original
    # fix used a LIFO stack, which mis-refunds under heavy+light interleaving). A
    # LIGHT run admitted BEFORE a HEAVY run, finishing FIRST, must refund only its
    # own 0.25 — leaving the heavy's 0.75 committed — or a second heavy would
    # wrongly co-admit (1.5 > budget), breaking the leave_room guarantee.
    s = Scheduler(budget=1.0, caps={"m": 4})  # base 0.25

    light = make_waiter(s, "m", req_id="light")  # cost 0.25
    s.enqueue(light)
    assert admitted(light)

    heavy = make_waiter(s, "m", req_id="heavy")
    heavy.cost = s.weighted_cost("m", 4.0)  # 0.75 (leave_room)
    s.enqueue(heavy)
    assert admitted(heavy)
    assert s.used == pytest.approx(1.0)  # 0.25 + 0.75

    # The light run finishes first. Refunding the wrong amount (e.g. the heavy's
    # 0.75 a LIFO pop would take, or a base 0.25-for-heavy elsewhere) would corrupt
    # `used`; the exact threaded refund leaves the heavy's 0.75 committed.
    s.release("m", cost=light.cost)
    assert s.used == pytest.approx(0.75)
    assert s.in_flight["m"] == 1  # the heavy is still running

    # A second heavy must NOT be admitted: 0.75 + 0.75 > 1.0. (With a wrong refund
    # `used` would read 0.25 here and this would wrongly admit — two heavies in the
    # pool at once, the exact thing leave_room forbids.)
    heavy2 = make_waiter(s, "m", req_id="heavy2")
    heavy2.cost = s.weighted_cost("m", 4.0)  # 0.75
    s.enqueue(heavy2)
    assert not admitted(heavy2)

    # The heavy finishes; refund its exact 0.75; now heavy2 (the reserved head) fits.
    s.release("m", cost=heavy.cost)
    assert admitted(heavy2)
    assert s.used == pytest.approx(0.75)

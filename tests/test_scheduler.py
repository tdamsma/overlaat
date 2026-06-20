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
    costs, pool_of, exclusive_seed = qp.load_model_info(path)
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
    # 'pool' is the newly-added column, last in the tuple.
    assert qp._EVENT_COLS[-1] == "pool"


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

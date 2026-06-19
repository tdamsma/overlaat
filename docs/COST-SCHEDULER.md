# Cost-weighted scheduler (PLANNED)

> **Status: NOT YET IMPLEMENTED — roadmap / design doc.**
> This describes the *next-generation* admission scheduler for Overlaat. None of
> it is in the shipping code today. Today Overlaat runs **independent per-model
> semaphores** (`overlaat/queue_proxy.py`): one `asyncio.Semaphore(cap)` per
> model, with `cap` read from `litellm-config.yaml::max_parallel_requests`. This
> document captures where we want to take that and *why*. It is a proposal to be
> reviewed, not a specification of current behavior. Experimental project, no
> support promise.

---

## 1. The problem with independent per-model caps

The current design treats each model as if it owned its own pool of compute. A
model with `cap=4` will admit 4 concurrent runs; a model with `cap=2` admits 2;
they do not know about each other. This is simple and it is the right *failure*
shape (a waiting FIFO queue instead of 429 reject-on-overflow), but it is
**dishonest about the hardware** in one specific way:

There is **one GPU**. Two models with independent caps can both be "within
budget" while collectively oversubscribing the single GPU's VRAM residency and
compute bandwidth. When that happens the symptom is *thrash*: weights get paged,
kernels contend, and aggregate throughput collapses below what either model
would have achieved alone. Independent caps cannot see this because no component
holds the global picture — each semaphore reasons only about its own model.

We want admission decisions to be made against **one shared budget** that
represents the single GPU, so the scheduler can refuse to oversubscribe even
when each model is individually under its own cap.

---

## 2. The model: one budget, cost-weighted admission

Replace the N independent per-model semaphores with **one global priority queue**
guarded by **cost-weighted admission** against a single shared budget `B = 1.0`.

`B = 1.0` is the whole GPU. Every model run consumes a **cost** equal to its
fraction of the GPU:

```
cost(model) = 1 / current_per_model_cap(model)
```

So a model whose backend cap is 4 has `cost = 0.25` (four of them fill the GPU);
a model whose cap is 2 has `cost = 0.5`; a model that wants the whole device to
itself has `cost = 1.0`. The cap is still read from the same place it is today
(`litellm-config.yaml::max_parallel_requests`) — see §6 on why cost is *derived*
from caps rather than tuned independently.

Admission loop, on every release and every new arrival:

```
used  = sum(cost(m) for each in-flight run m)      # currently committed budget
while queue not empty:
    pick the admittable request (see §3 packing policy)
    if none admittable:
        break
    used += cost(req.model)
    dispatch(req)                                   # acquire, hit the backend
# on completion of any run r:
#   used -= cost(r.model)   →  re-run the admission loop
```

A request is **admittable** only if BOTH constraints hold:

```
model_in_flight(req.model) < backend_cap(req.model)     # the hard per-model cap still binds
AND  used + cost(req.model) <= B                        # the shared budget has room
```

The first conjunct keeps the existing safety property: a backend that can only
service `cap` concurrent streams never gets a `cap+1`-th. The second conjunct is
the new global honesty: even if a model is under its own cap, it is refused when
the GPU as a whole is full.

### No preemption — by physics, not by choice

The scheduler **never preempts a running request.** Metal has no GPU preemption:
once a kernel is dispatched to the GPU it runs to completion, and a software
admission layer cannot reorder or evict GPU work. The only lever Overlaat holds
is **admission** — *which* request is allowed to *start*, and *when*. Everything
in this doc is about ordering the door, never about clawing back work already
inside. (This is the same reason the current proxy can cancel only *queued*
requests, never in-flight ones: releasing the slot on a client disconnect does
not stop a single-stream engine from decoding. See the disconnect/desync note in
the proxy docstring.)

---

## 3. The key policy decision: work-conserving packing + an anti-starvation guard

Given a budget with room and a priority-ordered queue, *which* request do we
admit? This is the crux, and it is a genuine tradeoff between two failure modes.

### 3a. Work-conserving packing (the throughput half)

We admit the **highest-priority request that FITS** the remaining budget — not
strictly the head of the queue. If the head is an expensive job (`cost = 1.0`)
and the budget only has `0.25` free, we do **not** idle that `0.25`: we scan down
and admit the highest-priority *cheap* job that fits, so leftover budget keeps
the GPU busy. This is "work-conserving" in the classic scheduling sense — never
leave a usable resource idle when there is runnable work.

The pure version of this policy has a well-known pathology.

### 3b. The starvation trap

If admission is *purely* "highest-priority that fits", a **steady drip of cheap,
high-priority jobs can starve an expensive job forever.** Picture a `cost = 1.0`
job sitting at the head. It needs the *entire* budget free at once. But cheap
`cost = 0.25` jobs keep arriving at equal-or-higher priority, and each time a
slot frees, one of them fits and is admitted — so the budget never fully drains,
and the big job waits indefinitely. Work-conservation, taken alone, is unfair to
expensive jobs.

### 3c. Reservation + aging (the fairness half)

To keep packing *and* bound the wait of an expensive job, add two guards:

1. **Reservation.** Once an expensive request reaches the **head** of the queue
   (it is the highest-priority waiter), the scheduler begins **reserving budget
   for it**: as in-flight cheap jobs complete and free their cost, that freed
   budget is *held* for the head job rather than handed to the next cheap
   arrival. Newly arriving cheap jobs are no longer admitted into the reserved
   portion. The reservation drains the GPU toward the `cost = 1.0` the head needs,
   then admits it. This caps the head job's wait at roughly the drain time of the
   currently-running cheap jobs, instead of "forever".

2. **Aging.** A request's **effective priority rises with its wait time.** A job
   that has been queued a long time climbs above fresher jobs of nominally equal
   priority, guaranteeing it eventually reaches the head (where reservation then
   protects it). Aging is what stops a *moderately* expensive job — one not big
   enough to always be the head — from being perpetually leapfrogged by a stream
   of fresh higher-priority cheap work.

The combination is the policy: **pack greedily for throughput, but reserve at the
head and age by wait so that no job — cheap or expensive — starves.** Reservation
bounds the expensive job's wait; work-conserving packing keeps the GPU busy with
cheap jobs *until* the reservation must bite; aging makes "reaches the head"
a guarantee rather than a hope.

> Reservation deliberately **sacrifices some throughput** (the reserved budget
> sits idle while the GPU drains for the head job) in exchange for a bounded
> worst-case wait. That tradeoff only triggers when an expensive job is actually
> starving; in the common case where the head fits immediately, packing is fully
> work-conserving and nothing is reserved.

---

## 4. Interactions with existing constraints

The cost scheduler does not replace the per-model backend caps or the swap-slot
mutex — it sits **on top of** them and must respect both.

### 4a. Backend hard caps still bind

As stated in §2, admission requires `model_in_flight < backend_cap` **as well as**
`used + cost <= B`. The cap is the backend's real concurrency ceiling (how many
streams its engine can actually service); the budget is the GPU's global ceiling.
A request must satisfy both. The cap protects an individual backend from
oversubscription even when the *global* budget has room (e.g. a small model that
is cheap per-run but whose engine still only handles `cap` streams).

### 4b. The exclusive swap-slot group = full budget

Overlaat already models a **swap-on-demand slot**: a set of large models behind a
mutex where only one is resident at a time, and loading another evicts the
current one (see `model_loads` in `schema.sql`). In the cost scheduler this is
modeled cleanly: **every member of the swap-slot group has `cost = 1.0`** (= full
budget). The arithmetic then enforces the mutex for free — admitting one
swap-slot model takes `used` to `B`, so no other run (swap-slot or otherwise) is
admittable until it completes. One scalar, one rule, and the "one big model at a
time" invariant falls out of the budget check rather than needing a separate
lock.

### 4c. Optional per-key priority (un-gameable)

Priority is not purely first-come. Each API key may carry a **`max_priority`** —
the ceiling of priority it is allowed to request. A request's **effective
priority** is:

```
effective_priority = min(requested_priority, key_max_priority)
```

This is **un-gameable by design**: a batch/background key is provisioned with a
*low* `max_priority`, so even if a batch client sets `priority: 9999` on every
request, its effective priority is clamped to the key's ceiling. Interactive keys
get a higher ceiling; batch keys cannot impersonate them. Priority becomes an
*operator-granted budget of urgency* attached to the key, not a free-text field
the client controls. (Keys are already fingerprinted on the call path as
`key_fp = sha256(bearer)[:8]`, matching `LiteLLM_VerificationToken.token[:8]`, so
the scheduler can resolve a key to its `max_priority` without handling the raw
secret.)

---

## 5. Honest caveat: a single scalar cost is an approximation

`cost = 1 / cap` collapses **two distinct, not-perfectly-correlated** resources
into one number:

- **VRAM residency** — how much of the GPU's memory the model's weights + KV
  cache occupy while loaded.
- **Compute bandwidth** — how much of the GPU's arithmetic throughput an active
  decode step consumes.

A model can be heavy on one and light on the other: a large-but-rarely-active
model hogs residency while contributing little compute pressure; a small model
under heavy concurrent decode is light on residency but saturates compute. A
single scalar `cost` cannot distinguish these — it assumes residency and compute
pressure move together, which is **only approximately true.** We accept this
approximation deliberately: a one-dimensional budget is tractable, explainable,
and good enough to kill the gross oversubscription that independent caps allow. A
two-dimensional budget (separate residency and compute ledgers, vector bin-pack)
is a possible future refinement, explicitly **out of scope** for this design. The
scalar is a *useful lie*, and naming it as such is part of the instrumentation
discipline.

---

## 6. Behavior change vs. today

| | Today (independent per-model semaphores) | Planned (shared budget, cost-weighted) |
|---|---|---|
| Resource model | N pools, one per model | 1 pool = the GPU (`B = 1.0`) |
| Admission test | `model_in_flight < cap` | `model_in_flight < cap` **AND** `used + cost <= B` |
| Cross-model awareness | none — models cannot oversubscribe each other on paper, but *can* in hardware | global — the GPU cannot be oversubscribed |
| Ordering | per-model FIFO | global priority queue (effective priority = `min(requested, key_max)`, aged by wait) |
| Swap-slot mutex | separate lock | falls out of `cost = 1.0` |
| Peak concurrency | **higher** (caps sum freely) | **lower** (capped at `B`) |
| Thrash under contention | possible | designed out |

The honest summary of the tradeoff:

> The shared budget is **more honest about the fact that there is one GPU, and it
> kills thrash** — the scheduler will refuse the admission that would page weights
> and collapse aggregate throughput. The price is **lower peak concurrency**: where
> independent caps would let two models run "4 + 2 = 6" concurrent streams on
> paper, the shared budget admits only as many as fit `B`. We are trading a higher
> *nominal* concurrency number — which under contention was a number the GPU could
> not actually honor — for a *real* concurrency the hardware can sustain. We
> believe that is the right trade for a single-GPU host; on a multi-GPU host the
> calculus changes and this design would need per-device budgets.

Costs are **derived, not tuned.** `cost = 1 / cap`, and `cap` comes from the same
backend config that drives admission today (`max_parallel_requests`, which in turn
reflects the backend's own concurrency settings). We do not introduce a new
free-floating tuning knob; the budget arithmetic is a *reinterpretation* of caps
the operator already sets, plus the single global `B`.

---

## 7. What stays exactly the same

This redesign touches **only the admission decision.** Everything that makes
Overlaat *Overlaat* is unchanged:

- **Waiting FIFO-spirit queue, not reject-on-overflow.** Requests that do not fit
  the budget *wait* (now in a global priority queue instead of per-model FIFO);
  they are never 429'd back to the client. The spillway shape is preserved.
- **One lifecycle event per request** to `request_events`, including queued and
  client-abandoned calls. The scheduler change does not alter the instrumentation
  contract — `t_enqueue` / `t_acquire` / `t_first_token` / `t_done` / `outcome`
  are emitted exactly as today, non-blocking via the background writer.
- **Cancel-while-queued only.** Queued requests never touched the GPU and stay
  safe to cancel (caller gets 499); in-flight requests remain non-cancellable for
  the no-preemption reason in §2.
- **Read-only dashboard / usage-API** over the same event + host-sample tables
  (`request_events`, `host_samples`, `model_loads`).
- **No silent cross-model failover.** A model works or fails cleanly; the
  scheduler decides *when* a request starts, never *which model* serves it.

---

## 8. Open questions (to resolve before implementation)

1. **Aging function.** Linear priority bump per second waited, or a step
   function past a threshold? Linear is simplest; a threshold avoids reshuffling
   the queue on every tick. Needs a measured queue to choose.
2. **Reservation trigger.** Reserve the moment an expensive job becomes head, or
   only once it has waited past some grace period? Eager reservation bounds wait
   tightest but costs the most throughput.
3. **Cost for models with no declared cap.** Today, no-cap models pass through
   without a queue. Under a shared budget a pass-through run still consumes the
   GPU. Options: assign a default cost, or keep no-cap models genuinely
   uncounted (and accept they can push `used` over `B`). Leaning toward a
   conservative default cost so the budget stays honest.
4. **Fairness across keys vs. across models.** Aging gives *time*-fairness;
   per-key priority gives *operator-intent* fairness. We have not specified
   *throughput*-fairness (a single key flooding the queue). May need a per-key
   in-flight cap as a separate guard.

---

## 9. Why this is worth doing

Independent per-model caps are a fine v1: they fix the reject-on-overflow problem
and give a waiting queue. But they let the operator *believe* the host can do
"sum of caps" concurrent work, when the single GPU cannot. The cost scheduler
makes the queue's promise match the hardware's reality: one budget, honestly
accounted, packed for throughput, guarded against starvation, with the swap-slot
mutex and per-key priority falling out of the same arithmetic. It is the same
philosophy as the rest of Overlaat — **instrument the call path once, derive the
rest, and never lie about what the machine is actually doing.**

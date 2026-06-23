# Cost-weighted scheduler

> **Status: IMPLEMENTED (since 0.0.3), ON BY DEFAULT.**
> This describes Overlaat's admission scheduler. It is the shipping behavior: the
> queue-proxy (`overlaat/queue_proxy.py`) wires every admitted request through the
> global cost-weighted scheduler in `overlaat/scheduler.py`. The scheduler is **on
> by default**; setting `OVERLAAT_SCHEDULER=off` is a kill-switch that restores the
> previous **independent per-model semaphores** (one `asyncio.Semaphore(cap)` per
> model, `cap` from `litellm-config.yaml::max_parallel_requests`) byte-for-byte.
> This document is the authority for the algorithm and *why* it is shaped this way.
> Experimental project, no support promise. §8 records the design decisions that
> were open during the proposal and are now baked into the code.

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

### Resource pools — the unit of admission (closes #11)

A single global budget assumes the whole host is **one** indivisible resource.
That is the right model for one GPU shared by everything, but it is too coarse
when the host actually has **physically independent** resources — e.g. a swap
engine that loads one big model at a time, plus a *separate* embeddings server on
its own device or process. With one global budget, filling the swap engine
(`used → B`) would wrongly block the embeddings model that shares none of its
hardware. That is issue #11: cross-backend interference that does not exist in
reality.

The fix is to make the unit of admission a named **resource pool** rather than the
whole host. Each model is assigned to exactly one pool via `model_info.overlaat_pool`
(default `default`); each pool `P` has its own budget `B_P` and its own committed
`used[P]`. The admission test becomes per-pool:

```
model_in_flight(req.model) < backend_cap(req.model)
AND  used[pool(req.model)] + cost(req.model) <= B_pool(req.model)
```

`cost(model) = 1 / cap` is now the model's fraction **of its pool**, not of the
whole host. A pool blocked on its own budget therefore **never idles a different
pool** — the work-conserving packing scans the global ordered queue and admits the
highest-priority waiter that fits *its own* pool's budget, so a busy swap engine
can no longer stall embeddings. The global priority queue, eager head reservation,
linear aging, and no-preemption (§3) are all preserved; they now operate **per
pool within one globally-ordered waiters list**. Each pool's head, if blocked *by
its pool's budget*, gets its own reservation of *that* pool's budget — reservations
are intra-pool and never bleed across pools.

Pools are declared in an OPTIONAL top-level `overlaat.pools` section of the
LiteLLM config (the proxy already parses that file):

```yaml
overlaat:
  pools:
    default:            # implicit; declare only to override its budget
      budget: 1.0       # default = OVERLAAT_BUDGET
    fat-slot:
      budget: 1.0
      exclusive: true   # see §4b
    embeddings:
      budget: 1.0       # its own budget → isolated from the swap engine (#11)
```

Rules: a model with no `overlaat_pool` is in `default`. A pool a model references
but does not declare is **auto-created** (non-exclusive, budget = `OVERLAAT_BUDGET`)
and logged at startup. `overlaat_cost` still overrides `1/cap` within a pool.
`OVERLAAT_BUDGET` is the budget of `default` and of any auto-created pool — there
is **no new env knob**. **The default is unchanged:** with every model in the
single `default` pool, the per-pool arithmetic is identical to the previous single
shared budget, so existing single-pool deployments behave exactly as before.

A separate per-model `model_info` knob, `overlaat_abort_on_disconnect` (bool, default
`true`), governs the **client-disconnect** path rather than admission cost. `true`
releases the slot the instant the client disconnects — correct for abort-honouring
engines that stop decoding when their upstream connection closes. Set it `false` for a
single-stream engine with **no abort path**: the proxy then holds the slot and keeps
draining the upstream to its natural end (bounded by the read-timeout) before releasing,
so slot accounting tracks the still-busy backend and the next call queues instead of
stalling on it. #28

Another per-model `model_info` knob, `overlaat_max_prompt_tokens` (positive int, default
unset = no ceiling), is a pre-admission size gate rather than a cost: a request whose
estimated prompt exceeds the ceiling is rejected with **413** (`outcome=rejected_oversized`)
*before* any slot or budget is taken — because the cost-clamp guarantees even a giant prompt
is otherwise admitted whole, this is the only thing that stops one oversized job from
collapsing the runtime. Set the same ceiling on every alias of a runtime (it is keyed by
model-name, not engine). #30

Another per-model `model_info` knob, `overlaat_breaker: { fails, cooldown_s }` (both
positive; default unset = off), is a **health gate** rather than a cost. Admission is
otherwise open-loop, so a wedged backend keeps being fed; the breaker watches the terminal
outcome the proxy already records and, after `fails` consecutive `upstream_error` outcomes
(upstream 5xx, connection errors, and read-timeouts all surface as `upstream_error`), trips
**open** and **fast-fails** new requests with **503** + a `Retry-After` header
(`outcome=rejected_unhealthy`) for `cooldown_s` seconds — it does *not* hold them in the
queue, which would only pile callers onto the wedge. After the cooldown a single **half-open**
probe is admitted; its success closes the breaker, its failure re-opens it with a fresh
cooldown. Zero-token completions are deliberately not treated as failures (too noisy). #31

### Prompt-size-weighted cost (closes #18)

A flat `cost = 1/cap` charges a 50-token prompt and a 33k-token prompt the same,
yet the huge prompt holds its slot far longer (and far more KV memory). Under FIFO
a handful of heavy prompts then fill every slot for ~a minute and the small
interactive calls behind them eat the queue wait. So the **cost is scaled by the
prompt size**:

```
cost(req) = clamp_to_pool( cost(model) × weight(estimated_prompt_tokens) )
```

- **Estimate** is cheap and deliberately coarse — the request body's message /
  prompt text length over ~4 chars-per-token (no tokenizer in the hot path). A body
  with no measurable prompt (`/embeddings`, `/rerank`) gets `weight = 1×`.
- **Weight** is a tier table, default `≤2k → 1×, 2k–8k → 2×, >8k → 4×`, overridable
  with `OVERLAAT_PROMPT_WEIGHT_TIERS` (e.g. `2000:1,8000:2,inf:4`). Set every
  multiplier to `1` to disable weighting — the exact pre-#18 behaviour.
- **Clamp** is the load-bearing safety rail: a weighted cost above the pool budget
  would make the request *un-admittable*, and as the eager-reserved head it would
  then starve its whole pool forever. So the cost is hard-clamped per pool via
  `heavy_max`:
  - `leave_room` (**default**) caps at `B_pool − cost(model)`, so at least one more
    base-cost run of that model always fits alongside even the heaviest prompt —
    the interactive fast lane never fully closes.
  - `full_pool` caps at `B_pool`, letting a giant prompt take the entire pool and
    run strictly alone (the *batch-job* intent). Declared per pool:
    ```yaml
    overlaat:
      pools:
        default: { heavy_max: leave_room }   # default if omitted
        batch:   { heavy_max: full_pool }
    ```
  The clamp never drops a cost below the model's base `cost(model)` (a cap-1 model,
  whose base already is the whole budget, simply stays at `1.0`).

The charged cost is logged as before in `request_events.cost`, so the weighting is
auditable against the `prompt_tokens` the backend later reports. **The default is
unchanged for small prompts** (≤2k tokens stay at `1×`); only heavy prompts move.

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

## Self-protection invariant (the oversized-prompt vector) (#24)

**The guarantee.** A single workload — even one API key — sending oversized
prompts at the model's concurrency cap **cannot collapse the shared backend or
starve the latency-sensitive fast lane.** A burst of giant prefills can occupy
*some* of the pool, but never *all* of it: there is always room for one
base-cost interactive call to keep flowing.

**How it is enforced — three layers in the Overlaat layer, all shipped enabled
by default:**

1. **Non-flat prompt-weight tiers price heavy prompts up.** The default tier table
   (`2000:1, 8000:2, inf:4`) makes a >8k-token prompt cost `4×` its base cost, so a
   heavy request consumes more of the pool and fewer of them run at once
   (`OVERLAAT_PROMPT_WEIGHT_TIERS`, see the #18 section above).
2. **A bounded pool budget makes the higher cost bind.** `OVERLAAT_BUDGET=1.0`
   (per pool) means the summed admission cost cannot exceed one GPU's worth, so a
   higher per-request cost directly translates to fewer concurrent admits — the
   budget is the lever that turns "costs more" into "runs fewer".
3. **`leave_room` heavy_max caps a single heavy prompt below the whole pool.** The
   default per-pool `heavy_max: leave_room` clamps a weighted cost at
   `B_pool − base_cost`, so even the heaviest prompt leaves room for at least one
   more base-cost run of that model. One fast-lane call *always* fits.

Layers 1 and 2 are co-dependent: weighting without a binding budget cannot
throttle anything (a higher cost against an unbounded budget still always admits),
and a binding budget without weighting charges a 50-token and a 33k-token prompt
the same. **Both** are required for self-protection, and both ship on by default —
so a fresh deploy is protected without setting a single env var.

**The startup warning (inert detection).** Because Layers 1 and 2 are
co-dependent, the protection is inert if **either** is missing — the condition is
`flat OR unbounded`, not `flat AND unbounded`:

- **Flat tiers** (every multiplier `1.0`): an oversized prompt costs the same as a
  tiny one, so no fast-lane slot is reserved in *any* pool, whatever the budget.
- **An effectively-unbounded pool budget**: so large, given the per-model caps, that
  the admit test can never fail even at the heaviest weighting — only the raw
  per-model cap binds. This is the *live* incident config (`OVERLAAT_BUDGET=9999`
  with the **default non-flat tiers**): the non-flat default does **not** rescue it,
  which is exactly why the check runs regardless of tier shape.

In either state a handful of oversized prefills pin every slot and the fast lane
starves. The proxy detects this at startup and writes a loud `stderr` warning naming
the failing condition(s) and the inert pool(s), with the remediation: set
`OVERLAAT_PROMPT_WEIGHT_TIERS` to a non-flat table (e.g. `2000:1,8000:2,inf:4`)
and/or lower `OVERLAAT_BUDGET` so the pool budget binds. (A pool budget is judged
"unbounded" only when every member has a finite cap and `sum(cap × cost × max_mult)
≤ B`; an uncapped member keeps the budget bindable, so that pool is flagged only
when the tiers are flat.)

**The ENGINE CONTRACT (#1 — operator deployment, NOT Overlaat code).** Token-level
prefill cost and true total wall-clock are only visible *inside* the inference
engine. The invariant therefore assumes the operator's engine run-script — which
Overlaat does not control and **cannot enforce** — satisfies:

- **Chunked prefill ON**, so a big prefill is interleaved with decode steps and
  does not starve the in-flight short requests at the engine level.
- **An authoritative total per-request deadline via the engine `--timeout`**, sized
  to the workload (e.g. ~240s), so a stalled or runaway generation is evicted
  GPU-side — this is the *real* total deadline, not anything Overlaat owns.
- **A sane `--max-tokens`**, so one request cannot decode unboundedly.

Overlaat cannot enforce any of these — they live in the engine. The proxy's
read-timeout (`OVERLAAT_UPSTREAM_READ_TIMEOUT`, default 300s) is the **inter-byte**
timeout (a slow token trickle never trips it), so it is only a *wedged-connection
backstop*, not a total deadline. It should sit just **above** the engine deadline
so the engine's own clean cancel wins and the proxy only cuts a connection the
engine has left fully wedged. See §10 of `docs/ARCHITECTURE.md` for the timeout
knobs and the deadline-ownership rule.

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

The cost scheduler does not replace the per-model backend caps or the exclusive
swap-slot mutex — it sits **on top of** them and must respect both.

### 4a. Backend hard caps still bind

As stated in §2, admission requires `model_in_flight < backend_cap` **as well as**
`used + cost <= B`. The cap is the backend's real concurrency ceiling (how many
streams its engine can actually service); the budget is the GPU's global ceiling.
A request must satisfy both. The cap protects an individual backend from
oversubscription even when the *global* budget has room (e.g. a small model that
is cheap per-run but whose engine still only handles `cap` streams).

### 4b. The exclusive swap-slot pool — honest 1/cap, not a forced 1.0 (closes #12)

Overlaat already models a **swap-on-demand slot**: a set of large models behind a
mutex where only one is resident at a time, and loading another evicts the
current one (see `model_loads` in `schema.sql`). The first cut modeled this by
**forcing every member to `cost = 1.0`** so admitting one took `used` to `B` and
the mutex fell out of the budget arithmetic. That was neat but **dishonest about a
member that can serve more than one stream**: a dual-engine member with `cap = 2`
would still be charged `1.0` and pinned to a single concurrent stream, even though
it can genuinely run two. That is issue #12.

The honest model makes the swap-slot an **exclusive pool**, and makes exclusivity
a **separate hard constraint** rather than a cost hack. Mark the pool
`exclusive: true` in `overlaat.pools`; assign every member to it with
`overlaat_pool`. Costs stay honest — `cost = 1/cap` like any other pool member.
The exclusivity rule is:

> At most **one distinct member id** may be active (`in_flight > 0`) in an
> exclusive pool at a time. While a member is resident, that member's own `cap`
> plus the pool budget govern how many of **its** streams run concurrently; a
> *different* member is hard-blocked (`wait_reason = exclusive`) until the
> resident fully drains, then the mutex hands off.

So a `cap = 2` member in an exclusive pool honestly costs `0.5` and, while it is
resident, runs **both** its streams (two of its requests pack into the pool budget,
`used → 1.0`). A `cap = 1` member costs `1.0` and runs one stream — byte-identical
to the old forced-1.0 behavior, so a single-stream fat slot is unchanged.

**The `0.5 + 0.5` trap this rule prevents.** Without the distinct-member mutex, two
different cap-2 members (`model-b`, `model-c`, each cost `0.5`) would *both* fit a
budget-1.0 pool — `0.5 + 0.5 = 1.0` — and the scheduler would admit both, putting
**two different big models resident at once**, exactly the swap thrash the slot
exists to prevent. With exclusivity as a separate constraint, once `model-b` is
resident at `used 0.5` there is `0.5` of *budget* headroom, but a `model-c` run is
still **rejected** (`wait_reason = exclusive`) because it is a different member.
Only a *second `model-b`* stream may take that headroom. When both `model-b`
streams drain, the pool goes idle and the next pump lets `model-c` become the
resident member. The budget arithmetic alone cannot express this ("there is room,
but not for *you*"), which is why exclusivity is a hard mutex layered on top of —
not folded into — the cost.

The legacy `model_info.overlaat_slot: NAME` key is **deprecated but bridged**: a
model with `overlaat_slot: NAME` (and no `overlaat_pool`) is treated as
`overlaat_pool: NAME` with that pool auto-marked `exclusive: true`. Old swap-slot
configs keep working unchanged.

### 4c. Per-key priority ceiling (un-gameable)

Priority is not purely first-come. Each API key may carry a **priority ceiling** —
the highest priority it is allowed to request. A request's **effective priority**
adds aging (§3c) to the ceiling-clamped request priority:

```
effective_priority = min(requested_priority, key_ceiling) + aging_rate * wait_s
```

This is **un-gameable by design**: a batch/background key is provisioned with a
*low* ceiling, so even if a batch client sets `priority: 9999` on every request,
its effective priority is clamped to the key's ceiling. Interactive keys get a
higher ceiling; batch keys cannot impersonate them. Priority becomes an
*operator-granted budget of urgency* attached to the key, not a free-text field
the client controls.

The ceiling is read from **LiteLLM key metadata**, not an env map: the
`LiteLLM_VerificationToken` table's `metadata` JSON, field **`overlaat_priority`**
(an integer). Keys are already fingerprinted on the call path as
`key_fp = sha256(bearer)[:8]`, matching `LiteLLM_VerificationToken.token[:8]`, so
the scheduler resolves a key to its ceiling without handling the raw secret.

**Hot-path safe:** the table is *never* read per request. A background task
refreshes a `key_fp -> ceiling` cache every ~60 s (mirroring how the usage-API
refreshes its key aliases). Resolution degrades gracefully: a key with no
`overlaat_priority`, or an unreachable table (e.g. a SQLite single-box deployment
that has no LiteLLM key table at all), falls back to `OVERLAAT_DEFAULT_PRIORITY`.
The request-supplied `priority` field (integer; absent → `OVERLAAT_DEFAULT_PRIORITY`)
is the *requested* side of the clamp.

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

| | Kill-switch (`OVERLAAT_SCHEDULER=off`, independent per-model semaphores) | Default (shared budget, cost-weighted) |
|---|---|---|
| Resource model | N pools, one per model | named **resource pools** (default: one `default` pool = the GPU, `B = 1.0`) |
| Admission test | `model_in_flight < cap` | `model_in_flight < cap` **AND** `used[pool] + cost <= B_pool` |
| Cross-model awareness | none — models cannot oversubscribe each other on paper, but *can* in hardware | global within a pool — a pool cannot be oversubscribed; separate pools are isolated (#11) |
| Ordering | per-model FIFO | global priority queue (effective priority = `min(requested, key_ceiling)`, aged by wait) |
| Swap-slot mutex | separate lock | an **exclusive pool**: honest `cost = 1/cap` + a one-distinct-member hard mutex (#12) |
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

## 8. Decisions (resolved for the 0.0.3 implementation)

These were the open questions during the proposal; the shipping defaults are:

1. **Aging function — linear, off by default.** `effective_priority` gains
   `OVERLAAT_AGING_RATE` priority units per second waited (recomputed on every
   pump from `enqueued_at`). The default `OVERLAAT_AGING_RATE=0.0` means aging is
   **off** — a single model with equal priority then reduces to pure FIFO, exactly
   matching the old semaphore. Set it > 0 to lift long waiters. (We picked linear
   over a step function because the queue is re-sorted on every pump anyway, so
   continuous aging costs nothing extra and avoids a threshold knob.)
2. **Reservation trigger — eager.** The head is reserved for the moment it is
   blocked by budget (`OVERLAAT_RESERVATION_GRACE` defaults to `0.0`). Eager
   reservation bounds the head's wait tightest at the cost of some throughput
   while the budget drains; that cost is paid only when an expensive job is
   actually starving. A non-zero grace is parsed (the env surface is stable) but
   reserved for a future refinement — the core reserves eagerly today.
3. **Cost for models with no declared cap — a conservative default.** No-cap
   models are **not** uncounted under the scheduler: each is charged
   `OVERLAAT_DEFAULT_COST` (default **1.0**, i.e. a no-cap run takes the whole
   budget) so a pass-through run cannot silently push `used` over `B`. The budget
   stays honest about the single GPU. (Override per model with
   `model_info.overlaat_cost`.)
4. **Fairness across keys vs. across models.** Aging gives *time*-fairness;
   the per-key ceiling gives *operator-intent* fairness. *Throughput*-fairness (a
   single key flooding the queue with many equal-priority requests) is still
   **not** specified — a per-key in-flight cap remains a possible future guard,
   out of scope for 0.0.3.

Other baked-in defaults: the scheduler is **on** (`OVERLAAT_SCHEDULER=on`),
`OVERLAAT_BUDGET=1.0` (the whole GPU), `OVERLAAT_DEFAULT_PRIORITY=0`, and the
budget admit test uses an epsilon (`used + cost <= B + EPS`, `EPS = 1e-9`) so an
exact packing like `3 * (1/3)` does not phantom-overflow on float drift.

---

## 9. Why this is worth doing

Independent per-model caps were a fine v1: they fix the reject-on-overflow problem
and give a waiting queue (and they remain a `OVERLAAT_SCHEDULER=off` kill-switch
away). But they let the operator *believe* the host can do "sum of caps" concurrent
work, when the single GPU cannot. The cost scheduler — now the default — makes the
queue's promise match the hardware's reality: one budget, honestly
accounted, packed for throughput, guarded against starvation, with the swap-slot
mutex and per-key priority falling out of the same arithmetic. It is the same
philosophy as the rest of Overlaat — **instrument the call path once, derive the
rest, and never lie about what the machine is actually doing.**

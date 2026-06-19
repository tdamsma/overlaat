# Overlaat docs

Design docs for Overlaat. Start with the top-level [`../README.md`](../README.md) for
what Overlaat is and how to run it; these go deeper on the internals.

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — the call-path and instrumentation design: how
  the queue-proxy sits in front of LiteLLM, the admission decision (cost-weighted
  scheduler by default, per-model FIFO semaphores as a kill-switch), and where the single
  lifecycle event per request is emitted.
- [`OBSERVABILITY.md`](OBSERVABILITY.md) — the `request_events` schema (including the
  `priority` / `cost` / `wait_reason` scheduler fields and budget utilization) and the
  three derived concurrency curves (**offered**, **active**, **queued**), plus the
  caveats about what the numbers do and don't mean.
- [`COST-SCHEDULER.md`](COST-SCHEDULER.md) — **implemented since 0.0.3, on by default.**
  The capacity-aware, cost-weighted admission scheduler: one global priority queue and a
  single shared GPU budget, with work-conserving packing, reservation + aging against
  starvation, and the swap-slot mutex / per-key priority falling out of the same
  arithmetic. `OVERLAAT_SCHEDULER=off` restores the independent per-model semaphores.

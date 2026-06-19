# Overlaat docs

Design docs for Overlaat. Start with the top-level [`../README.md`](../README.md) for
what Overlaat is and how to run it; these go deeper on the internals.

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — the call-path and instrumentation design: how
  the queue-proxy sits in front of LiteLLM, the per-model FIFO semaphores, and where the
  single lifecycle event per request is emitted.
- [`OBSERVABILITY.md`](OBSERVABILITY.md) — the `request_events` schema and the three
  derived concurrency curves (**offered**, **active**, **queued**), plus the caveats
  about what the numbers do and don't mean.
- [`COST-SCHEDULER.md`](COST-SCHEDULER.md) — **design-only roadmap, NOT implemented.** A
  proposal for a capacity-aware, cost-weighted admission scheduler. Today's code runs
  independent per-model semaphores; this describes where we might take that.

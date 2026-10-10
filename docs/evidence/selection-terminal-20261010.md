# Terminal selection review,2026-10-10

The two final follow-ups are terminal. Neither adds an accepted measurement.
The queue finished2026-10-09T16:38:25.565851Z (**2026-10-10 00:38:25 Beijing**).
Its exit0 means that the finite two-attempt list ended;both child audits are
`audited-failed`,with zero accepted cases. Manager and both child cgroups are
empty. No job remains queued or running from this increment.
[Verified terminal records](selection-terminal-20261010.json).

| Follow-up | Outcome | Effect on selection |
| --- | --- | --- |
| 21 PDG LibTorch CPU Attention prefill complete training | Ran,but reached24400s execution bound;no complete consumer result | CPU training baseline remains unavailable |
| 56 Settle LibTorch resident Attention prefill inference | Could not acquire11 free NPUs within120s admission policy;last poll136.013s;model never started | Missing Settle native timing remains unavailable |

CPU21's monitor wall time was24410.407s including termination/cleanup. Peak group
RSS393.451GiB stayed below the420.110GiB CPU lane allowance;the reported failure
was the execution time bound,not memory admission or an observed RSS breach.
This run changed the CPU memory mask toNUMA0..7 in a separate series while
retaining the original CPU affinity/thread counts. No completed warmup/measurement
result exists. The six-hour elapsed duration cannot be substituted for a measured
step or used to derive a precise CPU/NPU throughput ratio. External contention
and any internal phase bottleneck remain unattributed.

The NPU attempt never reached the consumer. It is a resource-availability failure,
not an NPU computation failure. The full-size monitor-fix retry therefore did not
run;the27 focused immutable regression checks remain its actual qualification.
Original failed attempts and budgets are preserved. No larger timeout or retry
has been scheduled;cells16/59 were deliberately omitted.

The [October9 selection report](selection-review-20261009.md) remains the timing
basis:61 accepted bound FP32 cells,8 separate historical unbound observations,
5 separately revalidated complete-over-bound observations and4 FP16 companions.
Its submission-time pending statements are superseded by this terminal review;
its raw scalar records and original outcome classifications are unchanged.

The practical guidance is unchanged:LibTorch Add generally favors CPU in measured
prefill cases;Attention prefill inference favors resident on the qualified
PDG/TimedDAG native and TimedDAG/Settle Python-owned paths. Python Add inference
has a different CPU/native boundary and can favor resident. Mixed-A is ahead of
resident for the retained PDG Attention training observation,while a complete
full-size CPU training comparison remains unresolved. Preserve all schedules and
presets for consumers and future machines;do not declare a universal winner.

Under the user's decision-sufficient criterion,local F6/F7 review is closed with
these explicit limits. Implementations and local CPU/NPU correctness evidence
remain qualified only in their declared profiles;the strict near-tie numerical
failure stays nonblocking and is not relaxed. Actual NVIDIA/x86_64/other-version
execution remains target-pending;downstream convergence and updated selector/
canon semantics are separate work. No more automatic experiment work is pending.
Protected historical stopped CPU work remains untouched.

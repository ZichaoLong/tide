# Comparisons and cost attribution

Choose a comparison that answers the current question. The
[benchmark contract](../../../../docs/benchmark-contract.md) owns the four joint
Transformer matches, parameter accounting and timing boundaries. Current
commands and public return shapes are in [instances](../../../../docs/instances.md).

| Comparison | What it can establish |
| --- | --- |
| Independent formula/solver or directional derivative | Selected forward or VJP correctness within its mathematical and numerical domain |
| Independent chronological evaluation | Time, state and continuation behavior for a declared instance |
| Same-function compact tensor implementation | An attainable cost with explicitly matched state, returns, gradients and update duties |
| No-grad, grad-enabled forward and complete training | Costs associated with graph building, reverse computation and optimizer work |
| Jointly matched Transformer or SSD | A different-function compute reference; parameter/count proximity is not mathematical or quality equivalence |

Keep three kinds of quantity separate: stored parameters, declared active
parameters, and actual work. Preparation, reception and regional networks can
run before Full selection. Shared weights may be stored once and multiplied
many times. Count selected and padded rows, queries/keys, messages and repeated
updates; k/n is a mass budget, not an observed active-node fraction.

Pair inputs and initial parameters with hashes when comparing devices or code
versions. Match state and optimizer boundaries where relevant. Training routes
may subsequently diverge; record the resulting work difference. A change of
support or workload prevents a strict same-work speedup claim.

Measure ordinary wall time separately from intrusive profiling. Caller timers
include dispatch and waits; overlapping worker sums and nested operator times
are not additive wall-time shares. Autograd inspection can count unique nodes,
private graphs inside custom Functions, saved storage and reverse executions;
those are different metrics. Inspect only the layer needed to resolve the gap.

Give elapsed time and its denominator. Input-position throughput counts actual
input sample/position pairs, not node updates or port messages. Timed pending and
flush are explicit boundaries. First-output latency requires timing the first
available output; a backbone without a task head does not measure language-model
service TTFT. Grad-enabled forward is not a training step.

Use useful shapes and finite samples. Warm up appropriately, retain raw samples
and investigate noise when it affects a decision. Small cases support frequent
checks; larger ones are useful at integration boundaries or when scale can change
the conclusion. Existing valid evidence need not be repeated to restore context.

Report actual dtype, operator, device, thread/affinity configuration, memory and
known contention. FP32 and BF16 comparisons use matched precision policies;
quantify any FP32 accumulation or stabilization island. Shared measurements stay
identified as shared observations. [Execution rules](../../../../docs/execution-flows.md)
own resource admission and durable jobs, not this diagnostic method.

Map isolated costs to current symbols and report the increment actually measured.
A one-sided small profile cannot explain an entire large-model gap. Retain failed
attempts, numerical deviations and unmeasured cells with their original scope;
an outer process success does not certify every inner result.

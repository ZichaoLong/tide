# Optional batched Aggregate autograd

Native `Options.aggregate_autograd = "replay" | "batched"` selects an execution
policy, outside graph/model/checkpoint identity. Default remains `replay`.
The Python native adapter accepts `aggregate_autograd="batched"`; the native
scale executable and exported PDG runner accept `--aggregate-autograd batched`.
Packed execution and a declared Aggregate capability are required. Unsupported
custom programs fail before execution, including in inference contexts.

## Scope and mechanism

The built-in sum, mean, positive weighted mean, active-source softmax and
all-source softmax implement this capability. Events are grouped by their
ordered logical-source signature. A multi-output first-order Function batches
source scaling, coefficient products and source-order sums. It returns each
event's summary and each source contribution as separate public roots.

Backward collects defined summary/contribution cotangents for each source,
combines them only when both roots are present, and batches input/scale/
coefficient VJPs. An untouched input atom or physical scale receives undefined;
a defined numerical-zero cotangent remains connected zero. Shared owners and
physical source aliases accumulate through their existing parameter identities.
A coefficient tensor owned by the caller has the ordinary dense tensor VJP.
Input/scalar packing saves snapshots. The normal immutable-input/state/parameter
contract applies while a graph is live; arbitrary in-place mutation is unsupported.

Softmax preserves an event axis through its normalization backward, accumulating
shared owner gradients afterwards. Summing event cotangents before the Jacobian
is mathematically linear but changes float cancellation near zero, which AdamW
can amplify. Positive weighted mean retains per-event normalization graphs for
the same reason. No normalization gradient is replaced by an exact mathematical
zero merely because its exact-real derivative vanishes.

The same implementation is used by native streaming, legal frontier/block,
independently scheduled native topologies and encoded/native SettleGraph.
`packed_sources` still supplies detached numeric transport for state kernels;
public source atoms, contributions and summary preserve independent gradients.
This optimization does not change clocks, routing, source identity, absent versus
zero messages, Next, state behavior, the scheduler or its legal time batches.

In `no_grad` and `inference_mode`, both policies call the existing numeric batch
kernel. `batched_aggregate_events` counts events on the new grad path;
`semantic_aggregate_replays` and its worker timer count actual old replay only.
Full policy is independent: use `full_autograd="batched"` separately to avoid Full
replay. State and Read replay remain unchanged.

## Differentiation boundary

CPU FP64/FP32, first-order public-root VJPs, shared/unused owners, independent
inputs and initial states, retained/detached cuts and optimizer updates are the
scope. Higher-order requests through the primitive fail explicitly. Forward AD,
compiled autograd and arbitrary intermediate adjoints are unqualified. Custom
upstream autograd programs must propagate undefined cotangents, as required by
[batched Full](full-batched-autograd.md). Scalar replay stays available.

## Verification and performance

Primitive tests use independent scalar formulas, finite differences, frozen and
aliased inputs, separate and combined roots, and None/connected-zero checks.
Graph tests compare complete observables and source roots with the independent
Python interpreter; native SGD/AdamW trajectories, checkpoints, policy switches,
serial/workers, streaming/frontier and native Settle are covered. Equal-source
near-zero-gradient regressions compare actual AdamW owner/state updates at the
existing tolerances, rather than accepting only small raw VJP differences.

Performance uses Full batched in both control and candidate. A separate diagnostic
pass measures Aggregate/state/Read worker times and coordinator phases. Formal
comparisons disable instrumentation, preserve graphs/weights/windows, and use
independent process repetitions. Small actual backward/optimizer/train-step
measurements have a separate workload; they do not certify wide-model training
throughput or large Attention performance. See the reviewed evidence when ready.

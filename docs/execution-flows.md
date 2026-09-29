# Independent complete execution flows

This is the active F1-F7 delivery contract in [ROADMAP](ROADMAP.md). Individual
implementation and qualification results are recorded separately; this document
does not claim that the full delivery is already verified.

Each flow independently consumes the same inputs, initial weights and declared
state. CPU reference outputs, routes, queue traces and gradients never become
candidate inputs. The CPU oracle is a correctness client only. A mixed flow
includes its CPU work, transfers and synchronization in its measured execution.
Device replay caches a computation structure and stable storage, not answers.

## Workload identity

`prepare_execution_flow.py` prepares a dependency-free, hashed packet. It leaves
the old foundation-v1/v2 capacity presets unchanged. The new ranked-local graph
has every body node on an input-to-output path. Adjacent local wires, sparse
cross-neighborhood wires and skip-layer wires expose locality without replacing
physical edge identity. Node reachability and actual selected work are distinct.

```bash
python scripts/prepare_execution_flow.py --preset smoke --memory add --output-dir NEW
python scripts/prepare_execution_flow.py --preset wide --memory attention --output-dir NEW
```

The common wide graph has 15 regions of 32 nodes, 480 reachable body nodes,
four local next-layer wires per source and one skip-layer wire where applicable.
It includes the same declared embedding/head in every flow. Parameter counts are
derived from actual module owners; padding with disconnected trainable nodes is
forbidden. Width, batch, window, budget, locality and layer counts are configurable.
`many-nodes` independently stresses graph scheduling with narrower tensors.

A rank-aligned body permits SettleGraph and its explicitly encoded TimedDAG/PDG
representations. Boundary nodes, clocks, source tags, output summation and owner
identity must survive encoding. Timed-local deliberately perturbs arrival delays;
it is a separate TimedDAG/PDG workload and explicitly cannot be selected as Settle.
Comparable parameter counts or separate family stress tests do not imply equal
functions. Historical PDG remains a separate preserved workload.

## Execution and measurement

The complete-flow interface keeps family, implementation, schedule, device,
dtype, device count, Read/control/ranking placement and CPU threads explicit.
Unsupported requests fail before timing. A named recommendation resolves to a
recorded configuration for the current hardware/software/workload identity;
another machine retains access to every qualified alternative.

All comparable cells use the same reset/continuation and training boundary.
Complete training includes forward/loss, backward, finite checks and optimizer
updates. End-to-end iteration timing includes required input preparation,
transfers, data-dependent host work, cleanup and completion synchronization.
Construction, reusable compilation/capture and warmup are separately measured;
cold and finite-run amortized times include them. Rebuilding per request belongs
inside that request's timing. Instrumented profiling and oracle comparisons are
separate passes, never mixed into throughput.

Mandatory correctness gates compare complete observables, physical identities,
int64 times/counts, stable ties, absent versus zero messages, isolated VJPs and
None versus zero gradients, then multiple optimizer updates. FP32/FP64 reference
contracts remain unchanged; FP16 has explicit floating tolerances and retains
exact discrete tests. Changed inputs and successive optimizer updates must be
tested through the actual replay path.

Full-size benchmarks follow those gates. Report reachable, touched, candidate
and selected nodes/events, edge work, output completion, parameter ownership,
memory and communication as well as throughput. Three independent processes
support a machine-specific recommendation; a capacity probe or single run does
not. Frozen failures, source/build identities and local Trackio records remain
part of the delivery. CUDA target execution stays target-pending on this server.

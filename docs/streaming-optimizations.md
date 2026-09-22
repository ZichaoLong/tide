# Optional streaming optimizations

These are schedules of the adopted `tide-core-3` semantics in
[semantics.md](semantics.md), not new local programs or graph versions.
All switches default off so the simple native schedule remains a comparison
anchor alongside the independent Python reference. They apply to Streaming and
its owned cursor; other native algorithms reject these switches.

`Options.parallel_regions` evaluates independent `(sample, region)` requests
using the existing bounded node pool. Each request reads the complete sorted
candidate set and that owner's prior history. Workers do not mutate the history
map. Workers validate their results. After all tasks join, the coordinator
publishes them in the original owner order and establishes comparison state. A thread's completion
order never resolves a selection tie. Node updates precede selection; selected
Full computations follow selection, with the original Next/clear behavior.
Region programs must be functional and safe to call concurrently, including
when parameters/program instances are shared. Mutable history belongs to the
request owner, never to a hidden shared kernel cache.

`Options.compact_events` sorts incoming atoms by their canonical identity and
groups contiguous owners directly, avoiding the intermediate map of copied
fibers. Parallel edges and all atom metadata are retained. Trace-free execution
moves final state handles into continuation and releases each temporary event
on one worker after delivery joins. Live continuation, outputs and pending
messages retain their own Tensor references. This does not detach gradients or
change state/history ownership. Trace mode retains complete events as before.
The implementation still uses maps for stored state and event indexes.

Exceptions drain the worker pool before escaping. As with the existing cursor,
an execution exception invalidates that cursor; restore a prior complete snapshot
into a new cursor. No transactional partial-region recovery is promised.

`DenseLinear(workers)` partitions a CPU FP32/FP64 linear projection by output
columns, runs independent `at::linear` calls, and concatenates in column order.
It never partitions the reduction dimension, replicates parameters, or changes
global BLAS settings. Ordinary slice/cat autograd accumulates shared weight and
input VJPs. All worker jobs inherit the caller's Torch thread-local state.
This helper is separate from graph semantics and is used for the vocabulary head.

The scale executable exposes `--head-workers`, `--parallel-regions`, and
`--compact-events`. Graph and head pools execute in separate stages; each has
its own persistent idle threads. The limit is 160 active worker × intra-op
threads per stage, not 160 total OS threads. Callers must also bound the actual
BLAS pool to avoid nested oversubscription. The executable reports ATen/inter-op,
OpenBLAS/MKL and OpenMP counts when available (`-1` means unavailable); these are
runtime-reported settings, not measured simultaneous CPU use. In particular,
an OpenMP OpenBLAS build can ignore `OPENBLAS_NUM_THREADS` in favor of OpenMP.

Validation: `test_stream_optimizations.py` compares complete traces, continuation,
isolated VJPs and structural absent gradients across independent Python,
simple native and optimized native schedules, including trace-free cleanup.
`test_dense_projection.py` checks dense forward/VJP, shared/strided inputs,
uneven columns and thread-local grad/inference modes. Native custom region
checks cover vector controls, empty selections, history and malformed results.
Performance acceptance requires fixed-window, fixed-work measurements; enabling
more workers is not itself evidence of a speedup.

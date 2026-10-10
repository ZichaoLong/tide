# Device-resident queue and backend control

`tools/device_online` contains separately tested building blocks and an
experimental [content-driven forward loop](content-flow.md) for an explicit
existing-module profile. The public [resident training owner](resident-training.md)
has its own graph, adjoint and optimizer gates; passing a component does not
certify that complete path. This backend isolates CANN runtime
models, raw ACLNN numerical stages and optional Ascend C kernels from the portable
core. The independent CPU scheduler and scalar qualification remain unchanged.

## CUDA source profile and target gate

The independent CUDA implementation shares the semantic kernel inventory, graph
representation, layouts and VJPs with CANN. Its static control blocks become a
CUDA12.8 conditional WHILE/SWITCH graph; dynamic branch indices are read by
CUDA kernels. It does not import the CPU oracle's event trajectory. Supported
source scope is the same declared resident module/profile set, including
positive-delay PDG feedback, TimedDAG, Settle, streaming/online greedy prefill,
FP32/FP16, inference and complete first-order training, continuation, checkpoint,
capacity handling, multi-owner/locality placement, SGD/AdamW and accumulation.
GPU execution is **target-pending**; local compilation/CPU contracts certify no
CUDA correctness, residency, multi-card behavior, memory calibration or speed.

Requirements: CUDA toolkit and driver support for12.8 conditional SWITCH graphs,
CC8.0+, matching CUDA LibTorch/Torch build and explicit architectures. Peer
owners require bidirectional P2P and native system-scope atomics; unsupported
requests fail. The initial shared-kernel adapter uses one execution thread per
logical worker and32KiB bounded scratch. cuBLAS owns an explicit4MiB workspace,
FP32 pedantic accumulation and disabled TF32. Scalar reductions and duplicate
scatter scans remain optimization work, without a claimed speed advantage.

On a fixed clean checkout, after activating the target's own CUDA toolchain,
choose new output directories outside the source and an explicit architecture
list (`80-real;90-real;100` is the local compile profile):

```bash
python scripts/build.py --backend cuda --build-dir "$CORE" --jobs 4
python scripts/build_device_control.py --core-build "$CORE" --build-dir "$STANDALONE" --runtime standalone --cuda-architectures "$ARCHS" --jobs 4
python scripts/build_device_control.py --core-build "$CORE" --build-dir "$PYTHON_RESIDENT" --runtime python --cuda-architectures "$ARCHS" --jobs 4
python scripts/build_resident_consumer.py --core-build "$CORE" --resident-build "$STANDALONE" --output-dir "$INSTALLED"
python scripts/build_online_consumer.py --core-build "$CORE" --resident-build "$STANDALONE" --output-dir "$ONLINE" --jobs 4
python scripts/verify_resident_target.py --device cuda:0 --python-core-build "$CORE" --python-resident-build "$PYTHON_RESIDENT" --standalone-resident-build "$STANDALONE" --online-build "$ONLINE" --installed-consumer-build "$INSTALLED" --output-dir "$GATE"
```

The complete gate requires at least3 visible consecutive logical devices for
1/2/3-owner continuation/repartition. It fails rather than skipping unavailable
hardware or missing binaries, checks exact build/source digests, runs the entire
registered standalone component inventory and all public resident consumers,
and rejects any skipped public test. Single-card component invocations through
`verify_device_control.py` are useful development steps, with explicitly partial
scope. The full gate preserves strict discrete checks and default strict control
comparison; it cannot convert a numerical near-tie witness into equivalence.

Actual residency requires a separate target trace. For a finite initial trace,
use Nsight Systems node-level CUDA Graph tracing on the warmed control/peer
checks, retain the `.nsys-rep` and export its CUDA API/kernel timeline:

```bash
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --output="$TRACE/control" "$STANDALONE/tide-device-control-check" --device=cuda:0 --dtype=float32
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --output="$TRACE/peer" "$STANDALONE/tide-device-peer-check" --device=cuda:0 --dtype=float32
nsys stats --report cuda_api_sum,cuda_gpu_kern_sum "$TRACE/control.nsys-rep"
```

Review device branch kernels and loop execution between boundary graph launch
and completion, and peer packet ordering on both devices. Component traces
include construction and assertions; they establish neither whole-model
throughput nor absence of all boundary copies. Before GPU selection experiments,
profile a representative complete consumer, calibrate memory and declare a fresh
finite serial measurement budget. No GPU timing is accepted on this CPU/NPU host.

## Runtime control

`DeviceProgram` binds a persistent stream to an explicit runtime model. Device
int32 indices choose labels; raw ACLNN tasks can perform packed FP32/FP16
arithmetic inside bounded device loops. One global target list is created before task emission, as CANN requires.
Local branch indices map onto that list on-device, allowing repeated targets
without assigning a label multiple CANN addresses. The stream remains bound until execution completes.
Labels, descriptors, payload owners and workspaces live through every execution.
A timed-out wait is not completion: release first tries to drain, and refuses
to free anything if completion remains unconfirmed. Destruction then quarantines
the entire owner until process exit. An errored worker must not resume execution.
A registered `RuntimeResource` lease remains live through raw-resource teardown
and is retained permanently by quarantine. The last `RuntimeSession.close()`
refuses finalization while such resources exist. After orderly resource close it
can be retried; quarantine requires a failed worker exit. New session/device
resolution and program construction/execution refuse a quarantined runtime.
An unsuccessful program close also forbids another execution. Direct vendor
finalization by an embedding application remains that application's responsibility.

Standalone checks own a `portable_torch::RuntimeSession` in `main`, outside all
tensor/program owners. Normal finalization must precede main-thread TLS teardown;
a former static finalizer could segfault in TorchNPU's current-stream lookup.
The lifecycle check covers nested owners, idempotent close and rejected reopen.
The separate `verify_device_failures.py` gate runs finite real CANN programs with
deterministically injected API errors: partial stream creation, build completion,
asynchronous submission, boundary wait and unbind. It checks that accepted work
executes once, failed owners cannot rerun, drains precede resource destruction,
and a retried close releases retained tensors. A separate process withholds all
completion confirmations, verifies retained owners and rejected finalization,
then exits with the explicit failure code 86. This tests library error handling;
it does not certify recovery from a physically hung device, driver reset or
external vendor calls. Exact development/qualification results are in STATUS.

The kernel hook submits work once during model construction. Subsequent execution
is a device task; it is not a per-iteration host callback. `run()` submits once
and waits at the complete call boundary. The caller finishes input writes first.
Branch producers must supply valid target indices, and arithmetic must stay in
its declared domain. Construction and inference use fixed contiguous buffers;
this interface deliberately provides no implicit autograd.

The checks cover changed loop limits, zero work, exhausted iteration budgets,
continuation, clocks above2^55 and packed index/FP32/FP16 arithmetic. An iteration
budget is separate from queue capacity or model memory. CANN9 rejected nesting a
captured NPUGraph through RIExecuteAsync on a model-bound stream; the failed
bridge is retained in the development artifacts, not exposed as a supported API.

### Precision and directed builds

Floating conversion is explicit: `DeviceProgram::cast` accepts matching shapes
with FP32/FP16 input and output, and refuses integer coercion. Queue metadata,
timestamps and counters never pass through this conversion. Copies continue
to require identical dtypes. Conversion checks include half rounding boundaries,
subnormals, overflow and reuse with changed inputs.

`PackedFull` accepts matching FP32 or FP16 parameters, payloads and state
comparisons. Its selected-action planner, stable indices, empty-work branches,
zero sentinels and chunk loop are shared. Parameter and numerical buffers retain
the selected dtype; byte budgets account for element size. ACLNN matmul keeps
the declared dtype. The FP16 component checks compare both CPU storage-dtype
arithmetic and an independent FP64 formula with explicit lower-precision
tolerances. Complete-session coverage is provided separately by `precision-flow`.

`PackedLhFull` uses the same selected-action planner for all nine activation /
normalization profiles in both dtypes, with FP32 normalization statistics and
unchanged epsilons. Its shared component fixtures retain the original FP32
thresholds; FP16 has a separate conditioning budget against the FP64 expression.
The complete FP32 LH graph gate still runs independently of these component tests.

Packed sum accepts matching FP32/FP16 payloads and source scales. Multiplication
and stable ordered accumulation use FP32; per-source contributions and final
summaries independently round when stored in the selected payload dtype. Scalar
and vector paths share the same metadata preflight and source order. Vector
loads/stores convert whole tiles, including non-aligned tails, without host
per-message work. Complete resident inference and VJP qualification remain
separate from these building blocks.

`PackedSwiGluFull` and `PackedEmission` also retain the requested FP32/FP16 dtype
through parameter banks, selected-row matmuls, activations, slot projections,
physical scales and payload stores. Their numerical byte budgets use the actual
element size; int64 planners, phase tests and edge identities are shared.

The state/Read building blocks accept FP32/FP16 payloads. Identity, EMA and
Add-repeat use the same scalar/vector algorithms; half arithmetic rounds after
each product/add, including every repeated tick and event. A vector time batch
therefore cannot retain extra precision merely because an intermediate state
stays in a local tile. Full-width state and message buffers keep the payload
dtype. Linear/norm Read products and reductions, selection scores/controls and
diagnostic event fields use FP32. This is an explicit scoring policy, compared
against an independent CPU FP32 Read on the stored state/content values.
It does not promise equality with a half-accumulated selector near a tie.

`state-read` checks the canonical CPU StateKernel steps, exact clocks above
2^55, observe/adopt/clear, scalar/vector execution, node-time batches versus
continued single-time windows, empty/NaN padding and failed commits. It is an
isolated component gate, not an independently scheduling graph engine.
The `emission`/`swiglu` FP16 cells are component-only; their FP32 cells also
retain the complete graph-window regressions. `precision-flow` integrates these
modules with FP16 HARD resident inference, dense/tiled event and fiber attention,
scalar/vector stages, both schedules and candidate-owned checkpoint continuation.
Its `--profile-smoke` selects two attention fixtures for bounded placement traces,
not throughput. FP16 backward and FP32-master publication remain pending.

Normalized Aggregate widens the stored model's mass/logit parameters into FP32
normalization banks. Softplus/softmax, denominators, coefficients and ordered
accumulation stay in FP32. Physical source products first enter the payload-dtype
contribution buffer; normalization multiplies those stored values in FP32,
then rounds contribution and summary stores independently. `aggregate-payload`
compares canonical CPU FP32 Aggregate on those stored products and a separate
FP64 formula. Logical-slot contributions are matched to physical rows by identity;
the two orderings are not interchangeable. These banks still need explicit
low-precision optimizer publication before complete FP16 training can be enabled.

`build_device_control.py --checks numerical full` builds only those standalone
components and their dependencies; the manifest records the requested subset.
The PackedFull target does not rebuild unrelated attention/VJP kernels.
Multiple requested checks use one aggregate CMake target so GNU Make does not
repeat CANN ExternalProject work for each separate top-level target.
Omitting `--checks` still builds the full backend. A subset result never replaces
the complete registered gate; verify it with the same explicit `--checks` list.

## Actual messages and closure

`PackedQueue` stores int64 `(sample,node,time,kind,physical source,position)`,
separate payloads and explicit valid bits. Zero-valued messages remain present.
Bulk append/replace compacts rows stably; overflow preserves the old queue and
sets a sticky error. Erasure reclaims live capacity, independent of cumulative
work. Packing sorts exact integer fields lexicographically, retaining duplicate
physical edges and tie order. Mutable storage is an inference primitive, not a
replacement for differentiable continuations.

`QueueClosure` precomputes only topology: minimum **nonempty** region-path delays,
including positive feedback returns. Each query uses actual live queue times to
certify the same greedy prefix as the host algorithm. Saturation precedes int64
addition. Its CPU tensor reference has an explicit approximate scratch budget
and chunks samples while preserving whole logical fibers. That budget does not
cover model state, gradients, optimizer, KV or the complete execution flow.

CANN9/TorchNPU2.10 reports int64 argsort on device-local AiCPU. That is distinct
from host CPU. More seriously, its scatter_reduce falls back to the host: the
tensor closure therefore **explicitly rejects NPU execution**. Exact integer
keys are never converted to float to avoid this limitation.

The optional `ascendc/tide_closure.cpp` computes readiness and its branch flag in
one AIV kernel, using preallocated metadata workspace and an exact int64 scalar
pipeline. This initial implementation prioritizes correctness, not optimized
parallel throughput. Invalid live coordinates set a device error before topology
indexing. Full graph dispatch, payload generation, grouped numerical contracts,
model state, peer completion, semantic VJPs and optimizer integration are outside
this readiness kernel's scope. The implemented profiles and their separate
qualification are described by the content-flow and resident-training contracts.
A readiness kernel alone is not a complete resident flow. Placement/profiling
results require actual target traces.

## Packed stages and bounded progression

`QueueTransaction` composes an AIV metadata proposal with a device commit/refuse
branch. Successful transactions retain survivors and append actual arrivals in
stable order; one bulk gather places payload rows. Invalid slots gather a zero
sentinel, so inactive NaNs never leak into padding. Refusal preserves every old
queue slot and its live/peak counters. For multiple queues, share one sticky
error buffer and append **all proposals before any commits**. Each proposal
snapshots payloads before any queue is overwritten, including cross-queue
aliases. Each commit rechecks the shared error on device. A later capacity
refusal therefore preserves earlier queues too. Errors remain sticky. Capacity describes
simultaneous occupancy and can be reused over more cumulative arrivals.

`BroadcastRouter` applies the explicit broadcast Full delivery contract to
already selected/computed actions. Device CSR traversal generates physical edge
identities, target coordinates and gather indices; payload gathering and edge
scaling are bulk operations. The CSR table contains only static topology. Cycles,
unequal delays and physical parallel edges need no special case. Arrival-time
overflow is checked before addition and fails, including beyond the current
window. Sparse/per-slot Full contracts are a separate missing delivery path.

`DeviceReady` connects the closure task to stable exact-int64 atom packing,
fiber offsets and complete region-time frame offsets. Fiber traversal supports
node-state sequences; a separate frame permutation groups complete candidate
sets for selection. A static per-region module capability can restrict a ready
prefix to its earliest whole frame; the filter runs on device before payload
packing and does not split candidate sets. Unrestricted regions keep their
certified time prefixes. Device lengths and valid bits identify real work. Packing
currently uses scalar AIV insertion ordering; it is not optimized parallel sort.
The integration check uses one submission to consume successive ready batches
and preserve pending messages across windows, with an explicit iteration bound.
It emits no messages and does not execute Tide numerical modules. That check
therefore certifies scheduling progression only, not a complete graph flow.

`FrameSelector` implements the `count-v1` and `positive-v1` region contracts
from actual ready frames and FP32 descriptors. Ranking preserves exact int64
counts, descending descriptor order and stable node-ID ties. Softmax uses every
candidate, including candidates excluded by positive-only selection. Separate
validity bits distinguish missing history from a present zero. Frames advance
history sequentially; the proposed history commits only after all downstream
queue/routing checks succeed. Nonfinite scores, malformed metadata and selected
count overflow explicitly fail. Selector scratch has a declared approximate
budget; other region programs remain unsupported by this component.

This selector component consumes supplied descriptors. The content-driven loop
connects sum Aggregate, content/old/proposal Read, identity/EMA/Add-repeat state and identity/tanh Full;
other Read/state/Full contracts still need integration. Its AIV metadata kernel
uses scalar loops; raw ACLNN performs the packed softmax and control gather.

`PackedFull` adds device selection of bounded physical chunks from actual active
actions. Bulk gathers feed FP32 KEEP_DTYPE batch matmul and tanh; index-copy writes
results to distinct destinations. Comparison snapshots precede selected clear.
Inactive owners are excluded from numerical work, with independent zero sentinels
for padding. The local Full scratch estimate can reduce the requested chunk rows;
it is not a complete model-memory budget. See [content-flow.md](content-flow.md)
for the finite contract and development coverage. Component placement does not
establish full-graph throughput.

The content flow separates `advance_device()` from CPU `snapshot()`/`result()`
exports. Callers may consume device output buffers across successive windows
without downloading the persistent graph state. Optional diagnostics remove the
event/message journals from the captured program; input validation/upload and
complete-window error reporting remain host boundaries. See
[content-flow.md](content-flow.md) for buffer lifetime, failure and export contracts.

`PackedSum` has selectable scalar/vector numerical implementations. The vector
path splits exact int64 metadata validation from parallel payload tiles, preserving
the stable message order inside every fiber. Validation errors prevent all payload
writes. Its directed check covers tail widths through2048, ragged and empty groups,
changed inputs on replay, unused NaN storage, malformed metadata and autograd refusal.
It is an inference component; qualification and placement status belong to STATUS.

State updates also offer vector payload tiles after metadata preflight. Add repeats
literal retention multiplies for the device-computed elapsed ticks, with a declared
per-candidate work limit and explicit refusal. Independent owner/width tiles share
no destination; each owner's time sequence remains ordered. Scalar state is an
explicit comparison option. Read's numerical preparation is still scalar device
work. See [lazy-add.md](lazy-add.md) for the formula and work-limit boundary.

These mutable stages provide no autograd. Model/state updates must eventually
share a commit boundary with successful delivery; committing state before an
overflowing queue transaction would violate the intended executor contract.
Byte budgets for parameters, state, KV, activations and communication are still
separate from these bounded metadata/payload capacities.

## Build and checks

Use a matching standalone NPU Tide/LibTorch-NPU package. The project build driver
verifies core source/binary fingerprints, CTests and non-Python/non-stub loader
closure. Ordinary builds use Ninja. CANN9's legacy Ascend C host-stub extraction
requires Unix Makefiles here because Ninja produces inconsistent literal object
paths. The kernel target name must be a valid C++ identifier, and consumers use
PIC objects because the vendor interface links PIE executables.

```sh
python scripts/build_device_control.py --core-build CORE --build-dir NEW --jobs 2
python scripts/build_device_control.py --core-build CORE --build-dir NEW \
  --ascendc-soc ACTUAL_SOC --jobs 2
NEW/tide-device-control-check --device=npu:0 --dtype=float32
NEW/tide-device-numerical-check --device=npu:0 --dtype=float16
NEW/tide-packed-queue-check --device=cpu --dtype=float64
NEW/tide-device-closure-check --device=npu:0 --dtype=float32
NEW/tide-packed-sum-check --device=npu:0 --dtype=float32
NEW/tide-device-add-check --device=npu:0 --dtype=float32
python scripts/verify_device_control.py --build-dir NEW --output-dir GATE \
  --device=npu:0
python scripts/profile_device_control.py --build-dir NEW --output-dir PROFILE \
  --device=npu:0 --check ready
python scripts/profile_device_control.py --build-dir NEW --output-dir LEAN_PROFILE \
  --device=npu:0 --check window --application-arg=--without-diagnostics
python scripts/verify_device_failures.py --build-dir NEW --output-dir FAILURES \
  --device=npu:0
```

Ascend C is optional and requires an explicit matching SoC; the closure check
rejects a different actual SoC. Its generated kernel library registers with CANN
at load time, so even its help check runs under a device lease. CPU CTests do not
launch it. The packed-queue NPU check covers queue operations only; the separate
Ascend C check covers NPU closure against the independently gated CPU reference.
Use `scripts/profile_device_control.py` for a bounded msprof placement trace
with exact binary/source checks and hashed exported operator records. Its scope
includes setup and CPU assertions, so task sums are not steady-state throughput.
Both markers and process termination matter: msprof may itself exit zero after
an application failure. The profile driver rejects its application-failure
warning even when an earlier acceptance marker was printed.
The active source, terminal results, failures and remaining work live in STATUS.
No resident-training or speed claim follows from these component checks.

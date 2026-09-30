# Experimental device queue and CANN control

`tools/device_online` contains separately tested building blocks and an
experimental [content-driven forward loop](content-flow.md) for an explicit
existing-module profile. It is not a general module or resident training backend. It isolates CANN runtime
models, raw ACLNN numerical stages and optional Ascend C kernels from the portable
core. The independent CPU scheduler and scalar qualification remain unchanged.

## Runtime control

`CannProgram` binds a persistent stream to an explicit runtime model. Device
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
model state, peer completion, semantic VJPs and optimizer integration still
require implementation and separate gates. A readiness kernel alone is not a
complete resident flow. Placement/profiling results require actual target traces.

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
connects sum Aggregate, content/old/proposal Read, identity/EMA state and identity/tanh Full;
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

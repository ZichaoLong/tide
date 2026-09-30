# Experimental device queue and CANN control

`tools/device_online` contains separately tested building blocks. It is not yet
an online graph executor or a resident training backend. It isolates CANN runtime
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
This failure-lifetime policy still needs its own injected-failure gate.

Standalone checks own a `portable_torch::RuntimeSession` in `main`, outside all
tensor/program owners. Normal finalization must precede main-thread TLS teardown;
a former static finalizer could segfault in TorchNPU's current-stream lookup.
The lifecycle check covers nested owners, idempotent close and rejected reopen.
Abnormal/quarantined-resource teardown remains part of the pending failure gate.

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
sets for selection. Device lengths and valid bits identify real work. Packing
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

This selector consumes supplied descriptors: real module Read, state updates
and selected Full computation still need integration. Its AIV metadata kernel
uses scalar loops; raw ACLNN performs the packed softmax and control gather.
Component placement does not establish full-graph throughput.

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

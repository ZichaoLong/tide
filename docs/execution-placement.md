# Public execution placement

`ExecutionPlacement` configures Read, control calculation, candidate ranking and
event progression separately. The implementation extends the public host-driven
schedulers; its verification status is in [STATUS](STATUS.md). The complete
resident, multi-device and performance requirements remain in
[execution-flows.md](execution-flows.md). This interface alone does not close them.

| Preset | Read | Control | Selection | Events |
| --- | --- | --- | --- | --- |
| `cpu` | CPU | CPU | CPU | CPU |
| `mixed-a` | CPU | CPU | CPU | CPU |
| `mixed-b` | Payload device | Payload device | CPU | CPU |
| `mixed-c` | Payload device | Payload device | Payload device | CPU |
| `resident` | Payload device | Payload device | Payload device | Payload device |

`cpu` requires CPU payloads; mixed/resident presets require an accelerator.
`native` retains the original layout: Read/control with payload, host selection
and events. Leaving the optional placement unset preserves the independent
original Read/selector implementation, including the CPU reference.

Each fine switch accepts `auto` (preset default), `cpu`, `payload`, or the exact
payload device. In an eager owner map, `payload` resolves independently at each
node or region owner; an explicit indexed switch must match that owner. Use
`payload` to follow a multi-device layout. Arbitrary extra scoring devices are
not silently introduced.
`place_model` and host schedulers explicitly reject device event progression.
GraphRuntime can select the optional [resident library](resident-library.md) with
an explicit matching `resident_library` build. Its inference/training and
multi-device scopes are recorded in [resident-training.md](resident-training.md)
and [STATUS](STATUS.md); it never simulates device event progression on CPU.

```python
from tidegraph import ExecutionOptions, ExecutionPlacement, GraphRuntime

options = ExecutionOptions(
    implementation="native", schedule="greedy", mode="hst",
    placement=ExecutionPlacement(preset="mixed-c"),
)
runtime = GraphRuntime(config, device="npu:0", options=options)
session = runtime.session(batch_size=2)
```

Both `streaming` and general online `greedy` use these adapters. Python execution
uses independent Python Read/Region programs; native execution uses C++ programs.
`GraphRuntime.manifest()` records requested options and resolved devices.
`Native(..., placement=...)` exposes the native adapter directly. C++ clients use
`tide::place_model(graph, model, request)` and `resolve_placement(...).record()`
from `tide/placement.h`, then construct Streaming/Greedy with the returned model.
No topology, input values, routes or event traces are executed in preparation.

Read packs each legal numerical node-time batch before transferring its values.
Control remains one differentiable region-time frame at a time. Ranking uses
stable tensor sorts on the requested device, preserving exact int64 selected and
affected counts, node-ID ties, positive-only eligibility and explicit budgets.
The host consumes a packed selection mask and updates its typed history. Therefore
`mixed-c` has device ranking but still has host event dispatch and history ownership.
Transfers, finite checks and that host work belong in complete timings.
On the current CANN 9.0/TorchNPU 2.10 development path, int64 `argsort` explicitly
reports AiCPU execution. The bounded development trace also places boolean
ScatterElements on AiCPU. That is an NPU engine, distinct from host CPU fallback;
this mixed adapter makes no all-AiCore claim. The exact integer ranking contract
must not be replaced by a lossy FP32 count conversion. A separate bounded trace
is available through `scripts/profile_execution_placement.py`; formal throughput
must be measured without profiling. The Ascend C resident selector is separately
implemented and qualified under the content-flow scope.

`scoring_dtype="profile"` retains the Read declaration; `payload`, `float32` and
`float64` are explicit numerical policies for linear Read. A named norm profile
keeps its declared precision and rejects an incompatible override. NPU cannot
consume FP64 descriptors in Read/control/ranking. NPU payloads with CPU FP64 Read,
CPU controls and CPU ranking are legal. Controls return to the payload dtype and
device; copies retain autograd. No conversion of int64 counters to floating point
is used to combine sort keys.

Placement preserves the original parameter leaves, aliases and checkpoint names.
Python execution views do not insert a registered wrapper prefix into state_dict.
Settle's existing identity-boundary encoding still has its own physical parameter
names. Differentiable softmax frames stay separate so independent output roots
do not acquire connected-zero gradients. Packed Read retains the existing
per-event semantic VJP replay; full resident backward remains a separate backend.
Only recognized built-in Read/Region programs are adapted. Custom/mismatched
handles, unsupported devices, conflicting precision and resident requests fail
explicitly instead of replacing a user's custom mathematical program.

## Eager payload owners

`GraphRuntime(..., node_devices=[...])` accepts one logical device per body node.
The first owner equals `device`; all owners use the same backend and dtype.
CUDA/NPU owners require explicit logical indices. This placement is independent
of schedule and graph/checkpoint identity. No CPU execution or route replay is
part of planning. Availability checks restore the runtime's default device,
including when a later owner is unavailable. An omitted map retains single-device
behavior. An explicit multi-device map uses the default Read/Region adapter if no preset
was supplied. `manifest()` includes actual node, region and Read/control devices.

```python
runtime = GraphRuntime(config, device="npu:0", options=options,
                       node_devices=["npu:0", "npu:1", "npu:1"])
optimizer = torch.optim.AdamW(runtime.model.parameters(), lr=1e-4)
```

For caller-owned models, use `tidegraph.place_payloads(graph, model, devices)`
**before constructing the optimizer**, then pass that model and the same map to
GraphRuntime. It stages tensor moves before publishing them and preserves alias
partitions. Shared node weights constrain those nodes to a common device; an
incompatible explicit map fails before allocation. Region parameters/history
have one owner (the first member by default, or a co-located shared parameter
owner). Shared port scales retain one canonical leaf and are copied at their
actual uses. A caller-supplied model is validated, never silently relocated.
C++ clients have the independent `tide::place_payloads` construction helper in
`tide/ownership.h`, followed by `place_model` and the ordinary executors.

Every node's Aggregate/state/KV/Full runs at its payload owner. Delivered messages,
including pending messages, live at their destination node. External records must
already be on their input node's device; Settle input and identity boundaries use
the first owner. Output values retain their output node's device. A consumer owns
any loss/head gather. Region descriptors gather to the configured control device;
selection sees the entire region-time frame and each control returns to its node.
Worker threads inherit the caller's stream for every model device. Host metadata,
queue progression and frame construction remain host work.

The initial owner implementation copies message tensors individually, preserving
the independent autograd edge of each message and canonical parameter leaf. It
does not yet claim packed cross-device transport or large-model admission. Those
performance/consumer obligations remain open under F5/F6. Existing legal local
time batching is unchanged. There is no implicit detach at a device or window
boundary. Inference/FP16/more policies and performance require their own evidence;
the directed owner gate covers FP32 (plus CPU FP64), HST Add/Attention, all three
families, streaming/greedy and complete updates.

Python checkpoint v5 remains a portable value boundary. Loading restores node
state/KV, region history, pending messages and optimizer slots to their current
owners, permitting a different map with the same aliases/graph. Scalar optimizer
step placement follows the optimizer's own policy. Serialization does not preserve
the preceding autograd segment. Standalone C++ in-memory continuation and optimizer
ownership are checked separately; its existing named-weight checkpoint is not a
serialization of the graph continuation.

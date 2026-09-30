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
payload device. This single-device adapter rejects other accelerator devices.
`place_model` and host schedulers explicitly reject device event progression.
GraphRuntime can select the optional [resident library](resident-library.md) with
an explicit matching `resident_library` build. That backend currently exposes
single-NPU FP32 HARD inference; it never simulates device event progression on CPU.

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

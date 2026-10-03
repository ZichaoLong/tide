# Eager cross-device packed-copy qualification

Clean implementation `a785d43cfd9662c7d7262c0fdd8c59ec44729f6f` is qualified for
CPU FP64/FP32 and two-device NPU FP32. [Machine audit](eager-packed-transfer-20261003.json)
checks12 accepted terminal jobs, all1511 frozen source hashes, archive members,
fresh links, installed consumers, loader closure, full gate counts and trace.
The failed standalone lazy-initialization run is retained separately.

| Gate | Passed scope |
| --- | --- |
| Builds | Three core/adapters (CPU,NPU Python,NPU standalone), two installed consumers |
| Python/native core | CPU352,NPU119;0 skipped |
| Standalone | CPU12 configurations per FP64/FP32;NPU36 FP32;two updates/four connected windows each, plus isolated copy roots |
| Actual consumers | CPU64/deselected8,NPU40/deselected9;0 skipped |
| Separate profile | Actual two-device Attention mixed-C/prefill, two complete AdamW updates and connected windows;no CPU reference/diagnostics |

Packed copies group only completed messages by source/target/dtype/shape, with
an8MiB limit; an indivisible larger row copies alone. Isolated and upstream roots,
missing versus connected-zero cotangents, frozen rows, aliases and pack boundaries
are checked. Actual consumers compare full records/gradients/parameter updates
with an independent CPU execution across all three families, Add/Attention,
both schedules and mixedA/B/C. Host metadata/scaling remains eager work;
this is not device-resident routing.

Each profiled update moves28 cross-device rows in24 copy groups (448bytes,
maximum group2). The trace records14,104 operators:11,374 AI_VECTOR_CORE,
1,622 MIX_AIV,980 AI_CORE and128 AI_CPU. AiCPU comprises80 Boolean/INT64
ScatterElements and48 INT64 Sort operations; exact integer semantics remain.
Both leased devices appear in the trace. No host tensor-compute fallback was
observed. Instrumented timings are not throughput or a speedup claim.

`packed-transfer-cpp-npu-clean01` hung during normal CANN compiler teardown and
ended at its900s bound, failed/exit124 with released lease/empty cgroup. Its saved
thread dump locates the main thread inside CANN's embedded Python finalization;
autograd workers were idle. On this TorchNPU2.10/CANN9.0 stack, explicit
`ACL_OP_INIT_MODE=0` initializes the compiler before worker use. The same binary
then passed the complete36-configuration gate and normal cleanup in31s under
`packed-transfer-cpp-npu-init01`. Actual consumer40 and the profile also passed
with that setting. No operator was disabled, no cleanup was skipped and no shared
package/environment was modified. The default lazy-init failure remains failed.
Static loader checks establish no linked CPython dependency; CANN can dynamically
embed its own Python compiler. [Launch contract](../accelerators.md) records this
version-scoped requirement.

Raw records are `TASK/runs/NAME/{status.json,task.log}`; accepted names are listed
in the machine audit. Frozen source `TASK/sources/packed-transfer-clean01`;
cores `TASK/builds/packed-transfer-{cpu,npu-python,npu-standalone}-clean01`;
installed consumers `TASK/builds/packed-consumer-{cpu,npu}-clean01`.
Re-audit with `python TASK/launchers/packed_transfer_evidence.py a785d43cfd9662c7d7262c0fdd8c59ec44729f6f`.
All accepted services are inactive/dead with empty cgroups and completed leases.
The affected gate does not replace the historical regression suite. Total-memory
calibration, original-scale CPU/mixed comparisons, eager FP16 training, CUDA
hardware and final F1–F7 integration remain separate work.

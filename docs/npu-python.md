# TorchNPU library path

The public `GraphRuntime(config, device="npu:0")` boundary places model, input,
state and optimizer tensors on the selected logical device. Python and native
schedules share graph/module semantics and are checked against an independent
CPU reference. FP32 is the supported NPU payload dtype; FP64 payloads and
`norm-fp64-v1` Read are explicitly rejected.

The [accelerator guide](accelerators.md) owns current build, qualification,
profiling, checkpoint and installation recipes. Native execution has two
separate consumers: a Python adapter using the wheel's TorchNPU registration,
and independent C++ using a matched standalone `libtorch_npu` SDK. They must not
load both vendor library flavors in one process.

The [2026-09-25 evidence](evidence/npu-python-20260925.md) remains a historical
finite Python smoke at `7811418`, including the then-unavailable standalone
SDK. It is not the acceptance record of the later public library extension.
Exact current environments/cases are in `.torch-portability/contract.json` and
its linked evidence. A card generation name or CANN version alone is not an
operator/backward compatibility guarantee. Profile the selected workload;
shared-installation owner warnings alone do not establish CPU fallback.

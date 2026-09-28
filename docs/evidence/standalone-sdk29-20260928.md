# Standalone TorchNPU2.9 SDK and local CANN runtime qualification

The additional standalone C++ SDK and project gates passed on CANN8.5.0,
8.5.1 and8.5.2. This extends the existing Torch2.10/CANN9.0 standalone evidence.
The project implementation is the unchanged clean `b4f26b3` snapshot; no Python
package or public core source changed for this work. [Exact identities and results](standalone-sdk29-20260928.json)
include SDK/build/binary/header hashes, job names, failures and operator traces.

## What was built and tested

The SDK comes from official Ascend/pytorch revision
`12d689a08941d4a6e45eab16e3ad6fef96a9affd`, matching TorchNPU2.9.0, against
Torch2.9.0+cpu/CANN8.5.0 on aarch64, GCC10.3.1, CXX11 ABI1. The same SDK and
matched Torch libraries were tested with three separately loaded CANN/ATB
runtimes. This is one SDK build and three runtime qualifications, not three
independent SDK builds. CANN8.5.1/.2 are local observations, not an extension of
the vendor's published compatibility guarantees. Hardware is Ascend910_9392/A3,
driver25.3.rc1. `TASK_QUEUE_ENABLE=0` is explicit.

| CANN runtime | Standalone core | Installed CMake consumer | Historical-topology consumer |
| --- | --- | --- | --- |
| 8.5.0 | ring/diamond passed | passed | 64 cells + analytic dispatch passed |
| 8.5.1 | ring/diamond passed | passed | 64 cells + analytic dispatch passed |
| 8.5.2 | ring/diamond passed | passed | 64 cells + analytic dispatch passed |

The core checks compare independent CPU/NPU schedules, full values and gradients,
shared owners, three AdamW updates, nondefault worker streams and checkpoint
weights/optimizer slots restored onto NPU and CPU, followed by another update.
The copied external application builds only against installed headers/targets and
passes allocation, chunking and backward. Loader closure excludes missing/stub
libraries, `libpython` and `libtorch_python`; static inspection also found no
undefined CPython API symbols. The consumer build passed all three CTests.

The192 consumer cells exercise two NPUs each, both models and optimizers,
CPU FP64/FP32 scoring, model FP32 scoring, independent ranking/queue choices,
small memory/locality placements, and real465-node D8/B1 forward/training cases.
Scope and numerical policy match [the dispatch/training qualification](accelerator-dispatch-training-20260928.md),
including strict tiny tests and explicit `basis-conditioned` wide isolated VJPs.
These are finite correctness gates; they do not certify full-size performance,
arbitrary topologies, all public-core modules through standalone registration,
or training convergence. The previous39 Python/43 native-adapter cases per CANN
version remain [separate broader evidence](accelerators-20260928.md).

## Packaging fixes and public modules

Initial attempts are preserved, with distinct successful follow-ups:

- The source/submodule fetch succeeded but the final direct compatibility-table
  request timed out. One recorded local-proxy retry retrieved the official table.
- The first independent C++ link could not find the Torch wheel's hashed Fortran
  dependency used by OpenBLAS. Adding the exact matched `torch.libs` directory
  to link/runtime search paths resolved it; no dependency was replaced.
- Project integration found the official2.9 SDK export lacked source-path ACL/HCCL
  headers and `HCCLUtils.hpp` required by public runtime/allocator headers.
  Forty-six exact-source headers complete that include closure. Their manifest
  supplements the original SDK hashes; SDK libraries and existing headers are unchanged.

Versioned public modules `libtorch-npu/2.9.0-cann8.5.0`,
`libtorch-npu/2.9.0-cann8.5.1` and `libtorch-npu/2.9.0-cann8.5.2` are published.
Load/unload, version conflicts, public read/traverse permissions and loader closure
were checked. All select the same matching Torch2.9 libraries. The existing2.10
SDK, base Torch/CANN installs and shared driver were left intact. Rebuild recipes,
source pin, patches, original failures and package README are retained with the SDK.

## Operator placement and limits

Each8.5 runtime passed a separate one-NPU tiny Attention D32/B4/T3 profile with
model FP32 Read/controls/ranking/events. Each trace contains6946 operator records,
including41 LpNormV2 on MIX_AIV,349 SoftmaxV2 on AI_VECTOR_CORE,12 int64 ReduceMin
on MIX_AIV,18 NonZero on MIX_AIV,55 int64 Sort on card-local AI_CPU and5 FP32 Sort
on MIX_AIV. This matches the engine placement observed with the2.10/CANN9.0 SDK.

Int64 sort is exact and runs on the card's AiCPU, distinct from host CPU and
AiCore. Host-owned count maps, tensor handles and C++ dispatch remain explicit;
these options do not make the full control loop device-resident. Profiles establish
placement for the named tiny workload, not full-size speed or absence of every
possible fallback. NPU FP64 and optional CSR pooling remain unsupported. CUDA
hardware and other host architectures still require their own target-machine gates.

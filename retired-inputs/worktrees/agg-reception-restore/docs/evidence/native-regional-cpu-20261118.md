# Standalone regional slice: fixed CPU checks

Implementation `89ce08ff858724648152790502f8c65d8d8b1222`, clean at build time.
The [machine-readable record](native-regional-cpu-20261118.json) preserves binary,
source, library, fixture and log identities. This is the complete regional scoring
Attention+FFN **subpath** described in [the native entry point](../native-regional.md),
not another complete Tide implementation. Fixed mathematics remains
`4f5b9f20a4030d68ee66ae2d26e34659a5e37234`; 108 files passed verification.

## Actual results

The native checker completed **29 cases**: 25 executing cases and four expected
rejections. Both FP32 and BF16 covered inference, grad-enabled forward and three
shared-owner AdamW updates with 1/2/4 workers. Each worker count then loaded its
checkpoint in a new native process and performed the real fourth update.
Outputs, input/shared-parameter gradients, updated parameters and both moments
matched the same-layout Python reference at the unchanged `rtol=2e-5, atol=2e-6`.
Maximum absolute differences across compared tensors were `2.384185791015625e-7`
for FP32 and `1.1920928955078125e-7` for BF16. These are finite-case numerical
results, not promises of bitwise or unrestricted trajectory equivalence.

All tasks shared one owner with 8,320 parameters. The fixture shape was
`[tasks=4,nodes=5,B=2,L=3,description_width=32]`, seed 1729. It included positive,
explicit zero, entirely empty and mixed prior cases. Visible zero priors retained
a nonzero right VJP; empty rows had the declared zero VJP. The separate zero-weight,
bias=1.001 case checked the BF16 Linear epilogue. Corrupt input, existing output,
invalid worker count and an unsupported explicit device were rejected.

Five independent Python tests passed in 0.38 seconds. They compare the active
`RegionNetwork(math)` values/shared VJPs, use an FP64 literal quotient formula and
right finite difference, and check that shared gradients equal the sum of each
task's contributions. FP64 is only a bounded diagnostic here. CTest's three CLI
checks passed for each build. No prior model precision qualification was changed.

## Build, precision and resources

Two independent binaries were built with GNU C++ 10.3.1, C++17, Release, aarch64,
LibTorch 2.12.0 and C++11 ABI1. The CPU binary actually loaded only the published
standalone CPU LibTorch closure; no Python runtime library was loaded. The NPU
binary links the published CANN 9.2 SDK from Ascend source
`6b88d20d143ef15243c44f3582f37c480e786f2e`, including its recorded packaging patch.
SDK static classification was `standalone-sdk-candidate`; the final loaders had
no missing libraries and the SDK inspection found no stub paths or Python coupling.

Both builds reported the optional missing-kineto warning. Required targets,
linking, loader closure and CPU operations passed; this does not qualify a profiler.
The SDK's variable-only CMake interface and misspelled CXX-flags variable were
recorded; the consumer explicitly applies the matched Torch flags target-locally.

BF16 uses real BF16 Q/K/V and FFN projections with the bias in the same Linear
operation. Parameters, inputs, gradients, optimizer moments, Attention products
and final scalar readout are FP32, with autocast caching disabled. Each task
restores TLS, grad, precision, device and stream context. CPU inter-op and each
task's intra-op count were one. Autograd aggregates all task contributions before
one shared AdamW step (`lr=.001`, `eps=.001`, `weight_decay=.01`).

The frozen build/check job ran in `background.slice`, affinity 72–79, CPU quota
400%, service memory maximum 16 GiB, sampled process-tree RSS limit 14 GiB and
600-second timeout. It ended normally after 97.59 seconds, peak sampled
process-tree RSS 2.240 GiB, with no remaining process. The check timings include
correctness work and are **not performance evidence**.

## Limits and retained records

No NPU device operation, operator-placement trace, same-stream concurrency,
multistream probe or formal performance comparison was run in this CPU record.
The NPU binary's build/loader result cannot qualify those paths. CUDA has no native
adapter in this finite experiment. Native checkpoints contain regional parameters,
moments and progress; full Tide memory, graph state and continuation are not part
of this subpath.

Raw evidence lives under
`artifacts/worktrees/capacity-bindings/artifacts/native-89ce08ff-cpu/` in the
integration workspace: `checks/status.json`, per-case tensors/logs/checkpoints,
`run.log`, `source-sha256.json`, SDK manifests, loader reports and binary hashes.
The JSON record includes exact absolute paths and SHA-256 identities. Initial
development build-v1's ambiguous `norm` overload failure remains recorded; the
local helper was renamed before the fixed implementation commit. It is not a
device or numerical failure. Reproduction commands are in
[native-regional](../native-regional.md); rebuild from the recorded source and
use new output paths plus current resource admission.

复现源码映射：原工作树 `89ce08ff` 的13个改动文件与主分支 `062a8ab6` 逐字节相同，
可从后者取得native切片及检查入口；构建仍会记录新实际提交/二进制身份。
完整文件清单见[集成身份核对](capacity-validation-20261118.md)。这不改写原二进制的构建身份。

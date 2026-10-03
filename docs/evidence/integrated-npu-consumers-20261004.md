# Integrated eager and resident NPU consumers

Clean source `e69b3bde3d53b0d6a019e89e82d1a78c3a91a7b8` passed **142 affected
checks, with no skips**. The current eager core and freshly compiled resident
backend now have a qualified installed standalone consumer in common.
[Audited identities, commands and terminal records](integrated-npu-consumers-20261004.json).

## Build and runtime boundaries

The former resident build embedded the older `d412541` core. This qualification
uses the current core C++ digest
`ca3597e96eb7a09dc68542b7b1c0904ec39f93375fe358524bf9b6273508868c` for both
runtime owners. The already qualified `a785d43` core binaries were verified
byte for byte and packaged with fresh CMake install metadata; they were not
recompiled. Resident host code and Ascend C kernels were freshly compiled from
the tested source against those packages.

The Python backend registers through the matching TorchNPU wheel. The standalone
backend uses the separate LibTorch NPU SDK. A new external C++ consumer builds
against installed public headers and packages, with eager and resident enabled
together. Its loader has no missing library, stub, simulator, `libpython` or
`libtorch_python` dependency. The installed core archives and resident library
match the verified build bytes. Environment: aarch64, Torch/TorchNPU 2.10,
CANN 9.0.0, Ascend910_9392; standalone execution uses `ACL_OP_INIT_MODE=0`.

## Affected qualification

| Gate | Checks | Time | Physical devices |
| --- | ---: | ---: | --- |
| Eager FP16 masters and complete consumers | 49 | 212.35 s | 6, 8 |
| Public resident training and complete consumers | 93 | 423.79 s | 6, 8, 12 |

The eager gate covers independent Python/native/standalone consumers, all three
families and both schedules, connected windows, physical sample splitting,
FP32 masters/slots, aliases, None/zero gradients and nonfinite rejection.

The resident gate covers FP32/FP16, all three families, both schedules, SGD and
AdamW, complete continued training/inference, head splitting, int64 boundary
coordinates, atomic refusal and checkpoint continuation. A fresh process restores
a two-owner checkpoint on three owners. Independent CPU oracles remain part of
these correctness tests; they supply no execution trace or gradients to candidates.

The three builds and both gates are terminal with zero exit codes, inactive
units and empty control groups. Both NPU leases were released. Snapshot inventory,
core/backend/client manifests, binaries, installed libraries and loader records
were audited. Shared-installation owner warnings, TorchScript/TypedStorage
deprecations and base-format allocation notices occurred; no host tensor-compute
fallback warning was observed.

## Retained failures and limits

The first standalone and Python packaging attempts failed before compilation
because the incremental core artifact directories had no `cmake_install.cmake`.
The second Python attempt also failed before compilation: Python-owned NPU
registration requires bindings enabled. Fresh, correctly configured package
directories resolved these issues. All three failed receipts and partial builds
remain failed and retained.

This qualification covers integration and small-model semantics. It does not
replace the outstanding full-size performance matrix, certify GPU execution or
relabel the [strict original-scale near-tie route failure](original-add-route-witness-20261004.md).
That numerical-sensitivity limitation remains explicit under the user's accepted
delivery policy. No comparison tolerance, tie rule, model or input was changed.
The already qualified full CPU gate and unrelated primitive suites were not
repeated.

Raw jobs are `build-integration-resident-standalone-clean02`,
`build-integration-resident-python-clean03`, `build-integration-online-clean01`,
`integration-eager-npu-clean01` and `integration-resident-npu-clean01` under
`TASK/runs`, where `TASK=/mi/data2T/zlong/tide-execution-flows`.

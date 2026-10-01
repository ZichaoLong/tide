# Resident FP16 building blocks

Source **466b4c3e89d71adf4ca48d188638b93089841748**, clean standalone C++ build,
aarch64 / Ascend910_9392 / CANN 9.0.0 / Torch and LibTorchNPU 2.10.0.
[Audited identities and results](resident-fp16-components-20261001.json).

All eight requested cells passed: numerical, Full, packed LH and sum, each in
FP32 and FP16. The normal build used `--checks numerical full packed-lh sum`,
two workers, fresh CANN archives, one applicable CTest and verified loader closure.
The device gate used the same explicit subset. It does not qualify the whole
resident library after the CMake refactor or enable public resident FP16 sessions.

| Component | Coverage per dtype |
| --- | --- |
| Full | 32 cases; width 1/7/33/257, selected rows, empty selection, inactive NaN, device chunking; CPU storage-dtype and FP64 formulas |
| LH | 40 cases, nine profiles, 536 normalized rows; unchanged epsilon, FP32 statistics, half storage/compute/output |
| Sum | 72 configurations, 144 replays, 18 refusals; width 1–2048, tails, empty/zero/NaN isolation, stable physical order |

Sum products and ordered accumulation use FP32; contribution and summary stores
round independently to the requested payload dtype. LH uses the existing
conditioning checks for near-zero normalization inputs. The half fixture's
CPU/device maximum absolute error relative to FP64 is 0.0276378; 270 half rows
and 34 float rows miss the strict component comparison but satisfy their explicit
conditioning budgets. FP32 thresholds and normalization epsilon are unchanged.

Separate bounded FP16 profiles passed: Full 544 AI_VECTOR_CORE + 24 AI_CORE
tasks; LH 5,364 AI_VECTOR_CORE tasks; sum 1,205 AI_VECTOR_CORE tasks. No AiCPU
task or logged CPU fallback was observed. These traces include construction and
CPU assertions; task durations are **not throughput measurements**.

Raw jobs under the project task store are `build-low-precision-components-clean01`,
`low-precision-components-clean01` and `low-precision-{full,lh,sum}-profile-clean01`.
All are terminal passed/exit 0. Source, reused core, binaries, loader, gate logs,
profile logs and CSV hashes were rechecked. Development runs remain separate.
Complete FP16 state/attention/Read/backward/master publication, multi-device
progression and performance remain pending under [F1–F7](../ROADMAP.md).

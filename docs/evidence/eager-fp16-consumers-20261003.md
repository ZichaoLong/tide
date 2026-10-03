# Eager FP16 consumer qualification

Implementation `7b1fae504ec779143655b9221c82d8c14a69b410` passed the affected
clean-source qualification on aarch64 CPU and Ascend NPU, Torch/LibTorch2.10
with CANN9.0. The [machine-readable audit](eager-fp16-consumers-20261003.json)
records source, installed binaries, tests, calibration, trace and terminal receipts.
All1535 frozen source files retain their hashes. Eight accepted jobs are
passed/exit0, with inactive services, empty cgroups and completed device leases.

The actual Python and independent standalone C++ consumers use FP16 payloads
and payload-dtype autograd accumulation, with FP32 loss, canonical masters and
SGD/AdamW slots. Static `--loss-scale` is explicit (default1); no automatic
retry or skipped update occurs. Aliases and None/connected-zero gradients remain
distinct. The standalone consumer's master state is process-local; this does
not extend the public C++ checkpoint schema. Resident FP32 adjoints remain a
separate precision policy.

| Check | Result |
| --- | --- |
| CPU FP16 and admission | 61 passed,22 deselected |
| Two-device NPU FP16 | 49 passed,no skips |
| CPU FP32/FP64 consumer regression | 55 passed,no skips |
| Fresh installed CPU/NPU clients | Both passed; matching qualified core archives and loader closure |
| CPU fresh-process calibration | Four passed |
| Two-NPU fresh-process calibration | Seven passed |
| Actual two-NPU consumer profile | Passed,separate from timing |

The directed gates compare independent complete records, gradients, losses and
updates across graph families, schedules and mixed placements, including forced
sample splitting and explicit loss scaling. They include ordinary same-dtype CPU
references and preserve discrete semantics. These are the affected gates, not a
rerun of every historical test.

Calibration covers Python/LibTorch Add/Attention at D256, plus C++ Add/Attention
and Python Attention at D2048 with six body nodes. Every process executes two
updates with two connected windows, scale1 and16GiB/device. Observed memory
remains within the declared estimates; FP32 envelopes and safety margins are
unchanged. CPU peaks are231.92,132.63,349.54 and259.99MiB. The largest NPU case,
D2048 C++ Attention, uses4017.41/1793.88MiB against6055.36/2900.00MiB estimates.
This does not certify480-node/B512 capacity or formal throughput.

The separate small Attention mixed-C/prefill trace includes two AdamW updates,
two connected windows, physicalB1×2 and scale128. It contains15180 operators:
12298 AI_VECTOR_CORE,1774 MIX_AIV,980 AI_CORE and128 AI_CPU. AiCPU consists of
80 BOOL/INT64 ScatterElements and48 INT64 Sort. Both assigned devices appear;
no host tensor-compute fallback was observed. Construction/head/loss/optimizer
are included, while CPU references and numerical diagnostics are outside the
trace. Operator counts are not a performance comparison.

The original `eager-half-cpu-dev01` duplicate-parametrization collection failure
is retained. The later test-entry fix skips only when no standalone binary is
selected; explicitly selecting an absent binary fails. That final entry change
passed15 checks before the implementation commit and the clean gates above.

Raw source: `TASK/sources/eager-half-clean01`. Installed clients:
`TASK/builds/eager-half-{cpu,npu}-clean01`. Job names and raw hashes are in the
audit; records live under `TASK/runs/NAME/{status.json,task.log}`. Standalone
NPU commands use explicit `ACL_OP_INIT_MODE=0`, as previously qualified.
Re-audit with `python TASK/launchers/eager_half_evidence.py 7b1fae504ec779143655b9221c82d8c14a69b410`.

Full original-scale CPU/mixed/resident comparisons and final integration remain
open. CUDA hardware,x86_64 and untested vendor versions remain target-pending.

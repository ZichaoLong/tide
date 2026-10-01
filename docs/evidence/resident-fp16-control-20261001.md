# Resident FP16 Emit and control qualification

Source `3e3397378f8b6545230b3ba543a9b0c72296cc94`; all seven immutable-source
jobs passed with exit 0. [Machine audit](resident-fp16-control-20261001.json)
authenticates source inputs, reused terminal archives, rebuilt checker/host
objects, CANN kernels, loader closure, logs and profiler CSVs.

FP16 HST/SOFTP inference now preserves half rounding of control probabilities,
`g-h`, products and final additions. Softmax probabilities and Read scores remain
FP32 internally. Local control/Read VJPs use real half forward values and FP32
cotangents/accumulation. Only identity boundary nodes bypass Emit: ordinary
identity Full still executes Emit and retains connected-zero control gradients.
Complete FP16 graph reverse, retained training and public training remain pending.

| Check | Result |
| --- | --- |
| Standalone and Python-owned builds | Both passed; source-matched terminal dependency reuse, distinct loader stacks |
| Control VJP, each FP32/FP16 | 36 cases, 108 replays; independent forward 3 cases, 9 replays |
| Whole forward flow, each FP32/FP16 | 36 configurations, 144 windows |
| FP32 complete control training | 98 trajectories, 1,568 windows, 392 updates |
| Python precision/event/fiber clients | 97 passed, no skips |
| FP16 control VJP profile | 2,499 AI_VECTOR_CORE, 144 MIX_AIV tasks |
| FP16 whole forward profile | 75,735 AI_VECTOR_CORE, 3,042 AI_CORE, 36 MIX_AIV tasks |

Coverage includes HST/SOFTP, streaming/online greedy, all three Read coordinates,
linear/norm Read, widths 1/7/257, partial frames, None/zero roots, zero zeta,
NaN padding, identity boundaries, ordinary identity Full, capacity refusal,
EMA/event/fiber attention, positive-delay feedback and continuation.
Half local roots are scaled by 256 to expose accumulation errors; VJP tolerances
are rtol 2e-3 / atol 2e-5, with FP32 unchanged at 1e-5 / 1e-6.
Whole forward half tolerance remains rtol 2e-2 / atol 2e-3.

Runtime: aarch64 Ascend910_9392, public LibTorch/TorchNPU 2.10.0 + CANN 9.0.0;
leased physical devices are recorded per job and remapped to logical `npu:0`.
Profiles include construction and CPU assertions and establish placement only.
No observed AiCPU or logged CPU fallback; no throughput or full-size speed claim.

Preserved failures: dev01 build lacked a control-training checker object; dev02
explicitly compiled it and rebuilt production. An uncommitted dev03 experiment
incorrectly bypassed Emit for ordinary identity Full: component checks passed,
but full FP32 training detected the Read None/zero mismatch. Production bytes
were restored to passing dev02 and checker coverage was strengthened. No failed
object was relabeled as a passing build and no tolerance was loosened.

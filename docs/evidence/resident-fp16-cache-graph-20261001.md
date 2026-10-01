# Resident FP16 attention graph and retained-cache qualification

Source `f26f3b07f5e24a2ea835dbbfe352f3acf478eb57`; all five immutable-source
jobs passed with exit0. The [audit](resident-fp16-cache-graph-20261001.json)
authenticates source, archives, four oracle/checker-support objects, CANN kernels,
loader closure, ten gate cells and profiler CSVs. Production source and kernels
are unchanged from the previously qualified implementation `25e996c`.

| Check, each FP32/FP16 | Completed result |
| --- | --- |
| Event attention retained reverse | 66 trajectories,264 windows |
| Same-fiber and mixed attention retained reverse | 152 trajectories,608 windows |
| Base graph / retained regression | 122 windows /42 trajectories,168 windows |
| Normalized Aggregate/LH/SwiGLU regression | 110 trajectories,440 windows |

Every trajectory uses four continued windows, including an empty final window,
and independent CPU FP32/FP64 Streaming autograd. Candidate and reference consume
the common public fixture separately. Checks cover complete forward observables,
all named-owner/input/state/initial KV/log-bias gradients, aliases, None/zero,
cache padding and exact replay after closing and poisoning live forward storage.
Coverage includes GQA, eviction, adopt/clear, all five fiber pools, mixed cache
groups, periodic clocks, feedback/parallel edges, both schedules, HARD/HST/SOFTP
and widths1/4/257. No CPU route or cache result becomes a candidate input.

The test-only half CPU state adapter uses actual half QKV/QK/output matmuls,
source products and query scaling, each half log-bias decay tick, and FP32 global
softmax/weighted accumulation. Dense CPU attention is independent of device key
tiling. Roots and cache adjoints are FP32; half roots use scale256. VJP tolerances
remain rtol2e-3/atol2e-5 for half and1e-5/1e-6 for FP32; discrete identities and
connection bits remain exact. Production code and tolerances were not changed.

The separate half placement profile covers two trajectories: mixed event/fiber
with HST and periodic fiber bias with SOFTP. It recorded9,557 AI_VECTOR_CORE,
430 AI_CORE and156 MIX_AIV tasks, with no observed AiCPU or logged CPU fallback.
This includes construction and CPU assertions; it is not throughput evidence.
Runtime: aarch64 Ascend910_9392, public LibTorch/TorchNPU2.10.0/CANN9.0.0.

The dev01 build failure is retained: a test helper named `matmul` conflicted
with ATen argument-dependent lookup. Renaming it `half_matmul` fixed the compile;
dev02 and clean qualification passed. No artifacts from the failed build were
reused. Only affected test objects were rebuilt against authenticated terminal
production dependencies; unchanged Python and portable-core gates were not rerun.

Public FP16 training/master/checkpoint lifecycle, device peer progression,
matrix screening and full-size performance remain separate pending work.

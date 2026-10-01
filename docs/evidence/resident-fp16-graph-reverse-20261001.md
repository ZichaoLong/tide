# Resident FP16 graph and retained reverse qualification

Source `25e996c6a55b56ede9b689c94b8085ff45a56908`; all seven immutable-source
jobs passed with exit0. [Audit](resident-fp16-graph-reverse-20261001.json) binds
source, production/oracle archives, rebuilt objects, CANN kernels, loader closure,
raw results, preserved failures and profiler CSVs.

Actual half forward tapes now feed the device graph reverse loop. Cotangents,
state/message carry and shared-owner gradients remain FP32. Independent CPU
Streaming uses test-only quantized forward kernels with FP32/FP64 autograd leaves;
no reference event, result or gradient feeds the candidate. Sum products and
ordered accumulation are FP32, with separate half contribution/result storage;
physical delivery rounds before the next Aggregate consumes it.

| Check | Completed result |
| --- | --- |
| Standalone/Python-owned builds | Both passed; source-matched terminal CANN/core dependencies reused |
| Graph reverse, each FP32/FP16 | 122 windows, two replays per window, FP32/FP64 references |
| Retained reverse, each FP32/FP16 | 42 trajectories,168 windows, replay after closing/poisoning live forward buffers |
| Fiber reverse regression, each dtype | 90 cases/180 replays;3 bias bridge cases/6 replays/6 refusals |
| FP32 complete control training | 98 trajectories/1,568 windows/392 updates |
| Python affected clients | 97 passed, no skips |
| Half graph profile | 124,237 AI_VECTOR_CORE,1,941 AI_CORE,1,176 MIX_AIV tasks |
| Half retained profile | 95,158 AI_VECTOR_CORE,1,666 AI_CORE,1,656 MIX_AIV tasks |

Integration coverage: sum Aggregate; identity/EMA/Add-repeat state; identity/tanh
Full; HARD/HST/SOFTP; old/content/proposal Read and mixed linear/norm; arbitrary
fixture feedback, self loops, parallel edges, DAG and edgeless topology;
streaming/online greedy; widths1/3/257; int64 above2^55; input/initial-state/
output/pending roots; None/connected-zero; aliases; empty continuation; replay;
budget, root dtype, malformed metadata and missing boundary refusals.
Half roots are scaled by256. VJP tolerance is rtol2e-3/atol2e-5; FP32 remains
1e-5/1e-6. Whole-forward half tolerance is rtol2e-2/atol2e-3, with exact discrete
identity and connectivity checks. Saved tape admission counts actual element sizes.

Runtime: aarch64 Ascend910_9392, public LibTorch/TorchNPU2.10.0/CANN9.0.0.
Each runtime job leases a device and uses logical npu:0. No observed AiCPU or
logged CPU fallback. Profiles include construction and CPU assertions; they
establish placement, not throughput. Remaining half module integration, public
training/master/checkpoint, peer progression and full-size performance are pending.

Preserved failures: dev01 build used a CMake template without the newly introduced
fiber checker target; dev02 rebuilt production and linked that checker explicitly.
The dev02 half gate then refused at CLI resolution because two checkers omitted
`allow_npu_float16`; dev03 enabled the explicit checker capability. Dev04 added
all Read coordinates and mixed norm/linear coverage and passed. No production
objects from the failed build were reused and no tolerances were loosened.

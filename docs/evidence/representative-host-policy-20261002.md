# Representative native worker and packing comparison

Tested source: `2222d9db25a234eb0c739baf11bdccfffa2c5f5e`, clean immutable
snapshot. The bounded pilot, three confirmation rounds and separate CANN profile
all passed. [Audited samples, configurations and artifact hashes](representative-host-policy-20261002.json).
The [CPU11/NPU10 gate](consumer-host-execution-20261002.md) qualifies the public
controls; this report measures one TimedDAG/LibTorch/prefill FP32 submatrix.

## Scope and policy selection

The unchanged representative packets have128 body nodes/544 edges,
D128/B8/T4/V257, clear=true, and8,995,632 Add or17,384,240 Attention parameters.
Inputs, parameter initialization, continuation, loss and optimizer are identical
across candidates. Each independently executes its own online schedule.

A28-cell pilot compares CPU default1, packed1/4/16 node workers, and mixed-a
default1, packed1/4 workers, in four Add/Attention inference/training groups.
Packed means both `--packed-sources` and `--batch-next`. Each pilot process
has one continued warmup and one measured step. All four groups select CPU16
packed and mixed-a4 packed from this finite search. Mixed-a was selected by the
[earlier five-preset screen](representative-preset-screen-20261002.md).

Confirmation uses36 fresh processes: three repeats of CPU16 packed, mixed-a4
packed and resident, across all four groups. Each has one continued warmup and
three measured steps, two windows and64 input tokens per step. Order rotates
between rounds. ATen/BLAS/OpenMP threads remain1; node workers are a separate
control. Each accelerator process leases one logical Ascend910_9392 device,
using LibTorch2.10/CANN9.0.0. Own heavy measurements run serially; the shared
server remains a source of variation. This is neither a global CPU optimum
nor a significance claim. No further worker search was performed.

## Steady complete-step timings

Seconds are the median of three process medians, with their minimum–maximum
in brackets. Throughput ratio is **CPU seconds / candidate seconds**; greater
than1 means the candidate is faster. Raw per-step values remain in the JSON.
Timers include input preparation/upload, online scheduling, graph/head/loss,
synchronization and, for training, backward, finite checks and AdamW.

| Work | CPU16 packed | Mixed-a4 packed | Resident | Resident/CPU throughput |
| --- | ---: | ---: | ---: | ---: |
| Add inference | 0.132 [0.132–0.132] | 1.169 [1.130–1.216] | 0.158 [0.158–0.160] | 0.833× |
| Add training | 0.560 [0.550–0.585] | 3.174 [2.694–3.511] | 0.362 [0.360–0.363] | 1.549× |
| Attention inference | 0.254 [0.250–0.260] | 2.473 [2.429–2.517] | 0.297 [0.296–0.299] | 0.855× |
| Attention training | 1.180 [1.165–1.208] | 5.186 [4.148–5.633] | 1.048 [1.046–1.048] | 1.126× |

Within this scope, CPU wins steady inference and resident wins complete training.
The earlier one-worker CPU comparison cannot represent the best measured host
policy. Mixed-a remains slower, with appreciable training variation. All FP32
candidate-event counts, output counts and final cuts match exactly; losses
agree within1e-5. These benchmark checks supplement the full-observable,
gradient and update qualification; they do not replace it.

## Startup and memory

Each complete process executes four real steps including warmup,256 input
tokens, and includes runtime startup/exit. CPU is faster in every group at this
short lifetime, including training. Do not extrapolate a universal crossover.

| Work | CPU process seconds | Mixed-a process seconds | Resident process seconds |
| --- | ---: | ---: | ---: |
| Add inference | 1.428 | 10.164 | 5.427 |
| Add training | 3.268 | 18.561 | 6.019 |
| Attention inference | 2.292 | 16.055 | 6.036 |
| Attention training | 6.175 | 27.177 | 9.393 |

Median construction/warmup seconds (CPU→resident): Add inference
0.552/0.118→1.088/0.285; Add training0.567/0.554→1.143/0.573;
Attention inference0.973/0.193→1.456/0.445; Attention training
0.987/1.090→1.605/1.194. Mixed values and each raw sample are in the JSON.

Maximum phase/repeat allocated HBM in GiB is0.068/0.155/0.112/0.299 for
mixed-a and0.128/0.779/0.427/2.386 for resident, in table order. CPU peak RSS
is0.229/0.377/0.315/0.660 GiB. Allocator figures exclude untracked vendor/driver
memory; RSS is a process-lifetime peak. All resident allocator peaks remain
inside admission estimates. This does not establish full-size capacity.

## Separate profile of the selected mixed policy

One Attention-training mixed-a4 packed profile covers construction, one warmup
and one measured step. It enables runtime/task/AiCPU collection, passes result
checks and exports the retained CSVs. No trace-loss/capacity warning was found.
Instrumented timings are excluded from the table.

The trace contains275,217 device tasks:253,719 AI_VECTOR_CORE,20,528 AI_CORE,
970 MIX_AIV, and no observed AiCPU task. It records265,753 host kernel-launch
API calls,16,502 synchronous memcpy calls and31,964 asynchronous memcpy calls.
The summed device-task duration is0.679 seconds; it is not wall time.
Host API levels nest and must not be added together.

Against the earlier default-worker mixed training profile, host launches change
from271,132 to265,753 while synchronous copies increase from15,873 to16,502.
The new controls do not remove the large submission/transfer count. This
supports host orchestration granularity as a remaining limitation; it provides
no evidence for an AiCPU explanation. Other earlier profiles retain their
original policy scope and are not reclassified as new-policy traces.

## Reproduction and limits

Use the public consumer and representative packet commands from the earlier
screen, with `--workers 16 --packed-sources --batch-next` for CPU and workers4
for mixed-a. Resident retains its device-owned packing and default host flags.
Keep `--threads 1 --warmup 1 --steps 3 --windows-per-step 2`, FP32, the same
resident capacities and aggressive splitting. Exact argv and binary/input
identities are retained under `TASK/runs/host-policy-pilot01`,
`host-policy-confirm0{1,2,3}` and `host-policy-profile01`, where
`TASK=/mi/data2T/zlong/tide-execution-flows`.

This closes the bounded host-policy comparison for this submatrix. Other graph
families, clients, streaming, full-size execution, FP16 policy tuning, CUDA and
other CANN versions remain unmeasured here. F6 as a whole remains incomplete.

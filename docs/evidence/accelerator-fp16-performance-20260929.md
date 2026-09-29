# Full-size FP32/FP16 paired comparison

All eight bounded cells completed on the clean consumer source
`595dccd0b23bd809bed28874591ae4f5f268512f`: two payload precisions for each
model/mode. This is one accepted pair per model/mode on a shared host. The
inference/Add-training pairs use FP32 then FP16; the final Attention-training
pair uses FP16 then FP32 after a retained memory-pressure failure. It is an
exploratory throughput/capacity observation,
not a causal speedup, stable process mean, convergence or arbitrary-scale claim.
Independent [small-tensor qualification](fp16-qualification-20260929.md)
preceded these full-size runs. Full-size acceptance requires complete windows,
finite results, clean exit and intact records; it does not compare all full-size
gradients against an independent CPU model.

The [adjacent JSON](accelerator-fp16-performance-20260929.json) preserves exact
configs, source/input/binary identities, timing policies and observations,
placement checks, work/memory/loss statistics, raw-record hashes and validation.
Its site path aliases resolve through the corresponding ignored
`artifacts/npu-performance-JOB/float32|float16` links. The matching core is built
at fc2a76f; its complete C++ source hash matches the consumer snapshot. Existing
[FP32 configuration screens](accelerator-performance-20260928.md) remain tied
to their older implementation and timing policy.

## Workload and timing

Both models retain D2048/B512/V50304,465 nodes,2208 logical/4418 physical edges,
seed7 and the original CPU FP32 owner initialization order, HARD signaling,
two body ticks/token, clear policy and full-domain softmax. Attention has
17,269,426,339 parameters; Add has9,468,020,899 (the historical binary-unit
"8.8B" label). FP16 casts the same FP32 seeded fixture. Locality placement and
resident state/KV/messages are used throughout. Each dtype pair holds one
physical allocation and the same node-to-shard mapping; embedding/head use
the same placement policy. Device counts are compute chips, not physical boards.

Runtime: aarch64 Ascend910_9392/A3,64GiB/chip, driver25.3.rc1,
Torch/TorchNPU2.10/CANN9.0, qualified standalone SDK/real loader closure.
Workers16, ATen/inter-op/OpenBLAS1, TASK_QUEUE_ENABLE=0, no exclusive host CPU
reservation. Other workloads remained on the shared host and devices. Before/
after inventories and, for the final attempt, periodic selected-device ownership
observations are retained. These are not exclusive device reservations.
No Python dependency or Trackio link is added to the C++ core.

Inference runs12 growing-context tokens, four warmup and eight measured.
The synchronized token timer includes embedding, graph, vocabulary head,
transfers and barriers; construction, ID creation and terminal validation/
recording are excluded. Values are ms/sample-token: batch token seconds*1000/512,
not single-request latency. The eight positions are not independent repetitions.

Training runs two complete12-token AdamW updates, first warmup and second
measured. Each window starts with empty graph state and retains full history;
parameters/slots continue across updates. The synthetic loss is FP32 mean
cross-entropy with targets `(input_id+1)%vocab`. AdamW uses lr1e-4,
betas0.9/0.999, eps1e-5 and weight decay0.01. FP16 uses FP32 masters/slots,
scale128; FP32 scale1. Timing includes zero_grad, window setup, forward/loss,
backward, unscale, finite guards, optimizer and master-to-payload copies.
Model construction, master/optimizer setup and previous-window graph disposal
are excluded. Executor checks/scalar transfers remain inside timed execution.
Training normalization divides the complete update by512*12 sample-tokens.
Each dtype has only one measured update; its reported zero sample standard
deviation does not establish stable performance.

## Paired observations

External processes were observed on allocated chips in 46 periodic observations of the final Attention attempt. Their timestamps, dtype stage and completed-update counts are retained in the JSON. This is an additional performance confound; successful completion does not establish exclusive device use.

All controls/ranking/events use CPU in these pairs. Inference uses CPU FP64
Read; Add training uses CPU FP32 Read; Attention training uses NPU FP32 Read.
Those choices were screened separately; this is not a repeated seven-way FP16
dispatch sweep. Model payloads remain on NPUs even when Read uses CPU.

| Model/mode | Chips | FP32 ms/sample-token | FP16 ms/sample-token | FP16 time change | FP32 peak GiB/chip | FP16 peak GiB/chip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Add inference | 2 | 17.412729 | 18.703945 | +7.42% | 19.868474 | 9.954680 |
| Attention inference | 4 | 54.611865 | 45.855257 | -16.03% | 22.286432 | 11.192269 |
| Add training | 4 | 39.482509 | 42.135417 | +6.72% | 41.058367 | 47.849997 |
| Attention training | 9 | 105.064709 | 110.660020 | +5.33% | 35.854781 | 40.230472 |

Peak is the maximum single-chip allocator peak over all warmup and measured
observations, not total device-driver memory. Lower payload precision does not
guarantee lower training memory: this implementation retains FP16 parameters/
gradients alongside FP32 masters/master gradients and FP32 optimizer slots.
The Add FP16 masters alone add35.27GiB across its four chips.

Inference distributions below describe the eight growing-context token positions.

| Inference | Dtype | Median | Min | Max | Population stdev |
| --- | --- | ---: | ---: | ---: | ---: |
| Add inference | float32 | 17.388898 | 16.651873 | 17.956683 | 0.410477 |
| Add inference | float16 | 18.542107 | 17.341599 | 20.097953 | 1.137270 |
| Attention inference | float32 | 54.983670 | 50.267647 | 59.478276 | 2.786010 |
| Attention inference | float16 | 45.614821 | 40.572114 | 49.910525 | 2.613783 |

Complete measured training-update phases are seconds; small timing overheads
mean the total need not equal the three named phases exactly.

| Training | Dtype | Update | Forward | Backward | Optimizer | Loss warmup -> measured | Gradient owners / retained slot owners |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| Add training | float32 | 242.580534 | 188.076774 | 53.752517 | 0.739750 | 12.79494095 -> 9.74259663 | 1356 / 1356 |
| Add training | float16 | 258.880002 | 197.853091 | 59.400982 | 1.609573 | 12.79888344 -> 9.74983501 | 1356 / 1356 |
| Attention training | float32 | 645.517575 | 536.783000 | 106.040362 | 2.658847 | 10.86651230 -> 10.22285366 | 2285 / 2310 |
| Attention training | float16 | 679.895161 | 567.715599 | 108.927951 | 3.214976 | 10.86676216 -> 10.22455883 | 2285 / 2310 |

FP16 can change near-tie decisions and therefore actual work despite identical
topology and placement. Work counters are retained for every observation.
For example, Add inference selects16896 events in both dtypes, but mean
candidate counts differ (74997.625 versus75002.75). This is not a claim of
cross-dtype route identity. Small-tensor parity uses a matching-dtype independent
CPU oracle and keeps exact discrete/None checks; it cannot prove convergence.

## Resource changes and record closure

The original eight-chip Attention training plan was cancelled while queued;
no cell started. Its replacement retained the same dimensions/window/batch on
seven chips after additional seven-chip correctness gates passed. Two seven-chip
attempts were then cancelled because unrelated processes entered allocated
physical1. Both had zero complete updates/metric observations, native exit-15,
wrapper143 and no residual descendant. Only project services were stopped.
These are resource interruptions, not capacity failures or successful cells.
The user initially chose to wait, then authorized a six-chip attempt with the
seven-chip plan retained as fallback. Six-chip FP32 failed at logical0/physical2
while an external process was present:66MiB requested,36.88GiB allocated,
38.04GiB reserved, only34.63MiB free of61.27GiB usable capacity. This is not a
clean intrinsic six-chip capacity rejection. Its automatically started FP16
child was cancelled; neither completed an update and both terminal records have
no remaining descendants. The successful six-chip correctness gates remain valid.

The user then authorized more chips, explicit sharing and autonomous opportunistic
allocation to prioritize completion. An eleven-chip waiter was cancelled before
any gate/cell when memory availability fell. The first shared nine-chip attempt
held physical0,2,3,4,5,11,12,13,14. FP32 passed both updates (measured717.549002s,
116.788575ms/sample-token, peak35.854233GiB/chip). FP16 passed a722.635070s
warmup, then failed in its measured update on logical0/physical0:82MiB requested,
39.39GiB allocated,41.19GiB reserved and47.53MiB free of61.27GiB usable capacity.
Other processes remained on that device. The failed FP16 cell has no measured
timing; its warmup and successful FP32 sibling are retained separately from the
accepted comparison. This is not a clean exclusive nine-chip capacity limit.

The successful retry replaces physical0 with physical1, which was idle at admission, keeping
nine logical shards, the same logical placement and every workload setting.
It deliberately runs FP16 first to expose the failing capacity path before
repeating FP32 on the same allocation. The prior eight nine-chip correctness
gates are reused at the unchanged source; no additional gate coverage is claimed.
Shared admission uses exact physical1,2,3,4,5,11,12,13,14, permits existing
processes and requires healthy chips, used HBM at most20480MiB, two snapshots
and cooperative locks; sampled AiCore utilization is allowed up to100%.
This is a task-local explicit eligibility exception, not a change to the public
runtime or the installed queue helper. Such admission cannot prevent later
allocations by unrelated tasks or guarantee exclusive memory/compute.
Exact attempt identities, reasons and record hashes remain in the adjacent JSON.
The 9-chip correctness gates and both unchanged full-size retry cells passed.
The seven-chip fallback reached its queue-wait limit without starting a cell; its terminal state remains failed.
The final device count is stated in the performance table;
its timings must not be compared directly with the old eight-chip row as
evidence of a dtype or software speedup.

Every pair passed source/topology/parameter-count/node-shard/clean-exit checks.
All eight terminal raw records passed schema, finite-metric and lifecycle
validation and have no remaining child processes. Every Trackio SQLite step
and metric matches its authoritative JSONL. Trackio0.35.0 used best-effort local
project tide-npu-performance with healthy recording; public logging offers no
per-event durable acknowledgement, so the explicit read-only database comparison
is retained separately. No dashboard was needed or exposed.

## Interpretation and support boundary

The observations support a per-workload precision choice, not a blanket FP16
speedup. Host orchestration, scalar extraction/barriers, small tensor operations,
Read/root replay, transfers and FP32 master handling are possible contributors.
The completed timing phases quantify their combined effect but do not isolate
full-size hardware bottlenecks. The separate [profiling analysis](accelerator-profile-analysis-20260929.md)
finds int64 Sort on AiCPU in an older tiny all-device trace; its14.62% task-time
fraction is neither end-to-end share nor a full-size bottleneck estimate.
The two new tiny FP16 Python/native traces have hardware kernels and no AiCPU
type or named CPU fallback, while host scalar extraction remains present. They
use a different scheduler/workload from the old standalone trace; this does not
show that changing dtype removes int64 Sort from AiCPU.

NPU ranking and event-key tensor implementations exist, but histories, returned
indices and dispatch remain host-owned. This is single-process model sharding
with device copies, not a fully device-resident control loop or DDP/HCCL.
Explicit transfer counters are requested bytes, not measured physical fabric
traffic. None of these profiles proves that most arithmetic uses AiCPU.

New FP16 qualification/performance covers TorchNPU2.10/CANN9.0 only; older
CANN8.5.0/.1/.2 retain their previous FP32 qualification. CUDA has a compiled
source path and host checks, but real GPU gates and other architectures/stacks
remain target-machine work. Standalone public owner checkpoints/NamedOptimizer
remain FP32/FP64; this full-size consumer owns masters and no checkpoint format.

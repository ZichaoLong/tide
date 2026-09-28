# Full-size NPU performance selection

The bounded local assessment is complete: all **54 new full-size cells** passed
on immutable implementation `b4f26b3659cec0549e18560a1a996d5bee0cae6d`.
It includes both seven-way inference/training screens, 2/4/8-device placement
measurements, three warmed training processes per model, three matched inference
pairs per model and one concurrent Add2/Attention4 workflow. The independent
[CPU/two/four/eight-device correctness gates](accelerator-dispatch-training-20260928.md)
preceded performance. Their small-tensor finite scope remains distinct from these
full-size workload results. Historical capacity failures remain below.
Full-size acceptance checks complete windows, finite outputs/loss and process/record
lifecycle; the linked reduced-tensor gates supply independent observable/VJP
comparisons.

The [record manifest](accelerator-performance-20260928.json) preserves exact
configurations, input/binary/record hashes, process observations, physical groups,
memory, transfers, decisions and terminal states. Raw records are under the
corresponding `artifacts/npu-performance-JOB/CASE` links. This standalone LibTorch
consumer does not change the foundation's public single-device API or defaults.
Performance here uses TorchNPU 2.10/CANN 9.0; the other three CANN stacks have
[separate correctness qualification](standalone-sdk29-20260928.md).

The consumer was built on the frozen implementation. It reuses the previously
qualified core: that build's original manifest revision is retained, while its
C++/CMake content hash matches the frozen tree exactly. See the
[core reuse qualification](accelerators-20260928.md). The final manifest records
both identities and the current consumer executable hash.

## Workload and interpretation


Both models retain the historical 465-node, 2208-logical/4418-physical-edge graph,
D2048/B512/V50304, FP32 payload, CPU seed7 in original owner order, HARD signaling,
two body ticks, clear state policy, all-softmax and four heads. Attention has
17,269,426,339 parameters; Add has 9,468,020,899 (the historical binary-unit
"8.8B" label). No shape, precision or history-window reduction hides failures.

Device counts refer to exposed NPU compute chips (`npu-smi` Phy-ID values);
the host has eight boards exposing sixteen such devices.

The host is aarch64 with Ascend910_9392/A3, 64 GiB per chip, driver25.3.rc1,
Torch/TorchNPU2.10 and CANN9.0. The standalone SDK and actual loader closure were
qualified independently. TASK_QUEUE_ENABLE=0 is the recorded workaround for the
previous cross-device backward failure. There are 16 requested node workers,
one ATen/inter-op/OpenBLAS thread, and no CPU affinity isolation. Device workers
are bounded by the device count. Unrelated host/device workloads remain present.

Inference processes 12 growing-context tokens, four warmup and eight measured.
Every timed token includes embedding, graph, vocabulary head, transfers and
device barriers. Construction, ID creation, previous-logit disposal and
final-logit validation/metrics are outside this timer. The reported ms/sample-token divides
the synchronized batch token time by512; it is not single-request latency.
The eight token positions are a series with growing context, not eight
independent fresh-process repetitions.

Training uses a complete 12-token sequence window from empty graph state,
mean cross-entropy targets `(input_id+1)%vocab`, full backward and AdamW
(lr1e-4, betas0.9/0.999, eps1e-5, weight decay0.01). Parameters and optimizer
slots survive successive updates; autograd/KV history stays connected throughout
each window. Synchronized update timing includes zero_grad, window setup, IDs,
forward/loss, backward and optimizer. Construction, previous-window graph disposal
and terminal loss validation/metrics are outside timing. These are synthetic workload timings,
not continuous-stream training or convergence evidence.

The exclusion of validation refers to the entrypoint's terminal output/loss
checks. Executor guards remain timed: `Resident::read` checks each descriptor's
device and finite value, with a scalar extraction for each NPU descriptor. Its
training path also retains scalar-root semantic replay, and differentiable
transfers preserve separate roots to distinguish absent gradients from connected
zeros. These choices add implementation and synchronization costs. The results
do not measure a hardware limit or an optimized fully device-resident executor.


## Scoring and dispatch screens

All cells use resident payload/state/message transport and locality placement.
`model` means the model's NPU. Each screen has one fresh process per configuration;
inference and Add training exploration overlapped other project work. These are
descriptive screens, not matched causal speedup estimates. Units throughout the
following table are **ms/sample-token**. Training columns are one cold complete
update, not warmed throughput.

| Name | Read / precision | Controls | Ranking | Events | Add2 inference | Attention4 inference | Add4 cold training | Attention8 cold training |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu64 | CPU / FP64 | CPU | CPU | CPU | 17.110 | 56.671 | 42.742 | 138.488 |
| cpu32 | CPU / FP32 | CPU | CPU | CPU | 18.419 | 49.512 | 41.679 | 140.461 |
| mixed32 | model / FP32 | CPU | CPU | CPU | 28.282 | 63.671 | 48.622 | 122.193 |
| score32 | model / FP32 | model | CPU | CPU | 39.415 | 59.956 | 57.142 | 130.592 |
| rank32 | model / FP32 | model | model | CPU | 35.383 | 60.560 | 53.073 | 155.031 |
| queue32 | model / FP32 | model | CPU | model | 40.650 | 71.442 | 56.014 | 138.272 |
| all32 | model / FP32 | model | model | model | 34.933 | 67.420 | 57.273 | 130.665 |

The fastest observed FP32 inference screen candidate was CPU32 for both models.
Cold training selected CPU32 for Add and the recorded Attention finalist below.
The training selector chose the smallest completed FP32 cold-update time, with
all candidate summary hashes retained. This selection does not establish that
the same configuration wins every warmed comparison or every workload.

All seven Add cold updates report 1356 gradient/optimizer-state owners; all seven
Attention cold updates report 2280. No loss, route or gradient tolerance was
changed to accept performance. Similar full-size losses do not replace the
independent full-observable and isolated-VJP correctness gates.

Int64 ranking/event sorting uses card-local AiCPU in the operator traces, while
FP32 reductions and softmax use accelerator cores. Histories, tensor handles and
C++ dispatch remain host-owned. `all32` eliminates the explicit timed vector-D2H
counter, but does not imply a fully device-resident control loop. Scalar extraction
and other implicit synchronization remain possible.

Transfer counters cover explicit client `Transfer` requests and metadata paths.
They exclude autograd reverse copies, scalar extractions and vendor-internal
transport. Synchronized times include such work when it occurs inside the timed
phase. Remote bytes describe logical forward payload, not measured fabric traffic.

Attention's CPU64 / CPU32 / mixed32 cold updates request 12555.781 / 12555.781 / 3.065 MiB of explicit D2H transfer per 12-token window. Their host peak RSS spans 97.132..97.310 GiB; these records do not establish a material host-memory saving from moving Read.

## Matched inference and configuration choice

The same physical subset was held for each model throughout three fresh process
pairs, with CPU64/CPU32 ordered AB/BA/AB. No other heavy project workload overlapped
these timings; unrelated host workloads remained. Each number below is a process
mean over eight measured growing-context tokens. The token positions are not
independent repetitions.

| Model | NPU devices | Read | Three process means | Median | Range |
| --- | --- | --- | --- | --- | --- |
| Add | 2 | cpu64 | 16.947 / 17.097 / 17.153 | 17.097 | 16.947..17.153 |
| Add | 2 | cpu32 | 16.924 / 16.975 / 17.427 | 16.975 | 16.924..17.427 |
| Attention | 4 | cpu64 | 54.802 / 53.377 / 48.298 | 53.377 | 48.298..54.802 |
| Attention | 4 | cpu32 | 53.169 / 56.962 / 54.756 | 54.756 | 53.169..56.962 |

The rule recorded before these measurements chooses CPU32 for the concurrent
pair only if all paired directions agree and its three-process range lies wholly
below CPU64; otherwise it retains CPU64. This is a conservative descriptive rule,
not a significance test or a universal claim. It selected **Add cpu64, Attention cpu64**. Public defaults remain CPU FP64 Read and CPU controls/dispatch.

## Placement and descriptive scaling

All rows use CPU FP32 Read and CPU controls/dispatch. These single observations
have different physical assignments and contention; they do not establish causal
scaling factors.

| Model | NPU devices | Placement | ms/sample-token | Cut edges | Remote MiB/token |
| --- | --- | --- | --- | --- | --- |
| Add | 2 | locality | 18.419 | 600 | 129.321 |
| Add | 4 | memory | 14.442 | 3444 | 512.504 |
| Add | 4 | locality | 16.609 | 898 | 169.346 |
| Add | 8 | locality | 12.284 | 1556 | 214.421 |
| Attention | 2 | locality | 73.028 | 616 | 111.724 |
| Attention | 4 | memory | 43.533 | 3370 | 485.726 |
| Attention | 4 | locality | 49.512 | 1040 | 182.480 |
| Attention | 8 | locality | 38.268 | 1306 | 228.267 |

Locality reduces Add4 remote payload by about 67%, but its observed time is higher.
Parameter loads span 8.003..9.673 GiB with locality versus 8.626..9.010 GiB with
memory balancing. Cut reduction, load balance and execution overhead must be
evaluated together. This finite assessment does not exhaust placement choices.

## Warmed complete training

Each selected configuration ran in three fresh processes on a fixed physical
group. Each process executed two complete 12-token AdamW updates; only the second
was measured, after the first complete update warmed the path and created slots.

| Model | NPU devices | Choice | Process | ms/sample-token | Update seconds | Forward seconds | Backward seconds | Optimizer seconds | Peak GiB/chip |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Add | 4 | cpu32 | 1 | 40.404 | 248.242 | 192.732 | 54.313 | 1.189 | 41.058 |
| Add | 4 | cpu32 | 2 | 43.123 | 264.946 | 200.026 | 63.734 | 1.173 | 41.070 |
| Add | 4 | cpu32 | 3 | 42.080 | 258.537 | 203.453 | 53.877 | 1.188 | 41.065 |
| Attention | 8 | mixed32 | 1 | 115.678 | 710.728 | 601.571 | 107.169 | 1.954 | 39.775 |
| Attention | 8 | mixed32 | 2 | 114.587 | 704.021 | 593.383 | 108.231 | 2.370 | 39.771 |
| Attention | 8 | mixed32 | 3 | 137.875 | 847.103 | 707.437 | 135.854 | 3.763 | 39.780 |

Add: median **42.080 ms/sample-token**, range 40.404..43.123, mean 41.869, population stdev 1.120. Median complete update 258.537 seconds; measured loss 9.74259663..9.74259758. Measured gradient-owner counts: 1356; retained optimizer-state-owner counts: 1356. A single count denotes all three processes; otherwise counts follow process order.


Attention: median **115.678 ms/sample-token**, range 114.587..137.875, mean 122.713, population stdev 10.730. Median complete update 710.728 seconds; measured loss 10.22285366..10.22285461. Measured gradient-owner counts: 2285; retained optimizer-state-owner counts: 2310. A single count denotes all three processes; otherwise counts follow process order.


Persistent optimizer slots raise the Add measured peak above its first update's
38.177 GiB. Raw records retain per-device memory; the manifest retains per-process
device maxima, host RSS and all phase observations. Active gradient-owner sets can change after a parameter
update, while previously established optimizer slots remain stored; the retained
slot count can therefore exceed the current gradient-owner count. A missing
gradient still skips that owner's update/decay, as qualified independently.
These two-update measurements are synthetic throughput,
not convergence, continuous-stream training or a warmed cross-configuration ranking.

## Controlled concurrent workflow

One Add2 process and one Attention4 process used the same disjoint physical groups
as matched inference, with the selected configurations above. Both children passed
all 12 tokens and their record/cleanup checks.

| Model | Choice | Measured ms/sample-token while concurrent |
| --- | --- | --- |
| Add | cpu64 | 17.138 |
| Attention | cpu64 | 50.652 |

The finite workflow completed **12,288 sample-tokens in 479.402 seconds**, or **25.632 sample-tokens/second**. This timer runs from starting both wrappers through both exits, including
construction, all 12 tokens, teardown and recording; outer queue wait and
inventory probes are excluded. It is not directly comparable with the warmed
per-token latency above. One pair does not establish a causal parallel speedup.


The first-to-last token-event envelopes overlap by 3 seconds. Event timestamps have one-second precision and include intervening work;
the manifest retains both envelopes and physical groups.

This records overlapping token-processing periods, not exact simultaneous
kernel intervals.

## Retained capacity evidence and record closure

The earlier two-card Attention model-FP32 Read/control grad-forward case failed
on token 7 with 58.66 GiB allocated, 61.05 GiB reserved and 1.71 MiB free when a
2 MiB allocation was requested. Its [failure evidence](accelerator-scoring-pilots-20260928.md)
remains intact; no unchanged retry is represented as success.

The four-card follow-up `pilot-attention-g1-n4-a3` passed 12 tokens/eight measured
samples at 136.025 ms/sample-token and 37.535 GiB peak allocation. Its distinct
source is `a12ee0c676faa80096db4034595b581db11a5f22`. This is a concurrent
grad-forward pilot with NPU FP32 Read/controls, separate from complete training.

The original `train-screen-attention-a4` waiter was cancelled before any benchmark
cell started when an external workload prevented its eight-chip allocation.
Its exit 143 and cancellation record remain; the completed replacement uses the
distinct name `train-screen-attention-a4b`.

All 54 new terminal records passed schema, finite-metric, identity and lifecycle
validation, reported healthy best-effort local Trackio recording, and left no
child process. Trackio has no per-event durable acknowledgement; raw records are
authoritative. No required performance workload remains live or queued. This
closes the finite local assessment; NVIDIA/CUDA and new host/software combinations
still require their own [target-machine gates](../accelerators.md#target-machine-acceptance).

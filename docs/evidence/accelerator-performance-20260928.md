# Full-size NPU performance selection

The completed exploratory screens below use immutable implementation
`b4f26b3659cec0549e18560a1a996d5bee0cae6d`, after the independent
[CPU/two/four/eight-device correctness gates](accelerator-dispatch-training-20260928.md).
All 28 reported cells passed, including three warmed Add training repetitions.
Matched repetitions, eight-device measurements and Attention full training remain
in [STATUS](../STATUS.md).
This report does not mark that remaining assessment complete.

The [record manifest](accelerator-performance-20260928.json) preserves each
configuration, exact input/binary/record hashes, raw timing observations,
physical allocation, memory, transfers and terminal state. Raw records are under
the corresponding `artifacts/npu-performance-JOB/CASE` links. This is a standalone
LibTorch consumer; it does not change the public single-device foundation API.

## Workload and interpretation

Both models retain the historical 465-node, 2208-logical/4418-physical-edge graph,
D2048/B512/V50304, FP32 payload, CPU seed7 in original owner order, HARD signaling,
two body ticks, clear state policy, all-softmax and four heads. Attention has
17,269,426,339 parameters; Add has 9,468,020,899 (the historical binary-unit
"8.8B" label). No shape, precision or history-window reduction hides failures.

The host is aarch64 with Ascend910_9392/A3, 64 GiB per chip, driver25.3.rc1,
Torch/TorchNPU2.10 and CANN9.0. The standalone SDK and actual loader closure were
qualified independently. TASK_QUEUE_ENABLE=0 is the recorded workaround for the
previous cross-device backward failure. There are 16 requested node workers,
one ATen/inter-op/OpenBLAS thread, and no CPU affinity isolation. Device workers
are bounded by the device count. Unrelated host/device workloads remain present.

Inference processes 12 growing-context tokens, four warmup and eight measured.
Every timed token includes embedding, graph, vocabulary head, transfers and
device barriers. Construction, ID creation, previous-logit disposal and
validation/metrics are outside this timer. The reported ms/sample-token divides
the synchronized batch token time by512; it is not single-request latency.
The eight token positions are a series with growing context, not eight
independent fresh-process repetitions.

Training uses a complete 12-token sequence window from empty graph state,
mean cross-entropy targets `(input_id+1)%vocab`, full backward and AdamW
(lr1e-4, betas0.9/0.999, eps1e-5, weight decay0.01). Parameters and optimizer
slots survive successive updates; autograd/KV history stays connected throughout
each window. Synchronized update timing includes zero_grad, window setup, IDs,
forward/loss, backward and optimizer. Construction, previous-window graph disposal
and validation/metrics are outside timing. These are synthetic workload timings,
not continuous-stream training or convergence evidence.

## Scoring and dispatch screens

All cells use resident payload/state/message transport and locality placement.
`model` means the model's NPU. These are concurrently observed exploratory
results, with one fresh process per cell; they cannot establish causal speedups.

| Name | Read / precision | Controls | Ranking | Events | Add2 inference | Attention4 inference | Add4 cold training |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| cpu64 | CPU / FP64 | CPU | CPU | CPU | 17.110 | 56.671 | 42.742 |
| cpu32 | CPU / FP32 | CPU | CPU | CPU | 18.419 | 49.512 | 41.679 |
| mixed32 | model / FP32 | CPU | CPU | CPU | 28.282 | 63.671 | 48.622 |
| score32 | model / FP32 | model | CPU | CPU | 39.415 | 59.956 | 57.142 |
| rank32 | model / FP32 | model | model | CPU | 35.383 | 60.560 | 53.073 |
| queue32 | model / FP32 | model | CPU | model | 40.650 | 71.442 | 56.014 |
| all32 | model / FP32 | model | model | model | 34.933 | 67.420 | 57.273 |

All three result columns are ms/sample-token. Training here is **one cold full
update**, not warmed throughput. CPU FP32 is the fastest completed FP32 candidate
for each screen, so it advances to repeated measurements. The CPU FP64 reference
and public defaults remain available; this screen alone does not justify a
default change or a claim that CPU FP32 is consistently faster than CPU FP64.

For the Add CPU FP32 cold update, forward/backward/optimizer take
191.218/62.352/2.502 seconds, total256.073 seconds. Peak allocated memory is
38.177 GiB on the most-loaded chip. All seven Add training cells report1356
gradient and optimizer-state owners; disconnected owners remain absent. The
losses differ only at about the1e-6 scale in this one update, which is descriptive
and does not replace independent gradient/route correctness qualification.

Putting these kernels on NPU does not make the whole control loop device
resident. Exact int64 sorting is executed on card-local AiCPU in the collected
traces; FP32 reductions and softmax use accelerator cores. Integer histories,
tensor handles and C++ dispatch remain host-owned. `all32` reports zero explicit
timed vector D2H copies, but still exchanges metadata, and validation/metrics
outside timing can copy tensors to CPU. Scalar extractions are not represented
by the explicit vector-copy counter.

## Placement and descriptive scaling

All rows below use CPU FP32 Read/controls/dispatch. A single observation per
placement/count, different physical assignments and overlapping workloads prevent
a causal scaling claim. The eight-card cells remain pending.

| Model | Cards | Placement | ms/sample-token | Cut edges | Remote MiB/token |
| --- | ---: | --- | ---: | ---: | ---: |
| Add | 2 | locality | 18.419 | 600 | 129.321 |
| Add | 4 | memory | 14.442 | 3444 | 512.504 |
| Add | 4 | locality | 16.609 | 898 | 169.346 |
| Attention | 2 | locality | 73.028 | 616 | 111.724 |
| Attention | 4 | locality | 49.512 | 1040 | 182.480 |
| Attention | 4 | memory | 43.533 | 3370 | 485.726 |

Locality cuts the observed Add4 remote message volume by about67%, yet this
measurement is slower. Its parameter loads span8.003..9.673 GiB, compared with
8.626..9.010 GiB for memory balancing. Communication reduction, load balance and
host/dispatch overhead must be evaluated together; cut count is not a latency
objective by itself. This finite screen does not tune or exhaust all placements.

## Warmed Add training

The selected CPU FP32 Read/CPU control/ranking/event configuration passed three
fresh processes on the same four-chip allocation. Each performed two complete
12-token AdamW updates, with the first update excluded as warmup.

| Fresh process | Measured ms/sample-token | Full update seconds | Forward seconds | Backward seconds | Optimizer seconds | Peak allocated GiB/chip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 40.404 | 248.242 | 192.732 | 54.313 | 1.189 | 41.058 |
| 2 | 43.123 | 264.946 | 200.026 | 63.734 | 1.173 | 41.070 |
| 3 | 42.080 | 258.537 | 203.453 | 53.877 | 1.188 | 41.065 |

Median42.080 ms/sample-token, range40.404..43.123; mean41.869, population
stdev1.120 across three process means. The full-update median is258.537 seconds.
Every measured update has1356 gradient and optimizer-state owners and loss
9.7425966..9.7425976. This two-update synthetic loss change is not convergence
or quality evidence. The retained optimizer slots raise the measured allocation
peak above the first update's38.177 GiB, to at most41.070 GiB. Other host workloads
were present; no causal improvement over the cold screen is inferred.

## Retained capacity evidence

The earlier two-card Attention model-FP32 Read/control grad-forward case failed
on token7 with58.66 GiB allocated,61.05 GiB reserved and1.71 MiB free when a2 MiB
allocation was requested. Its [failure evidence](accelerator-scoring-pilots-20260928.md)
remains intact; no unchanged capacity retry is disguised as success.

The four-card follow-up `pilot-attention-g1-n4-a3` passed all12 tokens/eight
measured samples at136.025 ms/sample-token and37.535 GiB maximum per-device peak
allocation. Its source is `a12ee0c676faa80096db4034595b581db11a5f22`; the manifest
records this distinct identity. It is a concurrent **grad-forward** pilot with
NPU FP32 Read/controls, not complete training or matched scaling evidence.

All28 new terminal records passed schema/finite-metric/lifecycle validation,
reported healthy best-effort local Trackio recording, and left no child process.
Trackio has no per-event durable acknowledgement; raw records are authoritative.
Live and queued work, including resource-only cancellations before any benchmark
cell starts, is recorded separately in STATUS.

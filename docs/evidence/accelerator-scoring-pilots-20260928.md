# Full-size NPU FP32 Read/control pilots

These four pilots tested immutable implementation
`a12ee0c676faa80096db4034595b581db11a5f22` after its
[CPU/two-NPU correctness qualification](accelerator-scoring-20260928.md).
The [record manifest](accelerator-scoring-pilots-20260928.json) preserves raw
artifact hashes, timestamps, dispersion, device assignments and terminal states.

## Workload and completed results

All cases use the historical 465-node topology, 2208 logical/4418 physical edges,
D2048/B512/V50304, FP32 payload and NPU FP32 Read/softmax controls, locality
placement and resident state/message transport. CPU initialization retains seed7
and the original owner/RNG order. There are two body ticks per token and twelve
growing-context tokens: four warmup and eight measured. The bound is 1800 seconds
of native process time and 512 GiB host RSS. Ascend910_9392, TorchNPU2.10/CANN9.0,
driver25.3.rc1, TASK_QUEUE_ENABLE=0. Each case uses two NPUs.

| Model | Mode | Terminal result | Mean ms/sample-token | Max per-device peak allocation |
| --- | --- | --- | ---: | ---: |
| Add 9,468,020,899 parameters | no-grad | passed, 12/8 observations | 41.6797 | 19.874 GiB |
| Attention 17,269,426,339 parameters | no-grad | passed, 12/8 observations | 100.0539 | 42.653 GiB |
| Add 9,468,020,899 parameters | grad-forward | passed, 12/8 observations | 64.5930 | 27.252 GiB |
| Attention 17,269,426,339 parameters | grad-forward | failed, NPU OOM | incomplete | see failure below |

The metric is synchronized batch token duration divided by 512, not latency of
one request. Timings include embedding, body, head, CPU coordination and timed
transfers; construction, ID creation, previous-logit disposal and metrics are
outside timing. Add's historical "8.8B" label uses binary units.
All three successful records have native exit0 and complete windows. All four
records, including the failed one, passed record validation; Trackio was healthy,
and no child process remained. This is standalone LibTorch execution linking the
installed core, without changing public graph semantics.

## Capacity failure and comparison limits

Attention grad-forward completed tokens0..6 and failed during token7 (the eighth
token). Logical device0 reported 58.66 GiB allocated, 61.05 GiB reserved and
1.71 MiB free when a 2 MiB allocation failed. No history detach is applied;
autograd and KV history grow with the window. This is a capacity failure of this
placement/window. It is neither an equivalence failure nor proof that every
two-device placement fails. Its partial timings are retained in raw records but
are not presented as a completed performance result.

Token processing overlapped across the successful pilots; the manifest preserves
first/last observation timestamps. These are individual concurrent pilots with
possible shared host-resource contention, not matched scaling repetitions or
controlled aggregate-throughput results. Earlier CPU FP64 Read pilots use
different cards and/or contention and cannot establish a scoring-placement
speedup. All current modes omit backward and optimizer updates; grad-forward is
not a complete training step.

The explicit vector D2H counter is zero, but CPU ranking still extracts scalars;
metrics/validation copies outside timing are not covered by that counter.
States/KV/messages reside on NPUs and cross-shard transfers use D2D. Discrete
node ranking, count histories and event scheduling remain CPU.

## CPU-control exploratory follow-up

A fresh two-device Add no-grad run (`pilot-add-g0-n2-cpuctrl-a3`) keeps NPU FP32
Read and moves FP32 softmax controls to CPU. It passed all twelve tokens/eight
measured observations, native exit0, healthy tracking and record validation,
with no remaining children. Mean27.9618 ms/sample-token, maximum per-device peak
allocation19.870 GiB. Source and other workload settings remain unchanged.
Its contention and device assignment are not matched to the earlier pilots;
this number alone does not establish a CPU-control speedup. Raw record hashes
and the full timing distribution are in the evidence manifest.

The four-device Attention capacity follow-up and its live state are in
[STATUS](../STATUS.md); the remaining acceptance work is in [ROADMAP](../ROADMAP.md).

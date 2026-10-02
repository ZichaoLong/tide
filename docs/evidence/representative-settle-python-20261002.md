# Representative Settle Python complete-flow comparison

Fixed source `80dae6e14d41614d0cdb1056bb39b57ca10d07ed`. Both Settle/Python
schedules passed: 40 screening and 72 confirmation processes across eight jobs.
[Audit, individual samples, ranges and receipts](representative-settle-python-20261002.json).
Together with [PDG](representative-pdg-matrix-20261002.md),
[TimedDAG LibTorch prefill](representative-host-policy-20261002.md) and
[the other five submatrices](representative-family-matrix-20261002.md), this
completes all ten required representative family/client/schedule submatrices.
Full-size and later memory optimizations are separate acceptance work.

Each common rank-aligned v2 packet has 128 body nodes/544 edges,
D128/B8/T4/V257, 8,995,632 Add or 17,384,240 Attention parameters, clear=true.
Each of five FP32 presets is screened once. CPU, the selected mixed preset and
resident then receive three fresh processes per workload. Each process has one
continued warmup and three measured complete steps, with two windows/64 input
tokens per step. Complete training includes loss, VJP, finite checks and AdamW.
Candidates independently consume inputs, parameters and continuation; the CPU
reference does not supply their events or decisions.

Python CPU/mixed use the default host scheduler; resident is the native C++/CANN
client in a Python-owned runtime. ATen/BLAS use one thread. This is not a comparison
against the tuned 16-worker packed LibTorch CPU baseline. The dense, unsliced
consumer predates subsequent memory optimizations. Own heavy measurements were
serial on a shared server; queue, process and parent bounds remain in the records.
No measured child was interrupted during the parent boundary holds.

Seconds below are medians of three process medians. Relative throughput is
CPU elapsed time divided by resident elapsed time: above 1 means faster resident
throughput. Cold ratios use outer process wall time, including startup and
construction as well as the same warmup and measured steps.

| Schedule | Memory/work | CPU seconds | Selected mixed seconds | Resident seconds | Warm resident/CPU | Cold resident/CPU |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| prefill | add-inference | 1.158286 | mixed-a 3.205247 | 0.173105 | 6.691× | 0.870× |
| prefill | add-training | 2.250212 | mixed-a 6.552147 | 0.391452 | 5.748× | 1.060× |
| prefill | attention-inference | 2.510821 | mixed-b 7.133758 | 0.283777 | 8.848× | 1.195× |
| prefill | attention-training | 6.148797 | mixed-a 14.481279 | 1.046711 | 5.874× | 1.892× |
| streaming | add-inference | 1.169841 | mixed-a 3.592898 | 0.433651 | 2.698× | 0.834× |
| streaming | add-training | 2.500623 | mixed-b 7.282921 | 0.740281 | 3.378× | 1.046× |
| streaming | attention-inference | 2.536939 | mixed-a 7.209269 | 0.554039 | 4.579× | 1.106× |
| streaming | attention-training | 6.180842 | mixed-a 15.136188 | 1.436353 | 4.303× | 1.647× |

Resident wins all warm comparisons here (2.698–8.848×), while cold Add inference
is slower. Cold Add training is only 1.046–1.060×; these short shared-server
measurements are not evidence of a robust startup-inclusive advantage.
Every selected mixed flow is slower than the Python CPU flow. The bounded pilot
does not establish a globally optimal mixed configuration.

Outputs (64/step), event counts and final cuts match the independent comparison;
loss differences are below 1e-5. Resident allocator peaks remain within their
estimates. No CPU-fallback warning was observed. This audit adds no operator
profile and makes no AiCPU attribution. Original wide execution, family-specific
stress graphs, FP16 performance, CUDA and other CANN versions are outside this
report's finite scope.

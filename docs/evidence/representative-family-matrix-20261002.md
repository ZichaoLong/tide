# Representative TimedDAG and Settle complete-flow comparison

Fixed source `80dae6e14d41614d0cdb1056bb39b57ca10d07ed`. Five completed
submatrices add100screening and180confirmation processes.
[Audit, all samples and ranges](representative-family-matrix-20261002.json).
This is the earlier dense, unsliced consumer, not a measurement of later memory
optimizations or original wide execution. [PDG](representative-pdg-matrix-20261002.md)
and [TimedDAG LibTorch prefill](representative-host-policy-20261002.md) are separate
completed evidence; Settle Python remains pending at this report's publication.

Each rank-aligned v2 packet has128body nodes/544edges,D128/B8/T4/V257,
8,995,632Add or17,384,240Attention parameters,clear=true. Five FP32 presets are
screened once; CPU, selected mixed and resident then receive three fresh
confirmation processes per workload, with one continued warmup and three
measured complete steps. A step spans two windows/64input tokens. Training
includes loss, VJP, finite checks and AdamW. Candidates execute independently;
reference execution and profiling are outside their timers.

LibTorch uses16CPU workers or4mixed workers with packed sources/batch-next.
Python uses its default single host scheduler and ordinary transport; resident
is the public native C++/CANN client in a Python-owned runtime. ATen/BLAS threads
are1. These policies differ, so comparing ratios across language rows does not
isolate a language effect. Own heavy measurements are serial on a shared server;
queue120s, child900s and cell120s bounds are retained. Two final TimedDAG Python
streaming confirmations used a new parent/device lease after an earlier parent
was cancelled at a completed-measurement boundary; no measured process was
interrupted or reused as a completed result.

Values are seconds per complete step, median of three process medians. Relative
throughput is CPU time/resident time; above1 means resident is faster. Full
process distributions, cold timings and memory observations are in the audit.

| Family/client | Schedule | Memory/work | CPU | Selected mixed | Resident | Resident/CPU throughput |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| timed-dag/libtorch | streaming | add-inference | 0.141278 | mixed-a 1.150564 | 0.419239 | 0.337× |
| timed-dag/libtorch | streaming | add-training | 0.618438 | mixed-c 3.032909 | 0.711715 | 0.869× |
| timed-dag/libtorch | streaming | attention-inference | 0.265373 | mixed-c 2.470875 | 0.539617 | 0.492× |
| timed-dag/libtorch | streaming | attention-training | 1.210626 | mixed-b 5.645675 | 1.412694 | 0.857× |
| settle/libtorch | prefill | add-inference | 0.134317 | mixed-a 1.113169 | 0.158612 | 0.847× |
| settle/libtorch | prefill | add-training | 0.552598 | mixed-a 2.785277 | 0.362827 | 1.523× |
| settle/libtorch | prefill | attention-inference | 0.256333 | mixed-a 2.464130 | 0.274964 | 0.932× |
| settle/libtorch | prefill | attention-training | 1.178398 | mixed-a 5.088511 | 1.018486 | 1.157× |
| settle/libtorch | streaming | add-inference | 0.143414 | mixed-a 1.153513 | 0.418392 | 0.343× |
| settle/libtorch | streaming | add-training | 0.634455 | mixed-b 3.415593 | 0.716437 | 0.886× |
| settle/libtorch | streaming | attention-inference | 0.265242 | mixed-c 2.454600 | 0.531788 | 0.499× |
| settle/libtorch | streaming | attention-training | 1.425247 | mixed-a 5.763370 | 1.414569 | 1.008× |
| timed-dag/python | prefill | add-inference | 1.106111 | mixed-b 3.455602 | 0.173057 | 6.392× |
| timed-dag/python | prefill | add-training | 2.244040 | mixed-a 6.682723 | 0.388522 | 5.776× |
| timed-dag/python | prefill | attention-inference | 2.508108 | mixed-a 6.850528 | 0.288871 | 8.682× |
| timed-dag/python | prefill | attention-training | 6.071561 | mixed-b 15.420298 | 1.046681 | 5.801× |
| timed-dag/python | streaming | add-inference | 1.145363 | mixed-a 3.635887 | 0.436057 | 2.627× |
| timed-dag/python | streaming | add-training | 2.455934 | mixed-a 6.918707 | 0.741498 | 3.312× |
| timed-dag/python | streaming | attention-inference | 2.526734 | mixed-a 7.036203 | 0.557184 | 4.535× |
| timed-dag/python | streaming | attention-training | 6.457174 | mixed-a 15.386267 | 1.461120 | 4.419× |

With the tuned LibTorch CPU baseline, resident prefill wins Settle training;
CPU wins its inference and all TimedDAG streaming comparisons. Settle streaming
Attention training is effectively tied:1.008× with CPU process medians ranging
1.262–1.548s, overlapping resident1.410–1.415s. That ratio does not justify a
stable speedup claim. CPU wins every LibTorch cold-process comparison here.

Against the Python default CPU workflow, resident wins all warm comparisons:
2.627–8.682×. Its construction/startup costs remain material; cold Add inference
is slower for both schedules. Mixed candidates are slower than CPU in all these
rows; the bounded pilot does not establish a globally optimal mixed setting.

Outputs64/step, observed event counts and final cuts match exactly, with loss
differences below1e-5. No CPU-fallback warning was observed. This report contains
no fresh operator profile and cannot attribute performance to AiCPU or particular
kernels. Original wide workloads, family-specific stress graphs, FP16 performance,
CUDA and other CANN tuples remain outside this evidence.

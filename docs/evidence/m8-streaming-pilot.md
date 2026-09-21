# M8 first bounded native streaming pilot

Source: aee0da48e4c6661d7a75fd97ca39ddaba654ee4d, clean.
CPU aarch64, Python 3.11.15, Torch/LibTorch 2.10.0+cpu, C++11 ABI.
This report qualifies one inference workload and its records, not broad scale,
LLM modules, prefill, training performance or a general parallel speedup.

## Reproduction and integrity

From that commit, run with the project environment (ATen/inter-op/OpenBLAS one
thread, TORCH_DEVICE_BACKEND_AUTOLOAD=0, Nice=10, two build jobs):

```sh
/home/zlong/anaconda3/bin/python scripts/job.py \
  --output-dir artifacts/streaming-pilot-20260921-2300 -- \
  /home/zlong/anaconda3/bin/python scripts/pilot_streaming.py \
  --device cpu --dtype float32 --jobs 2 --tracking best-effort \
  --output-dir artifacts/streaming-pilot-20260921-2300/comparison
```

Unit tide-foundation-streaming-pilot-20260921-2300 ended inactive/dead,
exit 0. Records: artifacts/streaming-pilot-20260921-2300/status.json,
comparison/pilot.json, each case's run.json/summary.json and metrics.jsonl.
Started 2026-09-21T22:57:48Z; finished 2026-09-21T22:58:16Z.
All 16 cases and 80 metric events passed. Portable record validation is retained
in record-validation.json (including validator identity), with no warnings.
Native metrics and their retained root copies are byte-identical.
All Trackio projections explicitly report degraded/ModuleNotFoundError at init;
local records are complete. No dashboard was launched or package installed.

Clean directed correctness gate: artifacts/bench-anchor-20260921-2302/,
15 passed, terminal exit 0, same source, 23:01:11Z to 23:01:23Z. Checks include
an independent Python literal recurrence, CLI rejection and record integrity.
Each native run additionally checks complete trace and continuation against the
functional anchor outside timing. See ../streaming-benchmark.md for the contract.

## Fixed workload and measurements

Shared EMA/tanh weights; eight-node rings with two parallel edges per hop.
Nodes 32/4096; four active rings, batch 4, width 16, ticks 32, seed 7;
two warmups and five measured repetitions. Each repetition has 512 candidates,
1024 visited edges, 128 retained states and 32 pending messages. Dormant nodes
increase while active IDs, work and values stay fixed.

Advance median [min, max] in milliseconds:

| API | Workers | Packed | Nodes 32 | Nodes 4096 |
| --- | --- | --- | --- | --- |
| functional | 1 | 0 | 63.107 [63.062, 63.138] | 62.787 [62.643, 62.886] |
| functional | 1 | 1 | 62.439 [62.295, 66.203] | 69.620 [69.454, 69.955] |
| functional | 3 | 0 | 67.606 [67.394, 68.040] | 68.092 [67.895, 68.443] |
| functional | 3 | 1 | 63.790 [63.526, 64.470] | 70.209 [67.450, 78.359] |
| cursor | 1 | 0 | 21.961 [21.889, 21.983] | 22.233 [22.199, 22.256] |
| cursor | 1 | 1 | 22.802 [22.780, 22.831] | 24.025 [23.383, 24.373] |
| cursor | 3 | 0 | 19.477 [19.329, 19.663] | 29.755 [29.678, 29.921] |
| cursor | 3 | 1 | 21.753 [21.627, 22.152] | 21.641 [20.989, 31.822] |

Graph identity size is 4,272 / 576,182 bytes. Serial/unpacked cursor reset
medians are 0.019 / 0.059 ms; explicit snapshot medians 0.218 / 0.253 ms.
Construction is approximately 66–81 / 329–365 ms (one observation per run).
Process peak RSS is approximately 127–135 / 138–147 MiB, including construction,
traced checks and warmup; this is not isolated advance memory or unshared weights.

The cursor has lower advance latency than functional export in these cases.
Parallelism and packing show no consistent benefit in this small matrix.
Original-LH correctness qualification ran concurrently; recorded host load
averages were approximately 64–66. Five repetitions under shared load do not
establish a general performance ratio. No adaptive sweep was run.

Next measurements need separate declared workloads for larger active sets,
more batch/width, Attention/SSM, time prefill and training/replay costs. Increasing
dormant topology here checks one sparse execution pattern, not arbitrary regions,
parameter counts, allocation complexity or the intended full scale.

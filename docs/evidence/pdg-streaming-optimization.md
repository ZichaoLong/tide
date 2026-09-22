# Canonical PDG streaming optimization: first controlled result

Source: `da5a17bdbda1196fb32e2352fba9aa3b95e6dde3`, CPU aarch64,
Torch/LibTorch2.10.0+cpu, FP32 performance and FP64/FP32 correctness.
Observed2026-09-23 Asia/Shanghai (raw timestamps are UTC).
The adopted authority remains `tide-core-3` in [upstream.json](../upstream.json).
Graph/profile/checkpoint versions and HARD/SOFTP/HST definitions are unchanged.

## Change and correctness

[Implementation contract](../streaming-optimizations.md): output-column parallel
dense head, independent `(sample,region)` selection followed by stable publication,
direct canonical event grouping, state-handle moves and parallel temporary-event
release in trace-free execution. All switches default off. The separate head
pool runs after graph workers join; no global BLAS settings are changed.

The directed development build passed479 tests in98.40s. Its source archive,
hash, logs and result are in `artifacts/pdg-opt-dev-20260923-0050/`.
After committing, all tracked files were frozen read-only at
`/var/tmp/zlong-graph-execution-foundation/qualification/pdg-opt-20260923-0100`.
The source-hash-matching directed binaries and original build manifest were
copied to an isolated `build/`; this was reuse of an identified build, not a fresh
clean compilation. The original manifest honestly retains the development base
revision. Full regression on the clean fixed source passed **6459 tests/716.61s**.

New checks compare independent Python, legacy native and optimized schedules:
complete traces, node state/slots, region history, pending messages, cuts, routes,
isolated first-order VJPs and structural absent gradients. They cover cycles,
parallel edges, mixed EMA/SSM/Linear/Delta/Attention, packed/scalar execution,
clear/selected-only/observe-all/control-blend and HARD/HST/SOFTP. Native custom
region checks cover vector controls, empty selection and invalid results.
Dense checks cover shared/strided parameters, uneven partitions, concurrent
callers and caller grad/inference modes. The actual-size head smoke at
`[512,2048] × [50304,2048]`, FP32,160workers had **max absolute error0** against
ordinary Linear; all weights required grad while execution used no_grad.
Three small optimized FP64/FP32/grad-forward scale checks passed before timing.

## Fixed comparison

Same binary, graph, seed7, fixed input tokens, fresh normal(std=.02) parameters,
17,269,426,339 elements, D2048/B512/V50304, four-head fiber Attention,
all-softmax pooling, SiLU/RMSNorm, clear on selection and row Emit. No LH weights
were loaded. Each case ran12steps with warmup4; means use indices4–11.
One token includes embedding, two body ticks, readout tick and vocabulary head.
Caches/history grow continuously. Construction and metric/checksum validation
are outside token timing. No backward, optimizer step or implicit detach occurs.

Both use CPUs160–319,160node workers, ATen1/inter-op1 and1280GiB address-space
limit. Native metrics report OpenBLAS1/OpenMP1; MKL introspection is unavailable.
The optimized head uses160workers versus1 for baseline. Independent graph/head
pools have idle threads;160 is the active compute budget per stage, not the
process's total OS thread count. The host was shared, not exclusively reserved.

| Configuration | Mean ms/sample-token | Median | Min–max | Sample stddev | Aggregate tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Same-binary baseline, all switches off | 34.30392 | 34.45497 | 33.16015–34.99514 | 0.57810 | 29.15119 |
| Head160, parallel regions, compact events | 29.65650 | 29.78136 | 27.93176–31.08598 | 0.92590 | 33.71942 |
| Historical original LH, same token indices | 24.58203 | 24.82520 | 22.28320–25.98242 | 1.24223 | 40.68012 |

The controlled pair reduces mean latency **13.5478%** and improves throughput
**15.6708%** (1.1567×). The earlier PDG result was33.75842ms, so today's baseline
is1.62% higher. Preserve that distinction: the earlier37.33% gap to LH was a
different run; this pair's baseline/optimized gaps are39.55%/20.64%.
Historical LH remains a comparable-module/scale reference with different weights,
random input tokens and potentially different cache occupancy. It is not an
exact workload or a fresh paired LH rerun. This is one repetition per new case,
with a short growing-context window; the observations are not independent
steady-state latency samples or a confidence interval.

## Where time changed

Wall seconds per batch token, averaged over4–11. These are coordinator intervals
including task preparation and barriers, not summed worker CPU time.

| Phase | Baseline | Optimized |
| --- | ---: | ---: |
| Input preparation | 0.00218 | 0.00215 |
| Event grouping | 0.34163 | 0.32016 |
| Aggregate/State/Read | 7.31296 | 7.50886 |
| Region selection/comparison | 1.22383 | 0.29325 |
| Next/Full/Emit | 5.97591 | 6.03488 |
| State/message commit | 0.40163 | 0.48652 |
| Temporary cleanup | 0.91583 | 0.44591 |
| Vocabulary head | 1.38147 | 0.06141 |
| Whole token, including other cursor work | 17.56361 | 15.18413 |

Head, selection and cleanup decreased1.32006s,0.93058s and0.46993s respectively.
Event grouping improved only0.02147s; commit increased0.08489s. Update and Full
were also slightly slower in this pair. These jointly enabled paths are not
separate causal ablations. The net result is2.37948s saved per batch token.
Update plus Full now consumes **89.20%** of token time. Further large gains need
operator/data-layout work in those phases, beyond reducing coordinator overhead.

Peak process RSS was110.45091GiB baseline and111.75505GiB optimized. Construction
was227.10996s/228.33796s, excluded. This run does not establish a memory reduction.
All12steps have **exactly matching model/work counters and logit checksums**, also
matching the older PDG work inventory:32body selections per sample-token.
Checksums do not prove full large-model state equality; small complete-state and
VJP comparisons above provide the semantic validation. Training correctness is
tested on small models; the large performance result covers no_grad inference.

## Records and reproduction

Unit `tide-pdg-opt-20260923-0100` terminated success/exit0, inactive/dead, MainPID0.
All6driver stages and the head smoke passed. Durable records are under
`artifacts/pdg-opt-20260923-0100/`: `status.json`, `pipeline.json`, `task.log`,
`qualification/{result.json,tests.log}`, `head-smoke.json`, five case directories,
`analyze.py` and `analysis.json`. The driver is
`artifacts/pdg-opt-runner-20260923-0100.py`; its hash and every tracked source hash
are retained in pipeline.json. Analysis verifies clean source, copied binaries,
input hashes, complete records, identical workload/configuration except switches,
phase accounting and all token counters. All five portable run records validate.
The full exact commands are retained in status.json/pipeline.json/run.json.

From a clean matching checkout/build, use the [scale entry point](../pdg-scale-benchmark.md)
with `--width 2048 --batch 512 --steps 12 --warmup 4 --workers 160 --threads 1
--packed 1 --grad 0 --profile 1 --seed 7`. Compare new output directories with
`--head-workers 1 --parallel-regions 0 --compact-events 0` and
`--head-workers 160 --parallel-regions 1 --compact-events 1`.
Retain the same topology file and the raw growing-context time series.

Trackio is best-effort/degraded because it is not installed; complete local
manifests, JSONL and logs remain authoritative. No dashboard was started.
Preserve the frozen source, build, inputs and all referenced artifacts.

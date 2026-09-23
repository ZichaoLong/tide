# Fiber execution options: CPU qualification and ablations

Source: `fa31ad8202b688dc220f814ef6c2ff6ba2c6e465`.
Frozen checkout: `/var/tmp/zlong-graph-execution-foundation/qualification/fiber-efficiency-20260923-113526`.
Records: `artifacts/fiber-efficiency-20260923-113526/`; fixed driver:
`artifacts/qualify_fiber_efficiency.py`; terminal audit:
`artifacts/audit_fiber_efficiency.py` and the run's `post-run-audit.json`.
Service `tide-fiber-efficiency-20260923-113526` is inactive/dead, MainPID0,
Result=success, ExecMainStatus0. All32 stages and persistent records passed.

## Correctness and delivery

The clean build and complete CPU FP64/FP32 regression passed **6719 tests in
832.99s**. Both engines rebuilt from the exported source kit after relocation to
a directory containing spaces. Optimized PDG smoke passed independent scalar
slot/scalar row/packed/parallel full-state checks; LH complete small logits equal
the earlier original-LH anchor exactly. The audit checks439 tracked source files
against Git archive,12 native build hashes,248 packet files, archive/manifest
hashes, prepared LH source/graph/binary identities and14 completed run records.
The freshly rebuilt kit PDG binary is byte-identical to the main qualification
binary. Reference repositories were not modified.

The five opt-in [execution/storage policies](../fiber-efficiency.md) cover CSR
post-attention pooling, immutable per-sample KV reuse, temporary head layout,
deferred old-state release and scale projection strides. Graph v13/checkpoint v5,
complete-fiber semantics, Aggregate, clocks, parameter owners and public VJPs are
preserved. Tests cover exact/single, serial/parallel, streaming/frontier, ragged
prefill, cycles/clear, HARD/SOFTP/HST, all pooling modes, absent/zero sources,
value/state/pending roots and disconnected gradients, snapshots/checkpoint policy
switches, strided shared parameters and optimizer steps. Packed VJPs retain the
existing semantic replay; this is not an optimized-backward performance claim.

Retained development evidence:375 passed/48 failed in
`artifacts/fiber-efficiency-dev-20260923-112506/`. The48 failures were the new
no-edge ragged fixture requesting a nonexistent pending-gradient objective;
values/states had already agreed. Corrected output/state objectives passed48
checks/2.96s against the unchanged C++ binary at
`artifacts/fiber-efficiency-retest-20260923-113224/`. Cyclic tests still compare
pending-root VJPs. The clean full gate covers the correction. Preserve the
single-case reproduction `artifacts/fiber-efficiency-first-failure/`, original
failed archive/audit, and the earlier cancelled-before-build candidate
`fiber-efficiency-dev-20260923-110604`; do not relabel them.

## Fixed inference workload

17,269,426,339 parameters; D2048/B512/V50304; FP32/no_grad; four attention heads;
12 tokens, warmup4, indices4–11 measured; seed7/fixed external IDs. Exact attention
packing, packed row emission, parallel regions and compact events are common.
All new options default off/old policy in the baseline. LH and PDG use the same
four graph blocks and comparable matrix arithmetic, with independently initialized
weights; this is not cross-engine complete function equivalence.

CPU aarch64, Torch/LibTorch2.10.0+cpu, GCC10.3.1, common affinity160–319, separate
sequential processes, 1024GiB address-space and1200s bounds per native case.
PDG requests160 or116 node/head workers in separate phases, ATen/OpenMP/BLAS1;
LH requests160 ATen/OpenMP workers using its original schedule. Requested workers
are not a guarantee of equal useful concurrency. No project build or other
project benchmark overlaps the large timing stages. Coarse coordinator timing
and work accounting are enabled; optional detailed operator timers are off.
Initialization is excluded. Divide batch elapsed time by512, as requested.

| Case | ms/sample-token | Delta from two-baseline mean | Peak RSS GiB |
| --- | ---: | ---: | ---: |
| LH, original schedule | 25.04663 | -13.545% | 149.623 |
| PDG baseline | 28.82426 | -0.506% | 110.046 |
| CSR pooling only | 29.22823 | +0.889% | 108.714 |
| Owned KV only | 28.48697 | -1.670% | 110.264 |
| Deferred release only | 28.76084 | -0.725% | 108.953 |
| Projection layout only | 29.55736 | +2.025% | 113.853 |
| Head-major attention only | 30.07597 | +3.815% | 110.950 |
| Baseline, 116 workers | 29.80582 | +2.882% | 105.908 |
| All options, 160 workers | 28.79132 | -0.620% | 111.616 |
| All options, 116 workers | 27.92335 | -3.616% | 106.957 |
| PDG baseline repeat | 29.11735 | +0.506% | 111.257 |
| All options, 160 repeat | 28.56860 | -1.388% | 112.237 |

The two baseline means average **28.97080**; the two all160 means average
**28.67996**, a **1.0039% latency reduction**. Baseline repetition differs1.0168%
from its first run; all160 repetition differs−0.7736%. This is limited evidence
for a small gain, not a substantial speedup or statistical uncertainty bound.
The repeated all160 mean remains14.5063% above this single LH observation.
All116 at27.92335 is3.6156% below the two-baseline mean, but has only one run;
its speed and106.957GiB RSS are a candidate observation, not a repeated result.

Each other ablation and LH also has one process. Tokens within a window have
different cache/work states; their standard deviation is not an independent
estimate of benchmark uncertainty. Raw token observations, medians, spread,
load averages and resources are retained in analysis.json and the run records.
Trackio was best-effort/degraded (unavailable); local records and validation are
complete. x86_64, long context, other widths/batches, prefill and training
performance remain unmeasured by this experiment.

One additional read-only `numastat -p` observation occurred during the pooling
case after8 completed tokens, plus an earlier initialization sample. The calling
command took1.207s; procfs page inspection can perturb execution. Its small
regression must not be treated as a clean causal estimate. The snapshot and
explicit limitation are in `numa-observations/`; no further such inspection ran
during this timing series. It observed uneven NUMA placement and motivates a
separate [controlled memory-policy comparison](fiber-numa.md), without itself
establishing the cause or performance effect.

## Interpretation and numerical limits

All11 PDG cases have exactly equal per-token `model/*`, `work/*` and `op/*`
inventories for all12 tokens. Baseline repeat, KV reuse, deferred release,
head-layout and worker-count-only cases have exactly matching output checksums.
Maximum absolute checksum differences are0.00513404 for CSR,0.01111816 for
projection strides and0.01461823 for combinations. Checksums are sums, not
full-tensor error bounds. Independent complete small values/states/routes and
VJPs establish numerical qualification; large counters/sums are additional
workload checks. The unchanged baseline also matches the prior diagnostic's
work and checksums exactly. No speedup comes from a reduced counted workload.

Mean coordinator seconds per batch token, averaged over both repetitions:

| Phase | Baseline | All160 |
| --- | ---: | ---: |
| Update | 7.323965 | 7.038536 |
| Full | 5.954737 | 6.153965 |
| State/message publication | 0.458460 | 0.378268 |
| Cleanup | 0.416187 | 0.417086 |

Update and publication decrease, while Full increases in these observations,
leaving a small total change. This does not causally assign the Full difference
to any one option. Update+Full still occupy about90% of the baseline's measured
token duration. The earlier summed worker timers must not be substituted for
these coordinator wall intervals or used to claim an intrinsic2× slower Emit.
The data do not support pooling, a single temporary layout or lower worker
count as the independently dominant source of the LH gap. They also do not
prove that the remaining gap is required by canonical PDG semantics.

Retain conservative defaults. All options remain available for controlled
machine/workload comparisons; more calls or fewer copies alone are insufficient
to choose them. The [memory-policy follow-up](fiber-numa.md) also finds no substantial PDG
gain from simple interleaving. Stable node/worker locality and batched state
storage remain questions, not established fixes from this series.

## Portable artifact

`artifacts/fiber-efficiency-20260923-113526/export/cpu-attention-compare.tar.gz`
(391597 bytes), SHA256:
`83c63e8df95a027d68c706d03f23ec43f922ebccac52c02e0c6ca6cca46aa62a`.
No commit checkout is required to use it. Extract and run the two existing
runners in a matching target Torch/LibTorch environment; see
[the packet instructions](../../tools/cpu_compare/README.md). A candidate run is:

```bash
python run_pdg.py --device cpu --threads 56 --attention-packing exact \
  --fiber-pooling csr --fiber-cache owned --defer-state-release 1 \
  --projection-layout linear --attention-layout head --output-dir runs/pdg-candidate
```

Compare sequentially with a default PDG run and `run_lh.py`, using identical
CPU affinity and new output directories. This command selects the validated
combination; it does not promise the160/116-worker local timings on a56-core
Intel server. The archive also supports individual switches and `--smoke`.

# CPU memory-policy follow-up

Source: `fa31ad8202b688dc220f814ef6c2ff6ba2c6e465`, reusing the clean checkout and
binaries from [fiber qualification](fiber-efficiency.md). No rebuild or numerical
implementation change. Records: `artifacts/fiber-numa-20260923-131329/`;
launch argv: `artifacts/fiber-numa-qualification.json`; frozen driver:
`artifacts/qualify_fiber_numa.py`; terminal audit:
`artifacts/audit_fiber_numa.py` and the run's `post-run-audit.json`.

Service `tide-fiber-numa-20260923-131329` terminated inactive/dead, MainPID0,
Result=success, ExecMainStatus0. All12 stages passed: two policy checks, an
interleaved checked small PDG run, four large cases and five record validations.
Terminal audit verifies439 source files,12 native binaries, unchanged original
LH prepared source/graphs/binaries, all five records and recomputed statistics.
The preceding full6719-test CPU gate remains the correctness qualification;
this follow-up does not claim another full regression or new algorithm.

## Question and protocol

The earlier pooling-run snapshot showed uneven *process* memory placement. It
did not isolate weights, establish the placement in other cases, or measure a
causal penalty. This bounded2×2 comparison changes only process memory policy
within each engine: default versus `numactl --interleave=4-7`.

CPU affinity160–319 covers NUMA nodes4–7 on this320-core aarch64 host. Both
policies use that same affinity and160 requested workers, a1024GiB address-space
limit and1200s native timeout. PDG retains all baseline execution options,
including exact packing, old fiber policies and ATen/OpenMP/BLAS1. LH retains its
original schedule and160 requested ATen/OpenMP threads. The inherited OS policy
was checked as default; the explicit policy check reports interleavemask4,5,6,7.
The policy applies to the process's allocations, including temporary tensors;
it is not a weight-only placement experiment. No procfs NUMA scans occurred
during these timed cases, and no project build/benchmark overlapped them.

Each case starts a fresh process:17,269,426,339 parameters, D2048/B512/V50304,
FP32/no_grad,12 tokens/warmup4, seed7/fixed external IDs. Initialization is
excluded; compare indices4–11 and divide batch duration by512. Order:
LH default, LH interleave, PDG interleave, PDG default. One process per cell;
there is no statistical-confidence claim from this small shared-host sample.

## Observations

| Engine | Memory policy | ms/sample-token | Peak RSS GiB |
| --- | --- | ---: | ---: |
| LH | default | 25.43408 | 148.781 |
| LH | interleave | 27.16406 | 149.252 |
| PDG | default | 29.35316 | 110.606 |
| PDG | interleave | 29.16130 | 115.070 |

Interleave changes LH latency by **+6.8018%** and PDG by **−0.6536%**.
PDG's0.19186ms difference is small compared with the earlier default baseline
variation. The observed PDG/LH gap falls from15.4088% under default policy to
7.3525% under interleave, primarily because LH becomes slower. That smaller
ratio is not evidence of a substantial PDG optimization or of matching the
best observed LH performance. Do not enable interleaving as a general default
or use the slower LH cell as the sole reference for a speedup claim.

Within each engine, every token's model/work/operator inventory and output
checksum match exactly between policies. PDG also exactly matches the earlier
baseline's work/checksums. These sums are additional checks, not a substitute
for complete small tensor/state/route/VJP qualification. Raw observations,
medians, spread, resources, commands and source identities are retained in each
run and analysis.json. Trackio was best-effort/degraded; all local records and
validators completed successfully.

This tests a simple allocation policy. It does not rule out benefits from
stable node/worker assignment, node-local state/cache storage, targeted placement
or a different hardware topology. A process-level interleave mask cannot make
those implementation questions equivalent. Those remain separate bounded
investigations in [ROADMAP](../ROADMAP.md); no additional experiment is active.

## Reproduction boundary

Use the already qualified portable source kit linked by the primary report.
Its runners inherit the caller's CPU affinity and memory policy. On this host,
prefix the same runner command with `taskset -c 160-319 numactl --interleave=4-7`
for the explicit policy, and omit numactl for the default comparison. Use the
same thread count, inputs and new output directories, and run sequentially.
The exact locally tested wrapper argv are in pipeline.json and the launch
record. CPU and NUMA IDs4–7 are machine-specific; this is not a command to copy
unchanged onto the user's Intel server. Intel execution remains unverified here.

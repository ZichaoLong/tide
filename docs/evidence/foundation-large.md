# Bounded large graph assessment

Clean source `3d2622a7d03afe505c980915c0b3c38f1470144c`, CPU aarch64,
Torch/LibTorch2.10.0+cpu, FP32. This covers the two frozen DAG/Settle large
presets, including limits; it does not certify successful target-scale runs.
Five stages completed, four timed out, and one larger stage was not launched.

## Provenance and resource boundary

Raw records: `artifacts/foundation-large-20260923-a/`, including status.json,
suite/, reviewed-audit.json, summary.json, report.json, terminal-inspection.json.
Exact launch/preflight: `artifacts/foundation-large-a-launch.json` and
`foundation-large-a-preflight.json`. Source checkout:
`qualification/foundation-large-3d2622a`; immutable build:
`qualification/foundation-bench-dev-20260923-a/build`. Its archived development
manifest retains494e6a9; this is reused matching C++ content, not a rebuild at
3d2622a. Adapter SHA256 is
`b92c0fa2517704d4e54a856dcdb15f133108a6515ed2b3b70819aacec2618f4c`.
All13 binary hashes and all recorded source hashes were checked. S6 rebuilds.

Unit `tide-foundation-large-20260923-a`: background.slice, Nice10,
RuntimeMaxSec4000. Sequential stages,32 node workers, ATen/OpenMP/BLAS1,
interop1, two state/cache/ledger-reset warmups, one measured repeat,300s work
limit including setup/warmups/profile, up to30s termination/reap grace.
No heavy task overlapped timing. Dynamic discovery found320 usable physical/
logical CPUs across eight NUMA nodes, no finite CPU quota; aggregate half-budget
160 CPUs and743.35GiB memory. Workers used a discovered160-CPU balanced affinity.
Peak combined RSS includes worker and coordinator; cgroup usage is also retained
but includes unrelated account activity. Address space is not reported as RSS.

## Workload and exact counts

Graph-only independent same-fiber Attention/SwiGLU owners, four attention heads,
seed7, width-scaled matrix initialization, FP32, B512/T6, no vocabulary head or
embedding. Four active ranked regions have32 (wide) or64 (narrow) candidates
and select one each; remaining regions are dormant. These are legal DAGs,
with positive delays; native Settle independently encodes two boundary nodes.

| Preset/body nodes | Parameters | Body edges | Estimated peak GiB |
| --- | ---: | ---: | ---: |
| wide D2048/N128 | 5,907,421,376 | 3,072 | 100.51 |
| wide D2048/N384 target | 17,722,251,712 | 3,072 | 188.54 |
| narrow D128/N256 | 46,391,680 | 12,288 | 11.10 |
| narrow D128/N1024 | 185,492,608 | 12,288 | 12.13 |
| narrow D128/N46912 target | 8,496,773,056 | 12,288 | 74.06 |

Each completed wide/narrow stage touches128/256 body nodes and65,536/131,072
node-sample state owners. Across3072 sample positions there are393,216/786,432
body candidates,12,288 selected body events, selection1/32 or1/64, and maximum
KV length6. Thus body selection is four events per sample position, not LH's
32 events per sample-token. TimedDAG logical Aggregate/Upd/Read/Next counts are
the body candidate count; native Settle adds6144 boundary events to Aggregate/
Read/Next and Full. Raw records retain complete logical/packing/transport counts.

Wide completed work has13,831,942,176,768 executed matrix/attention FLOPs;
narrow has107,911,053,312 under the declared operator-count scope. Executed vs
valid attention score elements are9,437,184/5,505,024 (wide) and
18,874,368/11,010,048 (narrow), including masked computation. Profiling is a
separate pass, not part of wall timing. Elementwise/norm/general backward FLOPs
are excluded. No large training or long-context claim follows from these runs.

## Terminal observations

Wall seconds below cover six measured batch positions in no_grad forward.
ms/sample-position = seconds*1000/(512*6); batch-position latency = seconds/6.
Only one independent repeat was budgeted; no speedup or variance claim follows.
RSS is the sampled combined process peak including setup, warmups and profiling.

| Graph / preset / nodes | Whole-run result | Forward seconds | Peak RSS GiB |
| --- | --- | ---: | ---: |
| TimedDAG / wide /128 | completed | 36.62812 | 50.007 |
| TimedDAG / wide /384 | timeout in second warmup | unmeasured | 99.305 |
| Settle / wide /128 | timeout in separate profile | 35.27225, partial run | 50.789 |
| Settle / wide /384 | not launched after prior limit | unmeasured | — |
| TimedDAG / narrow /256 | completed | 44.91222 | 5.280 |
| TimedDAG / narrow /1024 | completed | 44.84552 | 5.998 |
| TimedDAG / narrow /46912 | timeout in second warmup | unmeasured | 51.914 |
| Settle / narrow /256 | completed | 45.47293 | 5.404 |
| Settle / narrow /1024 | completed | 46.89877 | 6.116 |
| Settle / narrow /46912 | timeout during initialization | unmeasured | 48.373 |

TimedDAG wide128 gives11.92322ms/sample-position,83.86998 sample-positions/s
and6.10469s/batch-position. The two target TimedDAG models finished construction
(249.24s wide,189.81s narrow) but no formal measurement. Settle target narrow
has no completed construction record; its preflight count is not an observed
allocated-owner count. Settle wide128 retains a completed measured.json with
values and logical counts, while its operator profile remains incomplete.

TimedDAG stride is6. Settle stride is6 at the first stage,18 at narrow1024,
and would be735 at narrow target. Dormant ranks affect the Settle clock and KV
decay intervals, so equal active topology does not imply equal functions.

All nine manifests/metrics/summaries validate. Completed workers exited0;
four timeout workers exited-15 and every process audit has no remaining group
PID. The assessment wrapper exited0 (evaluated), unit inactive/dead/MainPID0;
this does not turn the four individual failures into passes. Startup-timeout
worker.json remains starting, superseded by its terminal enclosing summary and
process audit. Its post-construction affinity sample is unavailable; taskset's
recorded launch affinity was checked. Effective Torch pools and binary identity
were saved before construction. Audit-helper KeyError and correction are retained
in audit-startup-record-repro/; no benchmark records were rewritten. Trackio was
best-effort/degraded (ModuleNotFoundError); local records are authoritative.

## Reused PDG/LH evidence and conclusion

The original C++ LH and PDG17,269,426,339-parameter results remain in
[packed transport](packed-transport.md), with effective160-worker settings and
sequential resource control. LH has232 nodes per cortex (224 leaves+8 hubs);
independent inet/onet namespaces yield465 physical PDG nodes/4418 physical edges,
not232 total PDG nodes. This is comparable-scale inference with independent
initialization, not a complete two-graph continuation equivalence claim.
The [earlier narrow failure](pdg-scale-attention.md) retains5/8 steps before its
1800s limit, exit-15 and183.73GiB peak. Its actual summary reports16,608,289,021
parameters,57,856 static nodes per cortex,115,713 physical PDG nodes and
1,098,230 physical edges. Count audit:
`artifacts/foundation-pdg-narrow-count-audit.json`. This larger two-namespace
workload does not substitute for the frozen8.497B graph-only PDG narrow preset;
the bounded supplement below closes that gap. The historical expensive failure is retained. The DAG/Settle modules, parameters, clocks,
selected work and cache lengths above differ; no cross-family speed ratio is
computed from these records.

Together with [all12 medium configurations](foundation-medium.md), every graph
family has non-smoke performance evidence. Both frozen large presets have bounded
assessments across all three families, with the wide PDG/LH record reused.
Target-scale success is not claimed for the new timed-out/unlaunched targets.
Zero new bottlenecks or tuning candidates were added. Defaults remain conservative.
Relocated rebuild/smoke and [final correctness](foundation-final.md) are complete.


## Frozen PDG narrow supplement after exact-count audit

Clean source81a1b266af49d918aa6e1587e4ed9e0c4d4e5eb5, using the freshly rebuilt
and fully qualified relocated native binary. Source/build identity matches the
[final gate](foundation-final.md). Records:
`artifacts/foundation-pdg-narrow-20260923-a/` (suite, reviewed-audit.json,
report.json, terminal-inspection.json); exact command/preflight/resources in
`artifacts/foundation-pdg-narrow-a-launch.json`. This is the already-frozen
D128/B512/T6 graph-only preset, not another shape or tuning candidate. It ran
only after the final correctness/build unit had fully exited.

Half-effective launch budget160 CPUs/719.39GiB. The requested worker limit was32;
the fixed native-stream-packed variant explicitly uses one node worker,
constructor_threads_created0, ATen/inter-op/OpenMP/BLAS1. This is a serial baseline,
not a measurement of parallel PDG's narrow-graph ceiling. The medium P03 and
retained wide PDG runs cover node parallelism separately. Affinity is the same
resource-policy-derived160-CPU set. Two reset warmups, one measured repeat,
300s per stage including setup/profile, bounded cleanup and no heavy overlap.

| Body nodes / true parameters | Result | Forward seconds | Peak combined RSS GiB |
| --- | --- | ---: | ---: |
| 256 /46,391,680 | completed | 57.53553 | 3.521 |
| 1024 /185,492,608 | completed | 57.89133 | 4.254 |
| 46912 /8,496,773,056 | timeout in second warmup | unmeasured | 49.279 |

Target construction completed in175.21s, including the check of actual allocated
parameter count against preflight. It did not complete a formal timed phase.
The target worker exited-15; two completed workers exited0. All three records
validate, with empty remaining process groups. Unit
tide-foundation-pdg-narrow-20260923-a is inactive/MainPID0/exit0; its evaluated
wrapper status does not make the target run passed.

These PDG stages use stride1 and stop at logical cut6. Completed runs record
589,824 body candidates/updates and9,216 selections (actual1/64),256 touched body
nodes/131,072 node-sample owners,491,520 edge visits,1,536 outputs and98,304
pending messages. There are3,072 supplied sample positions and32,768 ledger keys;
the tail remains a valid continuation. ms/sample-position divides by supplied
positions, not completed outputs. Settle/TimedDAG's sealed clocks and larger
completed work differ. No timing ratio between those schedules is inferred.
A source-level measurement description is in [benchmark entry](../foundation-benchmarks.md).

Combined new large assessments have12 launched stages:7 complete and5 timeouts,
plus one explicitly unlaunched larger Settle stage. Historical PDG/LH results
and failures remain separately identified. The bounded assessment is closed;
there is no claim that every target size passed or that the serial narrow result
establishes a parallel limit. No further tuning follows this acceptance version.

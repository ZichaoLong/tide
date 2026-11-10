# Unique KV snapshots and graph census — 2026-11-10

Implementation: `1528df812ec4f102b3b6a8cbeef289355a682622`; mathematics: fixed
`0770593`. [Machine-readable observations](interface-snapshot-bank-20261110.json)
retain exact configurations, hashes, workload, lifecycle and scope.
The later read-only publication, row-unbind, recurrent Describe and scheduling
fixes are **not** part of this implementation or these measurements.

The increment retains one actual KV owner bank through untraced sequence VJP,
reuses overlapping query storage/projections and preserves independent snapshot
adjoints. Its scope and literal checks are in [the contract](../weighted-snapshot-bank.md).
The new [autograd census](../autograd-census.md) records outer and explicitly
registered private graphs separately. The repository workflows and public-boundary
fixes are described in [the initial audit](implementation-audit-20261110.md).

## Correctness and continuation

| Fixed-source check | Result |
| --- | --- |
| New bank, graph census, public boundaries and inventory | CPU 53 passed, 1 retained FP32 xfail |
| Query/sequence roots, prefix causality, layouts, copies and selected returns | CPU 248 passed, 2 existing xfails |
| NPU-configured bank/sequence/query/layout flow | 72 passed, 1 retained xfail; this includes CPU helper checks |
| Installed Python and Python→native consumers | CPU/NPU independent small training and exact restore passed |
| Standalone SDK, complete and selected returns | CPU FP64/NPU FP32 three steps and fresh-process exact restore passed; complete mode also checked independent roots |

Literal projection derivatives, independent scalar schedules, partial/zero roots,
shared parameters, ignored huge cache entries, int64 timestamps and version guards
answer different correctness questions. The full FC16 cost task still has no
independent full-scale numerical/learning qualification. The original scalar FP32
cat-evidence discrepancy is retained, with the same-material FP64 witness in
`test_snapshot_bank`; no tolerance was loosened.

First attempt `06b242e` built but failed two native Read gradient tests: an
unqualified helper named `bind` resolved through C++ ADL to `std::bind`, leaving
the shared projection without its VJP. A compile-time return-type assertion
confirmed the overload, and `attach_bank_vjp` fixed it in `1528df8`. This was an
implementation error, distinct from the retained numerical xfail. The failed job
and tests remain failed. A later header-location failure occurred after all CPU
checks passed; continuation located the installed header package without importing
its optional Python helper and did not rerun those completed checks.

## Finite cost observations

FC16: four serial regions of four nodes, adjacent regions fully connected;
B8/L128 = 1,024 input positions, d128, two attention heads, FFN512,
summary/adapter32, KV qry/Read+/cat/ind/context/cat/regional_total/quad,
fractional sending budget k2. Core ~4.564M parameters; complete task 5,092,788.
Same fixed downstream source `0439efd`, default window/packed-v2, minimal returns,
FP32, one native scheduler worker, CANN9.2/Torch2.12/TorchNPU2.12.

One independent process, warmup, parameter/optimizer reset and one measured step
per point. Complete training includes preparation, scheduling/packing, loss,
backward, clipping/finite check, optimizer, cleanup and synchronization.
The NPU was physical 11 selected by the idle-only cooperative queue; no shared
card permission was used. Project builds and timing did not overlap; other host
workloads were not controlled.

| Client / placement | Host intra-op threads | Full training s | Input positions/s | Inference/TTFT s |
| --- | ---: | ---: | ---: | ---: |
| Python→C++/LibTorch, CPU | 1 | 14.6283 | 70.00 | 3.1947 |
| Python→C++/LibTorch, CPU | 16 | 16.5985 | 61.69 | not measured |
| Python→C++/LibTorch, NPU mixed-c | 2 | 20.6338 | 49.63 | 4.3911 |

Inference TTFT is warm loaded local prompt creation/upload through greedy first
token return, excluding tokenizer/queue and subsequent decode. Its complete
request walls were 3.1961/4.3954 s, or 320.39/232.97 input positions/s.
Training work was 16,384 candidate updates, 8,344/8,345 Full computations and
25,060/25,064 internal messages for CPU/NPU. Same work counts alone do not prove
identical routes or gradients. Gradient norms were 1362.2411/1359.8636.

The [previous observation](fc16-cost-attribution-20261110.md) reported
CPU1/NPU training 17.4969/21.0647 s. CPU cost decreased in this observation;
NPU changed little and inference increased from the prior 4.2780 s request wall.
The earlier NPU used shared/different physical devices, so these are not a
controlled device-speedup claim. Sixteen CPU threads did not help this small
model. This does not predict the best thread count for a wide or billion-parameter
model. Peak RSS included warmup: CPU1 3490 MiB, NPU host 2912 MiB;
NPU peak allocated 1739 MiB, reserved 1858 MiB.

## What the graph diagnostic establishes

Separate B2/L8 instrumented processes, same FC16/d128 task, gave:

| Observation | CPU | NPU |
| --- | ---: | ---: |
| Outer graph nodes | 615 | 615 |
| Union including registered private graphs | 13,477 | 13,477 |
| Actual backward node executions | 12,324 | 12,068 |
| SelectBackward nodes | 3,200 | 3,200 |
| Read / Full private graph nodes | 6,976 / 4,366 | 6,976 / 4,366 |
| Regional private graph nodes | 908 | 908 |

Groups overlap and must not be summed; Python's outer view overlaps native
registration. Explicitly saved unique storage was 20,632,848 bytes; that is not
the allocator peak or complete native ATen saved storage. Counts explain why
32 outer query boundaries do not imply a small private graph. This supports
reviewing repeated row view transposes and publication, without equating a node
count reduction to wall-time speedup.

## Reproduction

Use the fixed commit, a clean checkout and separately installed wheel, with
matching native and standalone builds from [development](../development.md).
The JSON records exact source/build/config identities. Relevant tests are
`test_snapshot_bank`, `test_autograd_census*`, `test_kv_sequence_vjp`,
`test_kv_prefix_roots`, `test_weighted_query_vjp/layout`, `test_prefill_local_batches`,
`test_independent_copies`, `test_query_roots`, `test_selected_boundary_execution`
and the new public-boundary tests. Installed consumers use
`examples/consumer_core5/dense_query.py --no-trace`; SDK fixtures and checks follow
[the standalone reproduction commands](prefill-default-20261110.md#可独立复现),
with both `--return-mode complete` and `outputs`.

Cost: `benchmark_dense_query.py --device cpu --threads 1 --trace-records minimal
--native-library BUILD --downstream-repo READ_ONLY_DOWNSTREAM --output-dir NEW`.
Change threads to 16, or queue an idle NPU and use `--device npu:0 --threads 2`.
Inference adds `--mode inference`; separate diagnostics add
`--batch 2 --length 8 --profile stages --autograd-census`.
Each long workflow uses a bounded `background.slice` service. No retired run
directory, reference-selected trajectory or shared-card launcher is required.

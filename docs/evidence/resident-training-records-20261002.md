# Optional diagnostic exports during resident training

Tested source `ca26b472277eb9282509733aecaffe2b1893cc3c`, clean immutable
snapshot. All six build/gate/allocation jobs passed: 51 public tests and four
standalone component cells, without skips.
[Audit and exact source/binary/result identities](resident-training-records-20261002.json).

## Behavior and validation

Training preserves every journal needed by its VJP independently of the optional
`forward.diagnostics` / Python `trace` setting. Disabling diagnostics omits the
export-only message queue, weighted-contribution journal and history/source-scale
snapshots. Outputs, full continuation, backward, None/zero connectivity, optimizer
state and checkpoint behavior retain their contract. An observer requesting full
records enables these buffers and their admission allowance. No public limits,
checkpoint ABI, CANN kernel, logical capacity, dtype or update boundary changes.

The public gate covers six independent CPU-autograd trajectories across three
families, both schedules, HARD/HST/SOFTP and one/two devices with exports disabled;
eight FP32/FP16 Add/Attention complete-consumer diagnostic mode pairs across
standalone LibTorch and Python-owned clients; two observer-only cases; 24 existing
complete-training comparisons; eight forced-splitting updates; two CLI/failure
records and one preconstruction refusal. Full state/continuation, gradients,
None/zero flags and parameter updates are compared, not only scalar losses.

Standalone checks cover 43 FP16 basic and 40 FP16 cache trajectories, plus 32
sharded trajectories per dtype (FP32 and FP16). They retain independent CPU
FP32/FP64 VJP/master/optimizer comparisons and change diagnostic mode across
checkpoint restore. In total these component checks cover 2,352 windows and
588 updates; each sharded dtype covers 512 windows/128 updates.

Development passed the same affected scope. The initial observer-only test
failed because its CPU oracle disabled trace while the candidate explicitly
requested records; enabling the oracle's trace fixed the test setup. The failed
`training-records-observer-dev02` record is retained; corrected `dev03` and clean
qualification pass. No numerical tolerance was relaxed.

Builds reuse unchanged core/CANN dependencies only after identity checks. The
three affected archive units, two owner units and changed checker objects are
source/header verified before reuse from development; both runtimes are freshly
linked and loader-checked. The installed standalone consumer links the new
library. Unchanged full CPU/core suites are not rerun.

## Allocation observation

The same representative Attention workload and consumer as the prior
[fiber append comparison](resident-fiber-append-20261002.md) executes one FP32
AdamW step spanning two continued windows: D128/B8/T4/V257, 128 body nodes,
544 edges, 17,384,240 parameters, one NPU. All capacities and chunk policy match.
Instrumentation is bounded and separate from throughput measurement.

| Allocator peak | Bytes |
| --- | ---: |
| Before, clean80dae6e | 2,426,168,320 |
| After, cleanca26b47 | 2,409,123,840 |
| Reduction | 17,044,480 (16.26 MiB, 0.70%) |

Losses, output counts, final cut, precision and event/training statistics are
exactly equal. Both peaks stay below the conservative admission estimate. All
recorded allocations are freed; eight snapshots remain below the 256 MiB raw
limit and the history below 100,000 entries. TorchNPU allocator counters exclude
vendor/driver allocations outside that allocator. There is no throughput claim.

This is a modest concrete reduction, not full-size closure. Admission still
conservatively reserves the omitted buffers. Live KV, retained VJP journals,
reverse storage and CPU/mixed multi-device capacity remain separate F6 work.
Raw records are `TASK/runs/training-records-*-clean01`; baseline
`TASK/runs/fiber-append-allocator-clean01`, with
`TASK=/mi/data2T/zlong/tide-execution-flows`.

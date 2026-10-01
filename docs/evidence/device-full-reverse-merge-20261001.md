# Full reverse metadata merge on the device vector core

Implementation: `fe68065fbf41eebbba003051a84d3567e59028d1`.
All four immutable-source jobs passed: separate standalone/Python-owned builds,
the two-device FP32/FP16 retained VJP gate, and a three-device FP32 profile.
The [audit](device-full-reverse-merge-20261001.json) verifies source, binaries,
loaders, changed/reused objects, kernel hashes, terminal jobs and raw trace hashes.

The [preceding profile](device-full-reverse-20261001.md) found 120 AiCPU Bool
ScatterUpdate tasks when merging reverse connection flags. One Ascend C metadata
kernel now merges content/comparison/parameter connection bits, sticky error and
int64 chunk counts per nonempty shard. Numeric cotangents retain their existing
packed scatter. Padding cannot create connections; error/count bounds are explicit.

The unchanged independent CPU FP32/FP64 oracle passes 50 trajectories/200 retained
windows per payload dtype, including both schedules, all Full profiles, control,
normalized Aggregate, attention caches, after-close retention, replay and explicit
refusals. No tolerance changed. This internal merge does not affect the public
single-device path, so the previously qualified public/core gates were not repeated.

| Same two-fixture, three-device profiling scope | Previous | Fused |
| --- | ---: | ---: |
| AiCPU tasks | 120 Bool ScatterUpdate | 0 observed |
| Metadata merge tasks | 0 | 40 AI_VECTOR_CORE |
| AI_VECTOR_CORE tasks, whole profile | 11,562 | 11,418 |
| AI_CORE / MIX_AIV tasks | 366 / 196 | 366 / 196 |
| Host model submissions | 60 | 60 |
| Device notify records / waits | 336 / 336 | 336 / 336 |
| Device label switches | 1,713 | 1,673 |
| DMA tasks | 3,258 | 3,258 |

The runs use separately leased devices. Counts include construction, boundary
work and assertions; this comparison establishes dispatch changes, not throughput,
latency improvement or overlap. No logged host fallback was observed.

Two affected C++ objects were rebuilt in each runtime; the new merge kernel is
reused from a passed, source-identical development build. Other source/object/
archive reuse is byte-verified. No development failure occurred in this increment;
earlier failures remain in the preceding report. Local environment remains
aarch64 Ascend910_9392, CANN 9.0 and Torch/TorchNPU 2.10.

Frozen source/build: `TASK/sources/full-reverse-merge-clean01` and
`TASK/builds/full-reverse-merge-clean01`; Python build
`full-reverse-merge-python-clean01`, where
`TASK=/mi/data2T/zlong/tide-execution-flows`.
Queue/run/build limits are 120/600/900 seconds; profile storage is 512 MiB.

```bash
python "$TASK/launchers/full_reverse_merge_evidence.py" fe68065fbf41eebbba003051a84d3567e59028d1
```

Canonical device alias reduction, atomic multi-device optimizer publication,
public multi-device training and complete throughput comparisons remain pending.

# Public multi-device resident training qualification

Qualified source: **0d7c45eb417e827f6bc77c1ee9f876d9663ab215**.
The [audit](public-sharded-training-20261002.json) verifies ten terminal jobs,
standalone/Python runtime separation, recursive host dependency recompilation,
byte-identical object/archive reuse, loaders, installed consumer and raw profile
hashes. All ten passed; no numerical tolerance was relaxed.

## Delivered behavior

The public `ResidentTrainingSession` now exposes compact multi-device state/KV
and Full owners, consumer loss cotangents, canonical gradient shards, separate
backward and atomic SGD/AdamW step, and complete continuation. Its portable schema 1
checkpoint supports changing card count, placement, schedule, and restoring into
the original single-device implementation. Python remains a C++/CANN client.
See the [interface contract](../resident-sharded-training.md).

Initial model construction can explicitly stay on CPU before loading compact NPU
banks. It does not run a CPU event prepass or create a full forward state/KV replica
on the coordinator. Outputs/pending packets remain on the coordinator; returned
states and their loss roots remain owner-local. Static parameter aliases are
established before forward. Connection masks preserve None versus connected zero.
Window/session identity guards reject stale, foreign and out-of-order roots.

## Fixed-source checks

| Gate | Passed scope |
| --- | --- |
| Standalone, each FP32/FP16 | 32 trajectories / 512 windows / 128 updates per dtype; independent CPU FP32/FP64 reference |
| Placement/restore boundaries | Four cases, 64 windows / 16 updates: 3→legacy single with distinct Full/state maps, 2→3 FP16, 1→2 FP32, 1→legacy single FP16 |
| Python public and legacy clients | 41 passed, no skips; three graph families, both schedules, scalar oracle, loss roots and fresh-process checkpoint suffix |
| CPU interface checks | 25 passed, no skips |
| Installed public-header-only C++ client | Three inference windows and three retained training windows, two devices, optimizer restoration |
| Legacy standalone FP16 cache training | One trajectory / 16 windows / four updates |
| Isolated builds | Standalone and Python-owned shared libraries; current CMake package installation and independently compiled external consumer |

The C++ trajectory matrix covers streaming/prefill, SGD/AdamW, HARD/HST/SOFTP,
Full/state/cache profiles, actual repeated updates and forward publication,
aliases, None/zero, atomic refusals, capacity errors and after-close tensor lifetime.
Successful development checks and the eight earlier failures remain separate
from the clean-source qualification. The public struct ABI changed; bindings,
consumers and affected recursive header dependents were rebuilt.

## Profiling feedback

An earlier successful development trace contained **16 AiCPU Bool ScatterUpdate**
tasks while assembling the public loss-root connection masks. Replacing those
boolean scatters with device int32 flags and an exact bool cast removed that
fallback. Logical times and counters remain int64.

The separate clean-source two-card FP16 public training trace records **17,690
AI_VECTOR_CORE, 256 AI_CORE and 302 MIX_AIV**, with **no observed AiCPU operator**.
Both devices execute state VJP. There are 112 model executions, 816 matching device
notify record/wait pairs, 1,909 label switches and 7,063 DMA tasks. The trace covers
one trajectory, 16 windows and four updates, including setup, exports, checks and
refused updates. These counts are not throughput measurements or a CPU speedup.

## Preserved failures and limitations

The audit retains eight failed development jobs: two generated-kernel/header
lookup errors, a missing direct declaration dependency, a test const mismatch,
a task-local incremental link omission and its Python import failure, an attempted
install of unrelated unbuilt vendor archives, and a profile source-guard refusal.
The linker now rejects unresolved public symbols. The package check installs the
public package with the byte-verified shared library rather than claiming a fresh
build of all vendor archives.

The source-guard failure came from a concurrent vendor runtime writing
`fusion_result.json` in the frozen checkout. That failed record is unchanged.
Subsequent runtime commands use their own output directory; the final frozen
checkout remains clean. Successful development objects were reused only after
matching dependency and byte hashes, then freshly linked on the immutable source.
This is affected-source qualification, not a from-scratch vendor-kernel rebuild.

Local environment: aarch64 Ascend910_9392, CANN 9.0.0, Torch/TorchNPU 2.10.0.
The unchanged 8,954-check CPU suite was not rerun. Public scale consumers, full-size
CPU/mixed/resident timings, other CANN versions and target-machine CUDA remain
pending. The Python client is not an independent PyTorch device scheduler.

Local reproduction uses `TASK=/mi/data2T/zlong/tide-execution-flows`, frozen source
`sources/public-shards-clean01`, builds `public-shards-clean01` and
`public-shards-python-clean01`, and the retained run records named in the audit:

```bash
python "$TASK/launchers/public_shards_evidence.py" 0d7c45eb417e827f6bc77c1ee9f876d9663ab215
```

No new qualification job remains live. Queue120s/run600s/build900s bounds were used;
F1–F7 are not complete. Historical CPU Attention remains deliberately paused.

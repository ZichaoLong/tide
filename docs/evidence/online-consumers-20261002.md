# Continuous public model consumer qualification

Qualified source: **fe2d8869f2c0bdb5f59da525a7687d91e341fc78**.
The [audit](online-consumers-20261002.json) verifies five terminal passing jobs,
clean frozen source, newly compiled independent CPU/NPU clients, byte-verified
unchanged installed core, loaders, binary/packet/result hashes and profiler CSVs.
No numerical tolerance changed. See the [consumer contract](../online-consumers.md).

## Delivered and checked

The v2 consumer independently constructs real per-edge D×D projections, Add or
same-fiber Attention, learned embedding/head and exactly counted parameters from
the common packet. CPU FP32/FP64 and mixed A/B/C FP32 use the public schedulers,
with streaming or online greedy prefill and explicit placement switches. Complete
training includes next-token CE, finite-gradient agreement, SGD/AdamW, explicit
update-boundary detach and continuous state/history/KV/pending/input ledger.
Python execution, the Python native adapter and standalone LibTorch are distinct
clients. Candidate execution never consumes a CPU reference trajectory.

| Gate | Fixed-source result |
| --- | --- |
| CPU consumer/packet/affected Settle checks | 120 passed, no skips |
| Standalone CPU within that gate | 26 trajectories, 104 windows, 52 updates; FP32/FP64, three families, both schedules, Add/Attention, SGD/AdamW, delayed input |
| Directed NPU mixed checks | 18 passed, no skips; six family/memory/schedule/preset cases × Python/native/LibTorch, 72 windows and 36 updates |
| Independent installed clients | Two fresh CPU/NPU builds, isolated standalone runtime and checked loader closure |
| Separate actual-model profile | TimedDAG Attention/mixed-C/prefill, D32/N12/71,128 parameters, AdamW, four continued windows and two updates |

Checks compare outputs, full state/KV/history/pending/ledger, events, parameter
gradients including None, and parameters after both updates. They also exercise
unified CLI packet conversion and preserved failure records. The v1 reset-window
packet is still available and explicitly refused by the continuous runner.
Settle construction shares body owners without a second full parameter allocation.

## Profiling feedback

The separate instrumented run includes construction and disables diagnostics.
It records **18,927 AI_VECTOR_CORE, 1,712 AI_CORE, 2,974 MIX_AIV and 192 AI_CPU**
tasks. AiCPU consists of **72 INT64 Sort and 120 BOOL/INT64/BOOL ScatterElements**.
There is no observed unexpected host CPU fallback. The actual body candidate
counts are96 and97 across the updates; maximum state sequence3 and Full batch6.

The trace also contains23,805 host kernel launches,795 stream synchronizations,
2,785 stream synchronizations with timeout and7 device synchronizations with
timeout. These identify host round trips, small kernels and AiCPU operations as
investigation targets. Counts alone establish neither bottleneck share nor a
CPU/NPU speed ratio. Instrumented step timings are excluded from throughput claims.

## Scope and reproduction

This qualifies bounded real-model consumers and continuation. It does not qualify
full-size throughput, resident slot-affine training/sharding, consumer FP16 master/
head updates, CUDA or other CANN versions. Resident broadcast training evidence
cannot certify the model's per-edge projections. The parameter storage guard is
not a total peak-memory admission model. Further safe chunking and F6 remain due.

Local stack: aarch64 Ascend910_9392, CANN9.0.0, Torch/TorchNPU2.10.0. Builds use
unchanged byte-verified core archives rather than rebuilding unrelated kernels.
The unchanged8,954 CPU suite was not repeated. Both failed dev01 client builds
(missing tide/kernel.h include) remain intact alongside the fixed clean builds.

Use TASK=/mi/data2T/zlong/tide-execution-flows, frozen source
sources/online-consumer-clean01, builds online-consumer-{cpu,npu}-clean01, and the
five runs listed in the audit. Audit command:

```bash
python "$TASK/launchers/online_consumer_evidence.py" fe2d8869f2c0bdb5f59da525a7687d91e341fc78
```

All jobs used bounded build600s/run600s/queue120s limits and are terminal. The
historical CPU Attention job remains deliberately paused; no new formal full-size
performance conclusion is drawn.

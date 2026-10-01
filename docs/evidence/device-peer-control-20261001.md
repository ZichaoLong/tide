# Reusable packets inside two-NPU device loops

Implementation:`5c3662bd1162a9155c933f4953f8ff191b6fdc40`.
All four fixed-source jobs passed/exit0:affected component build,two-device
FP32/FP16 gate,single-device regressions and a separate FP32 two-device profile.
[Audit](device-peer-control-20261001.json) authenticates source,core/binaries,
loaders,terminal logs and raw CSVs. [Transport contract](../resident-peers.md).

## Verified behavior

Each dtype completes11 continued windows against independent CPU evolution.
The receiving NPU updates the counter returned to the sending NPU;that result
controls subsequent loop iterations and termination on device. The same programs
accept changed targets/increments/capacities without host event decisions.
Coverage includes empty windows,bounded partial progress and resumption,
129 iterations through one call site,exact keys/counters above2^55,bool masks,
FP32/FP16 payloads and contiguous views with nonzero storage offsets.
Packet capacity,shape/layout/dtype,overlapping destination and lifecycle guards
are checked. Four logical notifications suffice for the two directions;each
ready/pull/consumed handshake completes before buffer or notification reuse.

`CannProgram::submit()` separates submission from boundary waiting. All peers
are submitted on the constructing thread before any wait;the existing `run()`
remains submit+wait. Cross-thread calls fail before a CANN task is issued.
Duplicate submission and waiting without an outstanding submission are refused.
Existing control(8 cases),failure ownership(5 injected cases),and FP32/FP16
numerical checks passed as four regression cells. Unchanged portable core was
byte-verified and reused;no full CPU or vendor-kernel rebuild was necessary.

## Actual trace and limits

Local aarch64 Ascend910_9392,Torch/TorchNPU2.10,CANN9.0,isolated standalone SDK.
Qualification used physical1/9 and the independent profile physical11/13,
remapped to logical0/1. This is correctness/placement evidence,not a performance
comparison between those allocations. Both profile devices appear in raw CSVs.

The profile contains1,946 AI_VECTOR_CORE tasks and2 AI_CPU tasks. Both AiCPU
operations are INT64 OnesLike initializers and finish before their device's
first MODEL_EXECUTE;the audit checks timestamps. They are on-accelerator setup,
not host CPU fallback or evidence that the repeated loop uses AiCPU. No logged
host fallback was observed. The trace includes setup,CPU assertions and exports.

Host ACL counts are22 model submissions (11 windows×2 devices),4 notify creates,
4 notify records and4 notify waits. Device tasks include636 NOTIFY_RECORD,
636 NOTIFY_WAIT,614 label switches and1,535 MEMCPY_ASYNC tasks. These totals
include runtime-owned tasks and boundary copies;they are not pure payload byte
counts. The contrast between fixed host protocol construction and repeated
device tasks supports the intended execution location. Summed task time is not
end-to-end wall latency. No throughput or graph-scale claim follows.

## Scope and retained failures

This is an internal transport prerequisite. It does not certify a multi-device
graph scheduler,graph observables,distributed VJP/optimizer,parameter sharding
or the full-size performance matrix. Those remain required under ROADMAP F4–F7.
No CPU reference event or numerical result is supplied to the candidate.

Development dev01 failed in the CPU fixture's unsupported bool tensor initializer;
conversion from int64 fixed it. Dev02 submitted a raw CANN model from a fresh
host thread lacking context:107002 CONTEXT_NULL,then507046 peer wait timeout.
Resources were quarantined and the worker failed;this is not a passing sample.
The asynchronous same-thread submission and explicit thread guard fix that
boundary. Original failures/logs remain in the audit;no tolerance changed.

Snapshot/build:`TASK/{sources,builds}/peer-control-clean01`,where
`TASK=/mi/data2T/zlong/tide-execution-flows`. Build checks:peer/control/failure/
numerical,2 build threads. Gate/profile queues request2 NPUs;regressions1.
Queue waits120s,build900s,run600s,profile raw storage256MiB. Re-audit:

```bash
python "$TASK/launchers/peer_control_evidence.py" 5c3662bd1162a9155c933f4953f8ff191b6fdc40
```

All four qualification tasks are terminal. CANN9 merges device rows into one
export directory;device coverage is checked from CSV Device_id columns.

# Original-size resident Add: independently audited device slice

Source `e69b3bde3d53b0d6a019e89e82d1a78c3a91a7b8`. The original
D2048/B512/T12/V50304, 9,468,053,696-parameter LibTorch/TimedDAG/prefill
FP32 resident inference completed on eleven NPUs. It used the previously passed
owner map, physical four rows/128 groups and capacities. One cold update spans
two connected windows, with no CPU reference prepass or profiling in formal timings.

`profile-formal-resident-add01` passed/exit0 at 2026-10-04T02:04:27.114218Z.
The service cgroup is empty and its lease released. Source, binary, packet,
configuration, allocator bounds and all 444 trace files were audited; see the
[JSON report](fullsize-resident-add-profile-20261004.json). The independently
hashed complete inventory is retained under the report's TASK audit path.

The requested five-second collection spans **5.72370525 seconds** from first
task start to last task completion across all eleven devices, including staggered
collection/edge tasks. It contains **168,322 tasks**:

| Observed engine | Tasks | Fraction of summed device-task duration |
| --- | ---: | ---: |
| AI_VECTOR_CORE | 166118 | 87.1795% |
| AI_CORE | 2190 | 12.8173% |
| MIX_AIV | 14 | 0.0032% |
| AiCPU | 0 | Not observed in this slice |

The largest task-duration groups include ScatterUpdate (1.1746 summed seconds),
GatherV3 (0.7184), BatchMatMulV2 (0.6463), `tide_frame_select` (0.6155), another
GatherV3 variant (0.5239), and `tide_state_shard_pack` (0.2919). The trace also
contains `tide_queue_propose`, `tide_ready_pack`, Full packing and projection
planning. Shape/dtype fields for these low-level records are N/A; they must not
be reconstructed from kernel names as if directly observed.

Host statistics include 151 StreamSynchronize calls and 3.8591 summed seconds,
plus 1360 SetDevice calls. ACL/runtime layers can nest. These counts do not
identify each call's source location or establish per-event host scheduling.
Physical sample-group boundaries and input/output reporting remain host work.

This observation supports investigating vector gather/scatter, selection,
packing and boundary synchronization costs; it does **not** establish a single
cause for CPU/NPU performance differences. No AiCPU was observed in this slice,
which does not exclude it elsewhere. Device tasks overlap across streams/cards;
the percentages above are not end-to-end time shares or utilization. The
instrumented cold update took 340.721409 seconds and is excluded from the
continued unprofiled performance matrix. Full-update/training and FP16 profiles
remain separate work.

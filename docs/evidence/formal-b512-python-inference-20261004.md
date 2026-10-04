# Original B512 Python CPU and native resident inference

All three first formal processes passed on clean
`e69b3bde3d53b0d6a019e89e82d1a78c3a91a7b8`, 2026-10-04.
The terminal-parent audit checked source/input hashes, the actual Python package,
native binding and resident binaries, resolved placement, owner maps, memory
estimates, work counters and timing bounds.
[Audited records and exact commands](formal-b512-python-inference-20261004.json).

Each case is TimedDAG/prefill/FP32 with original D2048/B512/T12/V50304,
Add9,468,053,696 or Attention17,521,117,376 parameters. One fresh process performs
one continued warmup and one measured inference step, two connected windows per
step. No phase instrumentation, profiler, reference prepass or supplied routes
enters the timings. The CPU cell uses the independent pure Python implementation;
the Python-owned NPU cells call the native resident implementation. They are not
standalone LibTorch processes or pure Python device schedulers.

| Cell | Inference case | Construction s | Warmup s | Measured s | Input tokens/s | Actual candidate/resident events | Physical rows × groups | Peak growth GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 72 | Pure Python CPU Add | 29.474405 | 546.504512 | 625.406563 | 19.648019 | 1,188,500 | 32 × 16 | 72.635326 |
| 74 | Python-owned native resident Add,11 NPUs | 48.282122 | 341.847487 | 342.512730 | 35.876039 | 1,188,494 | 4 × 128 | 7.282186 |
| 80 | Python-owned native resident Attention,11 NPUs | 112.179349 | 419.783453 | 421.433023 | 29.157658 | 1,190,499 | 4 × 128 | 13.528637 |

All measured steps consume12,288 input tokens, produce12,288 outputs and end at
cut816. Both resident cases report7,168 stages and128 physical groups; CPU reports
544 greedy stages and16 physical groups. The counters describe their respective
algorithms and are not interchangeable units of work.

CPU used ATen16/workers1 and startup OpenMP/OpenBLAS16, with MKL1 and
OMP_THREAD_LIMIT32. A read-only live-process inspection confirmed those variables,
autoload0 and unchanged CPU0–319/memory-node0–7 affinity. The resident cases used
eight host ATen threads, startup BLAS1 and11 actual owners under fresh cooperative
NPU leases. All own heavy timings remained serial; unrelated server load was
uncontrolled. This round did not qualify resource isolation or concurrent timings.

Peak growth is CPU RSS high-water growth, or the largest tracked allocator growth
on one NPU. The corresponding maximum estimates are114.559198GiB CPU,
25.293424GiB resident Add and45.775907GiB resident Attention. All observed owners
fit their estimates and declared caps. NPU allocator observations exclude
untracked driver/vendor memory and do not establish total host RAM consumption.

For Add, this one-process Python CPU/native NPU pair shows higher native NPU
throughput, but also six fewer events and a different physical grouping.
It is a descriptive complete-flow comparison, not a strict equal-work speedup
or a configuration recommendation. The [standalone LibTorch Add samples](formal-b512-add-inference-npu-20261004.md)
have a different ordering: their CPU case is faster than resident. Language,
implementation, grouping and host work therefore belong in every comparison.
Resident event counts match the corresponding standalone Add/Attention samples;
matching counts alone do not prove complete-state equivalence or isolate binding
overhead. The [strict near-tie route failure](original-add-route-witness-20261004.md)
remains a separately accepted numerical limitation, with checks unchanged.

CPU retained1800s/update and4000s/process bounds; both resident cases retained
900s/update and2200s/process. Warmup and measured steps fit their original bounds
and3000s. Construction and the resource-free dependency wait remain separate.
The historical3000s/1.15 refusals for other cases are unchanged.

`formal-python-short01` ended exit0 at2026-10-04T00:20:12.114767+00:00;
its service is inactive, cgroup empty and both NPU leases completed/released.
All three planned cells completed; none was silently skipped.
Raw records: `TASK/runs/formal-python-short01/assessment/cell-N-repeat-1`.
Re-audit with a new output path:

```bash
python TASK/launchers/audit_formal_blas_group_v2.py \
  --name formal-python-short01 --output NEW_AUDIT_JSON
```

Together with the five prior audited cells, first-process coverage is now8/120
(0,1,2,3,8,72,74,80), with112 unmeasured cells and no three-process recommendation.
`TASK/audits/formal-matrix-review03` checks unique cell/repeat identities and keeps
all120 cells explicit. It covers these formal processes only, and does not erase
historical failures or replace reduced-batch/FP16 evidence. The [parallel resource
proposal](parallel-resource-review-20261004.md) is inspection-only. The user
requested a pause after this evidence commit/push; new experiments await authorization.

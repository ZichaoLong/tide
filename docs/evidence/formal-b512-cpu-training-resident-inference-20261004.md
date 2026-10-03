# Original B512 CPU Add training and resident Attention inference

Both first formal processes passed on clean `e69b3bde3d53b0d6a019e89e82d1a78c3a91a7b8`, 2026-10-04. The terminal-parent audit checked the actual source, binary, runtime, owner maps, capacity estimates, work counts and timing bounds. [Audited records and commands](formal-b512-cpu-training-resident-inference-20261004.json).

These are independent LibTorch/TimedDAG/prefill FP32 executions of the original D2048/B512/T12/V50304 packets. Add retains 9,468,053,696 parameters and Attention 17,521,117,376. Each fresh process performs one continued warmup and one measured step, with two connected windows per step. No phase instrumentation, profiler, diagnostic reference or supplied CPU routes enter these timings.

| Cell | Workload | Construction s | Warmup s | Measured s | Input tokens/s | Actual candidate/resident events | Physical rows × groups | Peak growth GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 3 | CPU Add complete training | 46.163312 | 893.228409 | 970.259169 | 12.664657 | 1,188,205 | 32 × 16 | 128.458973 |
| 8 | 11-NPU resident Attention inference | 147.308586 | 401.514133 | 402.400685 | 30.536727 | 1,190,499 | 4 × 128 | 13.528270 |

The rows have different models and modes; they do not form a CPU/NPU speed comparison. Each measured step consumes 12,288 input tokens, emits 12,288 outputs and ends at cut816. CPU training includes forward/loss, backward/accumulation, finite checks and SGD publication. Its measured loss was30.526611; Attention inference loss was20.897995. No training-quality or convergence claim follows from this benchmark objective.

CPU used ATen16/workers1 and OpenMP/OpenBLAS startup16; the live process environment was also inspected. Resident inference used11 owners and a fresh cooperative lease, with no CPU reference prepass. All own heavy work was serial, and no NPU lease remained during CPU training. Unrelated shared-server load remains uncontrolled.

Peak growth is process RSS high-water growth for CPU and the largest tracked allocator growth on any NPU. The corresponding largest admitted estimates were191.272808GiB and45.775907GiB. All observed owners, bounds and allocator checks passed; untracked driver/vendor memory is not included in allocator observations. The original capacities, aggressive-safe physical chunking and model geometry were unchanged.

Cell3 retained its3000s/update,6300s/process bounds. Cell8 retained900s/update,2200s/process bounds. Both warmup and measured steps fit those bounds and3000s. Construction and process overhead remain separate. The historical3000s/1.15 forecast refusals for other configurations are unaffected.

`formal-short-next01` ended exit0 at2026-10-03T23:31:02.569252+00:00; its cgroup is empty and its NPU lease released. The earlier resource-free dependency wait is part of the parent wall time, not either consumer timing. All two planned cells completed; none was silently skipped.

This brings the audited first-process coverage to5/120 cells:0,1,2,3,8. Three independent processes are still required before a formal configuration recommendation. The [strict near-tie route failure](original-add-route-witness-20261004.md) remains accepted as a separately listed limitation, not an equivalence pass. Actual work counts remain visible, and that witness does not explain every future discrepancy.

Raw records: `TASK/runs/formal-short-next01/assessment/cell-N-repeat-1`. Re-audit after creating the output parent:

```bash
python TASK/launchers/audit_formal_blas_group_v2.py \
  --name formal-short-next01 --output NEW_AUDIT_JSON
```

The dependent Python stage started only after this parent passed and became empty. Its native resident Add/Attention inference and pure Python CPU Add inference are separate measurements and remain uncertified by this report.

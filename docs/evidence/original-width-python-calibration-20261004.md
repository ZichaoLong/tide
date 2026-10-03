# Original-width Python CPU, resident and mixed inference calibration

All ten remaining Python pilots passed on clean `e69b3bde3d53b0d6a019e89e82d1a78c3a91a7b8`, 2026-10-04. This establishes original-width cost, continuation and memory observations for four CPU model/mode combinations, four Python-owned native resident combinations and two screened mixed inference paths. [Audited results, commands, work counts and provenance](original-width-python-calibration-20261004.json).

The original graph, D2048/T12/V50304 and parameter counts remain unchanged: Add 9,468,053,696 and Attention 17,521,117,376. Each fresh process uses a reduced logical batch, one continued warmup and one measured step, with two connected windows per step. These are phase-instrumented calibration runs, not original B512 performance or strict CPU/NPU equivalence evidence.

| Cell | Python-owned case | Batch/physical | Warmup/measured s | B512 forecast warmup/measured s, including 1.15 | Original 3000s forecast guard |
| --- | --- | ---: | ---: | ---: | --- |
| 72 | CPU Add inference | 64/32 | 75.278124/86.416619 | 692.558741/795.032892 | within |
| 75 | CPU Add training | 64/32 | 173.928417/183.942678 | 1546.406835/1659.059713 | within |
| 78 | CPU Attention inference | 64/32 | 245.493008/334.782519 | 2258.535670/3079.999173 | refused |
| 81 | CPU Attention training | 32/16 | 577.412944/686.670636 | 10340.528306/12475.200981 | refused |
| 74 | Native resident Add inference | 8/4 | 5.471506/5.349334 | 402.702861/393.710993 | within |
| 77 | Native resident Add training | 4/2 | 22.834645/20.011077 | 3134.373815/2726.292286 | refused |
| 80 | Native resident Attention inference | 8/4 | 6.685855/6.530239 | 492.078901/480.625579 | within |
| 83 | Native resident Attention training | 2/1 | 28.448047/25.587430 | 7613.998516/6767.210094 | refused |
| 73 | Mixed-B Add inference | 32/32 | 159.713249/191.703858 | 2938.723782/3527.350985 | refused |
| 79 | Mixed-A Attention inference | 16/16 | 137.927404/177.782961 | 5075.728482/6542.412981 | refused |

CPU uses ATen16/workers1 and explicit OpenMP/OpenBLAS startup16. CPU and resident pilots use two physical groups; mixed inference uses one. The one-group mixed pilots do not certify the eventual B512 multi-group lifetime. All work was serial. Each accelerator case acquired a fresh 11-card lease; CPU phases held no NPU lease. External shared-server load remains uncontrolled.

Every case produced the expected `24×batch` outputs and final cut816, with finite observations and allocator/RSS growth below its unchanged estimate and usable budget. Training includes forward/loss, backward/accumulation, finite checks and SGD publication. Forecasts scale measured sample work to B512 and add the measured once-per-step optimizer cost, then apply1.15. This extrapolation is a budget basis, not a measurement; nonlinear scaling and different actual work can change full-size time.

The four resident cases execute through Python-owned native C++/Ascend C libraries. Their observed package hash, native binding, resident libraries, runtime owner, resolved placement and owner maps were checked against the frozen source and current build manifests. Pure Python CPU/mixed records were checked separately; a matching standalone executable is a provenance prerequisite, not evidence that Python executed that binary. The two manifest schemas are handled explicitly, with no relaxed identity or numerical comparison.

Construction is separate from step timing. For native resident Attention it took180.253810s in inference and212.081803s in training; mixed Attention construction took210.874665s. These costs matter to cold or finite runs and are not included in the table's measured-step times. No formal Python-versus-C++ speed recommendation follows from these single, reduced-batch processes.

The terminal-parent audit verified the frozen inventory, source/build and actual runtime identities, helper/packet/result/log hashes, original parameter counts, physical group counts, output/cut boundaries, exact owner maps, observed memory, phase sums and all forecasts. Parent `remaining-python-calibration01` ended exit0 at2026-10-03T22:42:51.857555+00:00, with an empty cgroup and all leases released. No CPU reference routes, events, outputs or gradients fed any candidate. Existing strict numerical failures and the earlier Python B64 timeout remain unchanged.

Ten unexecuted budget02 files now have this audited basis: CPU Add inference1800s/update and training3000s; CPU Attention inference4500s and training16500s; resident Add/Attention inference900s, Add training4500s and Attention training9000s; mixed Add inference4500s and Attention inference9000s. Longer budgets retain every original3000s refusal. The existing mixed training budgets remain separately based on the [prior accelerator pilots](original-width-accelerator-calibration-20261004.md). An operating budget does not certify B512 execution.

Raw records: `TASK/runs/remaining-python-calibration01/assessment/cell-N`. Re-audit after creating the output parent directory:

```bash
python TASK/launchers/audit_python_calibration.py --output NEW_AUDIT_JSON
```

`audit_flow_runtime.py` supplies the checked manifest adapter. Its development checks used actual standalone, pure Python and native records. A separate formal-audit development attempt initially failed because its output directory did not exist; that attempt is retained in `TASK/audits/formal-add-runtime-schema-v2-attempt-01`. The unchanged audit then passed after directory creation. This was an audit-output error, not a failed consumer or calibration.

The full-size matrix and recommendation repeats remain open. The separately queued formal stage now runs resident Attention inference and CPU Add training only after this calibration parent passed and its cgroup became empty. Those formal processes are not certified by this report.

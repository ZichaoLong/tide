# Original B512 Attention complete-training feasibility

Clean source `29effaed064d59b9da930ec3acec810be58b2e1a`, 2026-10-03.
Original D2048/B512/T12/V50304 Attention with17,521,117,376 parameters completed
one independent FP32 SGD update on11 NPUs. Two connected windows per sample,
physical B1×512, TimedDAG/LibTorch/resident/prefill. The
[machine audit](original-b512-attention-training-20261003.json) checks exact
source/build/packet identities, terminal status and all acceptance artifacts.

| Measurement | Result |
| --- | --- |
| Complete update | 5695.490452595s |
| Sample forward/loss/backward work | 5692.808817143s |
| Once-per-update finite checks/optimizer | 2.681635452s |
| Construction, separate | 164.822193006s |
| Outputs / final continuation cut | 12288 /408 |
| Finite loss | 21.380962371826172 |
| Actual event / stage counts | 1,184,430 /28,672 |
| Largest framework allocation growth | 42.032GiB |
| Largest per-card estimate | 53.649GiB |
| Usable budget per card | 53.875GiB from60GiB cap |
| Largest saved continuation | 1,826,873,240bytes within4GiB/card |

All11 allocator peaks were below their unchanged estimates; all saved-context
peaks fit their pools. Operator rows stayed full16/emission1/aggregate8/attention1/
keys128/reverse1/head64, with the same explicit owner map, queues, KV/journals,
physical B1 and binary as the qualified original-width B4 pilot.

The retained [B4 diagnosis](original-width-attention-owner-diagnostic-20261003.md)
forecast7036.453031774s with coefficient1.15 and refused the original3000s gate.
Before launching this new run, a separate9000s complete-update budget and9480s
child timeout were declared from those measured phases. The actual/forecast
ratio is0.8094263441. The old3000s refusal still stands; no memory protection or
historical failure was removed or reclassified.

`wide-attention-b512-extended01` passed/exit0; its user service is inactive/dead
with empty cgroup and its11-card lease is completed. All1487 frozen source
hashes match. The audit also verifies source-matching installed resident library,
consumer sources, loader, exact packet and preserved pilot/plan hashes.
Raw records: `TASK/runs/wide-attention-b512-extended01/assessment`, including
`result.json`, `original/result.json`, `original/consumer/result.json` and logs.
Build: `TASK/builds/owner-map-consumer-clean01`. Re-audit:
`python TASK/launchers/wide_attention_b512_evidence.py 29effaed064d59b9da930ec3acec810be58b2e1a`.

This is a cold full-size feasibility run with phase synchronization, no warmup
or profiler. Finite CPU builds/gates/calibration overlapped; other current-task
NPU qualification/profiling waited for termination. It is not isolated formal
throughput, a speedup recommendation or a full-size CPU gradient/update oracle.
Candidates independently initialize and execute common inputs; independent CPU
numerical equivalence remains anchored by smaller qualified complete-training
cases. Original-scale CPU/mixed/resident comparisons, other required schedules/
families/clients, repeated recommendations and separate profiles remain open.

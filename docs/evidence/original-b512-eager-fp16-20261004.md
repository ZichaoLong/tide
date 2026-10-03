# Original B512 eager FP16 complete training

Both original-size models completed one independent FP16 SGD update on eight
NPUs, including two connected windows and all physical sample chunks. Memory,
finite-value and unchanged 3000-second update guards passed.
[Audited results, exact commands and artifact hashes](original-b512-eager-fp16-20261004.json).

Tested clean source: `7b1fae504ec779143655b9221c82d8c14a69b410`. The previously
qualified eager FP16 consumer is unchanged. This is mixed-A/LibTorch/TimedDAG
online prefill, aarch64 Torch/TorchNPU 2.10 and CANN 9.0.0. Physical devices
1, 2, 3, 4, 5, 7, 9, 11 map to logical devices 0–7.

## Actual original-size execution

Each model has 480 reachable body nodes and 2208 body edges, D2048/B512/T12/V50304.
Add has 9,468,053,696 learned parameters; Attention has 17,521,117,376.
The original hashed packets, initializer, online algorithm and update boundary
are unchanged. No CPU prepass supplies routes, events or gradients.

| Measurement | Add | Attention |
| --- | ---: | ---: |
| Physical sample rows × chunks | 32 × 16 | 8 × 64 |
| Construction seconds | 44.597335 | 80.146901 |
| Complete update seconds | **1254.205147** | **2334.246753** |
| Sample work seconds | 1251.427050 | 2330.434492 |
| Final finite checks / optimizer seconds | 2.778097 | 3.812260 |
| Loss | 30.625061 | 21.387640 |
| Candidate events | 1,183,449 | 1,184,548 |
| Maximum observed allocator growth, GiB/card | 17.633 | 33.753 |
| Maximum estimated peak, GiB/card | 37.864 | 51.312 |

Each update produces 12,288 outputs and ends at cut 408. All cards remain within
their individual estimates and the retained 60 GiB/card budget with safety
margin. CPU resource reservations account for the concurrently running CPU
Attention feasibility job and protected historical worker.

Payloads and ordinary autograd gradient accumulation use FP16. Loss, optimizer
masters and slots use FP32, with explicit static loss scale 128. All chunks
accumulate into one logical update. This differs from the resident backend's
explicit FP32-adjoint policy. There is no automatic loss-scaling retry or skipped
update.

## Interpretation and limits

These are cold, synchronized phase diagnostics: one update, no warmup, overlapping
CPU feasibility work. They are not formal steady throughput or an isolated FP16
speedup. The earlier FP32 mixed run used eleven cards; its Add/Attention event
counts were 1,183,427 / 1,184,436. Thus neither equal work nor cross-dtype route
equality is asserted. Lower precision preserves the declared mathematical
operations while rounding can change numerical scores and subsequent branches.

The previously measured phase forecasts, including the unchanged 1.15 factor,
were 1778.033584 / 2488.157225 seconds. Actual updates passed the original
3000-second guard. The older Attention physical-B4 cost refusal and physical-B16
capacity refusal remain retained; neither was overwritten by this successful
physical-B8 run.

`wide-eager-half-b512-01` is terminal passed/exit0, with an inactive unit, empty
control group and released eight-card lease. The immutable source inventory,
consumer bytes, pilot identities, plans, results and logs passed the audit. No
host tensor-compute fallback warning was observed. Raw records are under
`TASK/runs/wide-eager-half-b512-01/assessment/{add,attention}`, where
`TASK=/mi/data2T/zlong/tide-execution-flows`.

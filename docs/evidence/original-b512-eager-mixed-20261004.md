# Original Add and Attention B512 mixed training

Both independently initialized original-size mixed-A updates passed on clean
`c68609603c310f7121cb6f887afcadd019973209`.
The [audit](original-b512-eager-mixed-20261004.json) verifies all1530 frozen
source files,installed client,inputs,plans,terminal job and released11-card lease.

| Model | Parameters | Physical samples | Construction | Complete update |
| --- | ---: | --- | ---: | ---: |
| Add | 9,468,053,696 | B32×16 | 43.420s | 1287.284s |
| Attention | 17,521,117,376 | B8×64 | 73.953s | 2655.242s |

Both retain D2048/B512/T12/V50304,480 body nodes,LibTorch/TimedDAG/online
prefill,FP32 SGD,two connected windows and one whole-batch update. Mixed-A
places payload computation on11 NPUs and Read/control/selection/event progression
on CPU. Host execution uses ATen2/workers4,packed sources and batched Next.
Every observed device peak stays within its estimate and60GiB budget. Maximum
allocator growth was10,491,477,504bytes for Add and20,570,653,696bytes for Attention.

Each result has12,288outputs and final cut408. Add records1,183,427candidate /
208,896selected events and loss30.50037384033203;Attention records1,184,436 /
208,896 and loss21.38123893737793. Actual input preparation,online scheduling,
packing,communication,forward/loss,backward,finite checks,accumulation,optimizer
and synchronization are included. Neither process consumes CPU reference
events,routes or gradients.

Actual updates and margin-inclusive forecasts1727.950s/2820.988s pass the
unchanged3000s guard. Original capacities and the1.15 coefficient were retained.
The Add result matches the completed CPU Add result's four discrete work
counters,output count and cut;loss satisfies1e-6 absolute plus1e-5 relative.
This is not a complete gradient or trajectory comparison.

These are cold feasibility runs with overlapping diagnostic/correctness work,
without warmup,repeated timing or profiling. They do not establish formal
CPU/mixed/resident speedup recommendations. The earlier resident Add event/loss
difference remains under diagnosis;CPU Attention B512 is still running.

`wide-eager-mixed-b512-01` ended passed/exit0 at
2026-10-03T16:04:49.612858+00:00. Raw results are under
`TASK/runs/wide-eager-mixed-b512-01/assessment`;audit command is
`python TASK/launchers/wide_eager_mixed_b512_evidence.py`,where
`TASK=/mi/data2T/zlong/tide-execution-flows`.

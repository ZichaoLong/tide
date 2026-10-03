# Original-width eager CPU and mixed chunk calibration

Four original-width Add/Attention complete FP32 SGD updates passed on clean
`c68609603c310f7121cb6f887afcadd019973209`. The
[receipt audit](original-width-eager-chunks-20261003.json) verifies both terminal
jobs, source inventory, binaries, packets, plans and actual memory checks.
These processes retain D2048/T12/V50304,480 body nodes and the full9.468B/17.521B
parameter sets. They use two connected windows and independently initialized
candidates. The logical batch is reduced only for this declared calibration.

| Flow | Memory | Logical / physical batch | Update seconds | B512 phase forecast, ×1.15 |
| --- | --- | --- | ---: | ---: |
| CPU,ATen16/worker1 | Add | 64 /32×2 | 166.402 | 1502.146s |
| CPU,ATen16/worker1 | Attention | 64 /32×2 | 2559.648 | 23391.026s |
| Mixed-A,11 NPUs,ATen2/workers4 | Add | 64 /32×2 | 188.765 | 1727.950s |
| Mixed-A,11 NPUs,ATen2/workers4 | Attention | 16 /8×2 | 77.788 | 2820.988s |

All use LibTorch/TimedDAG/online prefill,packed sources and batched Next,with
80GiB parameter and4GiB head budgets. CPU total admission uses512GiB and NPU
admission60GiB per card,with unchanged aggressive safety margins. Every observed
peak remains within its estimate. CPU RSS growth is110.683/232.169GiB against
184.393/402.231GiB estimates for Add/Attention. Per-card mixed observations and
estimates are retained in the audit. Outputs are24×logical batch,final cut408,
and losses are finite; full-size gradient equivalence is not claimed.

The forecast scales only sample forward/loss/backward/continuation work to512
samples,charges the optimizer once,and then applies1.15. No forecast is an
actual B512 completion. The CPU Add and both mixed forecasts admit a separate
run under the existing3000s step guard. CPU Attention's forecast refuses that
guard; its refusal is preserved. No memory margin,capacity,precision,topology
or model semantic is relaxed to obtain admission.

CPU calibration overlapped bounded development work; mixed calibration briefly
overlapped CPU B512 feasibility. The Attention batches differ across backends.
Consequently this table supplies capacity/cost evidence,not a formal speedup
comparison. Formal comparisons require matched logical inputs,declared warmup/
continuation,serial heavy timing and independent repeats.

Raw jobs: `TASK/runs/wide-eager-{cpu,mixed}-chunk01`,both passed/exit0 with
inactive services and empty cgroups. The mixed11-card lease is completed.
Source: `TASK/sources/eager-rss-clean01`; builds:
`TASK/builds/eager-rss-{cpu,npu}-clean01`. The archived helper is
`assessment/launcher.py` in each run and its hash is in the audit.

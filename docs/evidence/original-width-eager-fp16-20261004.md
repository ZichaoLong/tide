# Original-width eager FP16 calibration

Clean implementation `7b1fae504ec779143655b9221c82d8c14a69b410`, 2026-10-04.
All three complete-update pilots passed, with eight-device memory observations
within admission estimates. [Machine audit](original-width-eager-fp16-20261004.json).
Both task services are inactive with empty cgroups and released leases.

Unchanged D2048/T12/V50304 and 480 body nodes: Add 9,468,053,696 parameters;
Attention 17,521,117,376. LibTorch/TimedDAG/prefill/mixed-A, eight NPUs, ATen2 /
workers4, FP16 payload and autograd gradients, FP32 loss/masters, static scale128.
One SGD update consumes two connected windows; each pilot uses two physical
sample chunks. Loss reduction and the optimizer boundary cover the logical batch.

| Model | Logical / physical batch | Complete update (s) | Peak growth, worst card (GiB) | B512 phase forecast ×1.15 (s) |
| --- | --- | ---: | ---: | ---: |
| Add | 64 / 32 | 195.599333427 | 17.406 | 1778.033584300 |
| Attention | 8 / 4 | 50.268560833 | 32.594 | 3418.386618841 |
| Attention | 16 / 8 | 71.174303463 | 32.951 | 2488.157225114 |

The rows4 Attention forecast still refuses the original 3000-second guard.
Original B512 static planning refuses rows16 at 63.887 GiB/card and admits rows8
at 51.312 GiB/card, under the 53.875 GiB usable budget. The additional bounded
rows8 pilot confirms actual execution at that physical size. Select Add32 and
Attention8 for the subsequent B512 feasibility run, retaining both refusals.
No memory margin, dtype, parameter count or logical model was changed to admit it.

Forecasts scale measured sample work and add one optimizer phase; they are not
B512 measurements. Timings are cold diagnostic observations overlapping CPU
feasibility, without warmup, repeats or profiler. They do not establish formal
throughput, cross-dtype route equality or a FP16 speedup. All outputs, cut408,
finite losses, source/build/packet/result hashes and terminal receipts were audited.
No host tensor-compute fallback warning appeared; this is not a new hardware trace.

Raw: `TASK/runs/wide-eager-half-chunk01` and
`TASK/runs/wide-eager-half-attention-rows8-01`; frozen source `eager-half-clean01`,
consumer `eager-half-npu-clean01`. Re-audit with
`python TASK/launchers/eager_half_followup_evidence.py`.

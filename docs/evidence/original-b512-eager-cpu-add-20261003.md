# Original Add B512 CPU complete training

One actual original-size CPU update passed on clean
`c68609603c310f7121cb6f887afcadd019973209`. The
[audit](original-b512-eager-cpu-add-20261003.json) verifies all1530 frozen source
files, installed consumer, input, admission, raw results and terminal service.

| Item | Actual execution |
| --- | --- |
| Model | Add9,468,053,696 parameters;D2048/B512/T12/V50304 |
| Flow | LibTorch/TimedDAG/online prefill,FP32 SGD,ATen16/worker1 |
| Update boundary | Two connected windows;physicalB32×16;one logical update |
| Construction | 50.482667954s,separate |
| Complete update | 1458.897225208s |
| Outputs / final cut | 12,288 /408 |
| Candidate / selected events | 1,183,427 /208,896 |
| Loss | 30.500370025634766 |
| Host RSS growth | 126,853,771,264bytes;within admission |

The candidate independently consumes the hashed packet; no precomputed reference
events,routes or gradients drive it. Packing,loss/backward,finite checks,gradient
accumulation,optimizer and synchronization are included. The measured chunk
pilot's phase forecast1502.145802685s and actual update both satisfy the existing
3000s guard. The1.15 margin and512GiB declared CPU memory budget were retained.

This was a cold feasibility run with overlapping diagnostic/correctness work,
without warmup or profiling. It is not a formal throughput recommendation.
Earlier resident B512 Add recorded1,183,429events and loss30.50836181640625;
that difference is unresolved. This report does not certify full-size
CPU/resident discrete or gradient equivalence. Matched comparisons,repeats,
profiles,CPU Attention completion and final integration remain open.

`wide-eager-cpu-add-b512-01` ended passed/exit0 at
2026-10-03T15:16:53.981916+00:00;service inactive and cgroup empty.
Raw:`TASK/runs/wide-eager-cpu-add-b512-01/assessment`.
Audit:`python TASK/launchers/wide_eager_cpu_b512_evidence.py`;
`TASK=/mi/data2T/zlong/tide-execution-flows`.

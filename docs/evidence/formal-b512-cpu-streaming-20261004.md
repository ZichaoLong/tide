# Original-B512 CPU Add streaming observations

The original-scale CPU-only group ended **failed** on a time allowance,with one formally accepted case. PDG/LibTorch streaming passed at570.014357s. TimedDAG/LibTorch completed its consumer and resource monitor,but its699.995797s measured step exceeded the declared600s allowance. The remaining Settle/LibTorch,TimedDAG/Python and Settle/Python cases never started. [Reviewed records](formal-b512-cpu-streaming-20261004.json).

Workload source is `e69b3bde3d53b0d6a019e89e82d1a78c3a91a7b8`;controller `103f5b6c8e9b2e6e46a7f2733185433de2cb5f3f`. Both observations use originalD2048/B512/T12/V50304 Add,9,468,053,696 parameters,physical rows32,one continued warmup and one measured step,two windows each. CPU lane80cores/NUMA0–3,ATen/OpenBLAS16,512GiB model budget and the existing RSS allowance are unchanged. No competing project heavy job or profiler ran.

| Cell | Family | Warmup s | Measured s | Input tokens/s | Formal acceptance |
| --- | --- | ---: | ---: | ---: | --- |
| 24 | PDG | 492.411856 | 570.014357 | 21.557352 | passed |
| 36 | TimedDAG | 529.082826 | 699.995797 | 17.554391 | failed600s step allowance |

Both consumers produced12288 output tokens,final cut816 and1188500 candidate events. Construction45.751610s/45.259418s is separate. Each consumer's source/runtime/packet identity,finite records,memory envelope and sampled process placement were inspected;the failed row is preserved as a completed overrun,not relabelled passed. This does not establish independent wide numerical parity. Neither row has three-process evidence;there is no formal recommendation.

The parent `formal-bound-cpu-streaming01` exited1 at2026-10-04T04:38:31.304689Z,with an empty cgroup. `audit_bound_matrix_v2.py` accepted cell24 and retained cell36 plus three unstarted groups. The original600s step and1600s child allowances,failed group/case receipts and raw result are unchanged. No automatic retry or allowance increase followed. Report the slower completed row alongside accepted rows to avoid hiding the overrun;do not pool it into a passed repeat group.

The accepted first-process aggregate is14/120,with no three-process cell. New results remain in `numa-bound-solo-v1`,separate from prior unbound timings. Pending cells continue independently;the failure does not supply routes or data to any subsequent run.

# First original-B512 CPU, mixed and resident Add inference comparison

All three independent processes completed on clean `e69b3bde3d53b0d6a019e89e82d1a78c3a91a7b8`,2026-10-04. For this LibTorch/TimedDAG/prefill/FP32 Add workload,CPU was fastest in the first measurement. The11-NPU resident process took1.447times the CPU time;the mixed-A process took2.760times. These are descriptive ratios for one process per configuration,with differing actual event counts and physical chunks. They are not strict equal-work speedups or three-process recommendations.

[Audited NPU results and exact commands](formal-b512-add-inference-npu-20261004.json),[audited CPU result](formal-b512-first-cpu-20261004.md). Original D2048/B512/T12/V50304,480body nodes/2208edges and9,468,053,696 parameters are unchanged. Each candidate independently consumes its packet,parameters and current state;no CPU reference routes,events,outputs or gradients feed either NPU path.

| Observation | CPU | Mixed-A,11 NPUs | Resident,11 NPUs |
| --- | ---: | ---: | ---: |
| Measured step | 226.376559s | 624.907705s | 327.613456s |
| Input tokens/s | 54.281238 | 19.663704 | 37.507617 |
| Throughput relative to CPU | 1.000 | 0.362 | 0.691 |
| Continued warmup | 207.571919s | 497.381831s | 327.117733s |
| Construction | 46.319800s | 42.950041s | 62.390691s |
| Physical rows × groups | 32×16 | 32×16 | 4×128 |
| Events:host candidate / resident events | 1,188,500 | 1,188,498 | 1,188,494 |
| Body candidates | 1,163,924 | 1,163,922 | separate counter unavailable |
| Outputs per measured step | 12,288 | 12,288 | 12,288 |
| Final cut | 816 | 816 | 816 |
| Measured loss diagnostic | 30.302034378 | 30.302227020 | 30.305915833 |

Each process performs one continued warmup and one measured step,two connected windows each. Counts and losses are for the measured step;loss here is an inference diagnostic,not a training result. All required input preparation,scheduling,packing,computation,transport and synchronization are included. Construction is separate. Phase instrumentation,diagnostic exports and profiling are disabled. CPU uses ATen16/workers1/BLAS16;the declared mixed and resident host settings remain in the exact commands. Own heavy measurements were serial;external shared-server load is uncontrolled.

Both NPU cases passed exact packet,source/binary,owner-map,output/cut,finite-value and allocator checks. The peak device growth was3.632GiB for mixed-A and7.281GiB for resident,within each unchanged device estimate and60GiB/card allowance. CPU RSS growth was42.769GiB,a different memory metric. Warmup and measured times met their separately declared1200s(mixed)/900s(resident) operating bounds and original3000s threshold. The original-width forecasts remain forecasts;the table contains actualB512 execution.

Mixed-A recorded20,524cross-device copy groups and7,870,636,032explicit cross-device bytes;these counters do not measure all host/device traffic. Resident recorded7,168stages,36,157Full chunks and56,320emission chunks across128physical groups. Stage/chunk definitions differ from the host counters. These results establish end-to-end cost for the chosen configurations;they do not isolate scheduling,chunk size,communication,host work or AiCPU as a cause. Separate resident/full-size profiling remains required.

The [accepted near-tie strict failure](original-add-route-witness-20261004.md) remains failed. No route tolerance,tie rule,input or dtype changed. This run was not a full-observable strict cross-backend comparison,and the earlier witness does not independently explain these new aggregate count/loss differences. Do not label the table strictly equivalent. Nominal input throughput remains useful when its actual work and numerical limits are disclosed.

The NPU parent `formal-add-inference-npu01` terminated exit0 at2026-10-03T21:30:38.378910+00:00,with an empty cgroup and both11-card leases released. The audit checked frozen source inventory,matching core and installed combined consumer,packet/plan/budget and measured-basis hashes,actual owner maps,raw result/log hashes and queue completion. Re-audit:

```bash
python TASK/launchers/audit_formal_blas_group.py \
  --name formal-add-inference-npu01 \
  --output NEW_AUDIT_JSON
```

Raw cells:`TASK/runs/formal-add-inference-npu01/assessment/cell-1-repeat-1` and`cell-2-repeat-1`. The120-cell FP32 matrix now has three first processes completed. Recommendation repeats,remaining family/client/schedule/model/mode cells and separate FP16/profile work remain open.

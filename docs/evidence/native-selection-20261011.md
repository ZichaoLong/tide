# Finite complete CPU/NPU selection, 2026-10-11

The current implementation has five new single-process unprofiled observations,
three accepted complete traces and two explicit resident budget refusals.
The finite local performance tranche is closed; no retries, additional split
search or full-size matrix is queued. [Audited sources, results and traces](native-selection-20261011.json).

## Scope and measurements

All complete executions use clean `00b1775ce15afa20fb0552d51f6afdf4392ccc1c`,
with exact matching a926703 standalone NPU core/resident/online binaries.
Later commits change checker/profiling tools and evidence, not these production
model paths. Core digest remains
`0e2d0baa8fb2e3db1aed807c20c4ef78e53838922295dd4f60fb7578313266f0`.
This is standalone LibTorch, without Python-owned execution or a Python core
library dependency. The launcher is Python. The separate
[pure PyTorch component observations](npu-batched-components-20261011.md)
retain their own implementation and source identities.

Workload: TimedDAG prefill,32 body nodes/112 edges,D64,logical B8,T4,V257,
FP32,2 connected windows per complete training step. One continued warmup
and one synchronized measured step per independent process;Add SGD and
Attention AdamW. ATen/BLAS and node worker counts are1. NPU uses2 owners.
Each candidate computes its own schedule,values,gradients and update.

| Model / route | Measured step s | Construction s | Warmup s | Physical batch |
| --- | ---: | ---: | ---: | ---: |
| add-cpu | 0.443972 | 0.098344 | 0.396758 | 8 |
| add-mixed-c | 2.987056 | 0.815221 | 3.111524 | 8 |
| add-resident | 0.484066 | 0.954675 | 0.717104 | 4 |
| attention-cpu | 1.396360 | 0.131852 | 1.491035 | 8 |
| attention-mixed-c | 8.616372 | 0.907538 | 8.779485 | 8 |

These are single observations under accepted external contention,not estimates
of a latency distribution. The five-case service completed in68.42s with
sampled aggregate RSS peak5,019,664,384B. Each case stayed within180s/12GiB;
all output stayed within128MiB. No profiler/diagnostics ran in these processes.
Construction excludes process/toolchain startup;whole-process durations are
also retained in JSON. Profiled step times below are diagnostic only.

Add reports103 outputs and1648 candidate events on all three routes;
Attention reports99 outputs and1590 candidate events on CPU and mixed-c.
These matching aggregate counts do not prove identical complete trajectories.
The original strict near-tie witness remains a failure;no route/tie/tolerance
check was relaxed. Resident physical B4 uses two sample chunks with logical
batch/loss denominator,Attention visibility,connected windows and optimizer
boundary preserved by the existing independently qualified chunking path.
The current resident composite includes34 public sample-chunk cases.

For this moderate workload,prefer CPU. Add resident is close to CPU and much
faster than mixed-c here;the10% observed CPU/resident difference is insufficient
for a stable ranking under shared load. Attention mixed-c is slower than CPU;
Attention resident has no accepted timing at the tested budgets. Keep CPU,
mixed-a/b/c,resident and fine controls. Do not extrapolate these D64/B8 results
to 20-tide scale,other families/schedules/dtypes or all clients. Earlier
large-model tables keep their original source/configuration and are not
performance ceilings. No new full-size run is justified by these observations.

## Complete profiles and the bottleneck evidence

Collection covers construction,warmup,one complete forward/backward/update
step and window boundaries. Every accepted trace has actual operators on both
selected physical devices and accepted per-device allocator calibration.

| Profile | Operators | SetDevice calls | StreamSynchronize calls | Observed growth MiB/card |
| --- | ---: | ---: | ---: | --- |
| add-mixed-c | 125063 | 113312 | 22000 | 37.15/36.35 |
| attention-mixed-c | 202008 | 275482 | 15628 | 50.61/49.72 |
| add-resident-rows4 | 51308 | 18974 | 997 | 122.88/104.68 |

Add mixed-c includes52,956 Mul kernels,9,906 IsFinite kernels,1,077 AiCPU
ScatterElements and571 AiCPU Sort operations. Its125,063 host node launches
and repeated device switches/synchronizations accompany only0.337s of summed
device task duration across both devices. The profiled measured step is4.340s;
the separately unprofiled step is2.987s. Task sums span construction/warmup and
may overlap;host API levels nest. Neither number is a wall-time percentage.
AiCPU executes on the accelerator and is distinct from host CPU fallback.

Consumer statistics explain an important batching limit:all1481 Add body
candidate events and1427 Attention body events enter the selected-clear
prefill fallback;causal selection/clear dependencies prevent unrestricted
time-axis grouping. Native State/Read batching is active,with1648/1590 batched
events respectively,but the largest state step batch is8. Semantic Next replay
also remains per event. These are not evidence that Full/Aggregate batching was
missing;both are active in the complete consumer. The component improvements
therefore do not remove all whole-graph dispatch and autograd costs.

The Add resident trace contains212 device queue tasks. Float GatherV3 accounts
for78.692ms of summed device tasks and ScatterUpdate62.345ms;other visible costs
include journals,state packing,frame selection and queue proposals. These are
useful future optimization targets,not proof that every gather or wrapper
should become a custom kernel. Existing repeated-source gather reuse remains.
No additional fusion was implemented without a targeted causal comparison.

The actual resident timeline has84 host RIExecuteAsync submissions and the
expected registered device branches. In8 host completion-wait intervals,
24–28 queue tasks per interval run fully inside the wait,with no same-thread
host kernel launch,copy,new model submission or label mutation inside it.
Together with the implementation and separate correctness gate,this supports
in-window device progression for this complete Add case. It does not certify
all profiles merely from kernel presence. CANN exports the sentinel Model Id
4294967295,so model attribution cannot be inferred from that field.

Allocator observations exclude untracked driver/vendor allocations. CPU native
RSS growth is80,867,328B for Add and181,977,088B for Attention. NPU peaks in the
table are incremental framework allocation,not total physical HBM usage.

## Capacity outcomes and finite stop conditions

The original resident physical B8 Add profile was refused before valid timing:
`invalid compact reverse packet capacity, mapping or budget`. Total HBM admission
and reverse tensor reservations are different checks. Under the unchanged
2GiB backward budget,2 windows and2 owners,the reverse pack reservation is
11,184,810B. Existing all-choice static traffic bounds imply at most1280
event/atom rows for B8 and640 for B4;the conservative pack estimates are
17,043,456B and8,523,776B respectively. These topology bounds contain no actual
reference trajectory and do not drive the online schedule.

A separately declared B4 configuration passed Add execution,trace and real
memory calibration without increasing any budget. Attention B4 passed that
reservation screening but was refused by its separate cache-adjoint boundary
reservation. The same static screening was never claimed to prove all other
budgets. Both original failed records remain failed. No further split search
or automatic budget increase followed. This is a configuration capacity limit,
not a missing Attention backward implementation:the declared Attention modules,
training,continuation and multicard cases have independent correctness evidence.

Profiles were serial with600s/12GiB/2GiB per case and8GiB/card admission.
Five new unprofiled cases had a separately declared180s-per-case budget;they
were not retries of failed profile configurations. Occupied NPUs were allowed
under the user's explicit authorization;selection recorded two snapshots,
health and10GiB headroom. No foreign or protected historical process was
stopped,signaled or cleaned. Original F-stage queues were not restarted.

## Audit, reproduction and remaining work

Artifacts under TASK from STATUS:
`runs/g2-npu-complete-profiles12`, `runs/g2-npu-complete-profiles13`,
`runs/g5-native-selection01`, `plans/g2-resident-split-admission01.json` and
`plans/g5-native-selection01.{json,py,sh}`. They retain exact commands,packets,
binary/source/loader hashes,sharing snapshots,raw traces and native results.
Use new output directories and a separately declared finite budget for any
future experiment;these retained failed and accepted directories are immutable.

`g5-final-audit01` failed its own API-name assertion because CANN prefixes names
with AscendCL@/Runtime@. Corrected `g5-final-audit02` streamed the existing traces,
checked sources/binaries/device coverage/calibration and completed in48.47s
with95,584,256B peak RSS. It executed no model or measurement. The extra
`residency-windows.json` review records the actual host-wait intervals. All
current stage task units are terminal with MainPID0,empty cgroups and no guarded
survivors. The protected historical record is excluded. systemd limits are
requested;legacy cgroup v1 does not prove their kernel enforcement. Affinity,
sampled summed RSS,deadlines and process-group cleanup use the existing guard.

The local G1–G5 scope has implementation/correctness closure,finite selection
advice and a GPU handoff. Remaining performance costs include custom/nondefault
replay,selected-clear/causal dependencies,semantic Next replay,small tensors,
private reverse budget sizing,gather/scatter and communication. They are
explicit limits and possible future work,not a new automatic experiment queue.
CUDA source,local compilation and CPU contracts are delivered. Actual GPU
correctness,residency,multicard,memory/performance and tuning of its initial
single-thread semantic adapter require hardware;follow [the target gate](../device-control.md).

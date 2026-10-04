# Original-B512 FP16 Add training and inference trace slices

LibTorch/TimedDAG/prefill resident Add complete SGD training passed on eight NPUs:
**2027.354117s per measured step,6.061102 input tokens/s**. One continued warmup
was2035.162127s;construction83.335077s is separate. This is one fresh process,
not a stable recommendation. [Reviewed audit references](formal-b512-fp16-training-profiles-20261004.json).

Clean workload e69b3bd/controller103f5b6 reused the qualified installed consumers.
D2048/B512/T12/V50304 and9,468,053,696 parameters are unchanged. Physical sample
rows2 create256groups. Each step has two connected windows;the measured step
produced12288outputs,final cut816,1187990actual events and loss30.183405.
FP16 payloads use FP32 adjoints,loss and optimizer masters. Explicit process-local
ACLNN_CACHE_LIMIT=0,owner map,step4500s/child9400s bounds and the93-check cache-policy
gate remain unchanged. Maximum observed allocator growth39.987278GiB stayed
within each estimate. The service ended exit0 at2026-10-04T06:49:59.658769Z,
with an empty cgroup and released eight-device lease. There is no CPU prepass.

Two separately instrumented original-B512 FP16 inference jobs also passed
terminal audit. Each collected a requested five-second cold slice on all eight
leased NPUs,using the corresponding audited inference packet,dtype,owners,
physical sample rows and cache policy. The exported device spans are5.850126s
(Add) and5.735082s(Attention),not whole-step throughput samples.

| Slice | AI_VECTOR_CORE tasks | AI_CORE tasks | MIX_AIV tasks | Observed AiCPU tasks |
| --- | ---: | ---: | ---: | ---: |
| Add | 232203 | 4936 | 25 | 0 |
| Attention | 375205 | 21030 | 45 | 0 |

The sampled work includes device scheduler/packing kernels,Gather,ScatterUpdate,
Cast and matrix multiplication. These slices support a scoped engine-location
observation;they do not exclude AiCPU outside the interval or establish complete
training profiling. Counts and overlapping per-stream task durations do not
measure end-to-end utilization. Full-size FP32 Attention profiling remains queued.

FP16 timings stay separate from FP32. Dtype,cache policy,card count and owner maps
differ from the primary eleven-device configuration;no isolated dtype speedup or
strict wide trajectory-equivalence claim follows. Attention complete training and
fresh repetitions remain pending in the unattended queue.

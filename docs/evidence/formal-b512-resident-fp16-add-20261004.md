# Original-B512 resident FP16 Add inference

The original-size LibTorch/TimedDAG/prefill resident FP16 Add process passed terminal audit on eight NPUs: **194.098070s per measured step,63.308203 input tokens/s**. This is one process in `numa-bound-resident-fp16-cacheoff-v1-8devices`,separate from the FP32 matrix. [Reviewed audit](formal-b512-resident-fp16-add-20261004.json).

Workload/source remain `e69b3bde3d53b0d6a019e89e82d1a78c3a91a7b8`,D2048/B512/T12/V50304,480body nodes/2208body edges and9,468,053,696 parameters. Physical sample rows4 produce128 physical groups. One continued warmup(194.477605s) precedes one measured step;each has two connected windows. Construction60.926761s is separate. The consumer produced12288 measured output tokens,final cut816 and1188301 actual events. No CPU precomputed events/routes/gradients or profiler feeds the candidate.

Explicit FP16 payloads retain the declared FP32 loss/adjoint/master policy and int64 timing/counts. The process uses `ACLNN_CACHE_LIMIT=0`,qualified by [93 affected correctness checks](resident-cann-cache-policy-20261004.md) and the original-width B4 reproduction. The default-cache full-size and B4 failures remain failed. No shared SDK or graph implementation was changed;900s step/2200s child allowances are unchanged.

The requested static envelope peaked at23.0107GiB/card;observed allocator growth peaked at3.934860GiB/card and stayed within every per-device estimate. The monitor verified the declared host RSS/NUMA lane and terminal cleanup. `formal-resident-fp16-add-inference03` ended exit0 at2026-10-04T05:23:52.193817Z,with an empty cgroup and released lease.

The first v3 auditor incorrectly expected an active SGD optimizer in inference;the consumer correctly recordsNone. That rejected audit and v3 helper are preserved. The separate v4 auditor requiresNone for inference and the configured optimizer for training,with all other checks unchanged. No consumer rerun or record mutation was needed.

This is execution/performance evidence under a scoped numerical policy,not strict wide FP16-versus-FP32 trajectory equivalence. Card count,owner map,dtype and cache policy differ from the eleven-device FP32 configuration;do not attribute a time ratio solely to dtype. Three fresh processes in one series are still required before recommending a configuration. [Attention inference](formal-b512-resident-fp16-attention-20261004.md) also passed;complete training and separate profiling companions remain open.

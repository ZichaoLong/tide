# Resident CANN operator-cache policy

Explicit process-local `ACLNN_CACHE_LIMIT=0` avoids the reproduced original-width FP16 copy failure on the tested Torch/TorchNPU2.10,CANN9.0.0,aarch64 Ascend910_9392 stack. The affected resident correctness gate passed **93 checks,with no skips**,in419.52s on unchanged source `e69b3bde3d53b0d6a019e89e82d1a78c3a91a7b8` and existing verified binaries. [Terminal audit and reproductions](resident-cann-cache-policy-20261004.json).

| Observation | Result |
| --- | --- |
| Original B512/default cache | failed cached TensorMove kernel lookup;no completed step |
| B4/default cache,same width/topology/owners/capacities | reproduced the same failure |
| B4/cache disabled,same binary and configuration | passed;construction58.868319s,one two-window step1.679584s,96outputs/cut408 |
| Basic captured copies,default cache | 30 single-device and60 two-device copies passed across five dtypes/six sizes;did not reproduce the full consumer failure |
| Resident gate/cache disabled | 93 passed,0 skipped;FP32/FP16,three graph families,two schedules,SGD/AdamW,independent CPU oracles,standalone/native consumers,continued windows and fresh2→3-owner checkpoint restore |

Only the process environment changed;no graph algorithm,payload dtype,precision rule,owner map,capacity or shared SDK file was modified. The installed TensorMove binary hashes match their metadata and export the named symbols. The vendor log identifies the cached function-handle lookup failure;the workaround does not establish its underlying vendor/integration cause. Preserve the [original failure](resident-fp16-copy-failure-20261004.md) and default-cache B4 reproduction as failed.

The build/probe/gate services are terminal with empty cgroups and released leases. A live environment sample confirmed `ACLNN_CACHE_LIMIT=0` in the gate wrapper;the launcher,source,inputs and logs are hashed. The current core,backend and installed consumer bytes were rechecked against their qualified manifests. The gate reused matching qualified libraries and reran only the resident paths affected by this environment change.

For this qualified resident configuration,export the setting before launching Python or standalone consumers:

```bash
export ACLNN_CACHE_LIMIT=0
```

Retain it alongside `ACL_OP_INIT_MODE=0` in the runtime record. Profiling must use the same cache policy as its unprofiled reference. Other CANN versions require their own verification;this report does not enable a silent global fallback or change public module files.

B4 is a failure-isolation run,not an original-B512 performance result. Full-size FP16 follow-up uses fresh records,unchanged900s step/2200s child bounds and a separately labelled cache-off series. It requires this passed audit before starting. Its result is tracked in STATUS;this report does not certify that pending run or a dtype speedup.

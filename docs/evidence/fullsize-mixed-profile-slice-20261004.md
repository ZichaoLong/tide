# Full-size mixed Add trace slice and retained profile timeout

Clean `e69b3bde3d53b0d6a019e89e82d1a78c3a91a7b8`, 2026-10-04. The full-size Add profiling process **failed its4500s execution bound without a complete consumer result**. The planned Attention profile never started. Its closed five-second trace was copied while the application ran and is independently auditable. [Exact launch, failure and CSV summary](fullsize-mixed-profile-slice-20261004.json).

The actual application used D2048/B512/T12/V50304,9,468,053,696 parameters,LibTorch/TimedDAG/prefill/mixed-A,FP32,11NPUs,physicalB32 and one cold SGD update across two connected windows. `msprof` requested runtime API/task-time/AiCPU recording,delay180s,duration5s,storage1GiB. Execution4500s and export180s/session were separately bounded. This run overlapped non-formal CPU diagnostics; it cannot establish throughput.

| Observation in the captured slice | Count or duration |
| --- | ---: |
| Device time span | 5.029450s |
| Devices with recorded tasks | all11 leased devices |
| AI_VECTOR_CORE | 133,285tasks |
| MIX_AIV | 4,421tasks |
| AI_CORE | 135tasks |
| AiCPU | none observed |
| Mul | 115,932tasks |
| Mul with2048-vector/scalar shape | 108,827tasks |
| MatMulV2 | 135tasks;3.203ms summed device duration |
| ACL SetDevice | 371,372calls;4.661s summed API duration |
| All device tasks | 137,841tasks;0.220692s summed duration |

The device interval is2026-10-03T18:31:33.714517–18:31:38.743967UTC. Batched shapes,MatMul,Softmax and Swish are present alongside many small operations. This supports investigating small-operation and host-submission costs in mixed-A. It does **not** establish the precise source call sites,timeout cause,full-update phase proportions or resident-backend behavior. Mixed-A intentionally retains host scheduling. Absence of AiCPU in this slice does not exclude it elsewhere.

Device durations overlap across devices/streams;their sum is neither wall time nor utilization. ACL/runtime API levels can nest;do not add their sums together. No complete update,optimizer result or formal timing can be recovered from this slice. EOF/semaphore cleanup messages after timeout are retained and do not establish the original cause. No automatic rerun or larger timeout is justified by this failure alone.

The audit rechecked the source-matching combined binary,packet/native encoding,helper and receipt hashes,all441 copied raw file hashes,CSV hashes and recomputed summaries/device coverage. `profile-fullsize-mixed01` is terminal failed/exit1 with an empty cgroup and released11-card lease. `profile-slice-inspect01` is terminal passed/exit0 with an empty cgroup. Re-audit: `python TASK/launchers/audit_fullsize_profile_slice.py`.

Separate full-size resident/FP16 profiling and broader phase coverage remain pending. Existing unprofiled cold feasibility and reduced-batch continued calibration remain separately scoped;this failure does not relabel either.

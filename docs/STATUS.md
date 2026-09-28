# Current handoff

Updated: 2026-09-28 (Asia/Shanghai). Branch graph-execution-foundation.

## Authorized scope and fixed workload

Implement LibTorch/NPU performance tests for historical 17B Attention and "8.8B"
Add: independent concurrent processes AND a model spanning devices. Exploit edge
locality; up to8/12 available NPUs. No subagents or pushes. Public core unchanged.
465 nodes,2208 logical/4418 physical edges,D2048/B512/V50304. Attention exactly
17,269,426,339 parameters; Add9,468,020,899 (historical8.818*1024^3). FP32,
CPU seed7 and original owner/RNG order,2 body ticks/token,clear,all-softmax,
4-head Attention,exact norm-fp64-v1 Read. Twelve growing-context tokens:
4 warmup+8 measured; no_grad and grad-forward without backward/optimizer/detach.
No smaller case may be reported as full size. Contract: docs/accelerator-scale.md.

## Implementation and numerical policy

edfb340/d69acba add a standalone installed-core consumer, resident/host transports,
memory/locality partitions, explicit copy and allocator metrics, durable wrapper.
Resident state/KV/Aggregate/Full/messages stay on assigned NPUs. Same-device
messages remain local; cross-device inputs use Torch D2D copies. CPU owns
metadata/selection and exact FP64 Read of necessary vectors. CSR pooling is
unrelated; the historical workload uses event pooling. No checkpoint import or
general heterogeneous-core claim. Placement aliases and None-vs-zero are checked.

Current candidate adds explicit SDK finalization after all local tensors/workers
are destroyed, before process statics. This fixes the observed intermittent
8-device exit crash in development. The core's later repeated finalization warns
"Please init npu device first!"; zero exit and all gates remain necessary.

Default --vjp-policy strict remains unchanged. Real-topology Add D8/B1 seed0
fails its first Full squared-norm VJP by7.7188e-6 (tolerance ratio3.06184).
An independent single-node Python reproduction, without graph execution or
transfers, yields exactly the same error. CPU FP32 vs FP64 itself differs by
9.20483e-6; NPU FP32 vs FP64 by1.37738e-5. Complete local Jacobian passes the
original componentwise tolerance (ratio0.03123). Explicit FP32 RMS expansion
(dev06) changed nothing and was removed; its snapshot/failure is retained.

Explicit --vjp-policy basis-conditioned reports any strict quadratic failure,
checks every root-coordinate VJP at original rtol1e-5/atol1e-6, and compares
quadratic VJPs against CPU FP64 contractions using sum(abs(J*cotangent)) scale.
Only roots<=64 coordinates qualify; all finiteness, None/zero, forward, routes
and ownership checks remain mandatory. No model math, dtype or seed changes.
This benchmark-specific numerical policy does NOT claim the strict gate passed.
Negative tests reject wrong Jacobians, None-vs-zero changes and nonfinite grads.

## Evidence, current jobs and next commands

Task root: /mi/data2T/zlong/tide-npu-performance. Frozen sources, builds, runs,
launchers, inputs and Trackio are separate. Artifact links are under
artifacts/npu-performance-*. Use launchers/freeze_run.py; never edit a snapshot.
Every job has tide-npu-performance-NAME.service in background.slice and
runs/NAME/{status.json,task.log}; NPU jobs have queue.json with physical mapping.
FIFO allocation uses the account's run_on_free_npu.py. All compiler threads bounded.

Clean a1 at d69acba: CPU/2-NPU eight-cell gates passed. Wrapper/Trackio success
and deliberate timeout cleanup passed (73 partial tokens, no remaining workers).
8-NPU a1 Attention finished checks/steps then crashed on exit: still FAILED.
Wide CPU Add passed; wide NPU Add strict VJP failed as above. GDB exit diagnostic
was not a workload pass. Tiny msprof trace contains1059 MEMCPY_ASYNC D2D operations
(240940 bytes), including internal copies; these are NOT all peer-link traffic.

Dev07 CPU/NPU builds passed. Wide2-card Add/Attention at seeds0/7 passed with
basis-conditioned policy. strict-repro-dev07 confirms default strict still fails
seed0 Add: enclosing diagnostic passed, native workload remains a strict failure.
Independent probes numerical-local-a1 and numerical-jacobian-a1 passed.

Dev08 CPU/NPU builds and new negative tests passed. Tiny CPU,2-NPU and8-NPU
all8 cells each passed strict policy (both models,transports,placements; full
observable and isolated zero/nonzero VJPs). Jobs check-cpu-dev08,
check-npu2-dev08,check-npu8-dev08 are terminal passed. topology-npu8-dev08 also
passed the real wide graph,D8/B1,models Add/Attention,seeds0/7,explicit
basis-conditioned policy, including all4 normal process exits. This is correctness only, not full-size performance.

Implementation committed bb0ecc6; clean source perf-a2 frozen at this commit.
Both clean builds and all immutable gates PASSED: build-cpu-a2/build-npu-a2,
check-cpu-a2/check-npu2-a2/check-npu8-a2 (24 tiny cells,strict),
topology-npu2-a2/topology-npu8-a2 (8 wide cells,basis-conditioned).
Evidence: docs/evidence/accelerator-scale-20260928.{md,json}. All process exits
were normal. Core qualification was not repeated.

Full-size pilots SUBMITTED from perf-a2 via launchers/launch-pilots-a2b.py:
- pilot-add-g0-n4-a2b
- pilot-attention-g0-n4-a2b
- pilot-add-g1-n4-a2b
- pilot-attention-g1-n4-a2b
Each unit is tide-npu-performance-NAME.service; records under runs/NAME and
artifacts/npu-performance-NAME. They use the same candidate physical devices
1,5,9,11 to run sequentially (queue verifies availability), not concurrently.
The original four *-a2 pilots were cancelled while queued before native startup,
because physical7 became occupied. Their cancelled records are retained.
Each replacement has1800s native timeout,512GiB RSS budget,7500s queue wait,workers16,
ATen threads1,locality,resident,FP32,D2048/B512/V50304,seed7,12 tokens/4 warmup.
--grad0 or1 means no_grad or grad-forward without backward/optimizer/detach.
Run wrapper uses /home/zlong/venvs/trackio/bin/python; module is the standalone
libtorch-npu2.10/CANN9.0 stack. The exact argv is in job status and, after
allocation, run/run.json. Input wide.txt SHA256:
d67fdff4b351ecaa1aeb69d42a5c8bff956aeca8c78e35940077b83048a592a4.

Inspect with python scripts/status.py and runs/NAME/{queue.json,task.log,
run/run.json,run/summary.json,run/stdout.log}. Stop a specific pilot with
systemctl --user stop tide-npu-performance-NAME.service. No pilot result yet;
verify current records rather than inferring completion from elapsed time.
After pilots: preserve OOM/timeouts. Do not repeat unchanged failed cells.
For completed modes compare2/4/8 cards,memory/locality,3 independent fresh
processes on matched resources. Independently concurrent models remain a
separate requested test (total occupancy<=8); not satisfied by these serial
pilots. Do not silently reduce dimensions or precision. Commit performance
evidence only from inspected terminal records.

## Runtime and completed foundation

Use user-selected public /opt modules, not the older guide's personal defaults.
Standalone module libtorch-npu/2.10.0-cann9.0.0; TorchNPU2.10/CANN9.0/driver25.3.rc1,
A3 Ascend910_9392. Public driver unchanged. Existing qualified core builds:
/mi/data2T/zlong/tide-accelerator/builds/native-{cpu,npu-sdk}-dev03.
Default async queue crashed at rtStreamWaitEvent on2 devices; retain its failures.
Explicit/default TASK_QUEUE_ENABLE=0 passes these gates and is recorded. This
changes dispatch synchronization, not CPU staging or model precision.
SDK allocator header supplement contains only2 unmodified official headers from
94f8a8e6b523d7ba553e1b80d5b5248478391526 with installation provenance.
Official EnvVariables.cpp at that same commit documents keep-origin FP32 default
on910B1 and later and disabled matmul HF32; no precision override is applied.

Trackio writer/viewer /home/zlong/venvs/trackio/bin/python,0.35.0; local root
 task-root/trackio,storage auto,project tide-npu-performance. No dashboard exposed.
Raw manifests/JSONL are authoritative. Completed foundation baseline1b0cb48:
CPU8636 tests+22 complex cells+installed consumers;4 CANN stacks passed328 positive
NPU gates+4 CSR rejections. Standalone2.10/CANN9.0 qualified. No CUDA hardware
results. Do not redo that core qualification for consumer-only changes.

Implementation is bb0ecc6. The next commit records only reviewed evidence,
ROADMAP and this handoff. No core/Python changes; full-size jobs use perf-a2.

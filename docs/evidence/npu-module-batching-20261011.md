# NPU module batching qualification — 2026-10-11

Clean **4f23993315554f68be3352d7d15533a0e445b260** accepts the complete public
module inventory: **236 positive cases and 2 explicit unsupported requests**.
The [audit](npu-module-batching-20261011.json) checks exact configurations,
required checks, runtime/build identities, tolerances and duplicate coverage.

| Implementation | FP32 | FP16 |
| --- | --- | --- |
| Python PyTorch | 57 passed | 57 passed |
| Python calling native LibTorch | 61 passed,1 CSR refusal | 61 passed,1 CSR refusal |

Each positive case checks complete observables, independent VJPs and None/zero,
chunk continuation, trace-disabled results, three AdamW updates, fresh-process
checkpoint and CPU handoff. Tolerances remain FP32 atol1e-6/rtol1e-5 and FP16
atol1e-3/rtol2e-2; discrete decisions remain exact. The native CSR request fails
explicitly with the declared event-pooling alternative; it is not silently changed.

The separate **standalone LibTorch** checker passed two graph cases, three
training steps, two workers, checkpoint handoff and a nondefault stream. A real
NPU operator doctor passed. These are distinct from Python/native execution.
The core was built at a926703 with exact source digest
`0e2d0baa8fb2e3db1aed807c20c4ef78e53838922295dd4f60fb7578313266f0`.

The original `g4-npu-modules07` stopped at its unchanged2700s ceiling with
exit124 after168 positive cases; its failure and interrupted nested records
are preserved. `g4-module-completion01` ran exactly the missing68 positive
cases and2 refusals plus the previously unstarted standalone checker. No
accepted case repeated. Its new correctness-only workload budget was1640s,
service1800s,6CPU,12GiB; observed guard elapsed1122.64s and peak summed RSS
10566848512B. Both terminal units have MainPID0,empty cgroups and no guard
survivors. The frozen source remains clean.

Physical9 mapped to logical0 under the user's occupied-device authorization.
Four launch-only CANN compiler-pool limits were1. Shared mappings contribute
to aggregate RSS; this is sampled enforcement, not a proven kernel allocation
cap. No performance or GPU result is claimed.

The later `g4-npu-training08` at9c3ffd3 separately passed four complete FP32
PDG training cases but failed its first FP16 case at the control-dtype assertion.
That failure does not invalidate this separately audited public module inventory;
complete scale training and resident qualification still require their own gates.

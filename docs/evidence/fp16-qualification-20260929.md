# Explicit FP16 qualification

The public Python/native API and standalone historical-topology consumer now
accept explicit FP16 payloads. Training uses FP32 masters/optimizer slots/loss
and static loss scaling; graph, HST, edge/owner identity and scheduling formulas
are unchanged. This is a finite implementation/qualification report, not a
convergence or arbitrary-configuration claim. See [precision.md](../precision.md).

Core/public implementation is fc2a76f. cb58110 orders profiler fixture/master
copies before its worker stream;595dccd strengthens the consumer oracle with
FP32 gradient normalization and master-weight comparisons. c8d2b61 adds early
CPU FP16 CSR rejection;4a7dec7 records that expected rejection in the named
suite (one clean CLI rejection case passed). Complete C++ core source hashes are identical across
these follow-ups, so their exact matching fc2a76f builds are reused. Every
qualification uses a clean immutable snapshot; all commands, source/binary hashes,
environment/terminal records and artifact locators are in the adjacent JSON.

| Gate | Accepted scope |
| --- | --- |
| Complete CPU regression | 8645 passed, FP64/FP32 required suite plus new FP16 tests;1629.46s |
| Public Python NPU FP16 | Complete named suite,39 positive cases |
| Public native-adapter NPU FP16 | Complete named suite,43 positive cases; one explicit CSR rejection |
| Public NPU FP32 regression | Five representative complete cases per implementation,10 total |
| Standalone consumer CPU | 36 FP32/FP16 configuration cells |
| Standalone consumer2 NPU | 36 FP32/FP16 cells: CPU64 and all32 policies |
| Standalone consumer4 NPU | 20 FP32/FP16 cells: CPU64 inference and CPU32 training |
| Standalone consumer6 NPU | Eight additional FP32/FP16 mixed32 training cells before the user-authorized six-chip attempt |
| Standalone consumer7 NPU | Eight additional FP32/FP16 mixed32 training cells, after available capacity changed |
| Standalone consumer8 NPU | Eight FP32/FP16 cells: mixed32 training |
| Consumer CTests | Four passed, including analytical master SGD, None/zero and overflow checks |
| CUDA compilation/host checks | CUDA-linked build and22 CPU/CLI tests; no GPU execution |
| CPU FP16 CSR boundary follow-up |20 focused tests passed on c8d2b61, including early rejection and FP16 family/checkpoint tests |

Local NPU qualification is aarch64 Ascend A3, Torch/TorchNPU2.10, CANN9.0,
driver25.3.rc1, public /opt modules and shared driver. Multi-device consumer
work uses TASK_QUEUE_ENABLE=0 and dynamically allocated2/4/6/7/8 physical devices.
Builds use two workers and isolated directories. Standalone loader closure
resolves real framework/vendor libraries, without Python or build-time stubs.
CUDA host build uses Torch2.10.0+cu128 / CUDA Toolkit12.8.1 on aarch64.

Public named cases compare complete values, state/history/pending, exact routes,
independent isolated VJPs and disconnected gradients, chunking, three SGD/AdamW
updates, fresh-process checkpoint continuation and CPU handoff. Consumer gates
compare independent CPU scalar-slot and placed packed-row schedules, logits,
Read/control precision and device, payload dtype/residency, nonzero/zero VJPs,
three complete updates, FP32 master trajectories and optimizer slots. These116
consumer configurations are bounded tensors; full-size performance is a separate
eight-cell experiment. The public scheduler remains host-owned; consumer tensor
ranking/event keys still return decisions to a host dispatch loop.

FP16 public tolerances are atol1e-3/rtol2e-2. Consumer FP16 uses explicitly
atol.002/rtol.02. FP32/FP64 tolerances retain their previous values. Tolerances do
not relax integer events/routes, owner identity or None-vs-connected-zero.
Matched-dtype CPU oracles are the default. An explicit quantized-fixture FP32
oracle is a cross-precision diagnostic and may fail on near-tie routes.

Retained development failures explain the policy rather than being relabelled:
initial Half test literals were unsupported by that ATen constructor; a CPU
Attention FP16-vs-FP32 diagnostic changed two routes; the first NPU cross-precision
training check failed numerically. A same-dtype VJP on both CPU and NPU narrowly
failed atol.001 (maxabs.001953125, maximum tolerance ratio1.02827), then passed
explicit atol.002 with discrete checks unchanged. CPU dev06 rejected an erroneous
two-device launch; corrected dev06b passed. An early CUDA launcher was cancelled
to correct its planned test path; the independent final host build/tests passed.
These original task directories remain under the task root, not overwritten.

CPU FP16 CSR sparse-dense matmul is unavailable in the local Torch2.10 CPU
backend; the public entry now rejects it before output creation. NPU CSR pooling
is unsupported at both dtypes. BF16/AMP and standalone FP16 owner checkpoints
remain outside this extension. Public native execution through Python uses the
Python Session/master-optimizer checkpoint boundary. A normal optimizer over
FP16 leaves does not acquire FP32 masters automatically.

Existing FP32 qualification across CANN8.5.0/.1/.2 and9.0 is unchanged. New FP16
evidence covers2.10/CANN9.0 only. CUDA real-device, x86_64 and other stack/dtype
combinations need their own gates; compilation is not hardware execution proof.
See [profiling evidence](accelerator-profile-analysis-20260929.md) for two new
FP16 hardware traces and the separately scoped older AiCPU analysis.

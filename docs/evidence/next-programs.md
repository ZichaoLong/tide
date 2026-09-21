# Complete Next programs and prefill guards, 2026-09-21

Clean source: `c4a5ce5a5b27284a50e343afed71a1a09e76ad55`.
Command: `python scripts/qualify.py --output-dir artifacts/next-20260921-1335 --jobs 2`.
Environment: aarch64 CPU, Python 3.11.15, Torch/LibTorch 2.10.0+cpu, GCC 10.3.1,
C++17 Release, two build workers, ATen/BLAS threads 1, node workers up to 3.
Build and tests exited 0: **2010 passed**, pytest 282.34 seconds.
Retain `artifacts/next-20260921-1335/{status.json,task.log,verification/}`.
Both manifests identify clean source and record native source/binary hashes.
Unit `tide-foundation-next-20260921-1335` is inactive, MainPID 0.
Development build `artifacts/next-build-20260921-1330/` passed; 122 related checks
passed in 17.52 seconds before the implementation commit.

Next receives old state, comparison, logical time, complete content, active flag
and control. Every candidate invokes it; absent nodes do not. Native independent
node calls run in the pool and commit canonically. Selected Full retains its
pre-Next comparison. Graph-owned clear policy remains independent of shared
weights. Python custom Next parameters are registered and native overrides are
explicitly guarded. Result validation covers clocks, tensor metadata/finiteness
and state-kernel slot layout.

Control blend's two-event analytic case ends at states (2.5,3.375), sum 5.875.
Initial-state gradients are 0.5625, first-input gradients 0.375, second-input
gradients 0.5, decay gradients (0.4375,0.9375), Read gradients (1.4375,0.5625).
It exercises a passive candidate, an absent node, independent sample leaves and
Full reading comparison instead of persistence. State-prefill blocks are zero
while Full batching remains present. The default adopt profile retains previous
behavior and the existing sharing tests with different clear policies pass.

A custom clock/content/control/active-sensitive Next updates SSM value and slots.
It consumes source-origin positions and has a shared learned gain. Its hand loss
is 40, input gradients [[4,4],[12,12]], gain gradient 20; selected clear changes
these to loss 10, gradients [[0,4],[0,12]], gain gradient 5. SSM skip gradients
(6,6), or (0,6) under clear, independently check the comparison input's VJP.
Native serial, packed parallel streaming and frontier match direct and encoded
boundary inputs. Idle/other-sample gradients remain None. Future state clocks
are rejected. Python additionally checks custom parameter AdamW/checkpoint state,
source-aware SettleGraph embedding and malformed state results.

Differential cases include selected-only adoption, clear, SSM/Linear/Delta slots,
first-order isolated roots, inference, fixed self-loop/chain schedules,
SettleGraph direct/encoded/chain execution, cursor cuts, explicit detach and
checkpoint profile rejection before weight changes. Growing attention caches
are explicitly rejected by the fixed-shape control-blend example.

Native graph identity format is v10, checkpoint payload remains v3. Next is
currently evaluated per event (native node parallelism); no joint Next batch is
claimed. Aggregate/state/Read/Full packing remains subject to its own contracts.
Full region histories/controls, LH numerical parity and workload performance are
subsequent gates. This report makes no speed or higher-order AD claim.

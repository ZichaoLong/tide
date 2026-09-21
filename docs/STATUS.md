# Current handoff

Updated: 2026-09-22 (Asia/Shanghai). Branch: graph-execution-foundation.

## Latest completed qualification

**3296 tests passed in 303.80s** on clean implementation
`e6188802d3e893fa31d5351a9602748e0543915c`; `evidence/pronounce.md`.
Unit tide-foundation-pronounce-20260921-175646 is inactive, MainPID 0, exit 0.
All five records in artifacts/pronounce-20260921-175646/ passed: outer status,
verification, oracle, oracle-release and pronounce-release. No active job.
Evidence is committed separately from the implementation.

Token-window adapters preserve sample/phase occurrence labels, connectivity and
two explicit clocks. Actual Pronounce defaults to identity normalization;
ALConfig has no norm field. RMS/Layer norm-only Full profiles are separate.
The rectangular vocabulary head stays outside the graph and runs rowwise.
Original Pronounce assertions-on FP64: 180 cases/1080 tokens/2430 sample outputs,
24 unavailable; FP32 and both assertions-off precisions: 204/1224/2754 each.
Full: 108 cases/756 rows per precision. Selector/Add/Attention also passed.
Original FP64 active-softmax diagnostic hardcodes an FP32 denominator. A separate
consistent assertions-off build fills numerical coverage; both builds retain
ordinary C/C++ asserts. No whole IOCortexNet claim follows.

## Next action

Implement `lh-iocortex-plan.md`: actual think/think_single_step oracle, four-block
unit-delay wiring, explicit local port ordering, proposal/selection/cache/history
checks, final activation-to-pending-message projection and token readout.
Begin with equal-width cortexes and supported local profiles. Single-PDG proof,
independent Python whole-model coverage and composite checkpoint remain separate.
No whole-model implementation is claimed yet. Follow implementation commit ->
clean qualification -> evidence commit.

## Source and retained evidence

LH snapshot artifacts/lh-source-20260921-1428: 69 files, identity
`ac7c878a56aeb55eec9f919da1962be6134fc5e3d1872f6d2303917306edb87f`; LH HEAD
`5fd237d40c9880ccb6e511e4bf20799c7022fd1e`, including actual dirty-source hashes.
All reference trees stay read-only. Preserve snapshot and cited artifacts.

Retain failures pronounce-dev-20260921-1744 (AnyModule accessor),
pronounce-dev-20260921-1751 (wrong RMS assumption), and successful follow-up
pronounce-dev-20260921-175316 (132 checks, both Pronounce variants).
Prior pooling: evidence/fiber-pooling.md, source 733983e648b18f3da0df215dfd7aedbef1d1d331,
3092 tests. Failed fiber-pool-dev-20260921-1709 and passed follow-up
fiber-pool-dev-20260921-1715 remain. Keep attention conditioning failures
fiber-dev-20260921-1601 and fiber-dev-20260921-1608; limits: fiber-attention.md.

## Execution policy

Default cache build/lh-oracle has runtime assertions on; separate
build/lh-oracle-release has them off. Reconfigure every run; retain unique logs
and source/library/CMake/binary hashes. Freeze source/shared caches during jobs.
No push. Delete only known obsolete project artifacts after dry-run inspection.
STATUS is the sole current handoff; ROADMAP owns backlog; Git retains old plans.
CPU aarch64, /home/zlong/anaconda3/bin/python, Python 3.11.15,
Torch/LibTorch 2.10.0+cpu, C++11 ABI. Backend autoload off; OMP/OpenBLAS=1;
two build jobs, Nice=10, background.slice. Graph v11, checkpoint v4.
Packed training promises tested first-order public-root VJPs with scalar replay;
inference does not replay. No higher-order AD or large-sparse speed claim.

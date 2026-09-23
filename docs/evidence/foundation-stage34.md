# Stages 3 and 4: modules, options and training

Clean frozen implementation `494e6a91ea134d9e36498aa14a51ce20bcdc0371`:
**7611 tests passed in1153.34s**, CPU FP64/FP32, test/job exit0.
Qualification checkout `qualification/foundation-stage34-494e6a9` remained clean.
Raw evidence `artifacts/foundation-stage34-20260923-b/`: status.json,
full-cpu/result.json, tests.log, retained test-tmp and reviewed-audit.json.
The audit verifies every tracked source against Git archive,13 binaries and36
unique fresh-process payload/manifests (pytest current symlink deduplicated).

The read-only build is `qualification/modules-dev-20260923-a/build`, from its
archived dirty development source. Its exact C++ content matches494e6a9; it was
not rebuilt at that commit. Adapter SHA256
`8960193033e7ea56f6ec9a502c216de540f37a2e3cc0b20270550e561917d0d1`.
S6 will independently rebuild final committed source. Environment: aarch64,
Torch/LibTorch2.10.0+cpu, Python3.11.15, GCC10.3.1, C++11 ABI, ATen/BLAS1.
Unit `tide-foundation-stage34-20260923-b`: background.slice/Nice10,
RuntimeMaxSec3600; inactive/dead, MainPID0/ExecMainStatus0, no descendants.
Only a small correctness build overlapped; these durations are not benchmarks.

S3 acceptance covers legal nontrivial frontier/native Settle sequence packing,
source transport, causal Next/reset waves, region workers, compact/deferred
cleanup, same-fiber policies, physical projection strides and explicit counted
fallback/rejection. Representative Attention/GQA/window, same-fiber Attention,
Linear Attention, ungated delta-rule-v1, existing gated delta, SSM and Full/Agg/
Emit modules retain scalar/Python anchors. The tiny explicit RMSNorm/RoPE/GQA/
SwiGLU adapter has layout/mask/cache/VJP checks; its cache is invalidated by an
optimizer update. It is a Python example, not arbitrary pretrained-model support.
Packed training still uses scalar semantic replay, with no backward speed claim.

S4 acceptance covers all six schedule classes with independent roots, initial
slots, shared/unused/connected-zero owners and multiple SGD/momentum/AdamW updates
(epsilon1e-5, decoupled decay), chunks and detached continuations. Fresh processes
compare uninterrupted trajectories with save/exit/load/continue for existing
single-graph v5, two-clock application partial-window bundles and native TIDENCK1
named model/optimizer values. Owners, aliases, groups, gradients, buffers, states,
ledger and subsequent updates align. Existing malformed/transactional rejection
and publication-failure gates pass. These formats retain their separate scopes;
none promises an entire training controller, data cursor or RNG restore.

The first stage34 attempt at5014c3d was cancelled, exit143, after finding the
Settle training helper used logical cut as an input position. Its earlier274-test
directed gate did not cover fresh later Settle windows. The failure and corrected
54-test gate are retained under `artifacts/settle-training-clock-repro-20260923/`.
This clean gate includes the494e6a9 correction and fresh-position assertion.
No tolerance or kernel was changed to hide the helper defect. Cancelled stage34-a
remains cancelled and is not part of this acceptance evidence.

S3/S4 are accepted for the finite declared graph/module/profile contracts. Full
C++ interfaces and schedule/encoding limits remain in stage2 and semantics docs.
S5 performance and S6 final rebuild qualification remain separate gates.

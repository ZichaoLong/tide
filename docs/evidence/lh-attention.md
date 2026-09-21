# LH same-fiber sum attention baseline qualification

2026-09-22 (Asia/Shanghai), clean implementation
`dfc2e5b622946614831962fb44683990fa609e8d`.
**2728 tests passed in 225.60 seconds**. Outer job, verification and original LH
oracle all passed with exit 0 and empty dirty status at that commit. Unit
`tide-foundation-fiber-20260921-1618` is inactive, MainPID 0.
Artifacts: `artifacts/fiber-20260921-1618/{status.json,task.log,verification/,oracle/}`.

Command: `python scripts/qualify.py --output-dir artifacts/fiber-20260921-1618
--jobs 2 --lh-snapshot artifacts/lh-source-20260921-1428`, through scripts/job.py
in background.slice. CPU aarch64, Python 3.11.15, Torch/LibTorch 2.10.0+cpu,
C++11 ABI; two build jobs, Torch/BLAS single-thread, backend autoload disabled.
Current source schemas at this commit: graph v11, checkpoint v4.

## Original inference oracle

Unmodified original C++ snapshot identity
`ac7c878a56aeb55eec9f919da1962be6134fc5e3d1872f6d2303917306edb87f`, with actual
dirty source inventory/hashes at LH HEAD `5fd237d40c9880ccb6e511e4bf20799c7022fd1e`.
Runtime assertions remain enabled. Reference worktrees were only read.

Each dtype passed **240 configurations, 5760 ticks, 16080 candidate updates**:
LOOP, PACKED, CACHEDMATMUL, CACHEDPACKED and CACHEDATTENTION; single/multi-sample
projection; clear on/off; decay 0/.01; width/head/bias cases (1/1/off, 4/2/on);
Tide serial, packed scheduling and frontier. Compare tick-by-tick and whole-window
proposals, K/V and log biases, encoded state/clocks and decoded idle cut state.
Cases have initial caches, absent samples/sources, real zero messages, idle prefix
and suffix, and cache growth past 16. Original block_size=2 does not evict.

Read is independently checked as an FP64 norm of Tide's proposal. Comparison to
the original norm inherits the proposal's payload tolerance; upcasting an FP32
proposal does not erase FP32 rounding. No route mismatch is accepted.

Original Selector (12 cases), Add (54) and Full (72 configurations) also passed
again in both dtypes in this same clean qualification. Exact domains are in their
prior reports. Oracle result records hashes of source snapshot, CMake, core library
and executables. `build/lh-oracle` is now a reusable build cache; binaries there
may be replaced later, while per-run records/logs remain independent.

## Tide training and schedules

- Independent formulas distinguish all-to-all same-fiber visibility from a
  triangular mask and summary-as-token; explicit analytic source/bias/decay VJPs.
- Ragged fibers, permuted local slots, multiple heads, scaled sources, initial
  caches longer than observation count and numerical zero rows.
- Python/native, native serial/node-parallel, batch/sequence loop fallbacks,
  streaming/frontier, cyclic execution, independent chain/self-loop schedules,
  direct/encoded SettleGraph and its one-input chain specialization.
- Isolated output, pending, state value and every cache-slot root; None versus
  connected-zero, clear with preserved comparison, observation overflow, idle
  cut decoding, owned cursor, detach, shared weights, AdamW and checkpoint restore.

The targeted development gate passed 210 fiber/counter checks. First failed run
`artifacts/fiber-dev-20260921-1601/` retains source tar and 29 failures: unsupported
dual-input specialization fixtures and ill-conditioned FP32 RMS compositions.
Fixed topology tests now honor their single-input contract; multi-source generic
SettleGraph remains tested. The attention schedule fixture isolates the state
program with tanh Full; RMS composition has a nonzero projection offset and
directional cotangents. No global tolerances changed, and arbitrary ill-conditioned
FP32 compositions are not certified. `fiber-attention.md` documents the numerical
investigation. The second failure (`fiber-dev-20260921-1608`) exposed the promoted
Read tolerance mistake corrected above; historical records remain failed.

This is the **scalar state-kernel baseline**. Native step batches heads, but state
batch and sequence methods still loop and report fallback counters. This does not
certify joint packed attention, CROSSBATCH, normalized/learned post-attention
Confluence, whole IOCortexNet/Pronounce, higher-order AD or performance. Tide's
training contract is independent of LH; original code is only run under no-grad.

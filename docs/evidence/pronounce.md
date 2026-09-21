# Token-window readout and original LH Pronounce

2026-09-22 (Asia/Shanghai), clean implementation
`e6188802d3e893fa31d5351a9602748e0543915c`.
**3296 tests passed in 303.80 seconds**. All five records passed with exit 0
and empty dirty status at that source: outer `status.json`, `verification/`,
`oracle/`, `oracle-release/` and `pronounce-release/` under
`artifacts/pronounce-20260921-175646/`. Unit
`tide-foundation-pronounce-20260921-175646` is inactive, MainPID 0, exit 0.

Command: `python scripts/qualify.py --output-dir artifacts/pronounce-20260921-175646
--jobs 2 --lh-snapshot artifacts/lh-source-20260921-1428`, via `scripts/job.py`
in background.slice. CPU aarch64, Python 3.11.15, Torch/LibTorch 2.10.0+cpu,
C++11 ABI; two build workers, OMP/OpenBLAS single-thread, backend autoload off.
Schemas: graph v11, checkpoint v4. Result manifests retain source, snapshot,
build/library, CMake and binary hashes. Unchanged 69-file LH snapshot identity:
`ac7c878a56aeb55eec9f919da1962be6134fc5e3d1872f6d2303917306edb87f`.

## Contract and independent checks

Independent Python/native token_inputs convert sealed body intervals into
token-clock inputs, retaining sample/phase labels and contiguous occurrence
positions. Missing rows remain absent, real zeros present, globally empty tokens
fail explicitly, and inputs/continuations are not mutated or detached.
See `../token-window.md`.

Actual original Pronounce performs its CSR phase/sample gather, state update,
identity normalization and rectangular vocabulary projection. ALConfig has no
norm field; ModuleUtils defaults to identity. Norm-only RMS/Layer Full profiles
are supported separately. Fixtures cover L=1/3, four samples including one
permanently absent, six tokens, missing phases, zeros, bias on/off, Add plus
five attention pools, original single/multi PACKED and CROSSBATCH, and Tide
serial/packed/frontier. Compare labeled logits, pre-head outputs, Add physical
hidden or complete KV/log-bias, token cuts versus whole window, ledger and clocks.

| Original build / dtype | Cases | Tokens | Sample outputs | Unavailable |
| --- | ---: | ---: | ---: | ---: |
| runtime assertions on / FP64 | 180 | 1080 | 2430 | 24 |
| runtime assertions on / FP32 | 204 | 1224 | 2754 | 0 |
| runtime assertions off / FP64 | 204 | 1224 | 2754 | 0 |
| runtime assertions off / FP32 | 204 | 1224 | 2754 | 0 |

The 24 excluded FP64 cases hit original active-softmax's hardcoded-FP32 diagnostic
denominator. The component oracle checks that exact exception; a consistent
separate build disables ENABLE_RUNTIME_ASSERTION and supplies numerical coverage.
Both variants retain ordinary C/C++ assertions via -UNDEBUG. No LH source changes;
unavailable cases are not numerical passes. Details: `fiber-pooling.md`.

Independent direct readout recurrence and Python/native schedules check values,
first-order isolated-root VJPs, None versus connected-zero, cuts/detach,
head/state checkpoints and momentum-SGD resumed updates (Add/all-softmax).
All earlier gates reran; original Full now covers 108 cases/756 rows per precision,
including 36 norm-only cases. Selector/Add/Attention also passed both precisions.

## Boundaries and retained failures

This is two-clock application composition, not a single-PDG proof or whole
IOCortexNet parity. Vocabulary projection runs rowwise. No higher-order AD,
optimized packed head or speed claim follows.

Retain failed pronounce-dev-20260921-1744 (846 tests + Full passed; inherited
AnyModule accessor compile failure) and pronounce-dev-20260921-1751 (850 tests +
Full passed; incorrect RMS assumption rejected by original weight import).
Source archives/hashes/logs preserve both failures. Corrected development
pronounce-dev-20260921-175316 passed 132 targeted checks in 6.45s and both
Pronounce variants before the clean commit. No referenced artifacts were removed.

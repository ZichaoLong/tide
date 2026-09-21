# Tick-repeat Add and original LH component qualification

2026-09-21; clean implementation `bb6590fe8a977e16590243094d0f645b41ccdcb0`.
**2436 tests passed in 219.01 seconds**. Outer job, verification and original-code
oracle records all passed with exit 0 and empty dirty status. Unit
`tide-foundation-lh-add-20260921-1511` is inactive with MainPID 0.
Artifacts: `artifacts/lh-add-20260921-1511/`, including `status.json`, `task.log`,
`verification/{result.json,tests.log}` and `oracle/{result.json,build.log,*-float*.log}`.

Command: `python scripts/qualify.py --output-dir artifacts/lh-add-20260921-1511
--jobs 2 --lh-snapshot artifacts/lh-source-20260921-1428`, via `scripts/job.py`
in `background.slice`. CPU aarch64, Python 3.11.15, Torch/LibTorch 2.10.0+cpu,
C++11 ABI; two build jobs, single-thread ATen/BLAS, backend autoload disabled.
Native source/library hashes and original snapshot inventory are checked.

## Original inference anchor

Unchanged snapshot identity:
`ac7c878a56aeb55eec9f919da1962be6134fc5e3d1872f6d2303917306edb87f`.
The manifest captures actual dirty LH files at HEAD
`5fd237d40c9880ccb6e511e4bf20799c7022fd1e`. No LH worktree files changed.
The oracle links original AccumulateLocal, Hidden, BatchHidden, Confluence and
ModuleUtils with assertions enabled, in addition to the Selector dependencies.

Each dtype passed **54 Add cases, 1296 ticks, 5400 candidate occurrences**:
original single hidden, no-grad cache and cache plus individual hidden; retention
0/.99/1; clear on/off; Tide serial, packed/node-parallel and frontier. It compares
proposal/pre-clear snapshots, FP64 norm descriptors and decoded physical hidden
at every cut, including nonzero initial states, present zeros, sparse source IDs,
absent samples and idle prefixes/suffixes. Whole-window results also match the
original tick loop. Sum Confluence is the original Add profile exercised here.
The original heap/tensor Selector oracle also passed again in both dtypes, each
with 12 cases, 288 ticks and 7368 candidate occurrences.

## Tide training and schedule checks

- Independent analytic value, observation/tick clock and physical decode VJPs;
  literal eager tick recurrence and retained zero versus absent gradients.
- Python/native, serial/node-parallel packed streaming, frontier sequence/causal
  paths, cycles, self-loop/chain specializations and SettleGraph direct/encoded/
  specialized schedules. Compare complete traces, pending, history and state.
- Separate output, pending, encoded state and decoded physical-state roots to
  parameters, source/input and initial-state leaves; observe-all, selected-only,
  clear, control-blend and stored old-mode Read.
- Cuts, idle cursor advance, detach, shared parameters, AdamW and checkpoint
  value/clock/alias round-trip; malformed state/retention and int64 counter limits.

Focused development qualification passed 278 tests. An earlier development run
used a pending-root fixture after its queue had drained; cut 7 now guarantees
that root. The corrected focused and full runs passed; failure logs remain in
`artifacts/lh-add-dev-20260921-1503/`.

This is component correctness, not whole-LH parity or measured speed. Repeat
cost grows with elapsed ticks; sequence state updates deliberately remain ordered
loops while Read/Full can batch. Eager LH interpretation requires fixed parameters
across composed windows. After retention changes, deferred decay follows the
declared lazy recurrence; it does not reconstruct historical parameter epochs.
Public-root first-order VJPs are the training boundary. See `../lazy-add.md`.

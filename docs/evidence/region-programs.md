# Region programs and typed history qualification

2026-09-21; implementation `eb328b36487d7acc4a11a21554c4959c12659d5a`.
Clean source: **2226 tests passed in 183.61 seconds**. Both outer job and
verification records say passed, exit 0. Unit
`tide-foundation-region-20260921-1417` is inactive with MainPID 0.
Artifacts: `artifacts/region-20260921-1417/{status.json,task.log,verification/}`.
`verification/result.json` records source, C++ fingerprint and binary hashes.

CPU aarch64, Python 3.11.15, Torch/LibTorch 2.10.0+cpu, C++11 ABI; two build jobs,
single-thread ATen/BLAS. Commands use `TORCH_DEVICE_BACKEND_AUTOLOAD=0`.
Qualification command: `python scripts/qualify.py --output-dir
artifacts/region-20260921-1417 --jobs 2`, under the durable job wrapper in
`background.slice`. Focused validation before commit: 310 tests plus native
custom-kernel checker in both FP64/FP32.

## Covered contract

- Independent Python/native region interfaces, graph-owned member layout and
  policy, complete candidates/controls, selected subset including empty.
- Default count/score/ID behavior and singleton connected-zero VJPs;
  positive-only selection; learned tensor-history scores and recurrence.
- Independent analytic history/route/control values and input, initial-history
  and region-parameter VJPs. None remains distinct from connected zero.
- Serial/node-parallel, streaming batch packing, frontier prefill/causal paths,
  observe-all/selected/clear/control-blend, cyclic/DAG specializations and
  SettleGraph direct/encoded/fixed-chain execution.
- History roots through SSM, Linear, Delta and event attention; local replay
  preserves isolated public-root gradient connectivity in tested cases.
- Cuts, detach, cursor history clone ownership, registration/sharing, AdamW and
  checkpoint v4. Invalid history, active subsets, controls and profile layouts
  fail. Identity/alias/history/weight errors precede checkpoint weight changes.
- Checked observation and selected counters reach maximum int64 without
  wrapping; incrementing beyond it fails in scalar/batch/sequence paths.
  Nonincrementing identity/passive counters may remain at that limit.
- A standalone native custom selector uses signed scalars, candidate maps,
  tensor history, vector controls and alternating empty selection. Its independent
  final history is 20 with derivatives `(2,2,1,1,4,9)` to four inputs, initial
  history and gain. Its selected output is 7.5; unrelated sample gradients are
  absent. Cursor import validates 128 histories; advancing one touched region
  makes one further history validation, without rescanning idle owners.

Graph identity v11 and checkpoint schema v4 intentionally reject older records.
The complete API and hand-computable tensor-history example are in
`../region-programs.md`. Training qualification concerns first-order public-root
VJPs, not arbitrary internal adjoints or higher-order AD. Region scanning remains
causal, with functional copies/validation of a touched history. These checks do
not establish workload speed, whole-LH parity, heterogeneous control records or
FP64 norm descriptor policy. No source in the LH worktree was modified.

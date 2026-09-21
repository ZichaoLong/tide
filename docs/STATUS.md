# Current handoff

Updated: 2026-09-21. Branch: `graph-execution-foundation`.

## Current work

M1/M2 `ema-ffn-v1` slice qualified: **105 tests passed** on clean `9c957fd`.
See `evidence/m1-streaming.md` for commands, scope and retained raw artifacts.
Native serial/node-parallel, packed batch and independent Python scheduling
agree in FP64/FP32, including trace, three VJP roots, cuts and checkpoint/AdamW.
M3 candidate: Python/native finite-input TimedDAG frontier with region quotient
cycles, batch/sequence Full and affine state scan. New Python tests: 32 passed.
Native qualification is pending. See `frontier-contract.md` for precise limits.

Planned unit: `tide-foundation-m3-20260921-0815`; command:
`python scripts/job.py --output-dir artifacts/m3-20260921-0815 -- python scripts/qualify.py --output-dir artifacts/m3-20260921-0815`.
Use the local Python below. Cwd is this checkout; keep it frozen during the job.
Inspect `artifacts/m3-20260921-0815/status.json` and `task.log`, and
`systemctl --user show tide-foundation-m3-20260921-0815`.
Stop with `systemctl --user stop tide-foundation-m3-20260921-0815`.

Environment inspected: aarch64, Python 3.11, Torch 2.10.0+cpu; native CPU
operator probe passed. Local Python: `/home/zlong/anaconda3/bin/python`.
Set `TORCH_DEVICE_BACKEND_AUTOLOAD=0` for CPU commands on this machine because
the installed NPU plugin otherwise auto-loads. No packages were changed.

## Next action

1. Finish native M3 build/qualification, fix failures and retain evidence.
2. Add independent topology specializations; implement M4 SettleGraph
   region-major execution and clock/boundary encoding.
3. Extend state kernels and packed sequence contracts (M5); retain exact limits.

## Important boundaries

- LH worktree contains user modifications; do not write there.
- LH is an inference compatibility target, never the training authority.
- No performance or broad model compatibility claims yet.
- Full user scope remains in the roadmap, including frontier execution,
  specializations, node parallelism and packed attention/recurrent modules.

# Current handoff

Updated: 2026-09-21. Branch: `graph-execution-foundation`.

## Current work

M1/M2 `ema-ffn-v1` slice qualified: **105 tests passed** on clean `9c957fd`.
See `evidence/m1-streaming.md` for commands, scope and retained raw artifacts.
Native serial/node-parallel, packed batch and independent Python scheduling
agree in FP64/FP32, including trace, three VJP roots, cuts and checkpoint/AdamW.
M3 frontier slice qualified on clean `06d9d3e`: **191 tests passed**, including
region quotient cycles, batch/sequence Full and affine state scan. See
`evidence/m3-frontier.md` and `frontier-contract.md` for scope and limits.
M4 candidate: direct SettleGraph, explicit identity-boundary embedding, Python
and C++ self-loop/chain specializations, standalone C++ smoke. New Python tests:
58 passed. All native changes await qualification.

Planned unit: `tide-foundation-m4-20260921-0830`; command:
`python scripts/job.py --output-dir artifacts/m4-20260921-0830 -- python scripts/qualify.py --output-dir artifacts/m4-20260921-0830`.
Cwd is this checkout, kept frozen during job. Inspect the output directory's
`status.json`, `task.log`, `verification/result.json`, or
`systemctl --user show tide-foundation-m4-20260921-0830`.
Stop: `systemctl --user stop tide-foundation-m4-20260921-0830`.

Environment inspected: aarch64, Python 3.11, Torch 2.10.0+cpu; native CPU
operator probe passed. Local Python: `/home/zlong/anaconda3/bin/python`.
Set `TORCH_DEVICE_BACKEND_AUTOLOAD=0` for CPU commands on this machine because
the installed NPU plugin otherwise auto-loads. No packages were changed.

## Next action

1. Complete native M4 qualification and fix any failures; record evidence.
2. Extend state/kernel interfaces for Attention/GQA, linear attention, DeltaRule,
   SSM and SwiGLU; qualify step/block, reset, clock and packed-state behavior.
3. M6/M7/M8 remain: broader training/sharing/truncation, LH C++ adapter, scale.

## Important boundaries

- LH worktree contains user modifications; do not write there.
- LH is an inference compatibility target, never the training authority.
- No performance or broad model compatibility claims yet.
- Full user scope remains in the roadmap, including frontier execution,
  specializations, node parallelism and packed attention/recurrent modules.

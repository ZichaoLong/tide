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
M4 qualified on clean `a1da7fe`: **301 tests passed**. Direct SettleGraph,
identity-boundary embedding, Python/C++ self-loop and chain specializations,
standalone C++ graph forward/backward all passed; see
`evidence/m4-settle-specialized.md`. No active jobs. Unit
`tide-foundation-m4-20260921-0830` completed with exit 0.

Latest qualification: clean `a2d833e`, **323 tests passed**. Parameter-alias
checkpoint v2, explicit detach, shared/random VJPs, inference-mode TLS and build
identity checks are covered. Cancellation lifecycle was separately checked.
See `evidence/m6-training-contracts.md`. No active jobs; all owned workers ended.

Environment inspected: aarch64, Python 3.11, Torch 2.10.0+cpu; native CPU
operator probe passed. Local Python: `/home/zlong/anaconda3/bin/python`.
Set `TORCH_DEVICE_BACKEND_AUTOLOAD=0` for CPU commands on this machine because
the installed NPU plugin otherwise auto-loads. No packages were changed.

## Next action

1. Extend state/kernel interfaces for Attention/GQA, linear attention, DeltaRule,
   SSM and SwiGLU; qualify step/block, reset, clock and packed-state behavior.
   Concrete design/navigation: `module-extension-plan.md`.
2. M6/M7/M8 remain: broader training objectives, LH C++ adapter, scale.

## Important boundaries

- LH worktree contains user modifications; do not write there.
- LH is an inference compatibility target, never the training authority.
- No performance or broad model compatibility claims yet.
- Full user scope remains in the roadmap, including frontier execution,
  specializations, node parallelism and packed attention/recurrent modules.

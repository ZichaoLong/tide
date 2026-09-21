# Current handoff

Updated: 2026-09-21. Branch: `graph-execution-foundation`.

## Current work

Building the CPU graph execution and equivalence foundation approved by the
user. Governance and upstream lock established; implementation starts at M1.
No runtime support is qualified yet. No active background jobs.

Environment inspected: aarch64, Python 3.11, Torch 2.10.0+cpu; native CPU
operator probe passed. Local Python: `/home/zlong/anaconda3/bin/python`.
Set `TORCH_DEVICE_BACKEND_AUTOLOAD=0` for CPU commands on this machine because
the installed NPU plugin otherwise auto-loads. No packages were changed.

## Next action

1. Implement M1 in `docs/ROADMAP.md`: sealed-window Python oracle, native
   streaming kernel, typed continuation, complete traces and differential tests.
2. Configure LibTorch from the chosen Python's `torch.utils.cmake_prefix_path`.
3. Validate FP64/FP32 forward/VJP and chunk composition; checkpoint in Git.

## Important boundaries

- LH worktree contains user modifications; do not write there.
- LH is an inference compatibility target, never the training authority.
- No performance or broad model compatibility claims yet.
- Full user scope remains in the roadmap, including frontier execution,
  specializations, node parallelism and packed attention/recurrent modules.

# Current handoff

Updated: 2026-09-21. Branch: `graph-execution-foundation`.

## Current work

M1/M2 candidate implementation is present: independent time-major Python oracle,
C++ sparse streaming, CSR/CSC, serial/node-parallel, packed batch, HST, trace and
value checkpoints. Python analytic suite: 11 passed. Native build/qualification
is next; no native support claim yet.

Planned durable build unit: `tide-foundation-build-20260921-0800`.
Command: `/home/zlong/anaconda3/bin/python scripts/build.py --jobs 2`.
Cwd: `/home/zlong/llm/graph-execution-foundation` (kept frozen during job).
Record: `artifacts/build-20260921-0800/status.json`; log: sibling `task.log`.
Inspect: `systemctl --user show tide-foundation-build-20260921-0800`.
Stop: `systemctl --user stop tide-foundation-build-20260921-0800`.
The job wrapper records exact committed source identity and exit status.

Environment inspected: aarch64, Python 3.11, Torch 2.10.0+cpu; native CPU
operator probe passed. Local Python: `/home/zlong/anaconda3/bin/python`.
Set `TORCH_DEVICE_BACKEND_AUTOLOAD=0` for CPU commands on this machine because
the installed NPU plugin otherwise auto-loads. No packages were changed.

## Next action

1. Complete the native build and fix any compiler/runtime failure.
2. Run `python scripts/verify.py --device cpu --dtype both --output-dir artifacts/verify-m1`.
3. Record exact qualification evidence, then implement M3/M4 frontier/embedding
   and independent topology-specialized anchors.

## Important boundaries

- LH worktree contains user modifications; do not write there.
- LH is an inference compatibility target, never the training authority.
- No performance or broad model compatibility claims yet.
- Full user scope remains in the roadmap, including frontier execution,
  specializations, node parallelism and packed attention/recurrent modules.

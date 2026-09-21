# Current handoff

Updated: 2026-09-21. Branch: `graph-execution-foundation`.

## Current work

Region programs/history are **qualified: 2226 passed in 183.61 seconds** on
clean implementation `eb328b36487d7acc4a11a21554c4959c12659d5a`.
See `evidence/region-programs.md` and the API in `region-programs.md`.
No active job or uncommitted implementation remains. The completed region plan
was removed; ROADMAP remains the full backlog. Status output summarizes old
failures without printing their stale dirty-file lists.

Implemented in this increment:
- Python RegionProgram and native RegionKernel, graph-owned membership/policy,
  default count, positive-only and learned tensor-history profiles.
- Typed region history, complete controls, empty selection in independent fixed
  topology loops, SettleGraph program sharing and initial-history embedding.
- Trace/history roots, cuts/detach, cursor clone ownership, parameter sharing,
  AdamW/checkpoint restoration and malformed-output checks.
- Checked int64 counters across built-in state step/batch/sequence and selection.

Native graph identity is **v11**; checkpoint payload is **v4**. Old identities or
schemas are rejected without implicit migration. Vector controls are supported
by custom local programs; built-in projection Emit/control-blend require scalars.
Region scans remain causal and scalar per frame; no joint selector batch or
performance gain is claimed. Named heterogeneous controls and explicit FP64 Read
precision policy remain future extensions.

Both development builds completed (exit 0, inactive, MainPID 0):
`tide-foundation-region-build-20260921-1406` and
`tide-foundation-region-build-20260921-1412`; artifacts use the corresponding
`artifacts/region-build-20260921-{1406,1412}` paths.

## Latest qualification

Unit `tide-foundation-region-20260921-1417` completed with exit 0, inactive,
MainPID 0. Both `artifacts/region-20260921-1417/status.json` and
`verification/result.json` record passed with a clean source identity; logs are
`task.log` and `verification/tests.log`. Evidence is tied to the implementation
commit above; evidence/documentation updates form a separate commit.

Previous qualification: 2010 passed at
`c4a5ce5a5b27284a50e343afed71a1a09e76ad55`; `evidence/next-programs.md`.

## Next action

1. Implement LH selector counters and explicit FP64 norm Read/descriptor policy (`lh-compatibility.md`, final section).
2. Compare both original LH C++ selector paths from an immutable dirty-source
   snapshot; then implement Add/same-fiber attention/Pronounce and compare complete
   mapped inference state/messages. LH never supplies the training contract.
3. Broader losses and performance qualification remain in ROADMAP: optimized
   packed backward, cache allocation, structured Delta chunks, large sparse work.

## Boundaries and environment

CPU aarch64; `/home/zlong/anaconda3/bin/python`, Python 3.11.15,
Torch/LibTorch 2.10.0+cpu. Commands need `TORCH_DEVICE_BACKEND_AUTOLOAD=0`,
`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`. Native builds use two jobs.
No packages changed, no push. Reference repositories remain read-only; LH has
user modifications. No original-LH numerical parity or workload speed is claimed.

The packed replay baseline preserves tested first-order public-root VJPs,
including None versus connected-zero, at a disclosed training cost. Inference
has no replay. Native SettleGraph executes its exact TimedDAG encoding; Python
is its graph compiler only. Existing qualified profiles include EMA, identity,
SSM, Linear/Delta, event attention/GQA/window and tanh/SwiGLU. Event attention is
not LH same-fiber attention or general pretrained-model compatibility.

Artifacts total approximately 584 KiB before this qualification; build products
19 MiB. No artifacts were deleted. Prior evidence is linked from ROADMAP.

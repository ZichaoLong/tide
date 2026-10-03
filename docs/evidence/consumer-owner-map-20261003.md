# Fixed consumer owner-map qualification

Implementation `29effaed064d59b9da930ec3acec810be58b2e1a`, clean immutable
`owner-map-clean01`, qualified on 2026-10-03. [Machine record](consumer-owner-map-20261003.json).

The offline planner and both actual consumer CLIs accept `--owner-map 0,1,...`
in encoded-node order, including both identity boundaries. Full and state use
that joint map. Every declared logical device must own nodes; malformed, short,
out-of-range and unused-device maps fail. Explicit maps never rebalance.
Per-card memory admission, operator cuts, optional physical-sample halving,
canonical parameter owners and safety margins remain unchanged. The result
records `owner_selection` and the actual maps. This makes original-width pilots
and larger logical batches comparable despite different continuation storage.

Three terminal clean jobs passed/exit0 with empty control groups and released
leases: `build-owner-map-consumer-clean01`, `owner-map-cpu-clean01` and
`owner-map-npu-clean01`. CPU45 covers independent Python/C++ planner parity,
fixed-map replay across sample geometries, invalid maps and exact index types,
offline/Python/native metadata and the actual standalone CSV parser. NPU22
covers twelve complete explicit-map candidates through both CLIs, two invalid
map refusals and eight automatic-owner regressions. No skips.

The twenty executed candidates independently consume the same inputs as CPU
streaming references. The explicit cases cover one/two devices, all three graph
families, streaming/prefill, Add/Attention, FP32/FP16, inference/full training,
ragged physical samples and continued windows with two updates. Comparisons
include loss, output/state/message/KV diagnostics, gradients and optimizer
values through the existing complete-consumer checks. Discrete identities and
None/zero connection checks remain exact; existing FP32/FP16 tolerances apply.
Every observed allocator peak fits its unchanged estimate.

Consumer objects were reused only after source, included-header and compiler
option identity checks, then freshly linked against an installed public package.
Core/CANN/resident backend bytes remain the qualified c38b72e dependencies;
loader closure contains no Python runtime, missing library or stub dependency.
This configuration-only change does not need a repeated backend profile or
allocator experiment. It makes no speed or original-scale training claim.

Retain `owner-map-cpu-dev01` as FAILED/exit1: a CPU-only metadata test tried to
construct an unregistered NPU device with vendor autoload disabled. The corrected
test supplies already-resolved logical-index metadata, and actual NPU resolution
is covered by both device CLIs. The original failed source/log/result remain.

Raw records live under `TASK/runs/NAME`, clean source under
`TASK/sources/owner-map-clean01`, build under
`TASK/builds/owner-map-consumer-clean01`, with `TASK` defined in STATUS.
Re-audit: `python TASK/launchers/owner_map_evidence.py
29effaed064d59b9da930ec3acec810be58b2e1a`.
Original Attention B512 training, eager mixed multi-card, formal performance
matrix and final integration remain open; CUDA requires target-machine evidence.

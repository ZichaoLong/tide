# Complete CPU integration qualification

Clean`78e9df6206f5cb95d17f433fafc440cb6ab41651` passed the complete CPU FP64/FP32
gate:**9458 passed,654 skipped**,in2022.25s. All12 registered standalone CTests
also passed,in12.61s. [Receipt and source audit](integrated-cpu-20261004.json).

This uses a fresh complete CPU core,including public CLI/check clients and the
Python module. Its build revision is`7b1fae5`;the eager consumer's is`c6ef224`.
The audit verifies every binary and exact relevant core/consumer source byte
against the tested`78e9df6` tree. Later differences are tests/documentation,not
an assumed binary match. The standalone consumers are CPU-only builds;optional
device/feature skips are not accelerator verification.

The earlier`integration-cpu-clean01` remains failed:120 failed,9337 passed,
654 skipped. It reused a narrow build that omitted standalone clients and
contained two obsolete test assertions. The replacement build supplies the
clients;test corrections retain nonboolean option rejection and expect the
already-defined resolved owner metadata. A targeted rerun first passed all120
previous failing nodes;the complete gate then passed on the same fixed source.
The old build and failed results were preserved.

The tests verify the foundation's CPU behavior and public interfaces. They do
not certify the later full-size CPU/NPU route comparison:the separately
recorded[FP32 near-tie witness](original-add-route-witness-20261004.md) remains a
strict discrete failure. Full-size timing,repeats,profiles and target-machine
CUDA/x86_64 execution have separate acceptance records.

All build/recheck/full/CTest services are terminal with empty cgroups.
Raw:`TASK/runs/integration-cpu-clean02/verification` and
`TASK/runs/integration-ctest-cpu-clean02`.
Audit:`python TASK/launchers/integration_cpu_evidence.py`,where
`TASK=/mi/data2T/zlong/tide-execution-flows`.

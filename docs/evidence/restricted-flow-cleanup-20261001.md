# Restricted consumer cleanup

The 24 pre-existing dirty files are preserved byte-for-byte in pushed archive
`archive/restricted-flow-20260930`, commit
`964bf628c67270200dabe55b1bca026bd403cd37`. Git contains the complete files.
The archive retains the restricted DAG/rank-aligned, reset-window prototype;
it is not the general online execution backend required by the current contract.

The main branch extracts three reusable fixes: mask inactive Full inputs before
nonlinear arithmetic, preserve CPU FP64 control precision, and release peer
transport bridges after non-retained backward. Targeted regressions additionally
compare independent Python/native schedules, full observables, isolated roots,
None/zero and three SGD/AdamW updates on small reachable packets.

Development checks passed: 56 Python FP32/FP64 cases; the new resident dtype
CTest in CPU and standalone-NPU-linked CPU processes; four CPU and four NPU
Add/Attention forward/backward/training cases; one actual two-device peer check.
Records are under `artifacts/execution-flows-cleanup-{cpu,npu}-dev03/`,
`artifacts/execution-flows-cleanup-python-dev01/` and the two isolated relink jobs.
Sources were frozen from 4b8ced4 plus the recorded dirty tree. Reused production
sources/objects were byte-checked; this is development validation, not clean
fixed-commit qualification or general-online backend/performance evidence.

The original dev01 build failures (new fixture used pre-compilation graph
identity) and dev02 gate failures (new overflow probe lacked Full configuration)
remain failed. Later fixture-only relinks and successful runs have distinct records.

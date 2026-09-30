# Public resident inference — 2026-10-01

Clean source **622dbb2** passes 8,954 CPU FP64/FP32 tests (23 optional-device
skips), all 33 standalone NPU component cells, 25 Python client cases and an
installed C++ consumer with three feedback windows. The separate backend
builds at **740fa87** use unchanged core sources at **d412541**; the
[manifest](public-resident-20261001.json) checks their source and binary hashes
and records each raw result. The standalone backend also passes four CTests.

The optional C++ `tide::ResidentSession` package and separately built
Python-owned plugin preserve state on a single NPU across windows. The Python
client never links the standalone NPU SDK into a torch_npu wheel process.
`advance_device()` returns borrowed device outputs; result/snapshot export is
explicit. CPU/NPU external payloads are packed at the boundary, and recursive
messages and scheduling decisions are generated on device. CPU reference
trajectories do not become candidate inputs.

The Python gate covers PDG, TimedDAG and Settle with streaming/greedy schedules,
EMA, event attention and fiber attention, changed inputs, non-default streams,
parameter freezing, capacity failures, explicit snapshots, schedule changes and
disk restore/reset. Invalid checkpoints preserve the live owner and weights;
device allocation failure after successful preflight leaves the old session
closed. The installed consumer uses only the public package/header, including
an installation prefix containing spaces.

The independent resident-check profile records 10,534 AIV and 16 MIX_AIV tasks,
with no AiCPU tasks or host CPU fallback diagnostic. It includes construction,
exports and assertions; summed device task time is neither wall time nor a
throughput measurement. This profile does not certify every attention variant.

Earlier development failures remain archived: a Python build target/interpreter
selection error, CPU checkpoint validation against an NPU model, and invalid
Settle-rank/fiber-GQA test fixtures. The accepted implementation corrected the
runtime boundary and fixtures without relaxing numerical tolerances.

This is **single-NPU FP32 HARD inference** qualification. Python is a client of
the native resident implementation, not an independent pure-PyTorch scheduler.
Resident training, FP16, peer progression and full-size performance remain
separate work. The later state-chain VJP is outside this evidence's source and
scope; F1–F7 are not complete.

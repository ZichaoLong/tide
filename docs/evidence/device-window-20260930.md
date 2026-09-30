# Device windows and optional diagnostics — 2026-09-30

Clean source `4e45072` passed the standalone build/four CPU CTests, all16 component
cells and a separate lean placement trace. The [manifest](device-window-20260930.json)
records source/component identity and hashes of existing job results; raw artifacts
remain under `artifacts/execution-flows-JOB`. No additional tracking system is required.

`advance_device()` preserves authoritative state/history/pending on NPU across
windows and returns borrowed device outputs/counters. CPU `snapshot()` and latest
window `result()` are explicit exports. Disabling diagnostics removes event and
message journals; trace capacity0 is legal. Window input validation/upload and
completion/error checks remain host boundaries. The API is synchronous.

The gate preserves640 existing complete-observable cases and adds384 windows across
four topologies, two input/state variants, both schedules, content/old/proposal Read
and both diagnostic settings. Three successive advances run without downloading
persistent state/history/pending. Delayed diagnostics agree with independent CPU
Streaming; mutating an exported CPU snapshot does not change subsequent NPU execution.
Failed execution refuses snapshots/results/re-entry. FP32 tolerance is rtol1e-5,
atol1e-6; discrete histories, coordinates, activity and pending membership are exact.

The lean trace runs192 windows plus one expected refusal:26378 AIV and360 AI Core
tasks, no journal task, no AiCPU task or host-fallback diagnostic.193 model submits
and193 model-boundary waits match the actual executions. Ordinary stream-sync calls
also occur during setup, boundary operations and verification exports. The profile
contains CPU assertions and is not throughput evidence.

This remains the declared single-device FP32 inference module profile. Vectorized
numerical work, other modules/attention, FP16 integration, public matrix, peer
progression, backward and optimizer are not certified by this source.

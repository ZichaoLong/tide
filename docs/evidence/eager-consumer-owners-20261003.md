# Actual eager consumers across payload owners

Qualified clean `e5d91d7a20321822fc69f8516331d1da371c0719` on 2026-10-03.
The [audit record](eager-consumer-owners-20261003.json) pins source, fresh installed
consumer binaries, reused unchanged qualified core/adapters, logs and device trace.
Five jobs passed/exit0 with empty cgroups and completed device leases.

The Python, native and standalone Add/Attention consumers independently construct
their named learned leaves on selected devices. Static parameter/locality planning
or explicit owner maps, per-device fixed-constant caches, head gathering, all-device
synchronization/allocator observations and finite-gradient agreement are covered.
The parameter planner is not total peak-memory admission. See the
[consumer contract](../online-consumers.md#eager-consumer-payload-placement).

- CPU: 73 affected FP64/FP32 consumer, host-option and static placement checks;
  48 explicitly deselected accelerator cases, no skips.
- Two NPUs: 40 actual consumer checks, 9 CPU-only planning cases deselected,
  no skips. Python/native/standalone, three families, Add/Attention, both schedules
  and mixed A/B/C are represented. Training uses two complete AdamW updates,
  two connected windows/update and physical B1×2. Inference and unified CLI
  replay, automatic/explicit ownership, native workers/packed-sources/batch-next
  are included.
- Independent CPU comparisons include full records, states, pending messages,
  events, gradients, updated parameters, losses, counts and continuation cuts.
  CPU trajectories never drive the accelerator candidates.
- CPU/NPU public-header-only clients are freshly compiled and linked against
  byte-verified qualified `55c3960` core archives installed into new prefixes.
  Standalone loaders contain no Python runtime, missing libraries or stub paths.
  All 1504 frozen source hashes remain unchanged.

The separate actual Attention D4/B2/T2/V7 mixed-C prefill trace includes model
construction, head/loss, backward and two AdamW updates on two devices, with two
connected windows and physical B1×2. It excludes the CPU reference and numerical
diagnostics. Both leased devices appear in the raw operator trace; the audited
JSON records operator engines, AiCPU types, runtime calls and CSV hashes. No host
fallback diagnostic was observed. This is not throughput or a scale extrapolation.

Raw source `TASK/sources/mixed-consumer-clean01`; builds
`TASK/builds/mixed-consumer-{cpu,npu}-clean01`; jobs
`build-mixed-consumer-{cpu,npu}-clean01`, `mixed-consumer-{cpu,npu}-clean01`,
`mixed-consumer-profile-clean01`, each with durable status/log and queue records
where applicable. `TASK=/mi/data2T/zlong/tide-execution-flows`.
Audit: `python TASK/launchers/mixed_consumer_evidence.py e5d91d7a20321822fc69f8516331d1da371c0719`.

Total eager peak admission, packed cross-card message transfer, original-scale
formal comparisons, eager FP16 training and CUDA target execution remain open.
This affected gate does not replace the full historical regression or close F5/F6.

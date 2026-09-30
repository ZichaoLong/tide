# Resident Add and vector state updates — 2026-09-30

Clean `bbe66e2` passes the standalone build/four CPU CTests, all18 component cells
and a separate CANN placement profile. The [manifest](device-add-20260930.json)
records source/component identity and hashes of existing durable results.

The forward loop now supports `lh-add-repeat-v1` alongside identity/EMA. Metadata
preflight checks clocks, counts and a positive repeat-work bound before vector
payload work. Disjoint owner/width tiles execute ordered state sequences; each
Add tick performs its literal FP32 multiply. The scalar device option remains.
Read preparation stays scalar device work. Pre-clear comparison feeds Full.

The Add gate compares560 windows against independent CPU Streaming/Greedy, covering
feedback, unequal delays, parallel edges, content/old/proposal Read, both schedules,
scalar/vector state, selected clear/adoption, int64 values above2^53, widths1–513,
lean diagnostics and restoration with a changed schedule/implementation. Two exact
rounding witnesses distinguish literal repeat from a power shortcut and check
parameter-snapshot isolation. Six repeat-work refusals and one counter overflow
fail explicitly; failed owners refuse snapshots/re-entry. Existing640 content
and384 window regressions also pass. FP32 tolerance remains rtol1e-5, atol1e-6;
discrete observables remain exact.

The trace contains90080 AIV and768 AI Core tasks, including727 Read tasks,
727 scalar/metadata state tasks and364 vector state tasks; no AiCPU task or host
fallback diagnostic. Queue transactions and gathers dominate this finite mixed
validation trace. It includes setup, exports and CPU assertions, so it provides
no complete-flow throughput ratio or CPU/NPU recommendation.

This source qualifies FP32 inference with default state clocks. Attention/KV,
periodic-clock integration, FP16, public matrix, peer progression and resident
backward/optimizer remain separate work. No convergence requirement is added.

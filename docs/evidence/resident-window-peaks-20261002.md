# Resident window capacity observations

Qualified source `475d4afc006117d0f63c297e25c94ead01d6c49a`.
The [audited record](resident-window-peaks-20261002.json) pins source, installed
consumer, unchanged standalone/Python-owned libraries, loader, tests and raw
observations. Audit: `TASK/launchers/window_peaks_evidence.py SOURCE`.

Both resident consumers now include maximum per-window event/stage/output
counts and the pending-queue high-water mark. They clone existing int64 device
counters, reduce them on device and transfer totals/peaks together after step
timing. No Result export, extra numerical input, per-event host read or change
to scheduling, model, VJP or optimizer is introduced. Settle counts include its
encoded identity boundaries; projected body diagnostics omit those nodes.
Pending peaks travel with saved continuations and may include earlier windows.

All **34 checks passed**, without skips: 32 complete sample-slicing cases across
both clients, FP32/FP16, three graph families, both schedules, Add/Attention and
SGD/AdamW, plus continued warmup/whole-batch compatibility and explicit saved-pool
refusal. Independent CPU checks retain outputs, full continuation, physical edge
identity, gradients and parameter updates. Maximum events/output counts are
checked against numerical diagnostics, with the encoded/body distinction for
Settle. Native exact-source objects were reused only after source, header and
compile-option checks; the installed client was freshly linked. Core libraries
and device kernels were unchanged, so no unrelated full regression or profile
was repeated.

The retained failed development run had six new assertions comparing 160
encoded Settle events to 120 body events. The assertion scope and explanation
were corrected; no numerical tolerance or library implementation changed.

These observations help size explicit capacities but do not prove bounds for
future inputs or parameter updates. No full-size training qualification,
throughput comparison or new AiCPU/profile claim follows from this report.

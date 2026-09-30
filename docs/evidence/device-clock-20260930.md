# Periodic state clocks in the device loop — 2026-09-30

Clean `49ff108` passed four CPU CTests/standalone loader checks, all19 component
cells and an independent CANN trace. The [manifest](device-clock-20260930.json)
links the existing frozen source/build and durable results.

Identity/EMA/Add Upd validates and converts actual event/previous timestamps with
int64 division/remainder on NPU. Add repeats only local ticks; stored timestamps,
Read/history/message coordinates and complete cuts stay global. Invalid event
phases refuse the device transaction; off-phase stored states reject restoration.
Identity boundary nodes retain the global clock required by the graph contract.

The clock gate compares480 windows against CPU Streaming/Greedy across feedback,
parallel edges, unequal delays, both schedules and scalar/vector state, all three
Read modes, int64 coordinates/counts above2^53 and continuation restoration. Another
18 multi-phase windows check an analytic Add result: reserved phases add no decay,
a two-local-tick work bound permits larger global gaps, and last_time remains
global. Twelve event/restore phase refusals are checked. Existing560 Add,640 content
and384 window comparisons pass. FP32 tolerances and exact discrete checks remain.

The profile records95457 AIV and746 AI Core tasks, with no AiCPU task or
host-fallback diagnostic. It includes construction, CPU assertions and exports;
it is placement evidence, not throughput. Resident training, attention/KV, FP16,
peer progression and the complete public matrix remain unqualified by this source.

The retained `device-clock-gates-dev01` failed before device execution because its
fixture incorrectly assigned a local clock to an identity boundary. The corrected
fixture keeps that boundary global and maps physical edge/input phases separately.
No runtime relaxation was made, and the original failure remains recorded.

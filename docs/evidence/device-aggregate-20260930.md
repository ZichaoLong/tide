# Device normalized Aggregate — 2026-09-30

Clean `f1b7168` passes the full standalone build, four CPU CTests, all 32 device
cells and independent CANN profiling. The [manifest](device-aggregate-20260930.json)
records exact source, component/core identity, raw hashes and placement counts.
This qualifies single-device FP32 HARD inference within the content-flow profile.

Sum, mean, positive weighted mean, active-source softmax and all-source softmax
retain their existing formulas. Actual ready-fiber metadata generates normalized
event chunks on device. Source scaling precedes normalization; only all-source
softmax includes absent logical sources in its denominator. Exclusive physical
aliases do not enlarge that domain. Present-zero messages/contributions remain
observable. Scalar and vector payload paths retain canonical projected atom order.
The optional coefficient stage and scratch enter the shared forward memory budget;
sum-only graphs omit it. Numerically zero weighted-mean mass refuses with code13
before live commits. The runtime does not substitute uniform weights or consume
a CPU numerical routing prepass.

The new check passes 180 analytic/restore cases, 160 general windows/restores,
four boundary checks and seven wide/empty-domain cases. Widths 1/33/257, physical
aliases, source scaling, zero messages, varied inputs, all Read modes, feedback,
origin projection, scalar/vector paths, 257-slot domains, extreme missing logits,
empty graphs and restore are covered. FP32 rtol 1e-5/atol 1e-6 and exact discrete
checks remain unchanged. The zero-mass failure invalidates the failed runtime.

The separate trace records 67,422 AIV, 139 MIX_AIV and 223 AI Core tasks, with no
AiCPU task or CPU fallback diagnostic. Counts include construction and assertions;
they establish placement, not throughput or a CPU speed ratio. The implementation
currently computes the initial sum before replacing normalized events, which is
a possible later performance improvement subject to independent regression tests.

Development build dev01 failed because Ascend C Muls could not infer a scalar
from a global-memory reference. Loading a local float fixed compilation; dev02
passed all gates and profiling. The original failed snapshot/log remains intact.
Resident VJPs/optimizer, remaining adapters, public presets/matrix and peer
progression remain separate work. F1–F7 are not complete.

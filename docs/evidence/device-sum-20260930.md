# Packed sum on NPU — 2026-09-30

Clean source `fbc6652` passed four CPU CTests, standalone loader checks, all17
component cells and a separate CANN trace. The [manifest](device-sum-20260930.json)
links existing durable records and raw-file hashes; no new experiment service.

The device validates physical sources, logical source uniqueness, fiber coordinates
and offsets before vector payload work. Independent fiber/width tiles use FP32
vector multiply/add in stable message order. Exact tails do not read padding;
unused NaN rows/scales remain absent. Scalar device sum stays explicitly selectable.

Acceptance covers72 scalar/vector cases,144 input-changing replays, nine malformed
or sticky refusals and autograd rejection, plus640 existing content and384 window
comparisons against CPU schedules. Existing FP32 tolerances and exact discrete
observables apply. The trace contains837 AIV tasks:72 scalar sums,81 metadata
preflights and81 vector sums, with no AiCPU task or host-fallback diagnostic.
It includes setup and CPU assertions; these task counts/times are not full-model
throughput or a CPU/NPU recommendation.

The original development build `build-device-sum-dev01` failed CANN scalar-template
deduction; its source/log remain. Loading the GM scalar into a local float fixed
that failure before the later passing development and immutable qualification.

This certifies the declared single-device FP32 inference profile. Other numerical
modules, FP16 integration, resident training, peer progression and the full public
performance matrix remain separate work.

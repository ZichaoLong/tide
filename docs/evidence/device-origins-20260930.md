# Device InputOrigin projection — 2026-09-30

Clean `1be3619`, with the matching standalone core built at `832a881`, passes
four build CTests, all 22 device component cells and a separate placement profile.
The [manifest](device-origins-20260930.json) fingerprints source, binaries through
the raw build result, gates and trace. Records reuse the existing task outputs.

The device projects Aggregate source metadata using the static edge/port/stride
table and stably sorts actual arrivals, retaining physical order on ties. Payload
contributions, receive scales, physical edges, pending messages and transmitted
messages retain physical identity. An off-lattice int64 source position refuses
with code 10; logical source collisions still refuse with code 2. Scalar and vector
sum both complete metadata preflight before numerical writes.

The origin gate passes 24 ordering cases, 128 continuation windows and 8 refusals.
Cases include 32 tied keys, scalar/vector sum, widths 1/33/257, large int64 ports
and times, strides 1/3, feedback/DAG, two inputs, both schedules and policy switches.
A cancellation witness distinguishes the required projected sum order from physical
order. Full observables use the existing strict FP32 and exact discrete comparisons.
The separate native stable-order correction has [its own evidence](native-origin-order-20260930.md).

The trace contains 27,256 AIV tasks, no AiCPU task and no host fallback diagnostic.
It includes setup, observation boundaries and CPU assertions; it is placement
evidence, not a CPU/NPU throughput measurement. Nonempty phase emission is explicitly
refused at this source rather than silently executed as broadcast. Later emission
work has separate tests and does not broaden this immutable qualification. FP16,
attention/KV, peer progression and resident training remain open.

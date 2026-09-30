# Selected LH Full on the device — 2026-09-30

Clean `cd03ca8` passes the standalone build/four CPU CTests, all 21 device
component cells and a separate CANN placement profile. The [manifest](device-lh-full-20260930.json)
records source/core/component identities and hashes of existing results.

All nine `lh-{relu|silu|identity}-{identity|rms|layer}-v1` Full profiles execute
batched CANN activation/normalization on actual selected comparison rows. A device
planner packs each declared kind and chooses subsequent chunks. Per-owner learned
norm parameters are gathered; sentinel padding reads zero and writes distinct
scratch. Inactive poisoned values/parameters do not enter the computation. Identity
and tanh Full can coexist. No content residual is added to LH Full; selected clear
retains the pre-clear comparison. Chunk counts and effective limits are explicit.

The new gate covers 40 component cases, widths 1/7/33/257, chunk limits 1/4,
empty/partial selection, zero variance and large int64 coordinates. The 96 complete
windows compare independent CPU Streaming/Greedy against the device, including
feedback/DAG, two input/state variants, content/old/proposal Read, both schedules,
continuation and switching schedule at a cut. Full graph tensor tolerances remain
rtol1e-5/atol1e-6; discrete/source/route observables remain exact. Existing failure,
lifecycle, control, queue, selection, vector and clock regression cells pass.

There is an explicit component precision boundary: 34 of 536 normalized rows in
the low-variance stress data miss the ordinary CPU/NPU tolerance. CPU FP32 itself
has maximum absolute error 1.31656e-5 against the independent FP64 formula; NPU
has 1.51344e-5. Both pass the input-conditioned engineering budget specified in
[content-flow](../content-flow.md), with maximum used fraction 0.0588963.
Well-conditioned component rows pass the ordinary comparison. This does not
certify arbitrary ill-conditioned graph compositions or change runtime dtype.
The original failed fixed-tolerance gate, diagnostic relink and bounded CANN mode
probe remain preserved. Modes 0/1 did not remove the difference; mode 2 refused
FP32 on this stack. Production uses the ordinary CANN normalization interface.

The separate trace contains 27,763 AIV tasks, including activation, RMSNorm,
LayerNorm and the selected-action planner; no AiCPU or host fallback diagnostic.
This trace includes setup, boundaries and CPU assertions. It is placement evidence,
not a CPU/NPU throughput comparison. Slot-affine signaling, attention/KV, FP16 flow,
public matrix, peer progression and resident backward/optimizer remain open.

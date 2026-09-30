# Device FP32 norm and vector Read — 2026-09-30

Clean `8a735ef` passes the standalone build/four CPU CTests, all 20 device
component cells and a separate CANN placement profile. The [manifest](device-read-20260930.json)
records exact source/core/component identities and hashes of existing results.

`linear-v1` and the explicit `norm-fp32-v1` Read may coexist in one flow.
FP64 norm is explicitly refused. Scalar device Read remains selectable;
`ContentLimits.vectorized_read` defaults to the packed vector implementation.
Metadata preflight validates int64 clocks and literal Add work bounds; independent
owner/width tiles prepare state and reduce dot/squared-norm partials, followed
by device reduction/sqrt. Each tile retains its owner's actual time order.
Selection checks finite descriptors before committing the transaction.

The norm gate compares 530 windows with independent CPU Streaming/Greedy, covering
content/old/proposal Read, scalar/vector execution, widths through 2048,
feedback, parallel edges, stable ties, zero norm, clocks above 2^53,
continuation and changing schedule/implementation at a cut. Two analytic checks,
two overflow refusals and two FP64 refusals pass. Earlier control, lifecycle,
packing, selection, content, window, Full, sum, Add and periodic-clock checks pass.
FP32 tolerance remains rtol 1e-5/atol 1e-6 and discrete observables remain exact.

The trace has 96,173 AIV and 1,083 AI Core tasks, including 410 vector Read,
410 Read-reduction and 802 scalar/metadata Read tasks; no AiCPU task or host
fallback diagnostic. The validation trace includes construction, input/output
boundaries and CPU assertions. Its host copies/synchronizations and kernel counts
are not complete-flow throughput measurements.

This qualifies the stated single-device FP32 inference profiles. Attention/KV,
FP16 flow, the public matrix, peer progression and resident backward/optimizer
remain separate work. Failed development norm fixtures and their diagnoses remain
preserved; they are not relabeled by this clean qualification.

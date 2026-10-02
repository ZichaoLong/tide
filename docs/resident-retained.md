# Retained device windows and boundary adjoints

`retain_reverse_tape` owns a snapshot of the actual device records used by the
[declared graph VJP](resident-graph-vjp.md). It copies dynamic values and
parameter banks on NPU and owns a copy of the static graph metadata. Repeated
TensorImpl references within one tape share one snapshot. Advancing, closing or
overwriting the original forward owner does not invalidate this saved tape.
Construction admits the complete declared tensor footprint before copying;
the caller must also budget all retained windows, reverse buffers and CANN workspace.

The standalone retention functions copy parameter banks for each tape. Guarded
public training owners additionally share immutable Full, emission and attention
parameter snapshots within one backward group. Attention snapshots include QKV,
output projections, parameter biases, decay and pool weights. They never include
KV, cache log-bias, lengths or journals; those still describe each actual window.
Fiber groups can contain fresh gathers, so the cache checks the underlying live
forward banks' identity/version/layout, plus group geometry and alias structure.
It clones the first group's values rather than borrowing writable forward banks.
CANN publication need not increment ATen versions; the owner's prohibition on
publication with outstanding windows is essential. Backward, explicit detach
and close discard the cache, and a later update captures the new values.
Full snapshots include tanh/LH/SwiGLU parameter banks and their static kind/mapping
tables, with identity, version, shape, stride, dtype and device checks. Actual Full
values, counts and event metadata remain independent for each window. The standalone
retention overloads retain their independent-copy behavior.
The dense pre-advance budget charges these shared parameters once and dynamic
records per window. Complete-consumer memory admission remains conservative
until separate allocator calibration justifies any change to its estimate.
The component does not silently detach state or pending messages. Its first-order
parameter accumulation assumes that retained forwards belong to one parameter
generation; it does not differentiate through intervening optimizer updates.

`append_window_bridge` connects consecutive windows. A device hash indexes the
later window's actual incoming boundary messages by all six physical coordinates:
sample, node, int64 time, kind, source and position. Each earlier pending message
must appear once among the later consumed fibers or remaining pending messages.
Missing or duplicate identity fails with sticky error22. Parallel edges retain
their physical source IDs; no numerical-zero or logical-slot shortcut merges them.

The bridge adds later boundary-message gradients to the earlier pending roots
and later initial-state gradients to the earlier final-state roots. The earlier
window's own output roots remain independent. Metadata alone writes connection
bits; vector tiles add only connected payloads. None, connected zero and poisoned
absent roots stay distinct. Empty windows propagate state and pending adjoints.
No event count, matching result or gradient connection returns to the host to
choose the continuation of this reverse program.

`append_parameter_accumulate` sums per-window owner gradients on device, checking
the same TensorImpl owners, aliases and offsets. It preserves connection by union
of the contributing flags and skips absent poisoned values. A finite list of
retained windows determines the outer recorded program structure; each window's
actual event/stage reverse decisions still execute on device.

The development checker retains four windows (including an empty final window),
closes and poisons the original owner, then reverses them in one device program.
Its 42 trajectories per dtype cover168 windows, feedback/self/parallel edges, both schedules,
widths3/257, large int64 times, independent/all/None/zero roots, shared parameters,
initial-state and every external-input gradient, replay, budgets and missing-boundary
refusal. HARD/HST/SOFTP are covered for sum Aggregate, identity/EMA/Add-repeat
state and identity/tanh Full. FP16 payloads remain half in the owned tapes;
journals, boundary roots and alias/parameter sums remain FP32. Byte admission
counts each actual tensor's element size, including mixed-precision tapes.
CPU FP32/FP64 Streaming autograd retains its own graph independently; its half
oracle preserves forward rounding with wide adjoints as described in
[graph reverse](resident-graph-vjp.md). Added checks are implementation scope
until accompanied by immutable-source qualification.

The separate `extended-retained` gate adds all four normalized Aggregate profiles,
nine LH activation/normalization profiles, SwiGLU and mixed modules. It retains
shared coefficients/projection/normalization owners across four windows and checks
None/zero roots, input/state gradients, physical messages and all named owners.
Its CPU half oracle preserves the normalized path's source-product rounding,
FP32 coefficients, separately rounded LH normalization/affine operations and
actual half SwiGLU matmuls/SiLU/products. This extends whole-graph qualification
coverage without changing the production scheduler or using reference routes.
For the half normalization oracle, forward statistics remain FP32 before half
storage, while CPU autograd differentiates normalization in FP64. A constant
input with linear upstream weights has an analytically zero middle gradient;
native CPU FP32 LayerNorm backward can introduce a6.1e-5 cancellation residual
there. The test includes that analytic anchor; existing VJP tolerances remain.

The `event-retained` and `fiber-retained` gates extend the same independent
Streaming comparison through attention caches. Test-only CPU state programs
preserve half QKV, QK and output matmuls, FP32 global normalization/weighted
accumulation, half source products and query scaling, and each actual log-bias
decay tick. They compute their own events and use a dense attention reference,
independently of device key tiling. Named-owner, physical-boundary and initial
state checks are joined by every initial key/value/log-bias gradient and actual
cache lengths. Roots and cache carry remain FP32, including empty connected
caches; poisoned padding must not acquire gradients.

Coverage includes event GQA/eviction, all five fiber pools, mixed event/fiber
groups, shared QKV/Full/decay owners, feedback and parallel edges, HARD/HST/SOFTP,
both schedules, periodic clocks and widths1/4/257. Four windows include an empty
final one. After retaining them, the checker closes the forward owner and
poisons its live cache journals, projection banks and payloads before running
reverse twice. The separate `--profile-smoke` option covers mixed attention and
periodic bias continuation for placement inspection; it does not replace the
complete correctness gate. Qualification is recorded separately in ROADMAP.

This is an internal retained-backward component. It does not supply the public
training lifecycle, optimizer-generation guards, checkpoint controller, additional
module integration, public FP16 training, peer progression or full-size throughput. Qualification and
remaining work are indexed in ROADMAP; submission or profiling alone is not a pass.

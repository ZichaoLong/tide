# Retained device windows and boundary adjoints

`retain_reverse_tape` owns a snapshot of the actual device records used by the
[restricted HARD graph VJP](resident-graph-vjp.md). It copies dynamic values and
parameter banks on NPU and owns a copy of the static graph metadata. Repeated
TensorImpl references within one tape share one snapshot. Advancing, closing or
overwriting the original forward owner does not invalidate this saved tape.
Construction admits the complete declared tensor footprint before copying;
the caller must also budget all retained windows, reverse buffers and CANN workspace.

This first implementation copies parameter banks for each tape. A public owner
can later share an immutable parameter generation after proving its lifetime.
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
Its 26 trajectories cover104 windows, feedback/self/parallel edges, both schedules,
widths3/257, large int64 times, independent/all/None/zero roots, shared parameters,
initial-state and every external-input gradient, replay, budgets and missing-boundary
refusal. CPU FP32/FP64 Streaming autograd retains its own graph independently.

This is an internal retained-backward component. It does not supply the public
training lifecycle, optimizer-generation guards, checkpoint controller, additional
module VJPs, FP16, peer progression or full-size throughput. Qualification and
remaining work are indexed in ROADMAP; submission or profiling alone is not a pass.

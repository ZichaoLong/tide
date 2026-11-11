"""Built-in task's tensor boundary; no event tensors in its differentiable path.

Input identities and the per-physical-port square objective are unchanged.
CPU observation records are decoded only after backward, outside the graph.
Plugins retain their declared record contract until they adopt a batch API.
"""
import torch
from ..weighted.column_layout import SequenceLayout
from ..weighted.message_columns import MessageColumns
from ..weighted.window_execution import WindowResult


def reason(owner):
    """Resolve before input allocation; runtime failures never cause fallback."""
    if owner.reference:
        return "independent scalar reference"
    if owner.spec.get("input_interface", "records") != "auto":
        return "explicit or frozen record interface"
    if owner.trace_records:
        return "complete event observations"
    if owner.resolved["task"].get("plugin"):
        return "task plugin declares the record interface"
    if owner.settle is None:
        return "general graph or non-position cuts"
    from ..weighted.window_execution import column_reason
    e = owner.execute
    return column_reason(owner.binding, owner.settle, e.implementation, e.placement, e.options)


class ColumnInputs:
    def __init__(self, plan, dtype):
        from ..planning.input_pattern import coordinates
        self.positions = plan["positions"]
        by_port, rows = {}, {}
        for b, t, p, _ in coordinates(plan):
            row = rows.setdefault((b, t), len(rows))
            by_port.setdefault(p, []).append(row)
        samples = [b for b, _ in rows]
        times = [t for _, t in rows]
        counts = torch.bincount(torch.tensor(samples, dtype=torch.int64), minlength=plan["samples"])
        self.layout = SequenceLayout(torch.cat((counts.new_zeros(1), counts.cumsum(0))),
                                     torch.tensor(times, dtype=torch.int64))
        # Match the original seeded FP32 material, including Python scalar
        # rounding before each add; dtype promotion happens afterwards.
        phase = torch.arange(plan["width"])*.19
        content = .2*torch.sin(phase[None, :]+torch.tensor([.3*t for t in times])[:, None]
                               +torch.tensor([.1*b for b in samples])[:, None])
        self.values = torch.cat((torch.ones((len(rows), 1)), content), 1).to(dtype=dtype)
        self.rows = {p: torch.tensor(ids, dtype=torch.int64) for p, ids in by_port.items()}

    def window(self, index, device, *, training):
        layout = SequenceLayout(self.layout.offsets, self.layout.times+index*self.positions)
        leaf = self.values.detach().clone().requires_grad_(training)
        values = leaf.to(device)
        # Share computation only for identical inputs; distinct ports keep
        # their physical identity in the dictionary used by Aggregate.
        cache, ports = {}, {}
        for p, rows in self.rows.items():
            key = tuple(rows.tolist())
            if key not in cache:
                value = values.index_select(0, rows.to(device))
                cache[key] = MessageColumns(rows, value[:, 0], value[:, 1:])
            ports[p] = cache[key]
        return layout, ports, leaf


def loss(result, scale, *, training):
    # Sum each port's own (cG)^2, not a square of an aggregated readout.
    # Match the record task: promote half operands before the product/square,
    # not after an overflowing or rounded half reduction.
    fields = [(m.strength.float(), m.content.float()) if m.content.dtype == torch.float16
              else (m.strength, m.content) for m in result.outputs.values() if len(m.rows)]
    terms = [(strength[:, None]*content).square().sum()*scale for strength, content in fields]
    if not terms:
        if training:
            raise ValueError("task.objective: window produced no differentiable output")
        return torch.zeros(())
    from ..weighted.placement_tensors import move
    return torch.stack([move(t, terms[0].device) for t in terms]).sum()


def output_records(result, spec):
    """Public CPU value observations; detached column copy before decoding."""
    records, owners = [], result.layout.owners()
    with torch.no_grad():
        for p, m in result.outputs.items():
            packed = torch.cat((m.strength.detach()[:, None], m.content.detach()), 1).cpu()
            times = result.layout.times[m.rows]*spec.stride+spec.rank(spec.graph.outputs[p])
            records.extend((b, t, p, packed[i]) for i, (b, t) in
                           enumerate(zip(owners[m.rows].tolist(), times.tolist())))
    records.sort(key=lambda x: (x[1], x[0], spec.graph.outputs[x[2]], x[2]))
    return records


def work(result):
    return dict(candidate_updates=result.stats["candidate_events"], full_calls=result.stats["selected_events"],
        messages=result.stats["visited_edges"], outputs=sum(len(x.rows) for x in result.outputs.values()),
        pending=len(result.continuation.pending), actual_stats=dict(result.stats),
        input_positions=len(result.layout.times), effective_window_execution=dict(result.execution))


def locations(result):
    from .locations import _counts, _owners
    q, c = result.continuation, result.columns
    messages = c.messages if c is not None else {}
    return dict(scope="tensor-window-v1 actual columns and raw continuation; logical bytes count aliases",
        state_values=_counts(s.value for s in q.states.values()),
        state_slots=_counts(x for s in q.states.values() for x in s.slots.values()),
        state_node_devices=_owners((v, s.value) for (_, v), s in q.states.items()),
        history_tensors=_counts(x for h in q.history.values() for x in h.tensors.values()),
        messages=_counts(x for m in messages.values() for x in (m.strength, m.content)),
        message_target_node_devices=_owners((c.graph.edges[e].target, m.content) for e, m in messages.items()),
        pending=_counts(a.value for a in q.pending),
        pending_target_node_devices=_owners((a.node, a.value) for a in q.pending),
        outputs=_counts(x for m in result.outputs.values() for x in (m.strength, m.content)),
        output_port_devices=_owners((p, m.content) for p, m in result.outputs.items()))

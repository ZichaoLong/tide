"""Default eager batch boundary; column composition or declared graph fallback.

The callable record API still accepts every legal PDG. Complete Settle position
windows additionally accept tensors directly, without constructing B×L events.
Fallback is chosen from capabilities before computing; runtime errors propagate.
"""
from dataclasses import dataclass, replace
import torch
from ..records import Continuation
from .column_layout import SequenceLayout
from .message_columns import MessageColumns, readout_columns


@dataclass(frozen=True)
class WindowResult:
    layout: SequenceLayout
    outputs: dict
    continuation: Continuation
    stats: dict
    execution: dict
    columns: object = None
    records: object = None
    dense_shape: tuple | None = None

    def readout(self, like):
        value = readout_columns(self.outputs, self.layout, like)
        return value.reshape(*self.dense_shape, -1) if self.dense_shape else value


def column_reason(binding, spec, implementation, placement, options):
    """Capability only: never inspect a computed numerical trajectory."""
    from ..placement import request, read_dtype
    from ..next import AdoptNext
    from .column_modules import module_for
    from .programs import Describe, RegionalTotal, WeightedAggregate, WeightedFull
    if options["streaming"] or options["packing"] == "scalar" or options["prefill_schedule"] == "greedy":
        return "explicit scalar/streaming/greedy schedule"
    if options["workers"] != 1 or options["query_execution"] not in {"auto", "dense-v1"}:
        return "explicit worker or query algorithm"
    devices = {n.bias.device for n in binding.model.nodes}
    if len(devices) != 1:
        return "multiple payload devices"
    device = next(iter(devices))
    p = request(placement).resolve(device)
    if p["events"].type != "cpu" or any(p[k] != device for k in ("read", "control", "selection")):
        return "placement requires split computation"
    for v, n in enumerate(binding.model.nodes):
        if n.bias.dtype not in {torch.float32, torch.float64} or read_dtype(spec.graph.nodes[v], n.bias.dtype, p) != n.bias.dtype:
            return "payload/scoring dtype requires graph adapter"
        if (type(n.read_program) is not Describe or type(n.full_program) is not WeightedFull
                or type(n.aggregate_program) is not WeightedAggregate or type(n.next_program) is not AdoptNext
                or spec.graph.nodes[v].clear or not spec.graph.regions[spec.graph.nodes[v].region].observe_all):
            return "custom local/state-adoption program"
        try:
            module_for(n)
        except ValueError:
            return "local memory has no column kernel"
    if any(type(r) is not RegionalTotal or r.config.support_policy != "strict" for r in binding.model.regions):
        return "regional numerical policy requires graph adapter"
    return None


class EagerExecutor:
    """One default client, with record and tensor-window input forms.

    ``__call__`` retains the complete PDG record contract. ``run_ports`` and
    ``run_positions`` declare ordinary column/parameter/input/left-state roots.
    Request complete intermediate roots through the record API. Rebuild this
    executor after replacing parameter owners, changing dtype or moving devices.
    """
    def __init__(self, record_run, binding, implementation, placement, library, options):
        self.record_run = record_run
        self.binding, self.implementation = binding, implementation
        self.placement, self.library, self.options = placement, library, options
        self.plans = {}

    def __call__(self, *args, **kwargs):
        return self.record_run(*args, **kwargs)

    def run_positions(self, spec, values, *, start_position=0, state=None, max_workspace_bytes=4*1024**3):
        if values.ndim != 3 or min(values.shape) < 1 or type(start_position) is not int or start_position < 0:
            raise ValueError("position window requires nonempty B,L,d and an integer start")
        b, length, width = values.shape
        if start_position+length > (2**63-1)//spec.stride:
            raise ValueError("position window exceeds int64 cut")
        layout = SequenceLayout(torch.arange(b+1, dtype=torch.int64)*length,
                                torch.arange(start_position, start_position+length).repeat(b))
        message = MessageColumns(torch.arange(b*length), values.new_ones(b*length), values.reshape(-1, width))
        result = self.run_ports(spec, layout, {p: message for p in range(len(spec.graph.inputs))},
                               stop_position=start_position+length, state=state, max_workspace_bytes=max_workspace_bytes)
        return replace(result, dense_shape=(b, length))

    def run_ports(self, spec, layout, ports, *, stop_position, state=None, max_workspace_bytes=4*1024**3):
        from .settle_columns import SettleColumnPlan
        from .column_checkpoint import validate_owner
        from types import SimpleNamespace
        binding, graph = self.binding, self.binding.graph
        if spec.binding != "ports" or spec.graph.identity != graph.identity:
            raise ValueError("position window requires the matching physical Settle binding")
        if type(max_workspace_bytes) is not int or not 0 < max_workspace_bytes < 2**63:
            raise ValueError("column workspace budget must be a positive int64 byte count")
        if type(stop_position) is not int or not 0 <= stop_position <= (2**63-1)//spec.stride:
            raise ValueError("invalid position window cut")
        q = Continuation(graph.identity, layout.samples) if state is None else state
        owner = SimpleNamespace(graph=graph, spec=spec, binding=binding)
        validate_owner(owner, q, layout.samples)
        if stop_position*spec.stride < q.cut or (len(layout.times) and
                ((layout.times < q.cut//spec.stride).any() or (layout.times >= stop_position).any())):
            raise ValueError("position window lies outside the sealed cut")
        if any(type(p) is not int or not 0 <= p < len(graph.inputs) for p in ports):
            raise ValueError("unknown physical input port")
        for payload in ports.values():
            payload.validate(layout, binding.model.width-1, binding.model.nodes[0].bias)
        key = (spec.ranks, spec.stride, max_workspace_bytes)
        if key not in self.plans:
            reason = column_reason(binding, spec, self.implementation, self.placement, self.options)
            plan = None if reason else SettleColumnPlan(binding, spec, implementation=self.implementation,
                native_library=self.library, max_workspace_bytes=max_workspace_bytes)
            self.plans[key] = plan, reason
        plan, reason = self.plans[key]
        if plan is not None:
            columns = plan.run_ports(layout, ports, left=None if state is None else plan.import_state(q))
            next_state = plan.export_state(columns, stop_position=stop_position, previous=q)
            return WindowResult(layout, columns.outputs, next_state, columns.stats,
                dict(interface="tensor-window-v1", schedule="settle-columns-v1", client=self.implementation,
                     fallback_reason=None, roots="columns", max_workspace_bytes=max_workspace_bytes), columns=columns)
        # The general engine is still the default for capabilities not yet
        # lowered to columns. These adapters run only on that declared branch.
        from .window_records import external, outputs
        records = self.record_run(q, external(spec, layout, ports, q), stop_position*spec.stride,
                                  sealed_until=stop_position*spec.stride)
        return WindowResult(layout, outputs(spec, layout, records.outputs, binding.model.nodes[0].bias),
            records.continuation, records.stats,
            dict(interface="tensor-window-v1", schedule="graph-record-adapter", client=self.implementation,
                 fallback_reason=reason, roots="columns", max_workspace_bytes=max_workspace_bytes), records=records)

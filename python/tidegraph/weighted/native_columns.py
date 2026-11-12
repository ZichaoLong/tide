"""Tensor-preserving adapter for the independently scheduled LibTorch columns."""
import torch
from .column_layout import SequenceLayout
from .kv_columns_state import KVStateColumns, KVWriteBank, KVWindowColumns
from .recurrent_column_state import RecurrentStateColumns, RecurrentWindowColumns
from .message_columns import MessageColumns


def layout(value):
    return SequenceLayout(torch.tensor(value.offsets, dtype=torch.int64), torch.tensor(value.times, dtype=torch.int64))


def state(value):
    if hasattr(value, "cache_layout"):
        return RecurrentStateColumns(**value.fields, inert=value.inert,
            last_times=torch.tensor(value.last_times, dtype=torch.int64),
            observations=torch.tensor(value.observations, dtype=torch.int64),
            cache_layout=layout(value.cache_layout) if value.cache_layout.offsets else None,
            cache_values=value.cache_values)
    return KVStateColumns(layout(value.layout), **value.fields, inert=value.inert,
        last_times=torch.tensor(value.last_times, dtype=torch.int64), observations=torch.tensor(value.observations, dtype=torch.int64))


class NativeColumns:
    def __init__(self, binding, spec, library=None, max_workspace_bytes=4*1024**3, placement=None):
        from ..native_loader import load_native
        from ..native_model import encode_model
        core = load_native(library, backend=binding.model.nodes[0].bias.device.type)
        if getattr(core, "weighted_column_placement_version", 0) < 3:
            raise RuntimeError("rebuild native library for the placed tensor-column interface")
        from .column_position_ops import POSITION_POLICY
        if getattr(core, "weighted_column_position_policy", None) != POSITION_POLICY:
            raise RuntimeError("rebuild native library for the bounded-common-origin column position policy")
        self.core = core
        graph, self.model = encode_model(core, binding.graph, binding.model)
        self.spec = core.SettleGraph(graph, spec.ranks, spec.binding, spec.stride)
        self.graph = binding.graph
        self.max_workspace_bytes = max_workspace_bytes
        self.placement = None
        if placement is not None:
            self.placement = core.ExecutionPlacement()
            for name, value in placement.to_dict().items():
                setattr(self.placement, name, value)

    def run_ports(self, domain, ports, left):
        from .settle_columns import SettleColumnResult, NodeColumns
        c = self.core
        encode = lambda l: c.ColumnLayout(l.offsets.tolist(), l.times.tolist())
        old = {}
        for v, s in (left or {}).items():
            if isinstance(s, KVStateColumns):
                old[v] = c.KVColumnState(encode(s.layout),
                    {name: getattr(s, name) for name in ("keys", "values", "summaries", "weights")},
                    s.inert, s.last_times.tolist(), s.observations.tolist())
            else:
                old[v] = c.RecurrentColumnState({name: getattr(s, name) for name in ("H", "Z", "W")},
                    s.inert, s.last_times.tolist(), s.observations.tolist(),
                    encode(s.cache_layout) if s.cache_layout is not None else c.ColumnLayout([], []), s.cache_values)
        values = {p: c.MessageColumns(m.rows.tolist(), m.strength, m.content) for p, m in ports.items()}
        result = c.weighted_settle_columns(self.spec, self.model, encode(domain), values, old, self.max_workspace_bytes, self.placement)
        def message(m):
            return MessageColumns(torch.tensor(m.rows, dtype=torch.int64), m.strength, m.content)
        nodes = {}
        states = {v: state(s) for v, s in result.states.items()}
        for v, node in result.nodes.items():
            w = node.window
            if isinstance(w, c.KVColumnWindow):
                window = KVWindowColumns(layout(w.layout), w.mass, w.content,
                    KVWriteBank(layout(w.bank_layout), **w.bank), states[v])
            else:
                window = RecurrentWindowColumns(layout(w.layout), w.mass, w.content,
                    layout(w.write_layout), w.bank, torch.tensor(w.times, dtype=torch.int64),
                    layout(w.cache_layout) if w.cache_layout.offsets else None, w.cache_values, states[v])
            nodes[v] = NodeColumns(torch.tensor(node.rows, dtype=torch.int64), window, node.descriptors, node.controls)
        return SettleColumnResult(self.graph, domain, states,
            {p: message(m) for p, m in result.outputs.items()}, {e: message(m) for e, m in result.messages.items()},
            nodes, result.stats, {p: x.rows for p, x in ports.items()})

"""Core-5 Settle §9–11: online whole-position windows with tensor boundaries.

Ranks order regions (independent regions may tie). Actual message supports
determine candidate groups, without a reference prepass or dense-fiber premise.
KV raw banks stay private; regional descriptors/controls and physical messages
remain columns throughout. This host schedule is independent of scalar engines.
"""
from dataclasses import dataclass
import torch
from .column_layout import SequenceLayout
from .column_modules import module_for
from .message_columns import MessageColumns, readout_columns


@dataclass(frozen=True)
class NodeColumns:
    rows: torch.Tensor
    window: object
    descriptors: dict
    controls: torch.Tensor


@dataclass(frozen=True)
class SettleColumnResult:
    graph: object
    layout: SequenceLayout
    states: dict
    outputs: dict
    messages: dict
    nodes: dict
    stats: dict
    input_rows: dict
    dense_shape: tuple | None = None

    def readout(self, like):
        value = readout_columns(self.outputs, self.layout, like)
        return value.reshape(*self.dense_shape, -1) if self.dense_shape else value


class SettleColumnPlan:
    """Host weighted composition, selected by actual graph/module capabilities.

    Ordinary column roots, terminal raw states and per-edge/port messages are
    returned. Legacy independent intermediate event-root requests continue to
    use the complete-record executor. This is not a resident queue.
    """
    def __init__(self, binding, spec, *, implementation="python", native_library=None, max_workspace_bytes=4*1024**3, placement=None):
        from .programs import Describe, RegionalTotal, WeightedAggregate, WeightedFull
        from ..next import AdoptNext
        if spec.binding != "ports" or binding.graph.identity != spec.graph.identity:
            raise ValueError("column Settle requires the matching physical-port binding")
        binding.model.validate_binding(binding.graph)
        if type(max_workspace_bytes) is not int or not 0 < max_workspace_bytes < 2**63:
            raise ValueError("column workspace budget must be a positive int64 byte count")
        self.max_workspace_bytes = max_workspace_bytes
        self.binding, self.spec, self.graph = binding, spec, binding.graph
        from ..placement import ExecutionPlacement, request, read_dtype
        from ..ownership import region_reference
        self.placement = request(placement) if placement is not None else ExecutionPlacement(selection="payload")
        self.node_locations = {v: self.placement.resolve(n.bias.device) for v, n in enumerate(binding.model.nodes)}
        self.region_locations = {r: self.placement.resolve(region_reference(binding.graph, binding.model, r).device)
                                 for r in range(len(binding.graph.regions))}
        self.locations = self.node_locations[0]
        if any(p["events"].type != "cpu" for p in (*self.node_locations.values(), *self.region_locations.values())):
            raise ValueError("column Settle is host scheduled")
        self.read_dtypes = {v: read_dtype(spec.graph.nodes[v], n.bias.dtype, self.node_locations[v]) for v, n in enumerate(binding.model.nodes)}
        if any(t not in {torch.float32, torch.float64} for t in self.read_dtypes.values()):
            raise ValueError("weighted tensor windows require explicit FP32/FP64 scoring dtype")
        g = self.graph
        self.regions = sorted(range(len(g.regions)), key=lambda r: (spec.ranks[r], r))
        self.inputs = {v: [p for p, node in enumerate(g.inputs) if node == v] for v in range(len(g.nodes))}
        self.edges = {v: [e for e, edge in enumerate(g.edges) if edge.target == v] for v in range(len(g.nodes))}
        self.members = [tuple(layout.members) for layout in g.region_layouts]
        self.modules = {v: module_for(n) for v, n in enumerate(binding.model.nodes)}
        if len({n.bias.dtype for n in binding.model.nodes}) != 1:
            raise ValueError("column Settle requires one payload dtype")
        if any(type(p) is not RegionalTotal or p.config.support_policy != "strict" for p in binding.model.regions):
            raise ValueError("column Settle requires the strict weighted regional binding")
        for v, node in enumerate(binding.model.nodes):
            if (type(node.read_program) is not Describe
                    or type(node.full_program) is not WeightedFull or type(node.aggregate_program) is not WeightedAggregate
                    or type(node.next_program) is not AdoptNext
                    or g.nodes[v].clear or not g.regions[g.nodes[v].region].observe_all):
                raise ValueError("column Settle requires supported modules and observe-all/adopt retained state")
        if implementation not in {"python", "native"}:
            raise ValueError("unknown column implementation")
        self.native = None
        if implementation == "native":
            from .native_columns import NativeColumns
            self.native = NativeColumns(binding, spec, native_library, max_workspace_bytes, self.placement)

    def placement_record(self):
        record = {k: str(v) for k, v in self.locations.items()}
        if len({p["payload"] for p in self.node_locations.values()}) > 1:
            record["scope"] = "first node summary; complete owner maps follow"
            record["nodes"] = {str(v): {k: str(t) for k, t in p.items()} for v, p in self.node_locations.items()}
            record["regions"] = {str(r): {k: str(t) for k, t in p.items()} for r, p in self.region_locations.items()}
        return record

    def import_state(self, continuation):
        from .column_checkpoint import import_state
        return import_state(self, continuation)

    def export_state(self, result, *, stop_position, previous=None):
        from .column_checkpoint import export_state
        return export_state(self, result, stop_position, previous)

    def run(self, values, *, start_position=0, left=None):
        """Unit-strength broadcast convenience binding, preserving both B/L axes."""
        if values.ndim != 3 or min(values.shape) < 1 or type(start_position) is not int or start_position < 0:
            raise ValueError("column Settle input must be nonempty B,L,d with an integer start")
        b, length, width = values.shape
        if start_position+length > 2**63-1:
            raise ValueError("column Settle position exceeds int64")
        rows = torch.arange(b*length, dtype=torch.int64)
        layout = SequenceLayout(torch.arange(b+1, dtype=torch.int64)*length,
                                torch.arange(start_position, start_position+length).repeat(b))
        payload = MessageColumns(rows, values.new_ones(b*length), values.reshape(-1, width))
        result = self.run_ports(layout, {p: payload for p in range(len(self.graph.inputs))}, left=left)
        from dataclasses import replace
        return replace(result, dense_shape=(b, length))

    def run_ports(self, layout, ports, *, left=None):
        """A sealed complete-position window; each port supplies its actual rows.

        ``layout.times`` are global positions; each node derives original int64
        logical times using its rank. Different samples/ports may have sparse,
        unequal schedules. An absent port is empty; a present zero is a candidate.
        ``left`` holds every node's typed raw state, never an effective projection.
        """
        model, g = self.binding.model, self.graph
        if any(type(p) is not int or not 0 <= p < len(g.inputs) for p in ports):
            raise ValueError("unknown physical input port")
        if left is not None and set(left) != set(range(len(g.nodes))):
            raise ValueError("column Settle requires every private left state")
        ref = model.nodes[0].bias
        if ref.dtype not in {torch.float16, torch.float32, torch.float64}:
            raise ValueError("column Settle requires FP16/FP32/FP64 payloads")
        if len(layout.times) and int(layout.times.max()) > (2**63-1-max(self.spec.ranks))//self.spec.stride:
            raise ValueError("column Settle logical clock exceeds int64")
        for p in ports.values():
            p.validate(layout, len(ref)-1, ref, allow_remote=True)
        if self.native is not None:
            return self.native.run_ports(layout, ports, left)
        initial = {v: self.modules[v].state_type.from_states([n.initial() for _ in range(layout.samples)])
                   for v, n in enumerate(model.nodes)}
        states = dict(initial if left is None else left)
        for v, state in states.items():
            ref = model.nodes[v].bias
            state.validate(model.nodes[v].kernel.config)
            payload = getattr(state, self.modules[v].reference_field)
            if len(state.last_times) != layout.samples or (payload.dtype, payload.device) != (ref.dtype, ref.device):
                raise ValueError("column Settle left state sample/payload binding mismatch")
            from .column_capacity import check
            scalar_bytes = torch.empty((), dtype=self.read_dtypes[v]).element_size()
            check(model.nodes[v].kernel, state, layout, self.max_workspace_bytes, max(ref.element_size(), scalar_bytes))
        messages, records = {}, {}
        empties = {v: MessageColumns(torch.empty(0, dtype=torch.int64), n.bias.new_empty(0),
                                   n.bias.new_empty((0, len(n.bias)-1))) for v, n in enumerate(model.nodes)}
        from . import column_transport
        transport_cache = {}
        candidate_count = full_count = calls = 0
        for r in self.regions:
            windows, descriptors, rows_by_node = {}, {}, {}
            aggregate_requests = []
            for v in self.members[r]:
                ref = model.nodes[v].bias
                sources = [(ports[p], self.binding.source_classes.inputs[p]) for p in self.inputs[v] if p in ports]
                sources += [(messages.get(g.edges[e].source, empties[v]), self.binding.source_classes.edges[e]) for e in self.edges[v]]
                sources = [(column_transport.message(p, ref, transport_cache), c) for p, c in sources]
                aggregate_requests.append((model.nodes[v].aggregate_program, sources))
            from .column_aggregate import aggregate_columns_many
            received_by_node = dict(zip(self.members[r], aggregate_columns_many(aggregate_requests,
                max_workspace_bytes=self.max_workspace_bytes)))
            preparations = {}
            for v, received in received_by_node.items():
                if received is None:
                    continue
                rows, mass, u = received
                positions = layout.select(rows)
                times = SequenceLayout(positions.offsets, positions.times*self.spec.stride+self.spec.rank(v))
                module = self.modules[v]
                preparations[v] = (module, model.nodes[v].kernel, times, mass, u, states[v])
                rows_by_node[v] = rows
            from .column_modules import prepare_many, full_many
            from . import column_placement
            prepared = prepare_many(preparations, max_workspace_bytes=self.max_workspace_bytes)
            read_requests = {}
            for v, window in prepared.items():
                module = self.modules[v]
                mode = self.binding.regions[r].read_mode
                read_requests[v] = (module, model.nodes[v], window, initial[v], mode,
                                   self.node_locations[v]["read"], self.read_dtypes[v])
                windows[v], states[v] = window, window.terminal
            descriptors = column_placement.read_many(read_requests, max_workspace_bytes=self.max_workspace_bytes)
            placement = self.region_locations[r]
            regional_fields = {v: column_transport.fields(d, placement["control"]) for v, d in descriptors.items()}
            from .regional_packing import RegionalPacking
            from .regional_groups import region_groups
            packing = RegionalPacking(rows_by_node)
            grouped = region_groups(model.regions[r], packing, regional_fields,
                                    control=placement["control"], selection=placement["selection"])
            controls = {v: column_placement.controls(value, model.nodes[v].bias) for v, value in grouped.items()}
            calls += len(packing.groups)
            fulls = full_many({v: (self.modules[v], model.nodes[v], window, controls[v])
                              for v, window in windows.items()}, max_workspace_bytes=self.max_workspace_bytes)
            for v, window in windows.items():
                full = fulls[v]
                rows = rows_by_node[v]
                messages[v] = MessageColumns(rows[full.rows], full.strength, full.content)
                records[v] = NodeColumns(rows, window, descriptors[v], controls[v])
                candidate_count += len(rows)
                full_count += len(full.rows)
        edge_messages = {e: column_transport.message(messages.get(edge.source, empties[edge.target]),
                         model.nodes[edge.target].bias, transport_cache) for e, edge in enumerate(g.edges)}
        outputs = {p: messages.get(v, empties[v]) for p, v in enumerate(g.outputs)}
        stats = dict(candidate_events=candidate_count, selected_events=full_count,
                     visited_edges=sum(len(p.rows) for p in edge_messages.values()),
                     region_column_calls=calls, state_column_calls=len(records))
        from ..finite import all_finite
        published = [x for n in records.values() for x in (*n.descriptors.values(), n.controls, n.window.mass, n.window.content)]
        for n in records.values():
            bank = n.window.bank if isinstance(n.window.bank, dict) else vars(n.window.bank)
            published.extend(x for x in bank.values() if isinstance(x, torch.Tensor) and x.is_floating_point())
        published += [x for p in (*outputs.values(), *edge_messages.values()) for x in (p.strength, p.content)]
        published += [x for state in states.values() for x in vars(state).values()
                      if isinstance(x, torch.Tensor) and x.is_floating_point()]
        if not all_finite(published):
            raise FloatingPointError("nonfinite column result")
        return SettleColumnResult(g, layout, states, outputs, edge_messages, records, stats, {p: x.rows for p, x in ports.items()})

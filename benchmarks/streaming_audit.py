"""Exact semantic stream totals with no retained per-time parameter sets.

The dense inventory is frozen before placement and contains Python metadata only.
Events use one bit per semantic parameter *atom*, never one entry per scalar.
Each real forward/flush is reduced and released before the next call. The public
WorkAudit remains useful for small, inspectable per-time formula witnesses.
"""
from collections import defaultdict
import copy
import hashlib
import json
from threading import RLock

from benchmarks.statistics import ParameterInventory, WorkAudit


class FrozenParameterInventory(ParameterInventory):
    """Detach ownership metadata from the dense model before sharding it."""

    def __init__(self, model):
        source = ParameterInventory(model)
        for name in ("sizes", "classes", "owner_atoms", "source_atoms", "whole_atoms", "local_nodes"):
            setattr(self, name, getattr(source, name))
        self._report = source.report()
        self.config = model.config.to_dict()
        # Deliberately do not keep source.model, source.parameters or tensor IDs.

    def report(self):
        return copy.deepcopy(self._report)


class StreamingWorkAudit(WorkAudit):
    """Exclusive, untimed audit of a contiguous stateful inference schedule.

    These ordinary Python method/forward hooks also run in region workers.
    Shared counters are locked; logical time is fixed until the joined forward
    returns. No caller-thread TorchDispatchMode is used or needs propagation.
    All timed samples are separate from this intrusive observation.
    """

    def __init__(self, model, *, inventory=None):
        inventory = FrozenParameterInventory(model) if inventory is None else inventory
        if inventory.config != model.config.to_dict():
            raise ValueError("frozen inventory configuration differs from model")
        if sum(p.numel() for p in model.parameters()) != sum(inventory.sizes.values()):
            raise ValueError("frozen inventory parameter total differs after placement")
        super().__init__(model, inventory=inventory)
        self.events = defaultdict(lambda: defaultdict(int))
        self._indices = {atom: index for index, atom in enumerate(inventory.sizes)}
        self._sizes = tuple(inventory.sizes.values())
        self._totals = defaultdict(lambda: dict(sum=0, min=None, max=0, covered_slots=0))
        self._union = 0
        self._slots = self._calls = self._peak_slots = self._peak_bytes = 0
        self._previous_end = self._batch = None
        self._digest = hashlib.sha256()
        self._lock = RLock()

    def _add(self, stage, key, atoms):
        bits = 0
        for atom in atoms:
            bits |= 1 << self._indices[atom]
        with self._lock:
            self.events[stage][key] |= bits

    def _work(self, name, values, offset=0):
        # No [sample,time] work dictionaries survive the current call.
        value = int(values.detach().sum().cpu())
        with self._lock:
            self.work[name] += value

    def _packed(self, kind, original, *args, **kwargs):
        result = original(*args, **kwargs)
        selected = 0 if result is None else result.node.numel()
        executed = 0 if result is None else result.data.numel()//result.data.shape[-1]
        with self._lock:
            row = self.packing[kind]
            row["calls"] += 1
            row["selected_rows"] += selected
            row["executed_rows"] += executed
        return result

    def _matrix(self, path, module, inputs, output, *, record_dispatch=True):
        with self._lock:
            super()._matrix(path, module, inputs, output, record_dispatch=record_dispatch)

    def _base_matrix_dispatch(self, path, module, inputs, *, out_features=None):
        with self._lock:
            super()._base_matrix_dispatch(path, module, inputs, out_features=out_features)

    def _count(self, bits):
        count = 0
        while bits:
            low = bits & -bits
            count += self._sizes[low.bit_length()-1]
            bits ^= low
        return count

    def _reduce_call(self):
        combined = defaultdict(int)
        for stage, events in self.events.items():
            for key, bits in events.items():
                combined[key] |= bits
        self._peak_slots = max(self._peak_slots, len(self.domain))
        self._peak_bytes = max(self._peak_bytes, sum((bits.bit_length()+7)//8
                              for events in self.events.values() for bits in events.values()))
        # Deduplicate equivalent supports within this call; release this cache
        # with the event maps. There is no cache growing with stream duration.
        counts = {0: 0}
        for stage, events in [*self.events.items(), ("__combined", combined)]:
            row = self._totals[stage]
            row["covered_slots"] += len(self.domain)
            for key in sorted(self.domain):
                bits = events.get(key, 0)
                if bits not in counts:
                    counts[bits] = self._count(bits)
                value = counts[bits]
                row["sum"] += value
                row["min"] = value if row["min"] is None else min(row["min"], value)
                row["max"] = max(row["max"], value)
                if stage == "__combined":
                    self._union |= bits
                    # Bind the per-time activity without serializing huge rows.
                    self._digest.update(json.dumps([*key, value], separators=(",", ":")).encode()+b"\n")
        self._slots += len(self.domain)
        self._calls += 1
        self.events.clear()
        self.domain.clear()

    def _run(self, original, x, state=None, *, record_work=False, inject=True):
        start = 0 if state is None else state.position
        if self._previous_end is not None and (start != self._previous_end or x.shape[0] != self._batch):
            raise ValueError("stream audit requires contiguous positions and a fixed batch")
        result = super()._run(original, x, state, record_work=record_work, inject=inject)
        self._previous_end, self._batch = result.state.position, x.shape[0]
        self._reduce_call()
        return result

    def _aggregate(self, stage):
        row = self._totals[stage]
        minimum = (0 if row["covered_slots"] < self._slots or row["min"] is None else row["min"])
        return dict(sum=row["sum"], mean=row["sum"]/self._slots if self._slots else 0,
                    min=minimum, max=row["max"])

    def report(self):
        if self.domain:
            raise RuntimeError("cannot report an incomplete forward audit")
        total = self._aggregate("__combined")
        total.update(unique_across_time_and_batch=self._count(self._union),
                     cumulative_per_input_position=total["sum"]/self.batch_inputs if self.batch_inputs else None)
        return dict(parameters=self.inventory.report(), effective_touched=total,
                    stages={stage: self._aggregate(stage) for stage in self._totals if stage != "__combined"},
                    definition="unique Parameter scalars in effective owner/group/source slices; not gradients or physical loads",
                    deduplication=("within sample/input step across settled layers" if self.model.settle else
                                   "within sample/logical tick across regions; repeated ticks accumulate; no output causal attribution"),
                    input_positions=self.batch_inputs, sample_time_slots=self._slots,
                    logical_axis="input_step" if self.model.settle else "tick", work=dict(self.work),
                    work_per_input_position={k: v/self.batch_inputs for k, v in self.work.items()} if self.batch_inputs else {},
                    query_packing=dict(self.packing), node_matrices=dict(self.matrices),
                    ffn_packing=self._ffn_packing_report(),
                    node_matrix_dispatch=self._matrix_dispatch_report(),
                    matrix_dispatch_definition="NodeLinear base dispatch; gather rows describe BMM operands, grouped rows count nonempty unique groups with separate bank-view span including empty groups; neither is copied bytes or bus traffic; private adapters remain separate in matrix cost",
                    matrix_cost_definition="multiply-add pairs of executed NodeLinear rows including padding/adapters; not complete FLOPs",
                    retention=dict(kind="one-call semantic-atom bitsets and scalar stream aggregates; no scalar-parameter axis",
                                   calls=self._calls, peak_sample_time_slots=self._peak_slots,
                                   peak_event_bitset_payload_bytes=self._peak_bytes,
                                   per_time_parameter_counts_sha256=self._digest.hexdigest(),
                                   digest_definition="ordered newline JSON [sample,logical_time,unique_parameters] per actual call; total combined only",
                                   per_time_rows_retained=False, dense_parameter_references_retained=False))

"""Python-owned control/ranking placement, independent of native selection."""
import torch
from .history import increment
from .region import RegionProgram, CountSelector, PositiveSelector, TensorHistorySelector, Selection
from .lh_selector import LHSelector


class PlacedRegion(RegionProgram):
    def __init__(self, spec, original, placement):
        super().__init__()
        if type(original) not in {CountSelector, PositiveSelector, TensorHistorySelector, LHSelector, PlacedRegion} or original.profile != spec.selector:
            raise ValueError("placement cannot replace a custom or mismatched Region program")
        if isinstance(original, PlacedRegion):
            original = original._original
        # Retain original parameter names and leaf/alias identity. The delegate
        # is not registered a second time as a nested module.
        object.__setattr__(self, "_original", original)
        self._parameters = dict(original._parameters)
        self._buffers = dict(original._buffers)
        self._modules = dict(original._modules)
        self.profile, self.placement = spec.selector, placement

    def initial(self, layout, reference):
        return self._original.initial(layout, reference)

    def validate_weights(self, layout):
        self._original.validate_weights(layout)

    def validate_history(self, history, layout):
        self._original.validate_history(history, layout)

    def step(self, r):
        nodes = [node for node, _ in r.candidates]
        if not nodes:
            raise ValueError("empty placed region input")
        p = self.placement
        descriptors = torch.stack([value.to(p["control"]) for _, value in r.candidates])
        scores = descriptors
        if self.profile == "tensor-history-v1":
            ids = torch.tensor([r.layout.slots[node] for node in nodes], device=self.bias.device, dtype=torch.int64)
            scores = descriptors + r.history.tensors["memory"].to(p["control"]) * self.bias.index_select(0, ids).to(p["control"])
        if not torch.isfinite(scores).all():
            raise ValueError("nonfinite region selector score")
        with torch.no_grad():
            numeric = scores.to(p["selection"])
            order = torch.argsort(numeric, descending=True, stable=True)
            if self.profile == "lh-count-affect-v1":
                counts = torch.tensor([r.history.node_maps["affected"].get(node, 0) for node in nodes], dtype=torch.int64, device=p["selection"])
                order = order[torch.argsort(counts[order], descending=True, stable=True)]
            if r.layout.spec.count_priority:
                counts = torch.tensor([r.history.node_maps["selected"].get(node, 0) for node in nodes], dtype=torch.int64, device=p["selection"])
                order = order[torch.argsort(counts[order], stable=True)]
            if self.profile == "positive-v1":
                eligible = (numeric > 0).to(torch.int64)
                order = order[torch.argsort(eligible[order], descending=True, stable=True)]
            take = torch.arange(len(nodes), device=p["selection"]) < r.layout.spec.budget
            if self.profile == "positive-v1":
                take = take & (numeric[order] > 0)
            bits = torch.zeros_like(take).scatter(0, order, take).cpu().tolist()
        active = {node for node, bit in zip(nodes, bits) if bit}
        # Each frame owns its softmax graph, preserving independent public roots.
        controls = scores.softmax(0).to(dtype=r.payload_dtype, device=r.payload_device)
        history = r.history.fork()
        history.last_time = r.time
        for node in active:
            history.node_maps["selected"][node] = increment(r.history.node_maps["selected"].get(node, 0))
        if self.profile == "lh-count-affect-v1":
            for node in nodes:
                history.node_maps["affected"][node] = increment(r.history.node_maps["affected"].get(node, 0))
        if self.profile == "tensor-history-v1":
            total = descriptors.sum().to(dtype=r.payload_dtype)
            memory = self.alpha.to(p["control"]) * r.history.tensors["memory"].to(p["control"]) + total
            history.tensors["memory"] = memory.to(r.payload_device)
        return Selection(active, dict(zip(nodes, controls.unbind())), history)

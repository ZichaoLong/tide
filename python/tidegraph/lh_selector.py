"""LH's prior-count/affect/score order, with Tide's declared softmax controls."""
import torch
from .history import History, increment
from .region import RegionProgram, Selection


class LHSelector(RegionProgram):
    profile = "lh-count-affect-v1"

    def initial(self, layout, reference):
        return History(node_maps={"selected": {}, "affected": {}})

    def validate_weights(self, layout):
        if not layout.spec.count_priority:
            raise ValueError("LH selector requires count priority")

    def validate_history(self, history, layout):
        if (history.scalars or history.tensors or history.node_maps.keys() != {"selected", "affected"}
                or any(c < 0 for counts in history.node_maps.values() for c in counts.values())):
            raise ValueError("invalid LH selector history layout")

    def step(self, r):
        counts, affected = r.history.node_maps["selected"], r.history.node_maps["affected"]
        order = sorted(r.candidates, key=lambda x: (counts.get(x[0], 0), -affected.get(x[0], 0),
                                                   -float(x[1].detach()), x[0]))
        active = {v for v, _ in order[:r.layout.spec.budget]}
        controls = torch.stack([d for _, d in r.candidates]).softmax(0).to(dtype=r.payload_dtype, device=r.payload_device)
        history = r.history.fork(); history.last_time = r.time
        for v, _ in r.candidates:
            history.node_maps["affected"][v] = increment(affected.get(v, 0))
            if v in active:
                history.node_maps["selected"][v] = increment(counts.get(v, 0))
        return Selection(active, {v: p for (v, _), p in zip(r.candidates, controls.unbind())}, history)

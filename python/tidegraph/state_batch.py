"""Independent step batches and ordered sample batches; no padded observations."""
import torch
from .records import State
from .history import increment


def step(kernel, weights, old, views, times):
    from .clocked_state import ClockedState
    from .matrix_memory import MatrixMemory
    from .lazy_add import LazyAdd
    from .packing import PackedSequence
    if type(kernel) is ClockedState:
        values = step(kernel.program, weights, [kernel.clock.local_state(s) for s in old],
                      views, [kernel.event(t) for t in times])
        return [kernel.clock.global_state(s) for s in values]
    h = torch.stack([v.value for v in views])
    if type(kernel) is LazyAdd:
        return kernel._batch_rows(weights, old, h, times, independent=False)[0]
    if type(kernel) is MatrixMemory:
        q, k, v = kernel.project(weights, h)
        matrix = torch.stack([s.slots["matrix"] for s in old])
        linear = kernel.kind == "linear"
        if linear:
            matrix = matrix + k.unsqueeze(-1)*v.unsqueeze(-2)
            z = torch.stack([s.slots["normalizer"] for s in old])+k
        else:
            beta = (h @ weights.extra["mem_beta"]).sigmoid()[:, None, None]
            decay = (h @ weights.extra["mem_decay"]).sigmoid()[:, None, None] if kernel.kind == "delta" else 1
            decayed = decay*matrix
            error = v - (k.unsqueeze(-2) @ decayed).squeeze(-2)
            matrix = decayed + beta*k.unsqueeze(-1)*error.unsqueeze(-2)
        values = (q.unsqueeze(-2) @ matrix).squeeze(-2)
        if linear:
            values = values / ((q*z).sum(-1, keepdim=True)+kernel.epsilon)
        values = values @ weights.extra["mem_out"]
        return [State(values[i], times[i], increment(s.observations),
                      {"matrix": matrix[i], **({"normalizer": z[i]} if linear else {})}) for i, s in enumerate(old)]
    batch = PackedSequence(h, list(range(len(old)+1)), [(i, 0) for i in range(len(old))], list(times), list(views))
    return kernel.packed_sequence(weights, old, batch)[0]

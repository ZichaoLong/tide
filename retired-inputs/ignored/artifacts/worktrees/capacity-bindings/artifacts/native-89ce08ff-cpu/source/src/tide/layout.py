"""Selected Full queries stay in one node/sample/query tensor."""
from dataclasses import dataclass
import torch


@dataclass
class Queries:
    nodes: torch.Tensor
    node: torch.Tensor
    batch: torch.Tensor
    time: torch.Tensor
    slot: torch.Tensor
    data: torch.Tensor
    positions: torch.Tensor
    valid: torch.Tensor
    original_shape: tuple

    def publish(self, result):
        n, b, length, d = self.original_shape
        out = result.new_zeros((n, b, length, d))
        return out.index_put((self.nodes[self.node], self.batch, self.time),
                             result[self.node, self.batch, self.slot])


def select_queries(u, control, times):
    active = control > 0
    nodes = active.any(dim=(1, 2)).nonzero().flatten()
    shape = u.shape
    if not len(nodes):
        return None
    active = active[nodes]
    counts = active.sum(-1)
    width = int(counts.max())
    ni, bi, ti = active.nonzero(as_tuple=True)
    slots = (active.long().cumsum(-1)-1)[ni, bi, ti]
    data = u.new_zeros((len(nodes), shape[1], width, shape[-1]))
    data = data.index_put((ni, bi, slots), u[nodes[ni], bi, ti])
    pos = times.new_zeros((len(nodes), shape[1], width))
    pos[ni, bi, slots] = times[nodes[ni], bi, ti]
    valid = torch.arange(width, device=u.device)[None,None,:] < counts[...,None]
    return Queries(nodes, ni, bi, ti, slots, data, pos, valid, shape)

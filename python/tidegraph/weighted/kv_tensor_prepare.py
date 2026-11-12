"""KV §1.2: positive writes on a node×row tensor boundary, without snapshots."""
import torch
from .tensor_kernel import Role, mapped_role, execute
from .tensor_describe import Algebra
from .tensors import norm
from .packing_observation import bucket


class Writes:
    fields = ("keys", "values", "summaries", "weights")
    shared = ()

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __call__(self, values):
        a = Algebra(values, self.shared)
        u = norm(values["u"], self.epsilon)
        projected = a.linear(u, "base")+a.linear(a.linear(u, "adapter_v"), "adapter_u")
        keys, vals = projected.chunk(2, -1)
        return dict(keys=keys, values=vals, summaries=a.linear(u, "summary"), weights=values["mass"].clamp_max(1))


def prepare(requests, prepared, ids):
    """Return actual new raw fields for a previously validated positive-write group."""
    width = max(len(prepared[i][1]) for i in ids)
    mappings, reverse = [], []
    for i in ids:
        rows = prepared[i][1]
        mapping = torch.full((1, width), len(requests[i][2]), dtype=torch.int64)
        mapping[0, :len(rows)] = rows
        mappings.append(mapping)
        reverse.append(torch.arange(len(rows), device=requests[i][3].device))
    roles = dict(u=mapped_role([requests[i][3] for i in ids], mappings),
                 mass=mapped_role([requests[i][2] for i in ids], mappings))
    for name in ("base", "adapter_v", "adapter_u", "summary"):
        values = tuple(getattr(requests[i][0], name) for i in ids)
        roles[name] = Role(values, shared=all(x is values[0] for x in values))
    bucket("kv_tensor_writes", width, len(ids))
    return execute(roles, tuple(reverse), Writes(requests[ids[0]][0].config.epsilon))

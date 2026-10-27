"""Recurrent §4.1–4.3: batched coefficients and affine prefix composition.

Only certified complete drivers enter here. Positive writes form ragged
segments; original int64 gaps, all intermediate states and raw zeros survive.
"""
import torch
from ..scan import affine_scan
from .packing import Writes, matrix_scan, project
from .tensors import norm


def basic_coefficients(memory, plan, x, decay):
    c = memory.config
    key = project(x, memory.key)
    key = key/key.norm(dim=-1, keepdim=True).clamp(min=1)
    value = project(x, memory.value)
    retain = torch.exp(-plan.weight*torch.nn.functional.softplus(x@memory.retain+memory.retain_bias))
    gain = plan.weight
    if c.memory_family == "delta":
        gain = gain*torch.sigmoid(x@memory.write+memory.write_bias)
        unit = torch.eye(c.key_width, dtype=x.dtype, device=x.device)
        a = (unit-gain[:, None, None]*key.unsqueeze(-1)*key.unsqueeze(-2))[:, None]
        a = a*(decay*retain[:, None])[:, :, None, None]
    else:
        a = (decay*retain[:, None])[:, :, None, None]
    b = (gain[:, None, None]*value.unsqueeze(-1)*key.unsqueeze(-2))[:, None].expand(-1, 2, -1, -1)
    return a, b, c.memory_family == "delta"


def packed(memory, old, batch):
    plan = Writes(old, batch)
    if not plan.ids:
        return plan.decode(old, batch, []), 0
    c = memory.config
    x = norm(plan.content, c.epsilon)
    delta = plan.gaps(old).to(x.dtype)
    decay = torch.exp(-delta[:, None]*x.new_tensor((c.lambda_short, 0.)))
    caches = None
    if c.basic:
        a, b, matrix = basic_coefficients(memory, plan, x, decay)
    else:
        from .gated_packed import coefficients
        a, b, matrix, caches = coefficients(memory, plan, old, x, delta)
    # p_v is a distinct scalar-formula branch. Sharing its normalization with
    # the matrix update changes FP32 adjoint summation before the RMS VJP.
    z = plan.weight[:, None]*project(norm(plan.content, c.epsilon), memory.summary)
    slots = [None]*len(plan.ids)
    for length, owners in plan.groups.items():
        ids = plan.group_index(length, owners)
        initial = torch.stack([old[i].slots["H"] for i in owners])
        H = matrix_scan(a[ids], b[ids], initial) if matrix else affine_scan(a[ids], b[ids], initial)
        Z = affine_scan(decay[ids].unsqueeze(-1), z[ids].unsqueeze(-2).expand(-1, -1, 2, -1),
                        torch.stack([old[i].slots["Z"] for i in owners]))
        W = affine_scan(decay[ids], plan.weight[ids].unsqueeze(-1).expand(-1, -1, 2),
                        torch.stack([old[i].slots["W"] for i in owners]))
        for row, owner in enumerate(owners):
            for j in range(length):
                k = plan.offsets[owner]+j
                slots[k] = dict(time=plan.times[k], H=H[j, row], Z=Z[j, row], W=W[j, row])
                if caches is not None:
                    slots[k].update(caches[k])
    return plan.decode(old, batch, slots), len(plan.groups)

"""Recurrent Read/Full column kernels; private matrix state never becomes KV."""
import torch
from torch.nn import functional as F
from .tensors import norm
from .normalization import scaled_rms
from .dense_describe import columns as describe
from .recurrent_column_math import snapshots, effective, features
from .kv_columns import FullColumns


def read_recurrent_columns(node, window, *, read_mode="+", initial=None):
    if read_mode not in {"0", "-", "+"} or read_mode == "0" and initial is None:
        raise ValueError("invalid recurrent Read snapshot or missing bound initial state")
    m, c, u = node.kernel, node.kernel.config, window.content
    rows = torch.arange(len(window.layout.times))
    layout, state, times, cache_layout, cache_values = snapshots(window, rows, read_mode, initial)
    H, Z, W = effective(m, layout, state, times)
    if c.memory_read == "state" and c.basic:
        q = norm(u, c.epsilon)@m.read_query.t()
        read = (H@q[:, None, :, None]).squeeze(-1)
        values = (m.read_out@read.unsqueeze(-1)).squeeze(-1)
        short, long = values.unbind(1)
    else:
        short, long = (Z/W.clamp_min(1)[..., None]).unbind(1)
        if c.memory_read == "state":
            q, _, _, _, _ = features(m, layout, u, cache_layout, cache_values)
            long = (H@q.unsqueeze(-1)).squeeze(-1).flatten(1)@m.read_out.t()
    return describe(node.read_program, m, window.mass, u, (short, long, W[:, 0], W[:, 1]))


def full_recurrent_columns(node, window, controls):
    m, c, p = node.kernel, node.kernel.config, node.full_program.content
    if (controls.shape != window.mass.shape or (controls.dtype, controls.device) != (window.mass.dtype, window.mass.device)
            or not torch.isfinite(controls).all() or (controls < 0).any()):
        raise ValueError("invalid recurrent Full controls")
    rows = (controls.detach() > 0).nonzero().flatten().cpu()
    ids = rows.to(controls.device)
    strength = controls.index_select(0, ids)
    if not len(rows):
        return FullColumns(rows, strength, window.content.new_empty((0, c.width)))
    u = window.content.index_select(0, ids)
    layout, state, times, cache_layout, cache_values = snapshots(window, rows, "+", None)
    H, _, _ = effective(m, layout, state, times)
    if c.basic:
        q = norm(u, c.epsilon)@p.query.t()
        read = (H@q[:, None, :, None]).squeeze(-1)
        mix = (p.out@read.unsqueeze(-1)).squeeze(-1).sum(1)
    else:
        q, _, value, gate, _ = features(m, layout, u, cache_layout, cache_values)
        read = (H@q.unsqueeze(-1)).squeeze(-1)
        if c.memory_family == "gated_delta":
            gated = scaled_rms(read, p.rho, c.output_epsilon).flatten(1)*F.silu(norm(u, c.epsilon)@p.output_gate.t())
        else:
            mixed = (read+p.direct[:, None]*value).flatten(1)*F.silu(gate)
            gated = scaled_rms(mixed, p.rho, c.output_epsilon)
        mix = gated@p.out.t()
    x = u+c.alpha_mem*mix
    n = norm(x, c.epsilon)
    value = x+c.alpha_ff*((F.silu(n@p.gate.t())*(n@p.up.t()))@p.down.t())
    return FullColumns(rows, strength, value)

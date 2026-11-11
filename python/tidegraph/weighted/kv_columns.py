"""KV §3–4: B×L column queries, explicit snapshots and selected Full tensors.

Position-free banks use the existing bank-sized Attention transpose.
No scalar Read/Full callback, event record or reference recurrence is invoked.
"""
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from .column_layout import SequenceLayout, gather_column
from .kv_columns_state import KVWriteBank
from .dense_describe import columns as describe
from .block_attention import attention
from .tensors import norm
from .numeric_time import passive_decay


def supported(memory):
    from .kv import KVMemory
    if type(memory) is not KVMemory:
        return False
    c = memory.config
    return (type(memory) is KVMemory and not c.read_frequencies
            and not any(c.full_frequencies) and not any(c.distance_slopes)
            and memory.support_policy == "strict"
            and memory.base.dtype in {torch.float32, torch.float64})


def _queries(window, bank, rows, *, before=False, horizon=0):
    # Selection changes actual query rows, not the bank or its normalization.
    # The one bool/index transfer is host scheduling metadata, never a numerical
    # CPU prepass. No B×L differentiable views are created here.
    rows = rows.detach().cpu()
    layout = window.layout.select(rows)
    device = window.content.device
    qids, reverse = layout.padded(device)
    kids, _ = bank.layout.padded(device)
    selected = rows.to(device)
    u = gather_column(window.content.index_select(0, selected), qids)
    mass = gather_column(window.mass.index_select(0, selected), qids)
    stamps = gather_column(bank.layout.times.to(device), kids)
    times = gather_column(layout.times.to(device), qids)
    age = times[:, :, None]-stamps[:, None]
    visible = (age > 0 if before else age >= 0) & (kids[:, None] < len(bank.layout.times))
    if horizon:
        visible &= age < horizon
    visible &= qids[:, :, None] < len(rows)
    weight = torch.where(visible, gather_column(bank.weights, kids)[:, None], 0)
    raw = {name: gather_column(getattr(bank, name), kids) for name in ("keys", "values", "summaries")}
    return u, mass, raw, weight, age, reverse


def read_kv_columns(node, window, *, read_mode="+", initial=None):
    """Describe actual Read+/Read-/Read0 snapshots; Read0 requires bound state."""
    memory, c = node.kernel, node.kernel.config
    if not supported(memory):
        raise ValueError("KV query columns require position-free FP32/FP64 strict memory")
    if read_mode not in {"+", "-", "0"}:
        raise ValueError("unknown KV Read snapshot")
    if read_mode == "0":
        if initial is None:
            raise ValueError("Read0 requires its explicitly bound initial state")
        initial.validate(c)
        if initial.layout.samples != window.layout.samples:
            raise ValueError("Read0 initial sample domain mismatch")
        bank = KVWriteBank.from_state(initial)
    else:
        bank = window.bank
    if not len(window.layout.times):
        raise ValueError("Read column group must contain a real query")
    u, mass, raw, weight, age, reverse = _queries(window, bank, torch.arange(len(window.layout.times)), before=read_mode == "-", horizon=c.window)
    short = weight*passive_decay(age.clamp_min(0), c.beta_short, u)
    totals = short.sum(-1), weight.sum(-1)
    if c.memory_read == "sum":
        values = tuple((w@raw["summaries"])/t.clamp_min(1)[..., None] for w, t in zip((short, weight), totals))
    elif (weight > 0).any():
        q = (norm(u, c.epsilon)@memory.read_q.t())[:, None]
        k, v = (raw["keys"]@memory.read_k.t())[:, None], (raw["values"]@memory.read_v.t())[:, None]
        values = tuple(attention(q, k, v, w)[:, 0]*t.clamp_max(1)[..., None] for w, t in zip((short, weight), totals))
    else:
        values = (u.new_zeros((*u.shape[:2], c.summary_width)),)*2
    fields = describe(node.read_program, memory, mass, u, (*values, *totals))
    return {name: value.flatten(0, 1).index_select(0, reverse) for name, value in fields.items()}


@dataclass(frozen=True)
class FullColumns:
    rows: torch.Tensor
    strength: torch.Tensor
    content: torch.Tensor


def full_kv_columns(node, window, controls):
    """Actual positive queries; payload is (c,G), never an already weighted cG."""
    memory, c, module = node.kernel, node.kernel.config, node.full_program.content
    if not supported(memory):
        raise ValueError("KV Full columns require position-free FP32/FP64 strict memory")
    if controls.shape != window.mass.shape or (controls.dtype, controls.device) != (window.mass.dtype, window.mass.device):
        raise ValueError("KV controls must match the window rows/dtype/device")
    if not torch.isfinite(controls).all() or (controls < 0).any():
        raise ValueError("KV Full controls must be finite and nonnegative")
    rows = (controls.detach() > 0).nonzero().flatten().cpu()
    strength = controls.index_select(0, rows.to(controls.device))
    if not len(rows):
        return FullColumns(rows, strength, window.content.new_empty((0, c.width)))
    u, _, raw, weight, _, reverse = _queries(window, window.bank, rows, horizon=c.window)
    b, length, _ = u.shape
    if (weight > 0).any():
        q = (norm(u, c.epsilon)@module.query.t()).reshape(b, length, c.heads, -1).transpose(1, 2)
        k, v = (raw[name].reshape(b, -1, c.heads, c.width//c.heads).transpose(1, 2) for name in ("keys", "values"))
        mixed = attention(q, k, v, weight).transpose(1, 2).reshape(b, length, c.width)
    else:
        mixed = torch.zeros_like(u)
    x = u+c.alpha_att*((mixed*weight.sum(-1).clamp_max(1)[..., None])@module.out.t())
    normal = norm(x, c.epsilon)
    value = x+c.alpha_ff*((F.silu(normal@module.gate.t())*(normal@module.up.t()))@module.down.t())
    return FullColumns(rows, strength, value.flatten(0, 1).index_select(0, reverse))

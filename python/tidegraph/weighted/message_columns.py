"""Architecture §1–2: actual message columns and physical-atom aggregation.

Rows identify existing messages, including zero-strength input messages. Source
classes select parameters; repeated edges/ports remain repeated atoms. The host
uses only int64 support metadata. No differentiable row views cross a module.
"""
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from .tensors import norm
from .mean import weighted_mean


@dataclass(frozen=True)
class MessageColumns:
    rows: torch.Tensor
    strength: torch.Tensor
    content: torch.Tensor

    def validate(self, domain, width, like):
        domain.select(self.rows)
        if self.content.shape != (len(self.rows), width) or self.strength.shape != (len(self.rows),):
            raise ValueError("message column dimensions differ from their binding")
        for x in (self.strength, self.content):
            if (x.dtype, x.device) != (like.dtype, like.device) or not torch.isfinite(x).all():
                raise ValueError("message columns require finite matching payload tensors")


def aggregate_columns(program, sources, *, cache=None):
    """Union of actual fibers → (global row IDs, received mass, mean content).

    A source is (MessageColumns, SrcClass). No numerical-zero test removes an
    atom. An all-zero fiber has a constant zero mean, with its mass roots kept.
    """
    sources = [(p, c) for p, c in sources if len(p.rows)]
    if not sources:
        return None
    signature = (program.alpha, program.epsilon, tuple((id(p),
        tuple(id(x) for x in program.transforms[str(c)].parameters()) if program.alpha else ())
        for p, c in sources))
    if cache is not None and signature in cache:
        return cache[signature]
    rows = torch.cat([p.rows for p, _ in sources]).unique(sorted=True)
    masses, values = [], []
    for payload, label in sources:
        z = payload.content
        if program.alpha:
            transform = program.transforms[str(label)]
            a, b = transform.matrices()
            z = z+program.alpha*(F.silu(norm(z, program.epsilon)@a.t()+transform.embedding)@b.t())
        ids = torch.searchsorted(rows, payload.rows).to(z.device)
        masses.append(z.new_zeros(len(rows)).index_copy(0, ids, payload.strength.clamp_min(0)))
        values.append(z.new_zeros((len(rows), z.shape[-1])).index_copy(0, ids, z))
    mass, value = torch.stack(masses, 1), torch.stack(values, 1)
    total = mass.sum(1)
    # Group the positive and zero branches once. A zero fiber must not create
    # an autograd dependency on its content or source-transform parameters.
    positive = (total.detach() > 0).nonzero().flatten()
    if len(positive):
        u = weighted_mean(mass.index_select(0, positive), value.index_select(0, positive))
        u = value.new_zeros((len(rows), value.shape[-1])).index_copy(0, positive, u)
    else:
        u = value.new_zeros((len(rows), value.shape[-1]))
    result = rows, total, u
    if cache is not None:
        cache[signature] = result
    return result


def readout_columns(outputs, domain, like):
    """§4.4 numerical projection; output existence remains in MessageColumns."""
    mass = like.new_zeros(len(domain.times))
    value = like.new_zeros((len(domain.times), like.shape[-1]))
    for payload in outputs.values():
        rows = payload.rows.to(like.device)
        strength = payload.strength.clamp_min(0)
        mass = mass.index_add(0, rows, strength)
        value = value.index_add(0, rows, strength[:, None]*payload.content)
    return value/mass.clamp_min(1)[:, None]

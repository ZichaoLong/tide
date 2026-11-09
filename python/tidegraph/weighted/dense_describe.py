"""Architecture §3, cat/ind/context/cat: actual shared-parameter GEMMs."""
import torch
from .packing import project
from .tensors import norm


def evaluate(program, memory, mass, u, values):
    short, long, ws, wl = values
    cur = norm(u, memory.config.epsilon)
    current = project(cur, memory.summary)
    stat = torch.stack((mass.log1p(), ws.log1p(), wl.log1p()), -1)
    features = torch.cat((cur, stat[:, :1], short, long, stat[:, 1:], current*short, current*long), -1)
    head = program.score
    score = head.bias+(torch.nn.functional.silu(project(features, head.matrix)+head.hidden_bias)*head.vector).sum(-1)
    evidence = norm(project(features, program.evidence_weight)+program.evidence_bias, memory.config.epsilon)
    recv = mass.clamp(max=1)
    fields = dict(w=recv, mu=(program.config.gamma_max*recv).clamp(max=1), score=score, evidence=evidence)
    return [dict(zip(fields, row)) for row in zip(*(value.unbind() for value in fields.values()))]

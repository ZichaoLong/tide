"""Allocation §3: staged Describe projections across compatible private nodes."""
import math
import torch
from .column_query_ops import linear
from .tensors import norm


def head(modules, values):
    hidden = linear(values, [m.matrix for m in modules])
    return [m.bias+(torch.nn.functional.silu(h+m.hidden_bias)*m.vector).sum(-1)
            for m, h in zip(modules, hidden)]


def encode(modules, values):
    hidden = [torch.nn.functional.silu(x+m.hidden_bias)
              for x, m in zip(linear(values, [m.in_weight for m in modules]), modules)]
    return [norm(x+m.out_bias, m.epsilon)
            for x, m in zip(linear(hidden, [m.out_weight for m in modules]), modules)]


def encoded(modules, groups):
    values = [encode([m.encoders[name] for m in modules], [g[name] for g in groups])
              for name in ("cur", "mem", "stat")]
    return list(zip(*values))


def columns(nodes, masses, inputs, memories):
    """Same configuration/row shapes, arbitrary parameter identities and values."""
    programs = [n.read_program for n in nodes]
    c = programs[0].config
    cur = [norm(x, n.kernel.config.epsilon) for x, n in zip(inputs, nodes)]
    current = linear(cur, [n.kernel.summary for n in nodes])
    groups, features, result = [], [], []
    for n, mass, u, s, values in zip(nodes, masses, cur, current, memories):
        short, long, ws, wl = values
        stat = torch.stack((mass.log1p(), ws.log1p(), wl.log1p()), -1)
        groups.append(dict(cur=u, mem=torch.cat((short, long), -1), stat=stat))
        features.append(torch.cat((u, stat[..., :1], short, long, stat[..., 1:], s*short, s*long), -1))
        result.append(dict(w=mass.clamp_max(1)))
    if c.regional:
        for r in result:
            r["mu"] = (c.gamma_max*r["w"]).clamp_max(1)
    else:
        gains = head([p.gain for p in programs], features) if c.send_gain == "context" else None
        for i, (p, r, mass) in enumerate(zip(programs, result, masses)):
            gamma = (1+(c.gamma_max-1)*gains[i].sigmoid() if gains is not None else
                     1+(c.gamma_max-1)*p.gain_logit.sigmoid() if c.gain_logit is not None else c.fixed_gain)
            r["mu"] = (gamma*mass).clamp_max(1)
    if c.local_score == "cat":
        scores = head([p.score for p in programs], features)
    else:
        modules = [p.score for p in programs]
        enc = encoded(modules, groups)
        joined = [torch.cat(row, -1) for row in enc]
        gate, up = linear(joined, [m.gate for m in modules]), linear(joined, [m.up for m in modules])
        scores = [m.bias+c.alpha_match*(row[0]*row[1]).sum(-1)/math.sqrt(c.selection_width)
                  +c.alpha_res*(torch.nn.functional.silu(g+m.gate_bias)*(u+m.up_bias)*m.vector).sum(-1)
                  for m, row, g, u in zip(modules, enc, gate, up)]
    for r, score in zip(result, scores):
        r["score"] = score
    if c.regional:
        if c.region_control_mode == "fixed":
            evidence = [f.new_zeros((*f.shape[:-1], c.region_control_width)) for f in features]
        elif c.region_evidence == "cat":
            evidence = [norm(x+p.evidence_bias, n.kernel.config.epsilon) for x, p, n in
                        zip(linear(features, [p.evidence_weight for p in programs]), programs, nodes)]
        else:
            modules = [p.evidence for p in programs]
            enc = encoded(modules, groups)
            maps = [linear([row[i] for row in enc], [getattr(m, name) for m in modules])
                    for i, name in enumerate(("cur", "mem", "stat"))]
            cross = linear([row[0]*row[1] for row in enc], [m.interaction for m in modules])
            evidence = [norm(a+b+d+x+m.bias, m.epsilon) for a, b, d, x, m in zip(*maps, cross, modules)]
        for r, value in zip(result, evidence):
            r["evidence"] = value
    if c.region_score == "att":
        xs = linear(cur, [p.current_weight for p in programs])
        if c.region_key == "content":
            keys = encode([p.key for p in programs], [g["mem"] for g in groups])
        else:
            key_inputs = [torch.cat((g["mem"], g["stat"][..., 1:]), -1) for g in groups]
            keys = [x+p.key_bias for x, p in zip(linear(key_inputs, [p.key_weight for p in programs]), programs)]
        for r, x, key, p in zip(result, xs, keys, programs):
            r.update(x=x+p.current_bias, key=key)
    return result

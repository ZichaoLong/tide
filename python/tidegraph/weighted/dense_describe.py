"""Architecture §3 / allocation §3: built-in Describe over leading row axes.

This numerical candidate keeps scalar Describe independent. The column map
returns every declared field; publication alone splits actual rows once.
"""
import math
import torch
from .config import RegionalConfig
from .packing import project
from .tensors import norm


def supported(config):
    """Descriptor algebra capability; independent of the memory/query family."""
    # RegionalConfig validates applicability at construction. All its finite
    # heads have a column implementation, regardless of the sending solver.
    return isinstance(config, RegionalConfig)


def linear(value, weight):
    """One shared matrix over B, L or flattened ready-row axes."""
    if value.ndim == 2:
        return project(value, weight)
    return project(value.reshape(-1, value.shape[-1]), weight).reshape(
        *value.shape[:-1], weight.shape[0])


def head(module, value):
    return module.bias+(torch.nn.functional.silu(linear(value, module.matrix)+module.hidden_bias)*module.vector).sum(-1)


def encode(module, value):
    hidden = torch.nn.functional.silu(linear(value, module.in_weight)+module.hidden_bias)
    return norm(linear(hidden, module.out_weight)+module.out_bias, module.epsilon)


def encoded(module, groups):
    # Parameter aliases alone do not justify merging inputs or gradient roots.
    return tuple(encode(module.encoders[name], groups[name]) for name in ("cur", "mem", "stat"))


def local_score(program, features, groups):
    score, c = program.score, program.config
    if c.local_score == "cat":
        return head(score, features)
    cur, mem, stat = encoded(score, groups)
    joined = torch.cat((cur, mem, stat), -1)
    gate = torch.nn.functional.silu(linear(joined, score.gate)+score.gate_bias)
    up = linear(joined, score.up)+score.up_bias
    return (score.bias+c.alpha_match*(cur*mem).sum(-1)/math.sqrt(c.selection_width)
            +c.alpha_res*(gate*up*score.vector).sum(-1))


def evidence(program, features, groups, epsilon):
    c = program.config
    if c.region_control_mode == "fixed":
        return features.new_zeros((*features.shape[:-1], c.region_control_width))
    if c.region_evidence == "cat":
        return norm(linear(features, program.evidence_weight)+program.evidence_bias, epsilon)
    module = program.evidence
    cur, mem, stat = encoded(module, groups)
    return norm(linear(cur, module.cur)+linear(mem, module.mem)+linear(stat, module.stat)
                +linear(cur*mem, module.interaction)+module.bias, module.epsilon)


def memory_key(program, groups):
    if program.config.region_key == "content":
        return encode(program.key, groups["mem"])
    # The received statistic belongs to score/evidence, never this key input.
    value = torch.cat((groups["mem"], groups["stat"][..., 1:]), -1)
    return linear(value, program.key_weight)+program.key_bias


def columns(program, memory, mass, u, values):
    """Same finite fields as scalar Describe, each shaped [*rows, *field]."""
    short, long, ws, wl = values
    cur = norm(u, memory.config.epsilon)
    current = linear(cur, memory.summary)
    stat = torch.stack((mass.log1p(), ws.log1p(), wl.log1p()), -1)
    c = program.config
    groups = dict(cur=cur, stat=stat)
    if c.local_score == "match" or c.region_evidence == "grouped" or c.region_score == "att":
        groups["mem"] = torch.cat((short, long), -1)
    features = torch.cat((cur, stat[..., :1], short, long, stat[..., 1:], current*short, current*long), -1)
    recv = mass.clamp(max=1)
    if c.regional:
        mu = (c.gamma_max*recv).clamp(max=1)
    else:
        if c.send_gain == "context":
            gamma = 1+(c.gamma_max-1)*head(program.gain, features).sigmoid()
        elif c.gain_logit is not None:
            gamma = 1+(c.gamma_max-1)*program.gain_logit.sigmoid()
        else:
            gamma = c.fixed_gain
        mu = (gamma*mass).clamp(max=1)
    fields = dict(w=recv, mu=mu, score=local_score(program, features, groups))
    if c.regional:
        fields["evidence"] = evidence(program, features, groups, memory.config.epsilon)
    if c.region_score == "att":
        fields.update(x=linear(groups["cur"], program.current_weight)+program.current_bias,
                      key=memory_key(program, groups))
    return fields


def evaluate(program, memory, mass, u, values):
    fields = columns(program, memory, mass, u, values)
    return [dict(zip(fields, row)) for row in zip(*(value.unbind() for value in fields.values()))]

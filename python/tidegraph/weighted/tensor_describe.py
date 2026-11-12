"""Allocation §3: descriptor algebra on [node,B,Q,*field] tensors.

All finite heads/keys/evidence options use the same numerical boundary. The
parameter role axis is independent of the query axes, and never expanded to
one matrix per event. Scalar formulas remain independent correctness anchors.
"""
import math
import torch
from torch.nn import functional as F
from .tensors import norm
from .packing_observation import projection


class Algebra:
    def __init__(self, values, shared=()):
        self.values, self.shared = values, shared

    def bias(self, name, rank=4):
        value = self.values[name]
        if name in self.shared:
            return value
        return value.reshape(value.shape[0], *((1,)*(rank-value.ndim)), *value.shape[1:])

    def linear(self, x, name):
        weight = self.values[name]
        if name in self.shared:
            projection("shared_block_mm", x.shape[0])
            return F.linear(x, weight)
        projection("tensor_block_bmm", len(weight))
        return torch.bmm(x.flatten(1, -2), weight.transpose(1, 2)).reshape(*x.shape[:-1], weight.shape[1])

    def head(self, name, x):
        hidden = F.silu(self.linear(x, name+".matrix")+self.bias(name+".hidden_bias"))
        return self.bias(name+".bias", 3)+(hidden*self.bias(name+".vector")).sum(-1)

    def encode(self, name, x, epsilon):
        hidden = F.silu(self.linear(x, name+".in_weight")+self.bias(name+".hidden_bias"))
        return norm(self.linear(hidden, name+".out_weight")+self.bias(name+".out_bias"), epsilon)

    def encoded(self, name, groups, epsilon):
        return tuple(self.encode(name+".encoders."+key, groups[key], epsilon) for key in ("cur", "mem", "stat"))


def fields(config):
    return ("w", "mu", "score")+(("evidence",) if config.regional else ())+(("x", "key") if config.region_score == "att" else ())


def describe(a, c, epsilon, u, mass, memories):
    short, long, ws, wl = memories
    cur = norm(u, epsilon)
    current = a.linear(cur, "kernel.summary")
    stat = torch.stack((mass.log1p(), ws.log1p(), wl.log1p()), -1)
    groups = dict(cur=cur, mem=torch.cat((short, long), -1), stat=stat)
    features = torch.cat((cur, stat[..., :1], short, long, stat[..., 1:], current*short, current*long), -1)
    recv = mass.clamp_max(1)
    if c.regional:
        mu = (c.gamma_max*recv).clamp_max(1)
    else:
        if c.send_gain == "context":
            gamma = 1+(c.gamma_max-1)*a.head("read.gain", features).sigmoid()
        elif c.gain_logit is not None:
            gamma = 1+(c.gamma_max-1)*a.bias("read.gain_logit", 3).sigmoid()
        else:
            gamma = c.fixed_gain
        mu = (gamma*mass).clamp_max(1)
    if c.local_score == "cat":
        score = a.head("read.score", features)
    else:
        ec, em, es = a.encoded("read.score", groups, epsilon)
        joined = torch.cat((ec, em, es), -1)
        gate = F.silu(a.linear(joined, "read.score.gate")+a.bias("read.score.gate_bias"))
        up = a.linear(joined, "read.score.up")+a.bias("read.score.up_bias")
        score = (a.bias("read.score.bias", 3)+c.alpha_match*(ec*em).sum(-1)/math.sqrt(c.selection_width)
                 +c.alpha_res*(gate*up*a.bias("read.score.vector")).sum(-1))
    result = dict(w=recv, mu=mu, score=score)
    if c.regional:
        if c.region_control_mode == "fixed":
            evidence = features.new_zeros((*features.shape[:-1], c.region_control_width))
        elif c.region_evidence == "cat":
            evidence = norm(a.linear(features, "read.evidence_weight")+a.bias("read.evidence_bias"), epsilon)
        else:
            ec, em, es = a.encoded("read.evidence", groups, epsilon)
            evidence = norm(a.linear(ec, "read.evidence.cur")+a.linear(em, "read.evidence.mem")
                +a.linear(es, "read.evidence.stat")+a.linear(ec*em, "read.evidence.interaction")
                +a.bias("read.evidence.bias"), epsilon)
        result["evidence"] = evidence
    if c.region_score == "att":
        x = a.linear(cur, "read.current_weight")+a.bias("read.current_bias")
        if c.region_key == "content":
            key = a.encode("read.key", groups["mem"], epsilon)
        else:
            key = a.linear(torch.cat((groups["mem"], stat[..., 1:]), -1), "read.key_weight")+a.bias("read.key_bias")
        result.update(x=x, key=key)
    return result

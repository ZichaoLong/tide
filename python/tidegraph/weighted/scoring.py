"""Pinned allocation §3.1–3.6: parameterized scoring/evidence/key/query maps.

Each module has its own parameters unless callers explicitly bind shared
Parameters. No state or scheduling enters these finite-dimensional functions.
"""
import math
import torch
from .tensors import norm, parameter


class ScalarHead(torch.nn.Module):
    def __init__(self, width, hidden, generator, dtype):
        super().__init__()
        self.matrix = parameter((hidden,width),generator,dtype)
        self.hidden_bias = parameter((hidden,),generator,dtype)
        self.vector = parameter((hidden,),generator,dtype)
        self.bias = parameter((),generator,dtype)

    def forward(self, value):
        return self.bias+(torch.nn.functional.silu(value@self.matrix.t()+self.hidden_bias)*self.vector).sum(-1)


class Encoder(torch.nn.Module):
    """Equation (2): normalized C SiLU(B z+b)+a, with arbitrary leading axes."""
    def __init__(self, width, hidden, output, epsilon, generator, dtype):
        super().__init__()
        self.in_weight = parameter((hidden,width),generator,dtype)
        self.hidden_bias = parameter((hidden,),generator,dtype)
        self.out_weight = parameter((output,hidden),generator,dtype)
        self.out_bias = parameter((output,),generator,dtype)
        self.epsilon = epsilon

    def forward(self, value):
        # Match the column-vector formula/native reduction for scalar calls;
        # retain leading batch axes without changing the mathematical map.
        if value.ndim == 1:
            hidden = torch.nn.functional.silu(self.in_weight@value+self.hidden_bias)
            return norm(self.out_weight@hidden+self.out_bias,self.epsilon)
        hidden = torch.nn.functional.silu(value@self.in_weight.t()+self.hidden_bias)
        return norm(hidden@self.out_weight.t()+self.out_bias,self.epsilon)


def encoders(memory, region, generator, dtype):
    return torch.nn.ModuleDict({name: Encoder(width,region.hidden_width,region.selection_width,
                                memory.epsilon,generator,dtype)
                                for name,width in (("cur",memory.width),("mem",2*memory.summary_width),("stat",3))})


class MatchScore(torch.nn.Module):
    """Equation (3): normalized content/memory match plus gated residual."""
    def __init__(self, memory, region, generator, dtype):
        super().__init__()
        self.encoders = encoders(memory,region,generator,dtype)
        self.gate = parameter((region.hidden_width,3*region.selection_width),generator,dtype)
        self.gate_bias = parameter((region.hidden_width,),generator,dtype)
        self.up = parameter((region.hidden_width,3*region.selection_width),generator,dtype)
        self.up_bias = parameter((region.hidden_width,),generator,dtype)
        self.vector = parameter((region.hidden_width,),generator,dtype)
        self.bias = parameter((),generator,dtype)
        self.config = region

    def forward(self, groups):
        cur,mem,stat = (self.encoders[name](groups[name]) for name in ("cur","mem","stat"))
        f = torch.cat((cur,mem,stat),-1)
        residual = (torch.nn.functional.silu(f@self.gate.t()+self.gate_bias)*(f@self.up.t()+self.up_bias)*self.vector).sum(-1)
        return (self.bias+self.config.alpha_match*(cur*mem).sum(-1)/math.sqrt(self.config.selection_width)
                +self.config.alpha_res*residual)


class GroupedEvidence(torch.nn.Module):
    """§3.4.1: three independent encoders and four combination matrices."""
    def __init__(self, memory, region, generator, dtype):
        super().__init__()
        self.encoders = encoders(memory,region,generator,dtype)
        for name in ("cur","mem","stat","interaction"):
            setattr(self,name,parameter((region.region_control_width,region.selection_width),generator,dtype))
        self.bias = parameter((region.region_control_width,),generator,dtype)
        self.epsilon = memory.epsilon

    def forward(self, groups):
        cur,mem,stat = (self.encoders[name](groups[name]) for name in ("cur","mem","stat"))
        return norm(cur@self.cur.t()+mem@self.mem.t()+stat@self.stat.t()
                    +(cur*mem)@self.interaction.t()+self.bias,self.epsilon)


class Query(torch.nn.Module):
    """Equation (5)'s unnormalized two-layer region query."""
    def __init__(self, width, generator, dtype):
        super().__init__()
        self.in_weight = parameter((width,width),generator,dtype)
        self.hidden_bias = parameter((width,),generator,dtype)
        self.out_weight = parameter((width,width),generator,dtype)
        self.out_bias = parameter((width,),generator,dtype)

    def forward(self, x):
        if x.ndim == 1:
            hidden = torch.nn.functional.silu(self.in_weight@x+self.hidden_bias)
            return self.out_weight@hidden+self.out_bias
        return torch.nn.functional.silu(x@self.in_weight.t()+self.hidden_bias)@self.out_weight.t()+self.out_bias

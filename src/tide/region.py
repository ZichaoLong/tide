"""Encoded node descriptions, regional attention and role allocation."""
import torch
from torch import nn

from .attention import weighted_attention
from .allocation import allocate
from .parameters import NodeLinear, NodeVector, norm


class Description(nn.Module):
    def __init__(self,nodes,d,groups=None,rank=8):
        super().__init__()
        self.encode = NodeLinear(nodes,2*d+3,d//2,groups,rank,bias=True)
        self.readout = NodeVector(nodes,d//2,groups)
        nn.init.normal_(self.readout.value,std=1/(d//2)**0.5)
        self.offset = NodeVector(nodes,1)

    def forward(self,u,received,memory,owners):
        p,ms,ml,ws,wl = memory
        features = torch.cat((norm(u),torch.log1p(received)[...,None],ms,ml,
                              torch.log1p(ws)[...,None],torch.log1p(wl)[...,None],
                              p*ms,p*ml),-1)
        z = norm(torch.nn.functional.silu(self.encode(features,owners)))
        score = (z*self.readout(owners)[:,None,None]).sum(-1)+self.offset(owners)[:,None,None,0]
        return z,score


def all_real_inputs_constant_zero(incoming_face, membership):
    """A node's received mass is fixed at zero if every real edge proves it."""
    if incoming_face is None:
        return None
    if membership is not None:
        incoming_face = incoming_face | ~membership[:, None, None, :]
    return incoming_face.all(-1)


class RegionNetwork(nn.Module):
    def __init__(self,d,backend="auto"):
        super().__init__()
        self.heads, self.backend = d//64, backend
        width = d//2
        self.q = nn.Linear(width,width,bias=False)
        self.k = nn.Linear(width,width,bias=False)
        self.v = nn.Linear(width,width,bias=False)
        self.o = nn.Linear(width,width,bias=False)
        self.first = nn.Linear(width,d)
        self.second = nn.Linear(d,width)
        self.readout = nn.Parameter(torch.empty(width))
        nn.init.normal_(self.readout,std=1/width**0.5)

    def forward(self,z,score,w,constant_zero=None):
        # Node becomes attention sequence; B and logical time remain batch axes.
        x = z.permute(1,2,0,3)
        shape = x.shape[:-1]
        def project(layer):
            return layer(x).reshape(*shape,self.heads,32).transpose(-3,-2)
        read = weighted_attention(project(self.q),project(self.k),project(self.v),
                                  weights=w.movedim(0,-1),backend=self.backend,
                                  _constant_zero_prior=None if constant_zero is None else constant_zero.movedim(0,-1))
        delta = self.o(read.transpose(-3,-2).flatten(-2))
        hidden = torch.nn.functional.silu(self.first(norm(x+delta)))
        # The final vector has only a scalar consumer: contract W2 with readout.
        # Regional scores decide exact support. Keep these scalar contractions
        # in the parameter dtype instead of rounding the whole score to BF16.
        # Projection/FFN matrices above still execute through BF16 autocast.
        with torch.autocast(x.device.type, enabled=False):
            correction = delta.to(self.readout.dtype) @ self.readout
            correction = correction + hidden.to(self.readout.dtype) @ (self.second.weight.T @ self.readout)
            correction = correction + self.second.bias @ self.readout
        return score + torch.where(w>0,correction.permute(2,0,1),torch.zeros_like(score))


def region_control(score, received, roles, budget, profile, partition, support_policy="strict",
                   *, return_constant_zero=False):
    if score.dtype in (torch.float16,torch.bfloat16):
        score = score.float()
    received = received.to(score.dtype)
    w = torch.where(received>=1,torch.ones_like(received),received.clamp_min(0))
    mu = torch.where(received>=0.5,torch.ones_like(received),2*received.clamp_min(0))
    hubs = roles[:,None,None]
    n_h = int(roles.sum())
    if n_h == len(roles) and partition == "singleton":
        return (mu, None) if return_constant_zero else mu
    if n_h == len(roles):
        k,kind = max(1,(n_h+1)//2),"quad"
    elif n_h:
        k,kind = budget+max(1,(n_h+1)//2),profile
    else:
        k,kind = budget,profile
    lower = torch.where(hubs,mu/4,torch.zeros_like(mu))
    options = {"support_policy": support_policy} if kind in ("shannon", "pool") and support_policy != "strict" else {}
    keep_face = return_constant_zero and kind == "quad" and n_h < len(roles)
    if keep_face:
        options["return_lower_face"] = True
    out = allocate(score.permute(1,2,0),w.permute(1,2,0),mu.permute(1,2,0),
                   lower.permute(1,2,0),mu.permute(1,2,0),k,kind=kind,**options)
    if keep_face:
        out, lower_face = out
    control = out.permute(2,0,1)
    if not return_constant_zero:
        return control
    # For quad the solver chooses the lower face at equality. On an ordinary
    # node its lower bound is the constant zero. Exclude fixed-through mu=0:
    # its declared derivative with respect to received mass can be nonzero.
    constant_zero = None
    if keep_face:
        constant_zero = ~hubs & lower_face.permute(2,0,1)
    return control, constant_zero

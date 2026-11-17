"""Messages are processed jointly in tensors [node,batch,time,edge,feature]."""
import torch
from torch import nn

from .attention import weighted_attention
from .parameters import NodeLinear, norm


class SourceEmbedding(nn.Module):
    def __init__(self, sources, width, groups=None):
        super().__init__()
        labels = sorted(set().union(*map(set, sources)))
        self.labels = {name:i for i,name in enumerate(labels)}
        groups = list(range(len(sources))) if groups is None else groups
        slots = {}
        lookup = torch.full((len(sources),len(labels)), -1, dtype=torch.long)
        for node, classes in enumerate(sources):
            for name in sorted(classes):
                key = (int(groups[node]), name)
                if key not in slots:
                    slots[key] = len(slots)
                lookup[node,self.labels[name]] = slots[key]
        self.register_buffer("lookup", lookup)
        self.weight = nn.Parameter(torch.zeros(len(slots),width))

    def forward(self, owners, classes):
        indices = self.lookup[owners[:,None], classes]
        return self.weight[indices]


class Reception(nn.Module):
    def __init__(self, sources, d, profile, groups=None, rank=8, backend="auto"):
        super().__init__()
        n = len(sources)
        self.profile, self.backend = profile, backend
        hidden = d if profile == "incoming_wide" else d//8
        self.a = NodeLinear(n,d,hidden,groups,rank)
        self.b = NodeLinear(n,hidden,d,groups,rank)
        self.embedding = SourceEmbedding(sources,hidden,groups)
        if profile != "source_residual":
            width = d if profile == "incoming_wide" else int(profile.rsplit("_",1)[1])
            self.heads = width//32
            self.sa_embedding = SourceEmbedding(sources,d,groups)
            self.q = NodeLinear(n,d,width,groups,rank)
            self.k = NodeLinear(n,d,width,groups,rank)
            self.v = NodeLinear(n,d,width,groups,rank)
            self.o = NodeLinear(n,width,d,groups,rank)

    def forward(self, content, mass, classes, owners):
        if mass.dtype in (torch.float16,torch.bfloat16):
            mass = mass.float()
        mass = mass.clamp_min(0)
        total = mass.sum(-1)
        y = content
        if self.profile != "source_residual":
            x = norm(content) + self.sa_embedding(owners,classes)[:,None,None]
            shape = x.shape[:-1]
            def heads(proj):
                return proj(x,owners).reshape(*shape,self.heads,32).transpose(-3,-2)
            q,k,v = heads(self.q),heads(self.k),heads(self.v)
            read = weighted_attention(q,k,v,weights=mass,backend=self.backend)
            read = read.transpose(-3,-2).flatten(-2)
            y = y + self.o(read,owners)
        hidden = self.a(norm(y),owners) + self.embedding(owners,classes)[:,None,None]
        value = y + self.b(torch.nn.functional.silu(hidden),owners)
        den = torch.where(total>0,total,torch.ones_like(total))
        aggregate = (value*mass[...,None]).sum(-2)/den[...,None]
        return total, aggregate

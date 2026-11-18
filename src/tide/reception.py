"""Messages are processed jointly in tensors [node,batch,time,edge,feature]."""
import torch
from torch import nn

from .attention import weighted_attention
from .parameters import NodeLinear, fused_node_linears, norm, owner_ids


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
        indices = self.lookup[owner_ids(owners)[:,None], classes]
        return self.weight[indices]


class Reception(nn.Module):
    def __init__(self, sources, d, profile, groups=None, rank=8, backend="auto", *, qkv_backend="separate"):
        super().__init__()
        if qkv_backend not in ("separate", "fused"):
            raise ValueError("reception QKV backend must be separate or fused")
        n = len(sources)
        self.profile, self.backend = profile, backend
        self.qkv_backend = qkv_backend
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

    @property
    def effective_qkv_backend(self):
        """Report the selected projection path, including placed fallbacks.

        Sharded owner wrappers retain their own device/parameter routing. They
        use their ordinary projections rather than bypassing that ownership.
        """
        if self.profile == "source_residual":
            return "none"
        if self.qkv_backend == "fused" and not all(isinstance(p, NodeLinear) for p in (self.q, self.k, self.v)):
            return "separate_owner_wrapper"
        return self.qkv_backend

    def forward(self, content, mass, classes, owners, edge_mask=None):
        """Aggregate real edges, preserving derivatives at genuine zero mass.

        ``edge_mask`` is static structural membership [node, edge], never a
        test of positive mass. A masked slot has no mathematical mass root;
        an existing edge with zero mass keeps its declared right derivative.
        """
        if edge_mask is not None and (edge_mask.dtype != torch.bool or
                edge_mask.shape != (content.shape[0], content.shape[-2])):
            raise ValueError("reception edge_mask must be boolean [node, edge]")
        if mass.dtype in (torch.float16,torch.bfloat16):
            mass = mass.float()
        if edge_mask is not None:
            mass = mass.masked_fill(~edge_mask[:, None, None], 0)
        mass = mass.clamp_min(0)
        total = mass.sum(-1)
        y = content
        if self.profile != "source_residual":
            x = norm(content) + self.sa_embedding(owners,classes)[:,None,None]
            shape = x.shape[:-1]
            if self.effective_qkv_backend == "fused":
                projections = fused_node_linears((self.q,self.k,self.v), x, owners).chunk(3, -1)
            else:
                projections = (projection(x,owners) for projection in (self.q,self.k,self.v))
            q,k,v = (value.reshape(*shape,self.heads,32).transpose(-3,-2) for value in projections)
            key_mask = None if edge_mask is None else edge_mask[:,None,None,None,None,:]
            read = weighted_attention(q,k,v,weights=mass,mask=key_mask,backend=self.backend)
            read = read.transpose(-3,-2).flatten(-2)
            y = y + self.o(read,owners)
        hidden = self.a(norm(y),owners) + self.embedding(owners,classes)[:,None,None]
        value = y + self.b(torch.nn.functional.silu(hidden),owners)
        den = torch.where(total>0,total,torch.ones_like(total))
        aggregate = (value*mass[...,None]).sum(-2)/den[...,None]
        return total, aggregate

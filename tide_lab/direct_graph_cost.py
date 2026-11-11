"""Non-TIDE dense DAG cost anchors, built only from ordinary Torch operations.

These are different functions: uniform incoming means, pre-RMS causal attention
and SwiGLU, with independent node weights. No weighted payloads, learned sending,
Read/Describe, general event scheduler, independent intermediate roots, or public
continuation contract. The exported static adjacency is the only foundation input.
This module does not import tidegraph and is not a replacement execution core.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class DenseRegion(nn.Module):
    def __init__(self, nodes, width, heads=2, ffn=None, attention=True,
                 initialization='fan_in', residual_scale=1.):
        super().__init__()
        self.nodes, self.width, self.heads = nodes, width, heads
        self.attention = attention
        if initialization not in ('fan_in','std015') or not 0 < residual_scale <= 1:
            raise ValueError('unknown direct initialization or residual scale')
        self.residual_scale = residual_scale
        if width % heads:
            raise ValueError('width must divide into heads')
        ffn = ffn or 4*width
        shapes = dict(up=(width, ffn), gate=(width, ffn), down=(ffn, width))
        if attention:
            shapes.update(q=(width, width), k=(width, width), v=(width, width), o=(width, width))
        self.weights = nn.ParameterDict({name: nn.Parameter(torch.empty(nodes, a, b))
                                        for name, (a, b) in shapes.items()})
        for w in self.weights.values():
            nn.init.normal_(w, std=w.shape[-2]**-.5 if initialization=='fan_in' else .15)
        self.norm_ffn = nn.Parameter(torch.ones(nodes, width))
        if attention:
            self.norm_att = nn.Parameter(torch.ones(nodes, width))

    @staticmethod
    def norm(x, scale):
        return x*torch.rsqrt(x.square().mean(-1, keepdim=True)+1e-6)*scale[:, None, None, :]

    @staticmethod
    def project(x, weight):
        # Independent-node parameter batch; samples and time remain matrix rows.
        n, b, l, d = x.shape
        return torch.bmm(x.reshape(n, b*l, d), weight).reshape(n, b, l, -1)

    def forward(self, x):
        # [nodes, independent samples, logical positions, width]
        n, b, l, d = x.shape
        if n != self.nodes or d != self.width:
            raise ValueError('wrong region shape')
        if self.attention:
            z = self.norm(x, self.norm_att)
            q, k, v = [self.project(z, self.weights[key]).reshape(
                n, b, l, self.heads, d//self.heads).permute(0, 1, 3, 2, 4)
                for key in ('q', 'k', 'v')]
            scores = (q@k.transpose(-1, -2))/math.sqrt(d//self.heads)
            mask = torch.ones(l, l, dtype=torch.bool, device=x.device).triu(1)
            weights = scores.masked_fill(mask, float('-inf')).softmax(-1)
            context = (weights@v).permute(0, 1, 3, 2, 4).reshape(n, b, l, d)
            x = x+self.residual_scale*self.project(context, self.weights['o'])
        z = self.norm(x, self.norm_ffn)
        hidden = F.silu(self.project(z, self.weights['gate']))*self.project(z, self.weights['up'])
        return x+self.residual_scale*self.project(hidden, self.weights['down'])


class DirectGraphLM(nn.Module):
    def __init__(self, graph, width=128, seed=11, attention=True, vocab_size=4096,
                 initialization='fan_in', final_norm=False, residual_scale=1.):
        super().__init__()
        self.graph, self.width, self.attention = graph, width, attention
        self.inputs = set(graph['inputs'])
        self.parents = [[] for _ in range(graph['nodes'])]
        for src, dst, delay in graph['edges']:
            if graph['ranks'][graph['node_regions'][dst]]-graph['ranks'][graph['node_regions'][src]] != delay:
                raise ValueError('anchor only admits same-position ranked DAG edges')
            self.parents[dst].append(src)
        self.region_nodes = [[i for i, r in enumerate(graph['node_regions']) if r == region]
                             for region in range(graph['regions'])]
        self.order = sorted(range(graph['regions']), key=lambda r: (graph['ranks'][r], r))
        if any(set(self.parents[i]) & set(nodes) for nodes in self.region_nodes for i in nodes):
            raise ValueError('within-region dependencies are outside this anchor')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.embedding = nn.Embedding(vocab_size, width)
            nn.init.normal_(self.embedding.weight, std=.02)
            self.blocks = nn.ModuleList([DenseRegion(len(nodes), width, attention=attention,
                                         initialization=initialization,residual_scale=residual_scale)
                                         for nodes in self.region_nodes])
            self.final_norm = nn.RMSNorm(width,eps=1e-6) if final_norm else nn.Identity()
            self.head = nn.Linear(width, vocab_size)
            self.head.weight = self.embedding.weight
            nn.init.zeros_(self.head.bias)
        self.last_metrics = {}

    def forward(self, ids):
        features = self.embedding(ids)
        values = [None]*self.graph['nodes']
        for r in self.order:
            packed = []
            for node in self.region_nodes[r]:
                incoming = [values[src] for src in self.parents[node]]
                if node in self.inputs:
                    incoming.append(features)
                if not incoming or any(x is None for x in incoming):
                    raise ValueError('unready or disconnected direct graph node')
                packed.append(torch.stack(incoming).mean(0))
            out = self.blocks[r](torch.stack(packed))
            for node, value in zip(self.region_nodes[r], out.unbind(0)):
                values[node] = value
        hidden = torch.stack([values[i] for i in self.graph['outputs']]).mean(0)
        b, l = ids.shape
        self.last_metrics = {'work/dense_node_positions': b*l*self.graph['nodes'],
            'work/edge_positions': b*l*len(self.graph['edges']),
            'work/attention_query_positions': b*l*self.graph['nodes'] if self.attention else 0,
            'work/largest_projection_rows_per_node': b*l,
            'work/largest_projection_parameter_batch': max(map(len, self.region_nodes))}
        return self.head(self.final_norm(hidden))

import torch
from tidegraph import Graph, Node, Region
from tidegraph.ops import Model
from tidegraph.ports import PortLayout

POOLS = ("mean", "linear", "active-softmax", "all-softmax")


def profile(kind):
    return f"lh-fiber-attention-{kind}-repeat-v1"


def analytic(dtype, kind, reverse=False):
    layout = PortLayout((), (), (2, 0, 1), (0,)) if reverse else None
    g = Graph((Node(0, memory=profile(kind)),), (), (Region(1),), (0, 0, 0), (0,), layout)
    m = Model(g, width=1, dtype=dtype); w = m.nodes[0]
    with torch.no_grad():
        w.extra["fiber_qkv"].fill_(1); w.extra["fiber_out"].fill_(1)
        w.extra["fiber_out_bias"].fill_(.7)
        for p in m.input_scale:
            p.fill_(1)
        if "fiber_pool" in w.extra:
            mass = w.bias.new_tensor([2., 3., 5.])
            w.extra["fiber_pool"].copy_(mass if kind == "linear" else mass.log())
    return g, m, w

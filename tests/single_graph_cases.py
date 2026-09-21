"""Small recurrent fixtures with parallel wires, missing phases and nonunit scales."""
import torch
from tidegraph import Edge, External, Graph, Node, Region
from tidegraph.ops import Model
from tidegraph.ports import PortLayout

PROFILES = ("add", "sum", "mean", "linear", "active-softmax", "all-softmax")


def fixture(dtype, pool, clear=False):
    memory = "lh-add-repeat-v1" if pool == "add" else f"lh-fiber-attention-{pool}-repeat-v1"
    nodes = tuple(Node(r, memory=memory, full="lh-silu-rms-v1", emission="slot_affine",
                       readout="norm-fp64-v1", query_heads=2, kv_heads=2, clear=clear) for r in (0, 0, 1))
    body = Graph(nodes, (Edge(2, 0, 1), Edge(2, 1, 1), Edge(0, 2, 1), Edge(1, 2, 1), Edge(1, 2, 1)),
                 (Region(1, selector="lh-count-affect-v1"), Region(1)), (2, 0), (0,),
                 PortLayout((1, 0, 1, 1, 0), (1, 0, 2, 1, 0), (3, 0), (0,)))
    layers = 3 if clear else 2
    readout = Graph((Node(0, memory=memory, full="lh-identity-identity-v1", query_heads=2, kv_heads=2),),
                    (), (Region(1),), (0,)*layers, (0,))
    model, read_model = Model(body, width=4, dtype=dtype), Model(readout, width=4, dtype=dtype)
    with torch.no_grad():
        for m in (model, read_model):
            for v, w in enumerate(m.nodes):
                if pool == "add":
                    w.extra["add_retention"].fill_(.79)
                else:
                    w.extra["fiber_decay"].fill_(.07)
                    w.extra["fiber_qkv_bias"].fill_(.2)
                    w.extra["fiber_out_bias"].fill_(.3)
                for name, value in w.extra.items():
                    if name.startswith("emit_w_"):
                        value.copy_(torch.eye(4, dtype=dtype)*(.13+int(name.rsplit("_", 1)[1])*.02))
                    elif name.startswith("emit_b_"):
                        value.fill_(.04*(v+1))
    xs = []
    for b, times in enumerate(((0, 4, 7), (1, 4), ())):
        for pos, t in enumerate(times):
            value = torch.sin(torch.arange(4, dtype=dtype)*.13+t*.1+b*.2)
            if b == 1 and pos == 0:
                value = torch.zeros_like(value)
            xs.append(External(b, 0, pos, t, value))
    xs.append(External(0, 1, 0, 3, torch.full((4,), .2, dtype=dtype)))
    return body, model, readout, read_model, xs, layers

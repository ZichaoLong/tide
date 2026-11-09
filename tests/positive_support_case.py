"""Two real regions: captured four-score face → tiny singleton / ordinary fan-in.

The independent scalar schedule sees the same inputs/weights, never a candidate
trajectory. This small witness is not the downstream HB16 training workload.
"""
import json
from pathlib import Path
import torch
from tidegraph import Continuation, Edge, External, Graph, Node, Region
from tidegraph.reference import run as reference
from tidegraph.weighted import KVConfig, RegionalConfig, bind
from tidegraph.placement import ExecutionPlacement
from tidegraph.weighted.execution import eager_runner


def fixture(device="cpu", policy="positive-floor-v1", read="qry"):
    data = json.loads((Path(__file__).parent/"data/tl_req_003_shannon_row.json").read_text())
    graph = Graph(tuple(Node(0 if v < 4 else 1) for v in range(6)),
                  (Edge(2,4,2), Edge(0,5,2), Edge(0,5,2), Edge(1,5,2), Edge(3,5,2)),
                  (Region(2),Region(2)), (0,1,2,3), (4,5))
    first = RegionalConfig(k=2, gamma_max=1., region_control_mode="fixed",
        total_fraction=data["target"]-1, allocation_temperature=data["temperature"],
        regularizer="ent", ent_alpha=1., support_policy=policy)
    # The downstream region also allocates, including an actual cap=delta.
    second = RegionalConfig(k=2, gamma_max=1., region_control_mode="fixed",
        total_fraction=.5, allocation_temperature=.25, regularizer="ent", ent_alpha=1.,
        support_policy=policy)
    b = bind(graph, memories=KVConfig(width=2,summary_width=2,adapter_rank=1,heads=1,
        ffn_width=3,window=8,memory_read=read,full_frequencies=((.2,),),
        read_frequencies=(.1,) if read=="qry" else ()),
        regions=(first,second),alpha_agg=.2,dtype=torch.float32,device=device)
    with torch.no_grad():
        for node, score in zip(b.model.nodes[:4],data["scores"]):
            node.read_program.score.vector.zero_()
            node.read_program.score.bias.fill_(score)
    xs = [External(sample,port,pos,time,torch.tensor([1.,.25+.04*port+.07*sample,-.3+.05*pos],
          dtype=torch.float32,device=device,requires_grad=True))
          for sample,times in enumerate(((0,1,3),(0,2)))
          for port in range(4) for pos,time in enumerate(times)]
    return b,xs,6


def execute(b, q, xs, stop, packed=True):
    if not packed:
        return reference(b.graph,b.model,q,xs,stop,sealed_until=stop)
    placement=ExecutionPlacement(preset="cpu" if b.model.nodes[0].bias.device.type=="cpu" else "mixed-c")
    run=eager_runner(b,"python",placement,packing="packed")
    return run(q,xs,stop,sealed_until=stop)


def start(b):
    return Continuation(b.graph.identity,2)


def roots(b,xs):
    return {**{n:p for n,p in b.model.named_parameters() if p.requires_grad},
            **{f"input.{i}":x.value for i,x in enumerate(xs)}}


def optimizer(b):
    return torch.optim.AdamW([p for p in b.model.parameters() if p.requires_grad],lr=.001)

def cast_tree(value):
    """CPU FP32 view for cross-precision/device comparison; int64 unchanged."""
    from dataclasses import fields, is_dataclass, replace
    if isinstance(value,torch.Tensor):
        return value.detach().cpu().to(torch.float32) if value.is_floating_point() else value.detach().cpu()
    if is_dataclass(value):
        return replace(value,**{f.name:cast_tree(getattr(value,f.name)) for f in fields(value)})
    if isinstance(value,dict):
        return {k:cast_tree(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return type(value)(cast_tree(v) for v in value)
    return value

"""Static workload -> independent public modules and physical inputs.

No reference forward, control trace or active-set prepass. The byte ceiling
guards known persistent tensors only; executors enforce their own capacities.
"""
from dataclasses import dataclass
import json
from pathlib import Path
from .workload import validate_plan
from .topology import positive
from .inventory import class_widths


@dataclass(frozen=True)
class PlannedWorkload:
    topology: str
    family: str
    width: int
    samples: int
    positions: int
    seed: int
    workload_sha256: str


def load(path):
    plan = json.loads(Path(path).read_text())
    validate_plan(plan)
    families = {c.get("memory_family", "kv") for c in plan["memories"]}
    config = PlannedWorkload(plan["graph"]["kind"], next(iter(families)) if len(families) == 1 else "heterogeneous",
                             *(plan[k] for k in ("width", "samples", "positions", "seed", "sha256")))
    return plan, config


def materialize(plan, *, max_bytes, dtype=None):
    import torch
    dtype = torch.float32 if dtype is None else dtype
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("materialization requires float32 or float64")
    inv = validate_plan(plan)
    positive("max materialization bytes", max_bytes, 2**63-1)
    floor = plan["memory_floor"]
    required = (floor["logical_parameter_bytes"]+floor["legacy_inert_buffer_bytes"])*(2 if dtype == torch.float64 else 1)
    if required > max_bytes:
        raise ValueError(f"materialization persistent tensors need at least {required} bytes, ceiling {max_bytes}; "
                         "this excludes initialization peak, optimizer, states and execution")
    from tidegraph import Edge, External, Graph, Node, Region
    from tidegraph.source_class import SourceClasses
    from tidegraph.weighted import KVConfig, RecurrentConfig, RegionalConfig, SourceBinding, bind
    # Full semantic config validation precedes learned tensor allocation.
    memories = tuple((RecurrentConfig(**c) if "memory_family" in c else KVConfig(**c)) for c in plan["memories"])
    regions = tuple(RegionalConfig(**c) for c in plan["regions"])
    source = SourceBinding(**plan["source_binding"])
    g = plan["graph"]
    graph = Graph(tuple(Node(r) for r in g["node_regions"]), tuple(Edge(*e) for e in g["edges"]),
                  tuple(Region(1) for _ in regions), tuple(g["inputs"]), tuple(g["outputs"]))
    if g["kind"] == "ranked-local":
        from tidegraph.settle import SettleGraph
        SettleGraph(graph, tuple(g["ranks"]), binding="ports", time_stride=plan["stride"])
    elif g["kind"] == "timed-local":
        graph.topological_order()
    classes = SourceClasses(tuple(plan["classes"]["edges"]), tuple(plan["classes"]["inputs"]))
    hidden = plan["source_hidden_width"]
    if isinstance(hidden, dict):
        hidden = class_widths(hidden)
    bound = bind(graph, memories=memories, regions=regions, source_classes=classes,
                 source_binding=source, source_hidden_width=hidden,
                 alpha_agg=plan["alpha_agg"], dtype=torch.float32, seed=plan["seed"])
    # Refuse drift between static shape/alias formulas and real public modules.
    groups = {}
    params = dict(bound.model.named_parameters(remove_duplicate=False))
    if params.keys() != inv["roles"].keys():
        raise ValueError("materialized parameter role mismatch")
    for name, x in params.items():
        row = inv["roles"][name]
        if list(x.shape) != row["shape"] or x.requires_grad != row["trainable"]:
            raise ValueError("materialized parameter shape/trainability mismatch: "+name)
        groups.setdefault(id(x), set()).add(name)
    if {frozenset(g) for g in groups.values()} != {frozenset(o["aliases"]) for o in inv["owners"].values()}:
        raise ValueError("materialized parameter alias mismatch")
    xs = []
    for sample in range(plan["samples"]):
        for position in range(plan["positions"]):
            value = torch.cat((torch.ones(1), .2*torch.sin(torch.arange(plan["width"])*.19+.3*position+.1*sample)))
            for port, v in enumerate(graph.inputs):
                origin = g["ranks"][g["node_regions"][v]] if g["kind"] == "ranked-local" else 0
                xs.append(External(sample, port, position, origin+plan["stride"]*position, value))
    from .precision import promote
    bound, xs = promote(bound, xs, dtype)
    return bound, xs, plan["span"]

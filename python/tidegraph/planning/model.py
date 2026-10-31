"""Resolved mathematical models, independent of execution and measurements.

0770593 architecture §2, allocation §S.1, KV/recurrent configurations. Runtime
Config classes own applicability; inventory independently predicts real roles.
"""
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import hashlib
import json
from ..weighted.config import KVConfig, RegionalConfig, finite
from ..weighted.recurrent_config import RecurrentConfig
from ..weighted.source_config import SourceBinding
from .inventory import inventory, repeated
from .legacy_topology import graph_counts, integer
from .workload import UPSTREAM, canonical, memory_floor, validate_plan


def fingerprint(record):
    return hashlib.sha256(canonical(record)).hexdigest()


def validate_graph(graph):
    """Static domain checks, including general same-rank independent regions."""
    n, nr = graph["nodes"], graph["regions"]
    integer(n, "nodes", maximum=200000)
    integer(nr, "regions", maximum=n)
    if len(graph["node_regions"]) != n or len(graph["ranks"]) != nr:
        raise ValueError("graph cardinality mismatch")
    for r in graph["node_regions"]:
        integer(r, "region", 0, nr-1)
    if set(graph["node_regions"]) != set(range(nr)):
        raise ValueError("regions must be nonempty")
    for rank in graph["ranks"]:
        integer(rank, "rank")
    for v in (*graph["inputs"], *graph["outputs"]):
        integer(v, "port owner", 0, n-1)
    if not graph["inputs"] or not graph["outputs"]:
        raise ValueError("workload requires input and output ports")
    if graph["kind"] not in {"ranked-local", "timed-local", "positive-delay"}:
        raise ValueError("unknown graph kind")
    for a, b, delay in graph["edges"]:
        integer(a, "source", 0, n-1); integer(b, "target", 0, n-1)
        integer(delay, "delay")
        if graph["kind"] == "ranked-local" and delay != graph["ranks"][graph["node_regions"][b]]-graph["ranks"][graph["node_regions"][a]]:
            raise ValueError("unaligned SettleGraph edge")
    if graph["kind"] != "positive-delay" and graph["counts"] != graph_counts(graph):
        raise ValueError("graph counts disagree")


def resolve_model(graph, memories, regions, *, classes=None, source_binding=None,
                  source_hidden_width=32, alpha_agg=1.):
    """Expand all defaults and reject inapplicable fields before tensor allocation.

    Lists permit heterogeneous node/region bindings; source labels classify
    parameters only. Every physical edge and port remains a separate atom.
    """
    graph = deepcopy(graph)
    validate_graph(graph)
    memory_configs = []
    for c in repeated(memories, graph["nodes"]):
        if c.get("memory_family", "kv") == "kv":
            c.pop("memory_family", None)
            memory_configs.append(KVConfig(**c))
        else:
            memory_configs.append(RecurrentConfig(**c))
    region_configs = [RegionalConfig(**c) for c in repeated(regions, graph["regions"])]
    if len(memory_configs) != graph["nodes"] or len(region_configs) != graph["regions"]:
        raise ValueError("node/region configuration count mismatch")
    sizes = [0]*graph["regions"]
    for j in graph["node_regions"]:
        sizes[j] += 1
    for c, size in zip(region_configs, sizes):
        c.capacity(size)
    source = SourceBinding(**(asdict(source_binding) if is_dataclass(source_binding) else source_binding or {}))
    finite("alpha_agg", alpha_agg, 0)
    classes = deepcopy(classes) if classes is not None else dict(
        edges=[a+1 for a, _, _ in graph["edges"]], inputs=[0]*len(graph["inputs"]))
    result = dict(schema="tide-weighted-model-v1", upstream=UPSTREAM, graph=graph,
                  memories=[asdict(c) for c in memory_configs], regions=[asdict(c) for c in region_configs],
                  source_binding=asdict(source), classes=classes,
                  source_hidden_width=deepcopy(source_hidden_width), alpha_agg=alpha_agg)
    inv = inventory(graph, result["memories"], result["regions"], classes=classes,
                    source_binding=result["source_binding"], source_hidden_width=source_hidden_width)
    result["inventory_sha256"] = inv["sha256"]
    result["model_id"] = fingerprint(result)
    return json.loads(canonical(result)), inv


def validate_model(model):
    if model.get("schema") != "tide-weighted-model-v1" or model.get("upstream") != UPSTREAM:
        raise ValueError("unknown model schema/upstream")
    resolved, inv = resolve_model(**{k: model[k] for k in (
        "graph", "memories", "regions", "classes", "source_binding", "source_hidden_width", "alpha_agg")})
    if model != resolved:
        raise ValueError("model is unresolved or has an inconsistent identity")
    return inv


def workload(model, *, samples=1, positions=1, seed=7, stride=None, recipe=None,
             input_pattern=None, input_origins=None):
    """Bind input windows, never a reference trajectory, to a resolved model."""
    inv = validate_model(model)
    for value, name in ((samples, "samples"), (positions, "positions")):
        integer(value, name)
    integer(seed, "seed", 0, 2**31-1)
    g = model["graph"]
    default_stride = (max(g["ranks"])+2 if g["kind"] == "ranked-local" else
                      g["counts"].get("longest_body_delay", g["counts"].get("feedforward_delay"))+3)
    stride = default_stride if stride is None else integer(stride, "stride")
    from .input_pattern import policy
    input_policy = policy(input_pattern)
    plan = {k: deepcopy(model[k]) for k in ("graph", "memories", "regions", "classes",
                                           "source_binding", "source_hidden_width", "alpha_agg")}
    plan.update(schema="tide-weighted-workload-v2", upstream=UPSTREAM, model_id=model["model_id"],
                width=plan["memories"][0]["width"], samples=samples, positions=positions, seed=seed,
                stride=stride, span=positions*stride, recipe=deepcopy(recipe),
                input_policy=input_policy,
                head=dict(kind="none", trainable_parameters=0),
                state_boundary="connected windows; detach at optimizer boundary")
    if input_pattern is not None:
        plan["input_pattern"] = input_pattern
    if input_origins is not None:
        plan["input_origins"] = deepcopy(input_origins)
        plan["schema"] = "tide-weighted-workload-v3"
    from .hierarchy import input_origins as resolve_origins
    origins = resolve_origins(plan)
    if input_origins is not None:
        plan["input_origins"] = list(origins)
    plan["inventory"] = dict(sha256=inv["sha256"], totals=inv["totals"], owners=len(inv["owners"]))
    plan["memory_floor"] = memory_floor(plan, inv)
    plan["sha256"] = fingerprint(plan)
    validate_plan(plan)
    return plan, inv

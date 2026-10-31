"""Allocation-free graph views and explicit mathematical topology changes.

Settle -> TimedDAG -> PDG inclusion is an identity embedding: a view never
rewrites a graph, input clock, model owner or workload identity. Delay changes
and feedback additions are separate graph constructors, hence separate models.
"""
from copy import deepcopy
from .legacy_topology import graph_counts, integer


def input_origins(plan):
    """Resolve each physical port's clock, retaining historical defaults."""
    graph = plan["graph"]
    default = ([graph["ranks"][graph["node_regions"][v]] for v in graph["inputs"]]
               if graph["kind"] == "ranked-local" else [0]*len(graph["inputs"]))
    origins = plan.get("input_origins", default)
    if not isinstance(origins, (tuple, list)) or len(origins) != len(graph["inputs"]):
        raise ValueError("input_origins must cover every physical input port")
    for origin in origins:
        integer(origin, "input_origins entry", 0, plan["stride"]-1)
    return tuple(origins)


def execution_view(plan, family="auto"):
    """Validate an execution view without relabeling the mathematical model.

    Engines must apply the declared frontend's domain checks to continuations
    as well. A view is not evidence that an independent scheduler was executed.
    """
    from .model import fingerprint
    from .workload import validate_plan
    validate_plan(plan)
    graph = plan["graph"]
    if family == "auto":
        family = {"ranked-local": "settle", "timed-local": "timed-dag",
                  "positive-delay": "pdg"}[graph["kind"]]
    if family not in {"settle", "timed-dag", "pdg"}:
        raise ValueError("execution family must be auto, settle, timed-dag or pdg")
    if family != "pdg":
        graph_counts(graph)  # Independent acyclicity/reachability, not kind.
    origins = input_origins(plan)
    if family == "settle":
        ranks, owners = graph["ranks"], graph["node_regions"]
        if (plan["stride"] <= max(ranks) or
                any(delay != ranks[owners[b]]-ranks[owners[a]] for a, b, delay in graph["edges"]) or
                any(origin != ranks[owners[v]] for origin, v in zip(origins, graph["inputs"]))):
            raise ValueError("settle execution requires aligned edges and original port rank clocks")
    record = dict(schema="tide-graph-execution-view-v1", family=family,
                  graph_kind=graph["kind"], model_id=plan.get("model_id"),
                  workload_id=plan["sha256"], physical_graph_id=fingerprint(graph),
                  input_clock=dict(stride=plan["stride"], origins=list(origins),
                                   time="origin[physical_port] + stride * global_position"),
                  embedding="identity; physical graph, owners, clocks and records unchanged")
    record["view_id"] = fingerprint(record)
    return record


def non_aligned(graph, *, edge_index=0, extra_delay=1):
    """Return a new acyclic model topology with one explicitly changed delay."""
    from .model import validate_graph
    validate_graph(graph)
    graph_counts(graph)
    integer(edge_index, "edge_index", 0, len(graph["edges"])-1)
    integer(extra_delay, "extra_delay")
    result = deepcopy(graph)
    result["edges"] = [list(edge) for edge in graph["edges"]]
    result["edges"][edge_index][2] += extra_delay
    integer(result["edges"][edge_index][2], "changed delay")
    result["kind"] = "timed-local"
    result["counts"] = graph_counts(result)
    validate_graph(result)
    return result


def with_feedback(graph, *, source=None, target=None, delay=1):
    """Add a physical closing edge to an acyclic model, producing a real PDG."""
    from .model import validate_graph
    validate_graph(graph)
    counts = graph_counts(graph)
    source = graph["outputs"][0] if source is None else source
    target = graph["inputs"][0] if target is None else target
    integer(source, "feedback source", 0, graph["nodes"]-1)
    integer(target, "feedback target", 0, graph["nodes"]-1)
    integer(delay, "feedback delay")
    forward = [[] for _ in range(graph["nodes"])]
    for a, b, _ in graph["edges"]:
        forward[a].append(b)
    reached, pending = {target}, [target]
    for node in pending:
        for neighbor in forward[node]:
            if neighbor not in reached:
                reached.add(neighbor); pending.append(neighbor)
    if source not in reached:
        raise ValueError("feedback edge must close an existing target-to-source path")
    result = deepcopy(graph)
    result["edges"] = [list(edge) for edge in graph["edges"]]
    result["edges"].append([source, target, delay])
    result["kind"] = "positive-delay"
    result["counts"] = dict(counts, body_edges=counts["body_edges"]+1,
                            encoded_edges=counts["encoded_edges"]+1,
                            feedforward_delay=counts["longest_body_delay"])
    del result["counts"]["longest_body_delay"]
    validate_graph(result)
    return result


def general_ports_graph():
    """Equal-rank regions, parallel edges and interior/duplicate physical ports."""
    graph = dict(nodes=3, regions=3, node_regions=[0, 1, 2], ranks=[1, 1, 3],
                 edges=[[0, 2, 2], [0, 2, 2], [1, 2, 2]],
                 inputs=[0, 1, 2, 2], outputs=[0, 2, 2], kind="ranked-local")
    graph["counts"] = graph_counts(graph)
    return graph


def general_ports_plan(*, width=8, family="kv", samples=2, positions=3, seed=7):
    """Small core-5 material; sparse per-sample ports keep original logical time."""
    from .model import resolve_model, workload
    from .recipes import memory_config
    graph = general_ports_graph()
    memories = [memory_config(family, width, 5, rank, stage="opening" if rank == 1 else "closing",
                              history=4) for rank in graph["ranks"]]
    model, _ = resolve_model(graph, memories, dict(selection_mode="regional_total", k=1,
                              region_control_mode="context", region_evidence="cat",
                              region_control_width=min(32, width), hidden_width=min(32, width)),
                              source_hidden_width=min(32, width))
    return workload(model, samples=samples, positions=positions, seed=seed,
                    input_pattern="staggered-v1", input_origins=[1, 1, 3, 3])

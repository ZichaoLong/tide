"""Static core-5 workload plans and honest lower-bound memory accounting."""
from copy import deepcopy
import hashlib
import json
from weighted_inventory import inventory
from weighted_topology import positive
from flow_topology import integer

UPSTREAM = "0770593838a37c790a568226930567ed60f99b83"
LEGACY_TARGETS = {"add-body": 9262008512, "attention-body": 17315072192}


def core3_body_target(graph, width, memory):
    """Core-3 complete-flow parameter formula, excluding its embedding/head.

    Same physical topology/width, different local functions and work. Historical
    anchor: flow_topology.make_packet; e*d*d+n*d+e+input_ports (+4*n*d*d).
    """
    positive("width", width, 1000000)
    if memory not in {"add", "attention"}:
        raise ValueError("unknown core-3 reference memory")
    n, e = len(graph["node_regions"]), len(graph["edges"])
    return e*width*width+n*width+e+len(graph["inputs"])+(4*n*width*width if memory == "attention" else 0)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def count(plan):
    return inventory(plan["graph"], plan["memories"], plan["regions"], classes=plan["classes"],
                     source_binding=plan["source_binding"], source_hidden_width=plan["source_hidden_width"])


def match_ffn(plan, target):
    """Nearest integer budget via real private SwiGLU widths; no dummy weights.

    Every node has 3*d*f trainable Full parameters. The remaining roles are
    unchanged. Widths differ by at most one, and the residual is recorded.
    """
    positive("target parameters", target, 2**63-1)
    plan = deepcopy(plan)
    for c in plan["memories"]:
        c["ffn_width"] = 1
    base = count(plan)["totals"]["trainable"]
    d, n = plan["memories"][0]["width"], len(plan["memories"])
    if target < base:
        raise ValueError(f"target {target} below fixed-role minimum {base}; change topology or declared widths")
    increments = (target-base+3*d//2)//(3*d)
    q, r = divmod(increments, n)
    for v, c in enumerate(plan["memories"]):
        c["ffn_width"] = 1+q+(v < r)
    actual = count(plan)["totals"]["trainable"]
    plan["parameter_match"] = dict(target_body=target, actual_body=actual, difference=actual-target,
        relative_difference=(actual-target)/target, method="private_ffn_integer_widths", ffn_widths=sorted({c["ffn_width"] for c in plan["memories"]}))
    return plan


def memory_floor(plan, inv, *, dtype="float32", optimizer="adamw"):
    """Lower bounds only: never certify queues, tapes, operator arenas or HBM.

    Role-expanded FP32 bytes are a useful duplicate-storage indicator, not an
    exact resident allocation estimate. No numeric pre-run decides capacities.
    """
    if dtype not in {"float16", "float32", "float64"} or optimizer not in {"sgd", "adamw"}:
        raise ValueError("unknown explicit dtype/optimizer")
    scalar = {"float16": 2, "float32": 4, "float64": 8}[dtype]
    learned, frozen = inv["totals"]["trainable"], inv["totals"]["frozen"]
    d, n = plan["width"]+1, len(plan["graph"]["node_regions"])
    cache=2*plan["span"]+2
    raw_float, raw_int=0,0
    for c in plan["memories"]:
        family=c.get("memory_family","kv");s=c.get("summary_width",2)
        if family=="kv":
            raw_float+=d+cache*(2*(d-1)+s+1);raw_int+=4+cache
        else:
            heads=c.get("heads",1);k=c.get("key_width",2);v=c.get("value_width",3)
            basic=family in {"affine","delta"}
            conv=0 if basic else heads*(2*k+v) if family=="gated_delta" else heads*v+2*c.get("groups",1)*k
            raw_float+=d+(2 if basic else heads)*k*v+2*s+2+cache*conv
            raw_int+=4+(0 if basic else cache)
    raw_bytes=plan["samples"]*(raw_float*scalar+raw_int*8)
    return dict(kind="lower_bounds_not_runtime_admission", dtype=dtype, optimizer=optimizer,
                logical_parameter_bytes=(learned+frozen)*scalar,
                fp32_master_bytes=4*learned, fp32_gradient_bytes=4*learned,
                fp32_optimizer_slot_bytes=(8 if optimizer == "adamw" else 4)*learned,
                role_expanded_fp32_bytes=4*inv["totals"]["role_elements"],
                legacy_inert_buffer_bytes=n*(d*d+3*d)*scalar,
                resident_default_cache=cache, resident_raw_one_snapshot_bytes=raw_bytes,
                resident_raw_initial_state_old_bytes=3*raw_bytes,
                excluded=["raw states/KV/convolution", "message queues", "retained journals/adjoints",
                          "communication", "temporary packing", "operator workspace", "allocator/vendor overhead"],
                complete_admission=False)


def make_plan(graph, *, family="kv", width=32, samples=4, positions=4, seed=7,
              summary_width=None, hidden_width=None, ffn_width=None, adapter_rank=2,
              selection="regional_total", k=None, memory_read=None, source_mode="region",
              source_rank=None, class_policy="parent", target=None,
              key_width=None, value_width=None, heads=None, groups=None, conv_size=None, conv_stride=None,
              match_core3=None):
    for name, value, limit in (("width", width, 1000000), ("samples", samples, 512),
                               ("positions", positions, 128)):
        positive(name, value, limit)
    integer(seed, "seed", 0, 2**31-1)
    if family not in {"kv", "affine", "delta", "gated_delta", "ssd"}:
        raise ValueError("unknown memory family")
    n = len(graph["node_regions"])
    nr = max(graph["node_regions"])+1
    if graph["kind"] == "ranked-local":
        stride = max(graph["ranks"])+2
    else:
        stride = graph["counts"].get("longest_body_delay", graph["counts"].get("feedforward_delay"))+3
    s = max(1, width//2) if summary_width is None else summary_width
    f = 2*width if ffn_width is None else ffn_width
    h = max(4, width//2) if hidden_width is None else hidden_width
    memory = dict(width=width, summary_width=s, ffn_width=f, epsilon=.3)
    if family == "kv":
        if any(x is not None for x in (key_width,value_width,groups,conv_size,conv_stride)):
            raise ValueError("recurrent dimensions/convolution do not apply to KV")
        memory.update(adapter_rank=adapter_rank, heads=1 if heads is None else heads, memory_read=memory_read or "sum", window=2*positions*stride,
                      beta_short=.1/stride, position_scale=stride)
    else:
        basic=family in {"affine","delta"}
        memory.update(memory_family=family, memory_read=memory_read or ("sum" if family in {"affine", "delta"} else "state"),
                      key_width=max(1,width//2) if key_width is None else key_width,
                      value_width=max(1,width//2) if value_width is None else value_width,lambda_short=.1/stride,
                      heads=(1 if basic else 2) if heads is None else heads,
                      groups=1 if groups is None else groups,
                      conv_size=(1 if basic else 3) if conv_size is None else conv_size,
                      conv_stride=(1 if basic else stride) if conv_stride is None else conv_stride)
    if class_policy not in {"parent", "single"}:
        raise ValueError("unknown source class policy")
    classes = dict(edges=[e[0]+1 if class_policy == "parent" else 0 for e in graph["edges"]], inputs=[0]*len(graph["inputs"]))
    member_counts = [0]*nr
    for r in graph["node_regions"]:
        member_counts[r] += 1
    regions = []
    for size in member_counts:
        c = dict(selection_mode=selection, hidden_width=h, gamma_max=1.6)
        if selection not in {"direct_attention", "unit_attention", "pooled_attention"}:
            c["k"] = size if k is None else k
            if type(c["k"]) is not int or not 1 <= c["k"] <= size:
                raise ValueError("sending k outside region")
        elif k is not None:
            raise ValueError("k does not apply to Attention sending")
        regions.append(c)
    memories = [dict(memory) for _ in range(n)]
    if family == "kv" and graph["kind"] == "ranked-local":
        for v, c in enumerate(memories):
            c["position_origin"] = graph["ranks"][graph["node_regions"][v]]
    plan = dict(schema="tide-weighted-workload-v1", upstream=UPSTREAM, graph=graph, memories=memories, regions=regions,
                classes=classes, class_policy=class_policy, source_binding=dict(mode=source_mode, rank=source_rank),
                source_hidden_width=width, alpha_agg=.2, width=width, samples=samples, positions=positions,
                seed=seed, stride=stride, span=positions*stride,
                input_policy="unit-strength; shared tensor per sample/position; sin content; physical ports separate",
                head=dict(kind="none", trainable_parameters=0), state_boundary="connected windows; detach at optimizer boundary")
    if match_core3 is not None:
        if target is not None:
            raise ValueError("choose one parameter target")
        target = core3_body_target(graph, width, match_core3)
    if target is not None:
        plan = match_ffn(plan, target)
        if match_core3 is not None:
            plan["parameter_match"]["reference"] = dict(semantics="core-3", memory=match_core3,
                topology="same physical graph", width=width, task_head_parameters_excluded=True)
    inv = count(plan)
    plan["inventory"] = dict(sha256=inv["sha256"], totals=inv["totals"], owners=len(inv["owners"]))
    plan["memory_floor"] = memory_floor(plan, inv)
    plan["sha256"] = hashlib.sha256(canonical(plan)).hexdigest()
    validate_plan(plan)
    return plan, inv


def validate_plan(plan):
    if plan.get("schema") != "tide-weighted-workload-v1" or plan.get("upstream") != UPSTREAM:
        raise ValueError("unknown weighted workload schema/upstream")
    data = {k: v for k, v in plan.items() if k != "sha256"}
    if hashlib.sha256(canonical(data)).hexdigest() != plan.get("sha256"):
        raise ValueError("weighted workload hash mismatch")
    for key, limit in (("width", 1000000), ("samples", 512), ("positions", 128),
                       ("span", 2**63-1), ("stride", 2**63-1)):
        positive(key, plan[key], limit)
    integer(plan["seed"], "seed", 0, 2**31-1)
    if plan["span"] != plan["positions"]*plan["stride"] or 9*plan["span"] >= 2**63:
        raise ValueError("window/time domain mismatch")
    if any(c.get("width", 4) != plan["width"] for c in plan["memories"]):
        raise ValueError("workload/memory width mismatch")
    graph = plan["graph"]
    if graph["kind"] not in {"ranked-local", "timed-local", "positive-delay"}:
        raise ValueError("unknown graph kind")
    if graph["nodes"] != len(graph["node_regions"]) or graph["regions"] != len(graph["ranks"]):
        raise ValueError("graph cardinality mismatch")
    if not graph["inputs"] or not graph["outputs"]:
        raise ValueError("workload requires physical input/output ports")
    for v in graph["outputs"]:
        integer(v, "output owner", 0, graph["nodes"]-1)
    for rank in graph["ranks"]:
        integer(rank, "region rank")
    if graph["kind"] == "ranked-local" and any(
            delay != graph["ranks"][graph["node_regions"][b]]-graph["ranks"][graph["node_regions"][a]]
            for a,b,delay in graph["edges"]):
        raise ValueError("unaligned SettleGraph edge")
    inv = count(plan)
    if plan["inventory"] != dict(sha256=inv["sha256"], totals=inv["totals"], owners=len(inv["owners"])):
        raise ValueError("weighted parameter inventory mismatch")
    if plan["memory_floor"] != memory_floor(plan, inv):
        raise ValueError("weighted memory lower bound mismatch")
    return inv

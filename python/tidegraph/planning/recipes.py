"""Core-5 FC/HB textbook-inspired model families and natural parameter scales.

Instances §4.2–4.4 determine the local dimensions/position bindings. LH supplies
only optional cross-region/skip connectivity. No task head or dummy parameters.
"""
from copy import deepcopy
import math
from .legacy_topology import integer
from .model import resolve_model, workload
from .topology import layered, lh_inspired

FAMILIES = ("kv", "affine", "delta", "gated_delta", "ssd", "mixed")


def frequencies(rotary_width):
    return tuple(10000.**(-2*a/rotary_width) for a in range(rotary_width//2))


def memory_config(family, width, stride, rank, *, rich=False, stage="middle", history=512):
    """Natural FFN=4d; fixed small summaries/adapters, heads scaled with d."""
    small = min(32, width)
    common = dict(width=width, summary_width=small, ffn_width=4*width, epsilon=1e-5)
    if family == "kv":
        head_width = min(64, width)
        rot = head_width if stage == "opening" else head_width//2 if stage == "closing" else 0
        return dict(common, heads=width//head_width, adapter_rank=small,
                    memory_read="qry" if rich and stage == "opening" else "sum",
                    window=history*stride if stage != "middle" else 0,
                    beta_short=math.log(2)/stride, position_origin=rank, position_scale=stride,
                    full_frequencies=(frequencies(rot),)*(width//head_width) if rich else (),
                    read_frequencies=frequencies(small) if rich and stage == "opening" else ())
    basic = family in {"affine", "delta"}
    return dict(common, memory_family=family, memory_read="state", key_width=min(64, width),
                value_width=min(64 if basic else 128, width), heads=1 if basic else max(1, width//128),
                conv_size=1 if basic else 4, conv_stride=1 if basic else stride,
                lambda_short=math.log(2)/stride)


def model_recipe(*, topology="hb", family="kv", width=512, region_width=32, depth=None,
                 branches=2, offsets=(0, 1), profile="baseline", sparse_denominator=8,
                 selection="regional_total", regularizer=None, ent_alpha=None,
                 history=512, source_mode="region", memory_overrides=None, region_overrides=None):
    """Return resolved public model and independent inventory; no numeric prepass.

    baseline: sum/cat/ind/context/cat; textbook: layered KV reads/positions/att,
    with FC §4.2 B or HB §4.2 positions and §4.4 regional control by default.
    Explicit overrides use the same Config validators as arbitrary public binds.
    """
    if family not in FAMILIES or profile not in {"baseline", "textbook"}:
        raise ValueError("unknown family/profile")
    integer(width, "width", maximum=65536)
    if width >= 64 and width % 64:
        raise ValueError("recipe content width must be a multiple of 64")
    if width < 64 and width % 2:
        raise ValueError("small recipe widths must be even")
    integer(history, "history"); integer(sparse_denominator, "sparse denominator")
    depth = (8 if topology == "fc" else 3) if depth is None else depth
    if topology == "fc":
        g = layered("fc", depth=depth, region_width=region_width)
    elif topology == "hb":
        g = layered("hb", depth=depth, branches=branches, region_width=region_width,
                    offsets=(tuple(offsets),)*(2*depth))
    elif topology == "lh":
        g = lh_inspired(depth=depth, region_width=region_width, branches=branches,
                        fanout=min(4, region_width))
    else:
        raise ValueError("recipe topology must be fc/hb/lh")
    stride = max(g["ranks"])+2
    memories, regions = [], []
    k = max(1, (region_width+sparse_denominator-1)//sparse_denominator)
    middle_index = 0
    for j, rank in enumerate(g["ranks"]):
        center = rank % 4 == 0 if topology == "fc" else rank == depth+1
        stage = "middle" if center else "opening" if topology == "fc" or rank <= depth else "closing"
        f = (("gated_delta", "ssd")[middle_index % 2] if center else "kv") if family == "mixed" else family
        middle_index += int(center)
        m = memory_config(f, width, stride, rank, rich=profile == "textbook" and topology != "fc",
                          stage=stage, history=history)
        if f == "kv" and profile == "textbook" and topology == "fc" and not center:
            m["full_frequencies"] = (frequencies(min(64, width)),)*m["heads"]
        m.update(memory_overrides or {})
        memories.extend(deepcopy(m) for _ in range(region_width))
        c = dict(read_mode="+", local_score="cat", region_score="ind", selection_mode=selection,
                 hidden_width=min(32, width), gamma_max=2.)
        if selection not in {"direct_attention", "unit_attention", "pooled_attention"}:
            c["k"] = k
        if selection in {"regional_total", "regional_threshold", "constrained"}:
            c.update(regularizer="quad" if regularizer is None else regularizer, ent_alpha=ent_alpha)
        elif regularizer is not None or ent_alpha is not None:
            raise ValueError("regularizer does not apply to this sending function")
        if selection.startswith("regional_"):
            c.update(region_control_mode="context", region_evidence="cat", region_control_width=min(32, width))
        if profile == "textbook" and topology != "fc" and not center:
            c.update(region_score="att", selection_width=min(32, width), region_key="content_stats")
        c.update(region_overrides or {})
        regions.append(c)
    return resolve_model(g, memories, regions, source_hidden_width=min(32, width),
                         source_binding=dict(mode=source_mode, rank=min(32, width) if source_mode == "adapted" else None))


def scaled_recipe(target, **options):
    """Nearest natural 128-wide step; never tune a private FFN to a count.

    Within this family the parameter count at multiples of 128 is quadratic:
    fixed topology/summary/adapter/key width, FFN=4d, scaled head count. Three
    exact independent inventories locate a bracket; actual neighbors choose the
    result. Overrides may break that polynomial and are refused for auto-sizing.
    """
    integer(target, "target parameters")
    if any(k in options for k in ("width", "memory_overrides", "region_overrides")):
        raise ValueError("auto-sizing requires natural dimensions; use an explicit width with overrides")
    probes = [model_recipe(width=128*i, **options)[1]["totals"]["trainable"] for i in (1, 2, 3)]
    a = (probes[2]-2*probes[1]+probes[0])/2
    b = probes[1]-probes[0]-3*a
    c = probes[0]-a-b
    root = (-b+math.sqrt(max(0, b*b+4*a*(target-c))))/(2*a) if a else (target-c)/b
    center = max(1, round(root))
    choices = []
    for multiplier in sorted({max(1, center-1), center, center+1}):
        model, inv = model_recipe(width=128*multiplier, **options)
        choices.append((abs(inv["totals"]["trainable"]-target), model, inv))
    _, model, inv = min(choices, key=lambda row: row[0])
    actual = inv["totals"]["trainable"]
    scale = dict(target_parameters=target, actual_parameters=actual, difference=actual-target,
                 relative_difference=(actual-target)/target, method="nearest_natural_width_multiple_128",
                 width=model["memories"][0]["width"], ffn_ratio=4, task_head_parameters=0)
    return model, inv, scale


def prepare(request, *, samples=1, positions=1, seed=7):
    """One JSON request produces a portable resolved workload and inventory."""
    request = deepcopy(request)
    target = request.pop("target_parameters", None)
    if target is None:
        model, inv = model_recipe(**request)
        scale = None
    else:
        model, inv, scale = scaled_recipe(target, **request)
    spec = dict(schema="tide-core5-recipe-v1", requested=request, scale=scale,
                sparse_budget=dict(requested_fraction=1/request.get("sparse_denominator", 8),
                    effective_k=[c["k"] for c in model["regions"]],
                    note="k is sending strength; actual support and Full counts require execution"))
    plan, inv = workload(model, samples=samples, positions=positions, seed=seed, recipe=spec)
    return plan, inv

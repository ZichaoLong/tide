"""Finite round-three model catalog and capacity/control matching.

The original 216 subjects remain immutable. The explicitly authorized 128-model
supplement varies mathematical allocation budgets and one controlled template
axis at a time. Seeds, optimizer recipes, exposure budgets, execution backends
and aliases do not create new subject identities. Counts are
obtained from real modules on the meta device; no allocated dummy parameters
are used to improve a match.
"""
from __future__ import annotations

import copy
from bisect import bisect_left, bisect_right
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import torch

from .models3 import ACTIVITY_VERSION, TIMED_LAST_NONEMPTY, build_model, parameter_counts, resolve_readout_policy
from .shannon_recipes3 import new_config


CATALOG_VERSION = "round3-tide-216-capacity-v2"
SUBJECT_QUOTAS = {"fc": 24, "lattice": 48, "exchange": 72,
                  "multiscale_settle": 36, "multiscale_timed": 36}
SUBJECT_WHOLE_PARAMETER_LIMIT = 1_000_000_000
SUBJECT_BACKBONE_PARAMETER_LIMIT = 32_000_000
CONTROL_MATCHER_VERSION = "round3-global-control-cover-v1"
SUPPLEMENTAL_CATALOG_VERSION = "round3-low-activity-128-v1"
COMBINED_CATALOG_VERSION = "round3-subjects-344-v1"
SUPPLEMENTAL_SUBJECT_QUOTAS = {"fc": 16, "lattice": 24, "exchange": 32,
                              "multiscale_settle": 28, "multiscale_timed": 28}
SUPPLEMENTAL_ALLOCATION_BUDGETS = (8, 4, 2, 1)
MAX_AUTHORIZED_SUBJECTS = 344
SUPPLEMENTAL_PARAMETER_REFERENCE = {"vocab_size": 8192, "backbone_parameters": 726_706_500,
                                    "whole_parameters": 860_924_228, "max_backbone_parameters": 22_682_912}
COMBINED_WHOLE_PARAMETER_REFERENCE = 1_843_506_106


def _number(values):
    mapping = {}
    return [mapping.setdefault(value, len(mapping)) for value in values]


def effective_model_identity(spec: dict) -> dict:
    """Canonical effective graph, clocks and parameter ownership, not raw JSON."""
    family = spec["family"]
    if family in ("transformer", "moe"):
        d = int(spec.get("d", 128))
        identity = dict(family=family, d=d, layers=int(spec.get("layers", 4)),
                        heads=int(spec.get("heads", 4)), ffn_width=int(spec.get("ffn_width", 4*d)),
                        window=int(spec.get("window", 128)))
        if family == "moe":
            identity.update(experts=int(spec.get("experts", 8)), top_k=int(spec.get("top_k", 2)))
        return identity
    if family != "tide":
        raise ValueError(f"unknown model family {family!r}")
    from tide import ModelConfig
    from tide.topology import build_topology
    c = ModelConfig.from_dict(copy.deepcopy(spec["config"]))
    graph = build_topology(c.topology)
    ordinary, hub = {"K": ("K", "K"), "S": ("S", "S"),
                     "X": ("K", "S"), "X_reverse": ("S", "K")}[c.memory]
    kinds = [hub if node.role == "H" else ordinary for node in graph.nodes]
    if c.kv_width is not None and "K" not in kinds:
        raise ValueError("kv_width has no effective K-memory owner")
    fixed = [all(graph.nodes[v].role == "H" for v in nodes) and
             (graph.config.partition == "singleton" or c.hub_allocation == "fixed_pass")
             for nodes in graph.regions]

    def groups(profile):
        if profile == "independent":
            return None
        keys = [(v.layer, v.role, kinds[v.id]) for v in graph.nodes]
        if profile == "region_shared":
            keys = [(v.region, key) for v, key in zip(graph.nodes, keys)]
        return _number(keys)

    main_groups = groups(c.sharing)
    memory_profile = c.memory_sharing or c.sharing
    memory_groups = groups(memory_profile)
    rkeys = [(graph.region_ranks[r], tuple(sorted({graph.nodes[v].role for v in nodes})))
             if c.sharing == "layer_shared" else r for r, nodes in enumerate(graph.regions) if not fixed[r]]
    region_groups = iter(_number(rkeys))
    regional = [-1 if skip else next(region_groups) for skip in fixed]
    ownership = {"main_groups": main_groups, "memory_groups": memory_groups,
                 "regional_groups": regional}
    if "K" in kinds:
        members = [v for v in graph.nodes if kinds[v.id] == "K"]
        keys = [(v.layer, v.region) if memory_profile == "region_shared" else
                (v.layer, tuple(sorted({graph.nodes[u].role for u in graph.regions[v.region]}))) for v in members]
        ownership["kv_groups"] = None if memory_profile == "independent" else _number(keys)
    clock = c.clock.resolve(graph.period, graph.config.timing == "settle")
    clocks = {"short_decay": clock.short_decay}
    if "K" in kinds:
        clocks.update(position_period=clock.position_period, kv_window=clock.kv_window)
    if "S" in kinds:
        clocks["ssd_stride"] = clock.ssd_stride
    # Pool ignores k; bounded all-H groups use their fixed quad budget. Only
    # ordinary/mixed regions use the configured allocation and ordinary budget.
    identity = dict(family="tide", d=c.d, input_dim=c.input_dim or c.d,
                    fc_unit_ingress=graph.config.kind == "fc",
                    nodes=[(v.layer, v.role, v.region) for v in graph.nodes],
                    edges=[(e.src, e.dst, e.delay, e.source_class) for e in graph.edges],
                    regions=graph.regions, input_nodes=graph.input_nodes, output_nodes=graph.output_nodes,
                    period=graph.period, timing=graph.config.timing,
                    readout_policy=resolve_readout_policy(spec, settle=graph.config.timing == "settle"),
                    partition=graph.config.partition if any(v.role == "H" for v in graph.nodes) else "group",
                    terminal_hubs=any(v.role == "H" and v.layer == len(graph.widths)-1 for v in graph.nodes),
                    memory_kinds=kinds,
                    ownership=ownership, fixed_regions=fixed, aggregation=c.aggregation,
                    allocation=c.allocation,
                    budget=None if c.allocation == "pool" else c.budget or max(1, graph.config.n//32),
                    ffn_width=c.ffn_width or 4*c.d, clocks=clocks)
    if "K" in kinds:
        identity["kv_width"] = c.kv_width or c.d
    if main_groups is not None or memory_groups is not None:
        identity["adapter_rank"] = c.adapter_rank
    return identity


def model_key(spec: dict) -> str:
    payload = json.dumps(effective_model_identity(spec), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def generate_subjects() -> list[dict]:
    """Deterministic 24/48/72/36/36 coverage within the preparation capacity cap.

    Small graphs and explicit FFN widths retain the registered 64/128/256
    content widths. Sharing remains an actual model choice with private
    adapters; it is never simulated by ignoring or removing counted weights.
    """
    from tide import ModelConfig
    from tide.config import edge_delay_pair
    rows = []

    def add(group, d, memory, variant, topology, changes=None, pair=None):
        config = dict(topology=topology, d=d, memory=memory, budget=1, ffn_width=d,
                      aggregation="source_residual", allocation="quad", sharing="independent")
        config.update(changes or {})
        config = ModelConfig.from_dict(new_config(config)).to_dict()
        spec = {"family": "tide", "config": config}
        if group == "multiscale_timed":
            spec["readout_policy"] = TIMED_LAST_NONEMPTY
        key = model_key(spec)
        rows.append({"id": f"{group}-d{d}-{memory.lower()}-{variant}-{key[:10]}",
                     "group": group, "model": spec, "model_key": key,
                     "catalog_version": CATALOG_VERSION, "coverage_variant": variant,
                     "edge_delay_pair": pair})

    for d in (64, 128, 256):
        for memory in ("K", "S"):
            for variant, topology, changes in (
                ("base", {"kind": "fc", "n": 2, "layers": 2}, {"ffn_width": 2*d}),
                ("depth3", {"kind": "fc", "n": 2, "layers": 3}, {"ffn_width": 2*d}),
                ("shared", {"kind": "fc", "n": 2, "layers": 2}, {"sharing": "region_shared", "ffn_width": 2*d}),
                ("ffn1", {"kind": "fc", "n": 2, "layers": 2}, {}),
            ):
                add("fc", d, memory, variant, topology, changes)
            lattice_base = {"kind": "lattice", "n": 2, "depth": 1}
            for variant, topology, changes in (
                ("base", lattice_base, {}),
                ("depth2", dict(lattice_base, depth=2), {}),
                ("ffn2", lattice_base, {"ffn_width": 2*d}),
                ("nodes4", dict(lattice_base, n=4), {}),
                ("receive64", lattice_base, {"aggregation": "incoming_light_64"}),
                ("receive128", lattice_base, {"aggregation": "incoming_light_128"}),
                ("memoryshared", lattice_base, {"memory_sharing": "region_shared"}),
                ("layershared", lattice_base, {"sharing": "layer_shared"}),
            ):
                add("lattice", d, memory, variant, topology, changes)
        for memory in ("K", "S", "X", "X_reverse"):
            exchange_base = {"kind": "exchange", "n": 2, "hubs": 1, "depth": 1, "rounds": 1}
            narrow = {"aggregation": "incoming_wide", "hub_allocation": "fixed_pass"}
            if memory != "S":
                narrow["kv_width"] = d//2
            for variant, topology, changes in (
                ("base", exchange_base, {}),
                ("shannon", exchange_base, {"allocation": "shannon"}),
                ("pool", exchange_base, {"allocation": "pool"}),
                ("capacity", dict(exchange_base, n=4), {"budget": 2}),
                ("independent", exchange_base, {"sharing": "independent"}),
                ("narrowwide", exchange_base, narrow),
            ):
                add("exchange", d, memory, variant, topology, {"sharing": "layer_shared", **changes})
            for variant, topology, changes in (
                ("group5", {"kind": "multiscale", "n": 2, "hubs": 1, "profile": "compact-5-dd4", "partition": "group"}, {}),
                ("singleton7", {"kind": "multiscale", "n": 2, "hubs": 1, "profile": "compact-7", "partition": "singleton"},
                 {"allocation": "shannon", "aggregation": "incoming_light_64"}),
                ("mixed9", {"kind": "multiscale", "n": 2, "hubs": 1, "profile": "compact-9", "partition": "mixed", "hub_mask": "checkerboard"}, {}),
            ):
                # Equal local clocks and schedule; only edge-delay binding and
                # resulting readout slots differ between each Settle/Timed pair.
                period = {"group5": 6, "singleton7": 8, "mixed9": 10}[variant]
                base = dict(topology=topology, d=d, memory=memory, budget=1, sharing="layer_shared", ffn_width=d,
                            clock=dict(position_origin="absolute", position_period=period,
                                       kv_window=128*period, short_decay=math.log(2)/(8*period), ssd_stride=period),
                            **changes)
                settle, timed = edge_delay_pair(ModelConfig.from_dict(base), period=period)
                pair_id = f"ms-d{d}-{memory.lower()}-{variant}"
                for group, config in (("multiscale_settle", settle), ("multiscale_timed", timed)):
                    resolved = config.to_dict()
                    top = resolved.pop("topology")
                    resolved.pop("d")
                    resolved.pop("memory")
                    add(group, d, memory, variant, top, resolved, pair_id)
    if Counter(row["group"] for row in rows) != Counter(SUBJECT_QUOTAS):
        raise RuntimeError("subject family quota mismatch")
    keys = [row["model_key"] for row in rows]
    if len(set(keys)) != 216:
        duplicates = [key for key, count in Counter(keys).items() if count > 1]
        raise RuntimeError(f"effective model aliases in subject catalog: {duplicates}")
    return rows


def _supplemental_allocation_metadata(spec):
    """Separate the ordinary k/n label from Hub rules and actual activity."""
    from tide import ModelConfig
    from tide.topology import build_topology
    config = ModelConfig.from_dict(spec["config"])
    graph = build_topology(config.topology)
    regions = Counter()
    hub_counts = []
    for members in graph.regions:
        hubs = sum(graph.nodes[node].role == "H" for node in members)
        label = "ordinary_only" if hubs == 0 else "hub_only" if hubs == len(members) else "mixed"
        regions[label] += 1
        if hubs:
            hub_counts.append(hubs)
    if not hub_counts:
        policy = "no_hubs"
    elif config.hub_allocation == "fixed_pass":
        policy = "fixed_pass_on_hub_only_regions"
    elif config.topology.partition == "singleton":
        policy = "fixed_pass_on_singleton_hubs"
    elif regions["mixed"]:
        policy = "bounded_hubs_in_mixed_regions"
    else:
        policy = "separate_hub_regions_use_quad_allocation"
    hub_budget = max(1, (max(hub_counts)+1)//2) if hub_counts else None
    return {"ordinary_nodes_per_region": config.topology.n,
            "ordinary_allocation_budget": config.budget,
            "nominal_ordinary_budget_ratio": config.budget/config.topology.n,
            "nominal_ratio_scope": "ordinary k/n only; not Full fraction or measured active-parameter ratio",
            "hub_nodes_requested_per_site": config.topology.hubs,
            "present_hub_nodes": sum(node.role == "H" for node in graph.nodes),
            "region_counts": {kind: regions[kind] for kind in ("ordinary_only", "hub_only", "mixed")},
            "hub_policy": policy,
            "separate_hub_nominal_budget": hub_budget if policy == "separate_hub_regions_use_quad_allocation" else None,
            "mixed_region_nominal_total_budget": config.budget+hub_budget if regions["mixed"] else None,
            "mixed_hub_bounds": "mu/4 <= hub control <= mu" if regions["mixed"] else None,
            "actual_parameter_activity": "requires qualified per-token measurements including always-active work"}


def generate_supplemental_subjects() -> list[dict]:
    """Authorized 32 one-axis templates x four allocation budgets = 128.

    Baselines are copied from the unchanged primary catalog, with common n=16
    and d=128. Within a template comparison n, d, allocation budget and timing
    agree. Allocation k is part of the mathematical model; optimizer/seed/token
    budgets are not additional model identities. No training or admission is
    performed here.
    """
    from tide import ModelConfig
    primary = generate_subjects()
    lookup = {(row["group"], row["model"]["config"]["d"], row["model"]["config"]["memory"], row["coverage_variant"]): row
              for row in primary}
    templates = {
        "fc": ("K", "base", (
            ("base", None, None), ("memory_s", "config.memory", "S"),
            ("depth3", "config.topology.layers", 3), ("region_shared", "config.sharing", "region_shared"))),
        "lattice": ("K", "base", (
            ("base", None, None), ("memory_s", "config.memory", "S"),
            ("depth2", "config.topology.depth", 2), ("memory_shared", "config.memory_sharing", "region_shared"),
            ("receive64", "config.aggregation", "incoming_light_64"), ("ffn2", "config.ffn_width", 256))),
        "exchange": ("X", "base", (
            ("base", None, None), ("memory_k", "config.memory", "K"), ("memory_s", "config.memory", "S"),
            ("memory_x_reverse", "config.memory", "X_reverse"), ("independent", "config.sharing", "independent"),
            ("shannon", "config.allocation", "shannon"), ("kv64", "config.kv_width", 64),
            ("hub_fixed", "config.hub_allocation", "fixed_pass"))),
    }
    multiscale = ("X", "group5", (
        ("base", None, None), ("memory_k", "config.memory", "K"), ("memory_s", "config.memory", "S"),
        ("singleton", "config.topology.partition", "singleton"), ("mixed", "config.topology.partition", "mixed"),
        ("checkerboard", "config.topology.hub_mask", "checkerboard"), ("shannon", "config.allocation", "shannon")))
    templates.update(multiscale_settle=multiscale, multiscale_timed=multiscale)
    rows = []
    for group, (memory, primary_variant, variants) in templates.items():
        parent = lookup[group, 128, memory, primary_variant]
        base = copy.deepcopy(parent["model"])
        base["config"]["topology"]["n"] = 16
        base["config"]["d"] = 128
        for variant, axis, value in variants:
            template = copy.deepcopy(base)
            reference_value = None
            if axis is not None:
                target = template
                pieces = axis.split(".")
                for piece in pieces[:-1]:
                    target = target[piece]
                reference_value = target[pieces[-1]]
                target[pieces[-1]] = value
            template_id = f"low-activity-{group}-{variant}"
            for budget in SUPPLEMENTAL_ALLOCATION_BUDGETS:
                spec = copy.deepcopy(template)
                spec["config"]["budget"] = budget
                # Check the upstream instance contract and fully preserve its
                # defaults, local clocks and versioned LM readout policy.
                spec["config"] = ModelConfig.from_dict(new_config(spec["config"])).to_dict()
                key = model_key(spec)
                pair = f"low-activity-ms-{variant}-k{budget}" if group.startswith("multiscale_") else None
                rows.append({"id": f"low-activity-{group}-{variant}-n16-d128-k{budget}-{key[:10]}",
                    "group": group, "model": spec, "model_key": key, "effective_key": key,
                    "catalog_version": SUPPLEMENTAL_CATALOG_VERSION, "cohort": "low_activity_supplement",
                    "coverage_variant": variant, "template_id": template_id,
                    "primary_baseline_id": parent["id"], "edge_delay_pair": pair,
                    "common_template_contract": {"n": 16, "d": 128},
                    "single_axis_control": {"reference_subject_id": None, "axis": axis,
                        "reference_value": reference_value, "variant_value": value,
                        "scope": "same supplemental family, n, d, allocation k and timing"},
                    "allocation": _supplemental_allocation_metadata(spec),
                    "allocation_budget_sweep": {"template_id": template_id, "axis": "config.budget",
                        "value": budget, "values": list(SUPPLEMENTAL_ALLOCATION_BUDGETS)},
                    "exposure": {"maximum_tuning_loss_tokens": 8*2**20, "formal_loss_tokens_per_trajectory": 12*2**20},
                    "model_counting": "one effective mathematical model; allocation k is a model axis, LR/seed/exposure/retry are not",
                    "control_matching": {"kinds": ["dense_total", "dense_active", "moe_joint"],
                                         "parameter_tolerance": .10, "ratio_tolerance": .02}})
    by_slot = {(row["group"], row["coverage_variant"], row["model"]["config"]["budget"]): row for row in rows}
    for row in rows:
        budget = row["model"]["config"]["budget"]
        if row["coverage_variant"] != "base":
            row["single_axis_control"]["reference_subject_id"] = by_slot[row["group"], "base", budget]["id"]
        row["allocation_budget_sweep"]["subject_ids_by_budget"] = {
            str(k): by_slot[row["group"], row["coverage_variant"], k]["id"] for k in SUPPLEMENTAL_ALLOCATION_BUDGETS}
        if row["edge_delay_pair"]:
            partner_group = "multiscale_timed" if row["group"] == "multiscale_settle" else "multiscale_settle"
            row["timing_pair"] = {"partner_subject_id": by_slot[partner_group, row["coverage_variant"], budget]["id"],
                "input_period": 6, "local_clocks_identical": True,
                "scope": "same graph/config/local clocks except timing binding; each binding retains its declared LM readout"}
    if Counter(row["group"] for row in rows) != Counter(SUPPLEMENTAL_SUBJECT_QUOTAS):
        raise RuntimeError("supplemental subject family quota mismatch")
    keys = {row["model_key"] for row in rows}
    if len(rows) != 128 or len(keys) != 128 or keys.intersection(row["model_key"] for row in primary):
        raise RuntimeError("supplemental catalog must contain 128 unique effective models disjoint from the original 216")
    return rows


def generate_combined_subjects() -> list[dict]:
    """Explicitly request the authorized 344-model catalog; never expand the default."""
    rows = generate_subjects()+generate_supplemental_subjects()
    if len(rows) != MAX_AUTHORIZED_SUBJECTS or len({row["model_key"] for row in rows}) != MAX_AUTHORIZED_SUBJECTS:
        raise RuntimeError("combined catalog must contain exactly 344 unique effective subjects")
    return rows


def subject_catalog_contract(*, include_supplemental: bool = False) -> dict:
    """Finite catalog authorization is separate from measured resource admission."""
    count = MAX_AUTHORIZED_SUBJECTS if include_supplemental else 216
    return {"schema": "pilot-r3-subject-catalog-contract-v1",
            "catalog_version": COMBINED_CATALOG_VERSION if include_supplemental else CATALOG_VERSION,
            "primary_catalog_version": CATALOG_VERSION,
            "supplemental_catalog_version": SUPPLEMENTAL_CATALOG_VERSION if include_supplemental else None,
            "primary_subject_count": 216, "supplemental_subject_count": 128 if include_supplemental else 0,
            "subject_count_limit": count, "control_count_limit": 3*count, "total_model_count_limit": 4*count,
            "primary_whole_parameter_limit": SUBJECT_WHOLE_PARAMETER_LIMIT,
            "primary_parameter_limit_applies_to_combined_catalog": False,
            "combined_whole_parameter_reference_at_vocab_8192": COMBINED_WHOLE_PARAMETER_REFERENCE if include_supplemental else None,
            "parameter_reference_is_storage_admission": False,
            "per_subject_backbone_parameter_limit": SUBJECT_BACKBONE_PARAMETER_LIMIT,
            "maximum_tuning_loss_tokens_per_subject": 8*2**20,
            "formal_loss_tokens_per_trajectory": 12*2**20,
            "resource_admission": {"required": True, "status": "requires_independent_measured_gate",
                "scope": "complete selected catalog plus globally reused controls, retained checkpoints, atomic peaks, quota, device and time budgets"}}


def exact_parameter_counts(spec: dict, vocab_size: int = 8192) -> dict:
    """Instantiate the executable model on meta, count real unique parameters."""
    with torch.device("meta"):
        model = build_model(spec, vocab_size)
    return parameter_counts(model)


def control_parameter_counts(spec: dict, vocab_size: int = 8192) -> dict:
    """Closed forms checked against executable modules; used for finite search."""
    c = effective_model_identity(spec)
    if c["family"] not in ("transformer", "moe"):
        raise ValueError("closed-form controls require transformer or moe")
    d, layers, hidden = c["d"], c["layers"], c["ffn_width"]
    shared = 4*d*d + 2*d
    if c["family"] == "transformer":
        total = active = layers * (shared + 3*d*hidden) + d
    else:
        shared += d*c["experts"]
        total = layers * (shared + 3*d*hidden*c["experts"]) + d
        active = layers * (shared + 3*d*hidden*c["top_k"]) + d
    return dict(backbone_total_parameters=total, mean_active_backbone_parameters=active,
                active_ratio=active/total, vocabulary_parameters=vocab_size*d,
                whole_model_parameters=total + vocab_size*d)


CONTROL_WIDTHS = (64, 96, 128, 192, 256, 384, 512, 768, 1024)
CONTROL_LAYERS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
CONTROL_EXPERTS = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)


def _near_width(value, d):
    # A finite registered structural interval, not an unlimited retry search.
    return sorted({max(16, min(16*d, 16*(math.floor(value/16)+offset))) for offset in (0, 1)})


def _dense_candidates(target, window):
    for d in CONTROL_WIDTHS:
        for layers in CONTROL_LAYERS:
            ideal = ((target-d)/layers - 4*d*d - 2*d)/(3*d)
            for hidden in _near_width(ideal, d):
                yield dict(family="transformer", d=d, layers=layers, heads=d//32,
                           ffn_width=hidden, window=window)


def _moe_candidates(total, active, window):
    for d in CONTROL_WIDTHS:
        for layers in CONTROL_LAYERS:
            for experts in CONTROL_EXPERTS:
                shared = 4*d*d + 2*d + d*experts
                ideal = ((total-d)/layers - shared)/(3*d*experts)
                for hidden in _near_width(ideal, d):
                    ideal_k = ((active-d)/layers - shared)/(3*d*hidden)
                    for top_k in sorted({max(1, min(experts-1, math.floor(ideal_k)+offset)) for offset in (0, 1)}):
                        yield dict(family="moe", d=d, layers=layers, heads=d//32,
                                   ffn_width=hidden, experts=experts, top_k=top_k, window=window)


def match_controls(subject: dict, activity: dict, vocab_size: int = 8192, *,
                   parameter_tolerance: float = 0.10, ratio_tolerance: float = 0.02,
                   window: int = 128) -> dict:
    """Finite three-way matching, failing closed on missing activity or mismatch.

    The MoE count is exact for equal-size experts with unique top-k dispatch;
    runtime calibration must still record its actual routes before admission.
    The returned nearest candidate is evidence when a match fails, not a pass.
    """
    spec = subject.get("model", subject)
    subject_id = subject.get("id", model_key(spec))
    if parameter_tolerance != 0.10 or ratio_tolerance != 0.02:
        raise ValueError("round-three matching tolerances are frozen at 10% and 0.02")
    counts = exact_parameter_counts(spec, vocab_size)
    result = {"subject_id": subject_id, "qualified": False, "controls": {},
              "parameter_tolerance": parameter_tolerance, "ratio_tolerance": ratio_tolerance,
              "search_version": "finite-control-grid-v1"}
    if not activity.get("qualified"):
        result["reasons"] = ["subject activity has no qualified per-token coverage"]
        return result
    total = counts["backbone_total_parameters"]
    active = activity.get("mean_active_backbone_parameters")
    if activity.get("backbone_total_parameters") != total or not isinstance(active, (int, float)) or not 0 < active <= total:
        result["reasons"] = ["activity identity/count does not agree with subject parameters"]
        return result
    ratio = active / total
    result["targets"] = dict(counts, mean_active_backbone_parameters=active, active_ratio=ratio)

    def choose(kind, candidates):
        evaluated = []
        for candidate in candidates:
            measured = control_parameter_counts(candidate, vocab_size)
            error_total = abs(measured["backbone_total_parameters"] - total)/total
            error_active = abs(measured["mean_active_backbone_parameters"] - active)/active
            error_ratio = abs(measured["active_ratio"] - ratio)
            relevant = error_total if kind == "dense_total" else error_active
            passed = relevant <= parameter_tolerance
            if kind == "moe_joint":
                relevant = max(error_total, error_active)
                passed = relevant <= parameter_tolerance and error_ratio <= ratio_tolerance
            # Passing rows precede failures. Capacity residuals dominate ties;
            # then use fewer layers/lower width and stable serialized identity.
            rank = (not passed, max(relevant/parameter_tolerance,
                                   error_ratio/ratio_tolerance if kind == "moe_joint" else 0),
                    relevant, candidate["layers"], candidate["d"], json.dumps(candidate, sort_keys=True))
            evaluated.append((rank, candidate, measured, error_total, error_active, error_ratio, passed))
        _, candidate, measured, error_total, error_active, error_ratio, passed = min(evaluated)
        actual = exact_parameter_counts(candidate, vocab_size)
        if actual["backbone_total_parameters"] != measured["backbone_total_parameters"]:
            raise RuntimeError("control formula disagrees with executable parameter count")
        return {"qualified": passed, "model": candidate, "model_key": model_key(candidate),
                "counts": dict(actual, mean_active_backbone_parameters=measured["mean_active_backbone_parameters"],
                               active_ratio=measured["active_ratio"]),
                "relative_total_error": error_total, "relative_active_error": error_active,
                "absolute_active_ratio_error": error_ratio, "candidates_examined": len(evaluated),
                "activity_requires_runtime_confirmation": True,
                "reason": None if passed else "nearest legal control exceeds frozen matching tolerance"}

    result["controls"]["dense_total"] = choose("dense_total", _dense_candidates(total, window))
    result["controls"]["dense_active"] = choose("dense_active", _dense_candidates(active, window))
    result["controls"]["moe_joint"] = choose("moe_joint", _moe_candidates(total, active, window))
    result["qualified"] = all(row["qualified"] for row in result["controls"].values())
    result["reasons"] = [] if result["qualified"] else [key for key, row in result["controls"].items() if not row["qualified"]]
    return result


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _control_tuple(spec):
    c = effective_model_identity(spec)
    return (c["family"], c["d"], c["layers"], c["heads"], c["ffn_width"], c["window"],
            c.get("experts", 0), c.get("top_k", 0))


def _control_spec(key):
    family, d, layers, heads, hidden, window, experts, top_k = key
    result = dict(family=family, d=d, layers=layers, heads=heads, ffn_width=hidden, window=window)
    if family == "moe":
        result.update(experts=experts, top_k=top_k)
    return result


def _control_errors(count, total, active):
    ct, ca = count["backbone_total_parameters"], count["mean_active_backbone_parameters"]
    # This product form is also used by gate3 for the independent measured
    # ratio audit. Subtracting two rounded ratios can move an exact boundary.
    return {"total_relative_error": abs(ct-total)/total,
            "active_relative_error": abs(ca-active)/active,
            "ratio_absolute_error": abs(ca*total-active*ct)/(total*ct)}


def _control_passes(kind, errors):
    if kind == "dense_total":
        return errors["total_relative_error"] <= .10
    if kind == "dense_active":
        return errors["active_relative_error"] <= .10
    return (errors["total_relative_error"] <= .10 and errors["active_relative_error"] <= .10
            and errors["ratio_absolute_error"] <= .02)


def _cover_candidates(candidates, required):
    """Deterministic weighted set cover; no validation/NLL input is consumed."""
    uncovered = required
    selected = []
    while uncovered:
        best = None
        reusable = any(row["available"] and row["coverage"] & uncovered for row in candidates)
        for row in candidates:
            gain = (row["coverage"] & uncovered).bit_count()
            if not gain or reusable and not row["available"]:
                continue
            if best is None:
                best = row, gain
                continue
            previous, previous_gain = best
            if reusable:
                wins = (-gain, row["cost"], row["key"]) < (-previous_gain, previous["cost"], previous["key"])
            else:
                lhs, rhs = gain*previous["cost"], previous_gain*row["cost"]
                wins = lhs > rhs or lhs == rhs and (-gain, row["cost"], row["key"]) < (
                    -previous_gain, previous["cost"], previous["key"])
            if wins:
                best = row, gain
        if best is None:
            break
        row, _ = best
        selected.append(row)
        uncovered &= ~row["coverage"]
    covered = required & ~uncovered
    # An early choice can become redundant. Removing it cannot weaken coverage
    # or create another training job. Prefer to retire new expensive choices.
    for row in sorted(selected, key=lambda r: (r["available"], -r["cost"], r["key"])):
        others = 0
        for candidate in selected:
            if candidate is not row:
                others |= candidate["coverage"]
        if others & covered == covered:
            selected.remove(row)
    return sorted(selected, key=lambda row: row["key"]), uncovered


def match_control_catalog(subjects: list[dict], activities_by_id: dict, vocab_size: int = 8192, *,
                          window: int = 128, available_controls=()) -> dict:
    """Globally reuse a finite legal control library against actual A3 inputs.

    Each activity must be the qualified, versioned last-sample aggregate from
    the prescribed A3 seeds, as emitted by ``round3.mean_activity``. Bare
    structural floors, initialization probes and hypothetical activity values
    are rejected. The caller owns the A3 evidence paths; gate3 subsequently
    re-reads them and the trained controls' actual A3 activity independently.

    Candidate dimensions/search intervals are unchanged from match_controls.
    Their union is covered jointly across dense-total and dense-active roles;
    MoE covers both targets and the absolute ratio tolerance. Existing eligible
    candidates, when supplied, have priority. New choices maximize newly
    covered relationships per estimated checkpoint byte, then coverage count,
    lower cost and the canonical structural tuple. No NLL or new forward is
    used. Available candidates are structural reuse hints, not proof of prior
    execution or learning qualification. ``qualified`` means all structural relationships have a candidate;
    it does not open the stage-B gate or certify control training/activity.
    """
    if type(vocab_size) is not int or vocab_size < 2 or type(window) is not int or window < 1:
        raise ValueError("positive window and integer vocab_size >= 2 required")
    ordered = sorted(subjects, key=lambda row: row["id"])
    ids = [row["id"] for row in ordered]
    result = {"schema": "pilot-r3-global-control-catalog-v1", "algorithm_version": CONTROL_MATCHER_VERSION,
              "qualified": False, "controls": [], "matches": [], "per_subject": {}, "reasons": [],
              "parameter_tolerance": .10, "ratio_tolerance": .02,
              "activity_requires_runtime_confirmation": True,
              "qualification_scope": "control structures matched to qualified subject A3 aggregates; independent control A3 gate audit still required",
              "catalog_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "subject_specs_sha256": _digest([{"id": row["id"], "model": row["model"]} for row in ordered]),
              "candidate_library_scope": "union of existing finite dense/MoE searches at subject total and observed A3 activity targets",
              "selection_rule": "reuse supplied eligible candidates; then new relationships/checkpoint-byte, more coverage, lower cost, canonical tuple; prune redundancy",
              "checkpoint_cost_scope": "12 bytes per unique parameter plus 16 MiB; selection cost only, not a measured storage admission"}
    if not ordered or len(ordered) > MAX_AUTHORIZED_SUBJECTS or len(set(ids)) != len(ids):
        result["reasons"].append(f"requires 1..{MAX_AUTHORIZED_SUBJECTS} distinct subject identities")
        return result
    if set(activities_by_id) != set(ids):
        result["reasons"].append("A3 activity mapping must cover exactly the requested subject IDs")
        return result
    targets = []
    effective = []
    for subject in ordered:
        sid, spec = subject["id"], subject["model"]
        if spec.get("family") != "tide":
            result["reasons"].append({"subject_id": sid, "reason": "subject is not Tide"})
            continue
        identity = model_key(spec)
        effective.append(identity)
        activity = activities_by_id[sid]
        valid = (isinstance(activity, dict) and activity.get("qualified") is True
                 and activity.get("version") == ACTIVITY_VERSION
                 and activity.get("aggregation") == "last prescribed sample of each fixed A3 seed"
                 and activity.get("method") == "actual-finish-controls-with-explicit-parameter-coverage"
                 and type(activity.get("measured_external_tokens")) is int and activity["measured_external_tokens"] > 0)
        if not valid:
            result["reasons"].append({"subject_id": sid, "reason": "qualified actual A3 aggregate required; estimates/probes are not admission evidence"})
            continue
        counts = exact_parameter_counts(spec, vocab_size)
        total, active = counts["backbone_total_parameters"], activity.get("mean_active_backbone_parameters")
        if (activity.get("backbone_total_parameters") != total or isinstance(active, bool)
                or not isinstance(active, (int, float)) or not math.isfinite(active) or not 0 < active <= total):
            result["reasons"].append({"subject_id": sid, "reason": "A3 activity/count does not match executable subject parameters"})
            continue
        if subject.get("model_key", identity) != identity or subject.get("effective_key", identity) != identity:
            result["reasons"].append({"subject_id": sid, "reason": "subject effective identity mismatch"})
            continue
        targets.append({"id": sid, "total": total, "active": active, "counts": counts})
    if len(set(effective)) != len(effective):
        result["reasons"].append("effective subject aliases are not distinct models")
    if result["reasons"]:
        return result
    pool = {}
    for target in targets:
        searches = (_dense_candidates(target["total"], window), _dense_candidates(target["active"], window),
                    _moe_candidates(target["total"], target["active"], window))
        for search in searches:
            for spec in search:
                key = _control_tuple(spec)
                if key not in pool:
                    count = control_parameter_counts(spec, vocab_size)
                    pool[key] = {"key": key, "spec": spec, "counts": count, "coverage": 0,
                                 "cost": count["whole_model_parameters"]*12+16*2**20,
                                 "available": False, "existing_id": None}
    existing_ids = {}
    for entry in available_controls:
        spec = entry.get("model", entry)
        key = _control_tuple(spec)
        if key not in pool:
            result["reasons"].append("available control is outside this finite candidate union")
            continue
        cid = entry.get("id")
        if cid and (cid in existing_ids and existing_ids[cid] != key or
                    pool[key]["existing_id"] not in (None, cid)):
            result["reasons"].append("conflicting available control identities")
            continue
        if cid:
            existing_ids[cid] = key
        pool[key].update(available=True, existing_id=cid)
    if result["reasons"]:
        return result
    result["input_sha256"] = _digest({"version": CONTROL_MATCHER_VERSION, "subjects": ordered,
        "activities": {sid: activities_by_id[sid] for sid in ids}, "vocab_size": vocab_size, "window": window,
        "available_controls": [{"model": _control_spec(key), "id": row["existing_id"]}
                               for key, row in sorted(pool.items()) if row["available"]]})
    # Index by target size first; this avoids a Cartesian all-candidate/all-
    # relationship scan while preserving the exact frozen predicates.
    dense_requirements = sorted((target[field], i*2+j) for i, target in enumerate(targets)
                                for j, field in enumerate(("total", "active")))
    moe_requirements = sorted((target["total"], i) for i, target in enumerate(targets))
    indices = {"transformer": dense_requirements, "moe": moe_requirements}
    sizes = {family: [value for value, _ in rows] for family, rows in indices.items()}
    for key, row in pool.items():
        family = key[0]
        total = row["counts"]["backbone_total_parameters"]
        low = math.nextafter(total/1.1, -math.inf)
        high = math.nextafter(total/.9, math.inf)
        left, right = bisect_left(sizes[family], low), bisect_right(sizes[family], high)
        for _, index in indices[family][left:right]:
            target = targets[index//2] if family == "transformer" else targets[index]
            kind = ("dense_total", "dense_active")[index%2] if family == "transformer" else "moe_joint"
            errors = _control_errors(row["counts"], target["total"], target["active"])
            if _control_passes(kind, errors):
                row["coverage"] |= 1 << index
    result["candidate_library_sha256"] = _digest([key for key in sorted(pool)])
    result["candidate_counts"] = {family: sum(key[0] == family for key in pool) for family in ("transformer", "moe")}
    chosen = []
    missing = {}
    for family, count in (("transformer", len(targets)*2), ("moe", len(targets))):
        rows = [row for key, row in pool.items() if key[0] == family and row["coverage"]]
        picked, uncovered = _cover_candidates(rows, (1 << count)-1)
        chosen.extend(picked)
        missing[family] = uncovered
    for row in chosen:
        spec = _control_spec(row["key"])
        actual = exact_parameter_counts(spec, vocab_size)
        if actual["backbone_total_parameters"] != row["counts"]["backbone_total_parameters"]:
            raise RuntimeError("control search formula disagrees with executable module count")
        key = model_key(spec)
        row["id"] = row["existing_id"] or f'control-{spec["family"]}-{key[:16]}'
        result["controls"].append({"id": row["id"], "group": spec["family"], "model": spec,
            "model_key": key, "effective_key": key, "counts": dict(actual,
                mean_active_backbone_parameters=row["counts"]["mean_active_backbone_parameters"],
                active_ratio=row["counts"]["active_ratio"]),
            "estimated_checkpoint_bytes": row["cost"], "reused_available_control": row["available"],
            "activity_requires_runtime_confirmation": True})
    controls_by_id = {row["id"]: row for row in result["controls"]}
    if len(controls_by_id) != len(result["controls"]):
        raise ValueError("selected control IDs collide")
    for i, target in enumerate(targets):
        local = {"subject_id": target["id"], "qualified": True, "controls": {}, "reasons": [],
                 "parameter_tolerance": .10, "ratio_tolerance": .02,
                 "targets": dict(target["counts"], mean_active_backbone_parameters=target["active"],
                                 active_ratio=target["active"]/target["total"])}
        for kind in ("dense_total", "dense_active", "moe_joint"):
            family, index = ("moe", i) if kind == "moe_joint" else ("transformer", 2*i+(kind == "dense_active"))
            candidates = [row for row in chosen if row["key"][0] == family and row["coverage"] & (1 << index)]
            if not candidates:
                reason = {"subject_id": target["id"], "kind": kind, "reason": "no candidate in finite union meets all frozen thresholds"}
                result["reasons"].append(reason)
                local["qualified"] = False
                local["reasons"].append(kind)
                result["matches"].append(dict(reason, qualified=False, control_id=None))
                continue
            def rank(row):
                errors = _control_errors(row["counts"], target["total"], target["active"])
                primary = errors["total_relative_error"] if kind == "dense_total" else errors["active_relative_error"]
                if kind == "moe_joint":
                    primary = max(errors["total_relative_error"]/.10, errors["active_relative_error"]/.10,
                                  errors["ratio_absolute_error"]/.02)
                return not row["available"], primary, row["cost"], row["key"]
            row = min(candidates, key=rank)
            errors = _control_errors(row["counts"], target["total"], target["active"])
            if not _control_passes(kind, errors):
                raise RuntimeError("selected control lost its qualifying relationship")
            result["matches"].append({"subject_id": target["id"], "kind": kind, "control_id": row["id"],
                "qualified": True, **errors, "activity_requires_runtime_confirmation": True})
            control = controls_by_id[row["id"]]
            local["controls"][kind] = {"qualified": True, "control_id": row["id"],
                "model": control["model"], "model_key": control["model_key"], "counts": control["counts"],
                "relative_total_error": errors["total_relative_error"],
                "relative_active_error": errors["active_relative_error"],
                "absolute_active_ratio_error": errors["ratio_absolute_error"],
                "activity_requires_runtime_confirmation": True, "reason": None}
        result["per_subject"][target["id"]] = local
    result["controls"].sort(key=lambda row: row["id"])
    result["qualified"] = not result["reasons"] and not any(missing.values())
    result["relationship_count"] = len(result["matches"])
    result["control_count"] = len(result["controls"])
    result["estimated_checkpoint_bytes"] = sum(row["estimated_checkpoint_bytes"] for row in result["controls"])
    if result["relationship_count"] != 3*len(targets) or result["control_count"] > 3*len(targets):
        raise RuntimeError("finite control coverage/count invariant violated")
    result["resolved_catalog_sha256"] = _digest({"controls": result["controls"], "matches": result["matches"]})
    return result

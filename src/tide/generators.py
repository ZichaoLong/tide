"""Deterministic development corpus for expanded SettleGraph Plans.

This module deliberately produces ordinary validated :class:`~tide.plan.Plan`
objects.  It is a development corpus, not a qualification fixture bundle: it
does not serialize parameters, inputs, expected traces, or evidence artifacts.
The default corpus broadens repeatable differential testing; callers may
materialize selected cases through :mod:`tide.fixtures`, but the complete
qualification corpus and coverage accounting described in ``docs`` remain
future work.
"""

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass
from typing import Callable, FrozenSet, List, Mapping, Sequence, Set, Tuple

from .builders import (
    build_chain,
    build_diamond,
    build_mixed_regions,
    build_multi_entry_terminal,
    build_single_layer,
    build_singleton,
    build_small_hb,
    build_unequal_path,
)
from .plan import EdgeSpec, NodeSpec, Plan, RegionSpec


DEFAULT_PLAN_CORPUS_SEED = 20_260_903
DEFAULT_PLAN_CORPUS_SIZE = 48
DEFAULT_INVALID_PLAN_CORPUS_SIZE = 24
CORE_V1_CANDIDATE_CORPUS_SEED = 20_260_903
CORE_V1_CANDIDATE_CORPUS_SIZE = 256
CORE_V1_CANDIDATE_VJP_SIZE = 64

_RUNTIME_DTYPES = {
    "hidden": "runtime",
    "parameter": "runtime",
    "state": "runtime",
    "readout": "runtime",
}
_PROFILE_TIMINGS = (
    ("N", "content"),
    ("SD", "content"),
    ("SD", "pre"),
    ("BO", "content"),
    ("BO", "pre"),
    ("BO", "post"),
)
_STATE_KINDS = ("ema", "gdn", "attention_window")
_AGGREGATES = ("mean", "edge_softmax", "edge_linear_mean")
_EMITS = ("hard", "hst", "softp")
_COMPUTES = ("identity", "affine_residual", "double_residual_swiglu")
_SCORES = ("fixed", "constant", "read_sum", "linear", "mlp")


@dataclass(frozen=True)
class PlanCorpusCase:
    """One reproducible Plan plus non-semantic generation metadata."""

    case_id: str
    motif: str
    ordinal: int
    generation_seed: int
    parameter_seed: int
    input_seed: int
    plan: Plan
    features: FrozenSet[str]
    vjp: bool = False


@dataclass(frozen=True)
class InvalidPlanCorpusCase:
    """One deterministic single-mutation Plan expected to fail validation."""

    case_id: str
    mutation_kind: str
    generation_seed: int
    base_plan_hash: str
    plan: Plan
    expected_codes: Tuple[str, ...]


@dataclass(frozen=True)
class _Template:
    motif: str
    variant: str
    factory: Callable[[int], Plan]


def _default_node(
    node_id: str,
    region_id: str,
    d_model: int,
    *,
    forced_active: bool = False,
) -> NodeSpec:
    return NodeSpec(
        node_id=node_id,
        region_id=region_id,
        hidden_shape=(d_model,),
        selector_read_shape=(d_model,),
        forced_active=forced_active,
        aggregate={
            "type": "mean",
            "formula_id": "agg.mean.v1",
            "output_shape": [d_model],
        },
        update={"type": "none", "formula_id": "update.none.v1"},
        selector_read={
            "type": "content",
            "formula_id": "read.selector.content.v1",
            "output_shape": [d_model],
        },
        ffn_read={
            "type": "zero",
            "formula_id": "read.ffn.zero.v1",
            "output_shape": [d_model],
        },
        node_compute={
            "type": "identity",
            "formula_id": "node.identity.v1",
            "output_shape": [d_model],
        },
        emit={
            "type": "hard",
            "formula_id": "emit.hard.v1",
            "output_shape": [d_model],
        },
    )


def _default_region(
    region_id: str,
    node_ids: Sequence[str],
    *,
    k: int,
    forced: bool = False,
) -> RegionSpec:
    ordered = tuple(sorted(node_ids))
    requested = 1 if forced else k
    return RegionSpec(
        region_id=region_id,
        node_ids=ordered,
        profile="N",
        selector_timing="content",
        k_max=requested,
        k_requested={
            "type": "fixed",
            "formula_id": "k.fixed.v1",
            "value": requested,
        },
        score={
            "type": "fixed",
            "formula_id": "score.fixed-by-node.v1",
            "values_by_node": {
                node_id: float(index) for index, node_id in enumerate(ordered)
            },
        },
        selector_context={"type": "none", "formula_id": "context.none.v1"},
        selector_history={"type": "none", "formula_id": "history.none.v1"},
    )


def _forced_backbone_base(d_model: int) -> Plan:
    """Build an optional branch beside a two-hop forced terminal backbone."""

    descriptions = (
        ("node.root", "region.root", True),
        ("node.optional.a", "region.optional", False),
        ("node.optional.b", "region.optional", False),
        ("node.backbone.0", "region.backbone.0", True),
        ("node.backbone.1", "region.backbone.1", True),
        ("node.out", "region.out", True),
    )
    nodes = tuple(
        _default_node(
            node_id,
            region_id,
            d_model,
            forced_active=forced,
        )
        for node_id, region_id, forced in descriptions
    )
    regions = (
        _default_region("region.root", ("node.root",), k=1, forced=True),
        _default_region(
            "region.optional",
            ("node.optional.a", "node.optional.b"),
            k=1,
        ),
        _default_region(
            "region.backbone.0", ("node.backbone.0",), k=1, forced=True
        ),
        _default_region(
            "region.backbone.1", ("node.backbone.1",), k=1, forced=True
        ),
        _default_region("region.out", ("node.out",), k=1, forced=True),
    )
    endpoint_pairs = (
        ("node.root", "node.optional.a"),
        ("node.root", "node.optional.b"),
        ("node.root", "node.backbone.0"),
        ("node.optional.a", "node.out"),
        ("node.optional.b", "node.out"),
        ("node.backbone.0", "node.backbone.1"),
        ("node.backbone.1", "node.out"),
    )
    edges = tuple(
        EdgeSpec(f"edge.{index:04d}", source, target)
        for index, (source, target) in enumerate(sorted(endpoint_pairs))
    )
    return Plan(
        plan_id="forced-backbone",
        d_model=d_model,
        dtype_roles=_RUNTIME_DTYPES,
        nodes=nodes,
        edges=edges,
        regions=regions,
        output_aggregate={
            "type": "mean",
            "formula_id": "agg.mean.v1",
            "output_shape": [d_model],
        },
        builder={
            "name": "development-forced-backbone",
            "version": "1",
            "config": {"d_model": d_model},
        },
    ).validate()


def _partition_layer(
    layer: int, node_ids: Sequence[str], rng: random.Random
) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
    """Partition one independent layer into stable regions of width 1 or 2."""

    partitions: List[Tuple[str, Tuple[str, ...]]] = []
    start = 0
    region_index = 0
    while start < len(node_ids):
        remaining = len(node_ids) - start
        width = 1 if remaining == 1 else 1 + rng.randrange(2)
        members = tuple(node_ids[start : start + min(width, remaining)])
        partitions.append((f"region.l{layer:02d}.r{region_index:02d}", members))
        start += len(members)
        region_index += 1
    return tuple(partitions)


def _generated_layered_dag(
    rng: random.Random, d_model: int, variant: int
) -> Plan:
    """Generate a bounded small DAG with every node on an entry-terminal path."""

    depth = 3 + (variant % 2)
    widths = [2]
    widths.extend(2 + rng.randrange(3) for _ in range(depth - 1))
    widths.append(2 + (variant % 2))
    layers = tuple(
        tuple(f"node.l{layer:02d}.n{index:02d}" for index in range(width))
        for layer, width in enumerate(widths)
    )

    region_records: List[Tuple[str, Tuple[str, ...]]] = []
    region_by_node = {}
    for layer, node_ids in enumerate(layers):
        for region_id, members in _partition_layer(layer, node_ids, rng):
            region_records.append((region_id, members))
            for node_id in members:
                region_by_node[node_id] = region_id

    nodes = tuple(
        _default_node(node_id, region_by_node[node_id], d_model)
        for node_ids in layers
        for node_id in node_ids
    )
    regions = tuple(
        _default_region(
            region_id,
            members,
            k=1 if len(members) == 1 else 1 + rng.randrange(len(members)),
        )
        for region_id, members in region_records
    )

    endpoints = set()
    for layer in range(len(layers) - 1):
        sources = layers[layer]
        targets = layers[layer + 1]
        # Every source has a child and every target has a parent.  Applying
        # this at each adjacent layer proves every node is on a complete path.
        for index, source in enumerate(sources):
            endpoints.add((source, targets[index % len(targets)]))
        for index, target in enumerate(targets):
            endpoints.add((sources[index % len(sources)], target))
        for source in sources:
            for target in targets:
                if rng.random() < 0.18:
                    endpoints.add((source, target))
    for layer in range(len(layers) - 2):
        if rng.random() < 0.7:
            sources = layers[layer]
            targets = layers[layer + 2]
            endpoints.add(
                (sources[rng.randrange(len(sources))], targets[rng.randrange(len(targets))])
            )

    edges = tuple(
        EdgeSpec(f"edge.{index:05d}", source, target)
        for index, (source, target) in enumerate(sorted(endpoints))
    )
    return Plan(
        plan_id=f"generated-layered-{variant}",
        d_model=d_model,
        dtype_roles=_RUNTIME_DTYPES,
        nodes=nodes,
        edges=edges,
        regions=regions,
        output_aggregate={
            "type": "mean",
            "formula_id": "agg.mean.v1",
            "output_shape": [d_model],
        },
        builder={
            "name": "development-layered-dag",
            "version": "1",
            "config": {
                "variant": variant,
                "widths": widths,
                "d_model": d_model,
            },
        },
    ).validate()


def _aggregate_config(kind: str, d_model: int) -> Mapping[str, object]:
    if kind == "mean":
        return {
            "type": "mean",
            "formula_id": "agg.mean.v1",
            "output_shape": [d_model],
        }
    if kind == "edge_softmax":
        return {
            "type": "edge_softmax",
            "formula_id": "TEST-AGG-EDGE-SOFTMAX-V1",
            "output_shape": [d_model],
        }
    if kind == "edge_linear_mean":
        return {
            "type": "edge_linear_mean",
            "formula_id": "TEST-AGG-EDGE-AFFINE-MEAN-V1",
            "bias": True,
            "output_shape": [d_model],
        }
    raise AssertionError(kind)


def _state_config(
    kind: str, d_model: int, variant: int
) -> Tuple[Tuple[int, ...], Mapping[str, object], Mapping[str, object]]:
    if kind == "none":
        return (
            (),
            {"type": "none", "formula_id": "update.none.v1"},
            {
                "type": "zero",
                "formula_id": "read.ffn.zero.v1",
                "output_shape": [d_model],
            },
        )
    if kind == "ema":
        state_dim = d_model + (variant % 2)
        return (
            (state_dim,),
            {
                "type": "ema",
                "formula_id": "state.ema.v1",
                "state_dim": state_dim,
                "decay": 0.35 + 0.05 * (variant % 5),
                "learnable_decay": False,
                "state_shape": [state_dim],
            },
            {
                "type": "state_default",
                "formula_id": "read.ffn.ema.v1",
                "output_shape": [d_model],
            },
        )
    if kind == "gdn":
        key_dim = 2
        value_dim = d_model
        return (
            (key_dim, value_dim),
            {
                "type": "gdn",
                "formula_id": "state.gdn.v1",
                "key_dim": key_dim,
                "value_dim": value_dim,
                "norm_eps": 1e-12,
                "state_shape": [key_dim, value_dim],
            },
            {
                "type": "state_default",
                "formula_id": "read.ffn.gdn.v1",
                "output_shape": [d_model],
            },
        )
    if kind == "attention_window":
        key_dim = 2
        value_dim = d_model
        window = 2 + (variant % 2)
        return (
            (window, key_dim, value_dim),
            {
                "type": "attention_window",
                "formula_id": "state.attention-window.v1",
                "key_dim": key_dim,
                "value_dim": value_dim,
                "window": window,
                "norm_eps": 1e-12,
                "state_shape": [window, key_dim, value_dim],
            },
            {
                "type": "state_default",
                "formula_id": "read.ffn.attention-window.v1",
                "output_shape": [d_model],
            },
        )
    raise AssertionError(kind)


def _selector_read_config(
    *,
    timing: str,
    state_kind: str,
    d_model: int,
    variant: int,
) -> Tuple[Tuple[int, ...], Mapping[str, object]]:
    if timing == "content":
        choice = variant % 3
        if choice == 0:
            return (
                (d_model,),
                {
                    "type": "content",
                    "formula_id": "read.selector.content.v1",
                    "out_dim": d_model,
                    "output_shape": [d_model],
                },
            )
        if choice == 1:
            return (
                (1,),
                {
                    "type": "content_norm",
                    "formula_id": "read.selector.content-rms.v1",
                    "out_dim": 1,
                    "output_shape": [1],
                },
            )
        read_dim = min(2, d_model)
        return (
            (read_dim,),
            {
                "type": "content_linear",
                "formula_id": "TEST-READ-PROJ-V1",
                "out_dim": read_dim,
                "output_shape": [read_dim],
            },
        )

    read_dim = 1 + (variant % min(2, d_model))
    if state_kind == "attention_window" or variant % 2:
        read_type = "content_state_summary_linear"
        formula_id = "TEST-READ-STATE-RMS-SUMMARY-PROJ-V1"
    else:
        read_type = "content_state_linear"
        formula_id = "TEST-READ-PROJ-V1"
    return (
        (read_dim,),
        {
            "type": read_type,
            "formula_id": formula_id,
            "out_dim": read_dim,
            "output_shape": [read_dim],
        },
    )


def _compute_config(kind: str, d_model: int) -> Mapping[str, object]:
    if kind == "identity":
        return {
            "type": "identity",
            "formula_id": "node.identity.v1",
            "output_shape": [d_model],
        }
    if kind == "affine_residual":
        return {
            "type": "affine_residual",
            "formula_id": "TEST-NODE-AFFINE-V1",
            "bias": True,
            "output_shape": [d_model],
        }
    if kind == "double_residual_swiglu":
        return {
            "type": "double_residual_swiglu",
            "formula_id": "TEST-NODE-SWIGLU-V1",
            "hidden_dim": 2 * d_model,
            "bias": True,
            "output_shape": [d_model],
        }
    raise AssertionError(kind)


def _emit_config(kind: str, d_model: int, variant: int) -> Mapping[str, object]:
    if kind == "hard":
        return {
            "type": "hard",
            "formula_id": "emit.hard.v1",
            "output_shape": [d_model],
        }
    if kind == "hst":
        return {
            "type": "hst",
            "formula_id": "emit.hst.v1",
            "zeta": 0.75 + 0.05 * (variant % 4),
            "output_shape": [d_model],
        }
    if kind == "softp":
        return {
            "type": "softp",
            "formula_id": "emit.softp.v1",
            "output_shape": [d_model],
        }
    raise AssertionError(kind)


def _score_config(
    kind: str,
    node_ids: Sequence[str],
    read_dim: int,
    variant: int,
) -> Mapping[str, object]:
    if kind == "fixed":
        center = 0.5 * (len(node_ids) - 1)
        tie = variant % 4 == 0
        return {
            "type": "fixed",
            "formula_id": "score.fixed-by-node.v1",
            "values_by_node": {
                node_id: 0.25 if tie else 0.35 * (index - center)
                for index, node_id in enumerate(node_ids)
            },
        }
    if kind == "constant":
        return {
            "type": "constant",
            "formula_id": "score.constant.v1",
            "value": 0.1 * ((variant % 5) - 2),
        }
    if kind == "read_sum":
        return {
            "type": "read_sum",
            "formula_id": "score.read-sum.v1",
            "context_dim": 0,
        }
    if kind == "linear":
        return {
            "type": "linear",
            "formula_id": "TEST-SCORE-LINEAR-V1",
            "bias": True,
            "shared_parameters": False,
            "context_dim": 0,
        }
    if kind == "mlp":
        return {
            "type": "mlp",
            "formula_id": "TEST-SCORE-MLP-V1",
            "hidden_dim": max(4, read_dim),
            "bias": True,
            "shared_parameters": False,
            "context_dim": 0,
        }
    raise AssertionError(kind)


def _configure_plan(
    base: Plan,
    *,
    case_id: str,
    motif: str,
    generation_seed: int,
    ordinal: int,
    offsets: Mapping[str, int],
    k_mode_hint: str,
    vjp: bool,
) -> Plan:
    d_model = base.d_model
    node_lookup = {node.node_id: node for node in base.nodes}
    configured_nodes = {}
    configured_regions = []
    mix_regions = motif in {
        "mixed-regions",
        "forced-backbone",
        "small-hb",
        "generated-dag",
    }

    for region_index, region in enumerate(base.regions):
        profile_index = (
            ordinal
            + offsets["profile"]
            + (region_index if mix_regions else 0)
        ) % len(_PROFILE_TIMINGS)
        profile, timing = _PROFILE_TIMINGS[profile_index]
        state_kind = (
            "none"
            if profile == "N"
            else _STATE_KINDS[
                (ordinal + region_index + offsets["state"]) % len(_STATE_KINDS)
            ]
        )
        selector_shape, selector_config = _selector_read_config(
            timing=timing,
            state_kind=state_kind,
            d_model=d_model,
            variant=ordinal + region_index,
        )

        for local_index, node_id in enumerate(region.node_ids):
            node = node_lookup[node_id]
            node_variant = ordinal + region_index + local_index
            state_shape, update, ffn_read = _state_config(
                state_kind, d_model, node_variant
            )
            aggregate_kind = _AGGREGATES[
                (node_variant + offsets["aggregate"]) % len(_AGGREGATES)
            ]
            compute_kind = (
                _COMPUTES[1 + (node_variant % 2)]
                if vjp
                else _COMPUTES[
                    (node_variant + offsets["compute"]) % len(_COMPUTES)
                ]
            )
            emit_kind = (
                _EMITS[1 + (node_variant % 2)]
                if vjp
                else _EMITS[(node_variant + offsets["emit"]) % len(_EMITS)]
            )
            epsilon = 1e-6 * (1.0 + (ordinal + 1) / 1000.0)
            configured_nodes[node_id] = dataclasses.replace(
                node,
                state_shape=state_shape,
                state_owner=node_id if state_shape else None,
                selector_read_shape=selector_shape,
                input_norm={
                    "type": "rmsnorm",
                    "formula_id": "norm.rms.v1",
                    "eps": epsilon,
                },
                ffn_norm={
                    "type": "rmsnorm",
                    "formula_id": "norm.rms.v1",
                    "eps": epsilon * 1.125,
                },
                aggregate=_aggregate_config(aggregate_kind, d_model),
                update=update,
                selector_read=selector_config,
                ffn_read=ffn_read,
                node_compute=_compute_config(compute_kind, d_model),
                emit=_emit_config(emit_kind, d_model, node_variant),
            )

        forced = any(configured_nodes[node_id].forced_active for node_id in region.node_ids)
        k_mode = "fixed" if forced else k_mode_hint
        if k_mode == "input":
            k_requested = {
                "type": "input",
                "formula_id": "k.input.v1",
                "field": "requested_k",
                "minimum": 1,
                "maximum": region.k_max,
            }
        else:
            fixed_value = int(region.k_requested.get("value", region.k_max))
            k_requested = {
                "type": "fixed",
                "formula_id": "k.fixed.v1",
                "value": fixed_value,
            }
        score_kind = (
            ("linear", "mlp")[region_index % 2]
            if vjp
            else _SCORES[
                (ordinal + region_index + offsets["score"]) % len(_SCORES)
            ]
        )
        configured_regions.append(
            dataclasses.replace(
                region,
                profile=profile,
                selector_timing=timing,
                k_requested=k_requested,
                score=_score_config(
                    score_kind,
                    region.node_ids,
                    selector_shape[0],
                    ordinal + region_index,
                ),
            )
        )

    output_kind = "node_softmax" if (ordinal + offsets["output"]) % 2 else "mean"
    output_aggregate = (
        {
            "type": "node_softmax",
            "formula_id": "TEST-AGG-TERMINAL-SOFTMAX-V1",
            "output_shape": [d_model],
        }
        if output_kind == "node_softmax"
        else {
            "type": "mean",
            "formula_id": "agg.mean.v1",
            "output_shape": [d_model],
        }
    )
    return dataclasses.replace(
        base,
        plan_id=case_id,
        nodes=tuple(configured_nodes[node.node_id] for node in base.nodes),
        regions=tuple(configured_regions),
        output_aggregate=output_aggregate,
        builder={
            "name": "deterministic-plan-corpus",
            "version": "1",
            "config": {
                "seed": generation_seed,
                "ordinal": ordinal,
                "motif": motif,
                "source_plan": base.plan_id,
            },
        },
    ).validate()


def _features(plan: Plan, motif: str, *, vjp: bool) -> FrozenSet[str]:
    features = {f"motif:{motif}", "topology:dag"}
    if plan.topology_kind == "hb":
        features.add("topology:hb")
    if len(plan.entry_node_ids) > 1:
        features.add("boundary:multi-entry")
    if len(plan.terminal_node_ids) > 1:
        features.add("boundary:multi-terminal")
    if any(node.forced_active for node in plan.nodes):
        features.add("routing:forced-active")
    if vjp:
        features.add("direction:vjp")

    for node in plan.nodes:
        features.add(f"state:{node.update['type']}")
        features.add(f"aggregate:{node.aggregate['type']}")
        features.add(f"emit:{node.emit['type']}")
        features.add(f"compute:{node.node_compute['type']}")
    for region in plan.regions:
        features.add(f"profile:{region.profile}/{region.selector_timing}")
        features.add(f"score:{region.score['type']}")
        features.add(f"k:{region.k_requested['type']}")
        if region.k_requested["type"] == "fixed":
            requested = int(region.k_requested["value"])
            if requested == 1:
                features.add("budget:top-1")
            if requested == 2:
                features.add("budget:top-2")
            if requested == len(region.node_ids):
                features.add("budget:all")
    features.add(f"output-aggregate:{plan.output_aggregate['type']}")
    return frozenset(features)


def _templates() -> Tuple[_Template, ...]:
    return (
        _Template("singleton", "forced", lambda width: build_singleton(d_model=width)),
        _Template(
            "single-layer-r2",
            "top-1",
            lambda width: build_single_layer(receiver_count=2, k=1, d_model=width),
        ),
        _Template(
            "single-layer-r2",
            "top-2-all",
            lambda width: build_single_layer(receiver_count=2, k=2, d_model=width),
        ),
        _Template(
            "single-layer-r8",
            "top-1",
            lambda width: build_single_layer(receiver_count=8, k=1, d_model=width),
        ),
        _Template(
            "single-layer-r8",
            "top-2",
            lambda width: build_single_layer(receiver_count=8, k=2, d_model=width),
        ),
        _Template(
            "single-layer-r8",
            "all",
            lambda width: build_single_layer(receiver_count=8, k=8, d_model=width),
        ),
        _Template("chain", "length-3", lambda width: build_chain(length=3, d_model=width)),
        _Template("chain", "length-5", lambda width: build_chain(length=5, d_model=width)),
        _Template(
            "diamond", "top-1", lambda width: build_diamond(d_model=width, branch_k=1)
        ),
        _Template(
            "diamond", "all", lambda width: build_diamond(d_model=width, branch_k=2)
        ),
        _Template("unequal-path", "base", lambda width: build_unequal_path(d_model=width)),
        _Template(
            "multi-entry-terminal",
            "base",
            lambda width: build_multi_entry_terminal(d_model=width),
        ),
        _Template("mixed-regions", "base", lambda width: build_mixed_regions(d_model=width)),
        _Template("forced-backbone", "base", _forced_backbone_base),
        _Template("small-hb", "base", lambda width: build_small_hb(d_model=width)),
    )


def generate_plan_corpus(
    seed: int = DEFAULT_PLAN_CORPUS_SEED,
) -> Tuple[PlanCorpusCase, ...]:
    """Return 48 deterministic, validated development cases for ``seed``.

    The topology motifs are repeated with different compatible local formulas.
    Three additional bounded layered DAGs exercise generation rather than only
    mutation of the readable builders.  No module-global random state is read
    or changed.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("plan corpus seed must be a nonnegative integer")
    rng = random.Random(seed)
    scheduled: List[Tuple[_Template, int]] = [
        (template, repetition)
        for repetition in range(3)
        for template in _templates()
    ]
    rng.shuffle(scheduled)

    offsets = {
        "profile": rng.randrange(len(_PROFILE_TIMINGS)),
        "state": rng.randrange(len(_STATE_KINDS)),
        "aggregate": rng.randrange(len(_AGGREGATES)),
        "emit": rng.randrange(len(_EMITS)),
        "compute": rng.randrange(len(_COMPUTES)),
        "score": rng.randrange(len(_SCORES)),
        "output": rng.randrange(2),
    }
    vjp_ordinals = frozenset((5, 13, 21, 29, 37, 45))
    cases: List[PlanCorpusCase] = []

    for ordinal in range(DEFAULT_PLAN_CORPUS_SIZE):
        d_model = 2 + ((ordinal + rng.randrange(2)) % 2)
        if ordinal < len(scheduled):
            template, repetition = scheduled[ordinal]
            motif = template.motif
            variant = f"{template.variant}-r{repetition}"
            base = template.factory(d_model)
            k_mode_hint = "input" if repetition == 1 else "fixed"
        else:
            variant_index = ordinal - len(scheduled)
            motif = "generated-dag"
            variant = f"layered-{variant_index}"
            base = _generated_layered_dag(rng, d_model, variant_index)
            k_mode_hint = "input" if variant_index % 2 else "fixed"

        case_id = f"corpus.{ordinal:03d}.{motif}.{variant}"
        vjp = ordinal in vjp_ordinals
        plan = _configure_plan(
            base,
            case_id=case_id,
            motif=motif,
            generation_seed=seed,
            ordinal=ordinal,
            offsets=offsets,
            k_mode_hint=k_mode_hint,
            vjp=vjp,
        )
        cases.append(
            PlanCorpusCase(
                case_id=case_id,
                motif=motif,
                ordinal=ordinal,
                generation_seed=seed,
                parameter_seed=rng.getrandbits(63),
                input_seed=rng.getrandbits(63),
                plan=plan,
                features=_features(plan, motif, vjp=vjp),
                vjp=vjp,
            )
        )

    assert len(cases) == DEFAULT_PLAN_CORPUS_SIZE
    return tuple(cases)


_CORE_V1_MANUAL_MOTIFS = (
    "singleton",
    "single-layer-r2",
    "single-layer-r8",
    "chain",
    "diamond",
    "unequal-path",
    "multi-entry-terminal",
    "mixed-regions",
    "forced-backbone",
    "small-hb",
)


def _core_v1_candidate_manual_base(
    motif: str, variant: int, d_model: int
) -> Tuple[Plan, str]:
    """Build one readable fixed-slot topology for the larger candidate set."""

    if motif == "singleton":
        return build_singleton(d_model=d_model), "forced"
    if motif == "single-layer-r2":
        k = (1, 2)[variant % 2]
        return (
            build_single_layer(receiver_count=2, k=k, d_model=d_model),
            f"k{k}",
        )
    if motif == "single-layer-r8":
        k = (1, 2, 8)[variant % 3]
        return (
            build_single_layer(receiver_count=8, k=k, d_model=d_model),
            f"k{k}",
        )
    if motif == "chain":
        length = 3 + (variant % 3)
        return build_chain(length=length, d_model=d_model), f"length-{length}"
    if motif == "diamond":
        branch_k = 1 + (variant % 2)
        return (
            build_diamond(d_model=d_model, branch_k=branch_k),
            f"k{branch_k}",
        )
    if motif == "unequal-path":
        return build_unequal_path(d_model=d_model), "base"
    if motif == "multi-entry-terminal":
        return build_multi_entry_terminal(d_model=d_model), "base"
    if motif == "mixed-regions":
        return build_mixed_regions(d_model=d_model), "base"
    if motif == "forced-backbone":
        return _forced_backbone_base(d_model), "base"
    if motif == "small-hb":
        return build_small_hb(d_model=d_model), "base"
    raise AssertionError(motif)


def _core_v1_candidate_rng_seed(
    seed: int, ordinal: int, retry: int = 0
) -> int:
    """Mix a case-local seed without reading process-global RNG state."""

    mask = (1 << 64) - 1
    value = (
        (seed & mask)
        ^ (((ordinal + 1) * 0x9E3779B97F4A7C15) & mask)
        ^ (((retry + 1) * 0xD1B54A32D192ED03) & mask)
    )
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & mask
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


def _topology_signature(plan: Plan) -> Tuple[object, ...]:
    """Return a formula-independent identity for generated-topology retries."""

    return (
        plan.topology_kind,
        plan.entry_node_ids,
        plan.terminal_node_ids,
        tuple(
            (node.node_id, node.region_id, node.forced_active)
            for node in plan.nodes
        ),
        tuple(
            (edge.edge_id, edge.source, edge.target, edge.label)
            for edge in plan.edges
        ),
        tuple(
            (
                region.region_id,
                region.node_ids,
                region.control_dependencies,
                region.line,
                region.phase,
            )
            for region in plan.regions
        ),
    )


def generate_core_v1_candidate_corpus(
    seed: int = CORE_V1_CANDIDATE_CORPUS_SEED,
) -> Tuple[PlanCorpusCase, ...]:
    """Return 256 deterministic fixed-K executor-equivalence candidates.

    The fixed slots follow the eleven topology motifs in the core-v1 plan:
    sixteen cases for each readable motif and ninety-six bounded generated
    DAGs.  Exactly sixty-four cases are marked for VJP comparison.  These are
    validated logical Plans and reproducible seeds, not serialized fixture
    bundles or a claim that any qualification capability cell has passed.

    This entry point is intentionally independent of :func:`generate_plan_corpus`;
    changing its seed, size, schedule, or identity does not alter the default
    48-case development corpus.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(
            "core-v1 candidate corpus seed must be a nonnegative integer"
        )

    schedule_rng = random.Random(seed ^ 0xC0E5_1DAD)
    offsets = {
        "profile": schedule_rng.randrange(len(_PROFILE_TIMINGS)),
        "state": schedule_rng.randrange(len(_STATE_KINDS)),
        "aggregate": schedule_rng.randrange(len(_AGGREGATES)),
        "emit": schedule_rng.randrange(len(_EMITS)),
        "compute": schedule_rng.randrange(len(_COMPUTES)),
        "score": schedule_rng.randrange(len(_SCORES)),
        "output": schedule_rng.randrange(2),
    }
    metadata_rng = random.Random(seed ^ 0x5EED_256)
    d_models = (2, 3, 4, 7)
    cases: List[PlanCorpusCase] = []
    generated_signatures: Set[Tuple[object, ...]] = set()

    for ordinal in range(CORE_V1_CANDIDATE_CORPUS_SIZE):
        d_model = d_models[ordinal % len(d_models)]
        retry = 0
        if ordinal < 16 * len(_CORE_V1_MANUAL_MOTIFS):
            motif = _CORE_V1_MANUAL_MOTIFS[ordinal // 16]
            variant_index = ordinal % 16
            base, variant = _core_v1_candidate_manual_base(
                motif, variant_index, d_model
            )
        else:
            motif = "generated-dag"
            variant_index = ordinal - 16 * len(_CORE_V1_MANUAL_MOTIFS)
            while True:
                local_rng = random.Random(
                    _core_v1_candidate_rng_seed(seed, ordinal, retry)
                )
                base = _generated_layered_dag(
                    local_rng,
                    d_model,
                    variant_index + retry * CORE_V1_CANDIDATE_CORPUS_SIZE,
                )
                signature = _topology_signature(base)
                if signature not in generated_signatures:
                    generated_signatures.add(signature)
                    break
                retry += 1
            variant = f"layered-{variant_index}-retry-{retry}"

        case_id = f"core-v1-candidate.ql-{ordinal:04d}.{motif}.{variant}"
        vjp = ordinal % 4 == 0
        plan = _configure_plan(
            base,
            case_id=case_id,
            motif=motif,
            generation_seed=seed,
            ordinal=ordinal,
            offsets=offsets,
            k_mode_hint="fixed",
            vjp=vjp,
        )
        plan = dataclasses.replace(
            plan,
            builder={
                "name": "core-v1-executor-equivalence-candidate",
                "version": "1",
                "config": {
                    "seed": seed,
                    "ordinal": ordinal,
                    "motif": motif,
                    "source_plan": base.plan_id,
                    "variant": variant,
                    "topology_retry": retry,
                },
            },
        ).validate()
        candidate_features = {
            "corpus:core-v1-candidate",
            f"shape:d{d_model}",
        }
        for node in plan.nodes:
            candidate_features.add(f"selector-read:{node.selector_read['type']}")
            candidate_features.add(f"ffn-read:{node.ffn_read['formula_id']}")
            candidate_features.add(f"input-norm:{node.input_norm['formula_id']}")
            candidate_features.add(f"ffn-norm:{node.ffn_norm['formula_id']}")
        features = _features(plan, motif, vjp=vjp) | frozenset(
            candidate_features
        )
        cases.append(
            PlanCorpusCase(
                case_id=case_id,
                motif=motif,
                ordinal=ordinal,
                generation_seed=seed,
                parameter_seed=metadata_rng.getrandbits(63),
                input_seed=metadata_rng.getrandbits(63),
                plan=plan,
                features=features,
                vjp=vjp,
            )
        )

    if len(cases) != CORE_V1_CANDIDATE_CORPUS_SIZE:
        raise AssertionError("core-v1 candidate schedule has the wrong size")
    if sum(case.vjp for case in cases) != CORE_V1_CANDIDATE_VJP_SIZE:
        raise AssertionError("core-v1 candidate VJP schedule has the wrong size")
    plan_hashes = {case.plan.canonical_hash() for case in cases}
    if len(plan_hashes) != len(cases):
        raise AssertionError("core-v1 candidate logical Plan hashes are not unique")
    if any(
        region.k_requested.get("type") != "fixed"
        or region.k_requested.get("value") != region.k_max
        for case in cases
        for region in case.plan.regions
    ):
        raise AssertionError("core-v1 candidate corpus contains a non-fixed K")
    return tuple(cases)


def _invalid_plan_mutations(base: Plan) -> Mapping[str, Tuple[Plan, Tuple[str, ...]]]:
    first_node = base.nodes[0]
    first_region = base.regions[0]
    cycle_edge = EdgeSpec(
        "edge.mutant.cycle",
        base.terminal_node_ids[0],
        base.entry_node_ids[0],
    )
    duplicate_edge = dataclasses.replace(base.edges[0])

    member = first_region.node_ids[0]
    intra_region = dataclasses.replace(
        base,
        edges=base.edges
        + (EdgeSpec("edge.mutant.intra-region", member, member),),
    )
    wrong_shape = dataclasses.replace(
        base,
        nodes=(
            dataclasses.replace(
                first_node,
                hidden_shape=(base.d_model + 1,),
            ),
        )
        + base.nodes[1:],
    )
    invalid_k = dataclasses.replace(
        base,
        regions=(
            dataclasses.replace(
                first_region,
                k_requested={
                    "type": "fixed",
                    "formula_id": "k.fixed.v1",
                    "value": 0,
                },
            ),
        )
        + base.regions[1:],
    )
    wrong_timing = dataclasses.replace(
        base,
        regions=(
            dataclasses.replace(
                first_region,
                profile="N",
                selector_timing="post",
            ),
        )
        + base.regions[1:],
    )
    unstable_id = dataclasses.replace(
        base,
        nodes=(dataclasses.replace(first_node, node_id=" invalid.node"),)
        + base.nodes[1:],
    )
    wrong_terminal = dataclasses.replace(
        base, terminal_node_ids=(base.entry_node_ids[0],)
    )
    return {
        "cycle": (
            dataclasses.replace(base, edges=base.edges + (cycle_edge,)),
            ("plan.topology",),
        ),
        "duplicate-edge-id": (
            dataclasses.replace(base, edges=base.edges + (duplicate_edge,)),
            ("plan.topology",),
        ),
        "intra-region-edge": (intra_region, ("plan.topology",)),
        "hidden-shape": (wrong_shape, ("plan.formula",)),
        "fixed-k-zero": (invalid_k, ("plan.formula",)),
        "profile-timing": (wrong_timing, ("plan.formula",)),
        "unstable-node-id": (unstable_id, ("plan.schema",)),
        "wrong-terminal-set": (wrong_terminal, ("plan.topology",)),
    }


def generate_invalid_plan_corpus(
    seed: int = DEFAULT_PLAN_CORPUS_SEED,
) -> Tuple[InvalidPlanCorpusCase, ...]:
    """Return 24 named, deterministic validator mutants for development.

    Each case applies one mutation operation to a valid source Plan.  A
    mutation may necessarily violate several leaf invariants, but every leaf
    remains within the single stable failure phase/code recorded by the case.
    This is not the 96-mutant qualification corpus.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("invalid Plan corpus seed must be a nonnegative integer")
    rng = random.Random(seed ^ 0x51A7E)
    scheduled = []
    for repetition in range(3):
        base = build_chain(
            length=3 + repetition,
            d_model=2 + (repetition % 2),
        )
        for mutation_kind, (plan, expected_codes) in _invalid_plan_mutations(
            base
        ).items():
            scheduled.append(
                (repetition, base.canonical_hash(), mutation_kind, plan, expected_codes)
            )
    rng.shuffle(scheduled)
    cases = tuple(
        InvalidPlanCorpusCase(
            case_id=f"invalid.{ordinal:03d}.{mutation_kind}.r{repetition}",
            mutation_kind=mutation_kind,
            generation_seed=seed,
            base_plan_hash=base_hash,
            plan=dataclasses.replace(
                plan,
                plan_id=f"invalid.{ordinal:03d}.{mutation_kind}.r{repetition}",
            ),
            expected_codes=expected_codes,
        )
        for ordinal, (
            repetition,
            base_hash,
            mutation_kind,
            plan,
            expected_codes,
        ) in enumerate(scheduled)
    )
    assert len(cases) == DEFAULT_INVALID_PLAN_CORPUS_SIZE
    return cases


__all__ = [
    "CORE_V1_CANDIDATE_CORPUS_SEED",
    "CORE_V1_CANDIDATE_CORPUS_SIZE",
    "CORE_V1_CANDIDATE_VJP_SIZE",
    "DEFAULT_INVALID_PLAN_CORPUS_SIZE",
    "DEFAULT_PLAN_CORPUS_SEED",
    "DEFAULT_PLAN_CORPUS_SIZE",
    "InvalidPlanCorpusCase",
    "PlanCorpusCase",
    "generate_core_v1_candidate_corpus",
    "generate_invalid_plan_corpus",
    "generate_plan_corpus",
]

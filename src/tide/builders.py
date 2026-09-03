"""Readable, completely expanded SettleGraph fixture builders."""

from __future__ import annotations

import itertools
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

from .plan import EdgeSpec, NodeSpec, Plan, RegionSpec


_RUNTIME_DTYPES = {
    "hidden": "runtime",
    "parameter": "runtime",
    "state": "runtime",
    "readout": "runtime",
}


def _node(
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


def _region(
    region_id: str,
    node_ids: Sequence[str],
    *,
    k: Optional[int] = None,
    forced: bool = False,
    line: Optional[int] = None,
    phase: Optional[str] = None,
) -> RegionSpec:
    ordered = tuple(sorted(node_ids))
    if not ordered:
        raise ValueError("a builder region must contain at least one node")
    requested = len(ordered) if k is None else k
    if forced:
        requested = 1
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
        line=line,
        phase=phase,
    )


def _edge(
    edge_id: str, source: str, target: str, *, label: str = "data"
) -> EdgeSpec:
    return EdgeSpec(edge_id=edge_id, source=source, target=target, label=label)


def _plan(
    plan_id: str,
    d_model: int,
    nodes: Iterable[NodeSpec],
    edges: Iterable[EdgeSpec],
    regions: Iterable[RegionSpec],
    *,
    topology_kind: str = "general",
    builder_name: str,
    builder_config: Mapping[str, Any],
) -> Plan:
    return Plan(
        plan_id=plan_id,
        d_model=d_model,
        dtype_roles=_RUNTIME_DTYPES,
        nodes=tuple(nodes),
        edges=tuple(edges),
        regions=tuple(regions),
        output_aggregate={
            "type": "mean",
            "formula_id": "agg.mean.v1",
            "output_shape": [d_model],
        },
        topology_kind=topology_kind,
        builder={
            "name": builder_name,
            "version": "1",
            "config": dict(builder_config),
        },
    ).validate()


def build_singleton(*, d_model: int = 4) -> Plan:
    """Build the minimum forced-active entry/terminal Plan."""

    node_id = "node.0"
    region_id = "region.0"
    return _plan(
        "singleton",
        d_model,
        [_node(node_id, region_id, d_model, forced_active=True)],
        [],
        [_region(region_id, [node_id], forced=True)],
        builder_name="singleton",
        builder_config={"d_model": d_model},
    )


def build_single_layer(*, receiver_count: int = 2, k: int = 1, d_model: int = 4) -> Plan:
    """Build one competition region whose receivers are all boundaries."""

    if receiver_count < 1:
        raise ValueError("receiver_count must be positive")
    if not 1 <= k <= receiver_count:
        raise ValueError("k must lie in [1, receiver_count]")
    region_id = "region.0"
    node_ids = [f"node.{index:04d}" for index in range(receiver_count)]
    return _plan(
        f"single-layer-r{receiver_count}-k{k}",
        d_model,
        [_node(node_id, region_id, d_model) for node_id in node_ids],
        [],
        [_region(region_id, node_ids, k=k)],
        builder_name="single-layer",
        builder_config={
            "d_model": d_model,
            "receiver_count": receiver_count,
            "k": k,
        },
    )


def build_chain(*, length: int = 4, d_model: int = 4) -> Plan:
    """Build a forced-active chain with one receiver per region."""

    if length < 1:
        raise ValueError("length must be positive")
    nodes: List[NodeSpec] = []
    regions: List[RegionSpec] = []
    for index in range(length):
        node_id = f"node.{index:04d}"
        region_id = f"region.{index:04d}"
        nodes.append(_node(node_id, region_id, d_model, forced_active=True))
        regions.append(_region(region_id, [node_id], forced=True))
    edges = [
        _edge(
            f"edge.{index:04d}",
            f"node.{index:04d}",
            f"node.{index + 1:04d}",
        )
        for index in range(length - 1)
    ]
    return _plan(
        f"chain-{length}",
        d_model,
        nodes,
        edges,
        regions,
        builder_name="chain",
        builder_config={"d_model": d_model, "length": length},
    )


def build_diamond(*, d_model: int = 4, branch_k: int = 1) -> Plan:
    """Build fan-out, competing branches, and fan-in aggregation."""

    if branch_k not in {1, 2}:
        raise ValueError("branch_k must be 1 or 2")
    nodes = [
        _node("node.root", "region.root", d_model, forced_active=True),
        _node("node.branch.a", "region.branches", d_model),
        _node("node.branch.b", "region.branches", d_model),
        _node("node.out", "region.out", d_model, forced_active=True),
    ]
    regions = [
        _region("region.root", ["node.root"], forced=True),
        _region(
            "region.branches",
            ["node.branch.a", "node.branch.b"],
            k=branch_k,
        ),
        _region("region.out", ["node.out"], forced=True),
    ]
    edges = [
        _edge("edge.root-a", "node.root", "node.branch.a"),
        _edge("edge.root-b", "node.root", "node.branch.b"),
        _edge("edge.a-out", "node.branch.a", "node.out"),
        _edge("edge.b-out", "node.branch.b", "node.out"),
    ]
    return _plan(
        f"diamond-k{branch_k}",
        d_model,
        nodes,
        edges,
        regions,
        builder_name="diamond",
        builder_config={"branch_k": branch_k, "d_model": d_model},
    )


def build_unequal_path(*, d_model: int = 4) -> Plan:
    """Build a two-hop and three-hop path that merge at one terminal."""

    descriptions = [
        ("node.root", "region.root", True),
        ("node.short", "region.split", False),
        ("node.long.0", "region.split", False),
        ("node.long.1", "region.long", True),
        ("node.out", "region.out", True),
    ]
    nodes = [
        _node(node_id, region_id, d_model, forced_active=forced)
        for node_id, region_id, forced in descriptions
    ]
    regions = [
        _region("region.root", ["node.root"], forced=True),
        _region("region.split", ["node.short", "node.long.0"], k=2),
        _region("region.long", ["node.long.1"], forced=True),
        _region("region.out", ["node.out"], forced=True),
    ]
    edges = [
        _edge("edge.root-short", "node.root", "node.short"),
        _edge("edge.short-out", "node.short", "node.out"),
        _edge("edge.root-long0", "node.root", "node.long.0"),
        _edge("edge.long0-long1", "node.long.0", "node.long.1"),
        _edge("edge.long1-out", "node.long.1", "node.out"),
    ]
    return _plan(
        "unequal-path",
        d_model,
        nodes,
        edges,
        regions,
        builder_name="unequal-path",
        builder_config={"d_model": d_model},
    )


def build_multi_entry_terminal(*, d_model: int = 4) -> Plan:
    """Build input broadcast and output aggregation with two boundaries each."""

    entries = ["node.in.a", "node.in.b"]
    terminals = ["node.out.a", "node.out.b"]
    nodes = [
        *[_node(node_id, "region.in", d_model) for node_id in entries],
        *[_node(node_id, "region.out", d_model) for node_id in terminals],
    ]
    edges = [
        _edge(f"edge.{source[-1]}-{target[-1]}", source, target)
        for source in entries
        for target in terminals
    ]
    regions = [
        _region("region.in", entries, k=len(entries)),
        _region("region.out", terminals, k=len(terminals)),
    ]
    return _plan(
        "multi-entry-terminal",
        d_model,
        nodes,
        edges,
        regions,
        builder_name="multi-entry-terminal",
        builder_config={"d_model": d_model},
    )


def build_mixed_regions(*, d_model: int = 4) -> Plan:
    """Build independent competition and forced regions at one graph depth."""

    nodes = [
        _node("node.root", "region.root", d_model, forced_active=True),
        _node("node.optional.a", "region.optional", d_model),
        _node("node.optional.b", "region.optional", d_model),
        _node("node.backbone", "region.backbone", d_model, forced_active=True),
        _node("node.out", "region.out", d_model, forced_active=True),
    ]
    regions = [
        _region("region.root", ["node.root"], forced=True),
        _region(
            "region.optional",
            ["node.optional.a", "node.optional.b"],
            k=1,
        ),
        _region("region.backbone", ["node.backbone"], forced=True),
        _region("region.out", ["node.out"], forced=True),
    ]
    edges = [
        _edge("edge.root-a", "node.root", "node.optional.a"),
        _edge("edge.root-b", "node.root", "node.optional.b"),
        _edge("edge.root-backbone", "node.root", "node.backbone"),
        _edge("edge.a-out", "node.optional.a", "node.out"),
        _edge("edge.b-out", "node.optional.b", "node.out"),
        _edge("edge.backbone-out", "node.backbone", "node.out"),
    ]
    return _plan(
        "mixed-regions",
        d_model,
        nodes,
        edges,
        regions,
        builder_name="mixed-regions",
        builder_config={"d_model": d_model},
    )


Coordinate = Tuple[int, ...]


def _coordinates(branch_factor: int, length: int) -> List[Coordinate]:
    return list(itertools.product(range(branch_factor), repeat=length))


def _coordinate_key(coordinate: Coordinate) -> str:
    if not coordinate:
        return "root"
    return ".".join(f"{digit:04d}" for digit in coordinate)


def build_hb_lattice(
    *,
    branch_factor: int = 2,
    expansion_depth: int = 2,
    platform_lines: int = 1,
    region_size: int = 2,
    include_shortcuts: bool = True,
    include_mirrors: bool = True,
    d_model: int = 4,
) -> Plan:
    """Build a bounded-degree expansion--platform--contraction HB Plan.

    ``platform_lines`` counts extra peak-width Lines after the first peak
    Line, matching the semantic document's ``P_plat`` convention.
    """

    if branch_factor < 2:
        raise ValueError("branch_factor must be at least 2")
    if expansion_depth < 1:
        raise ValueError("expansion_depth must be positive")
    if platform_lines < 0:
        raise ValueError("platform_lines must be nonnegative")
    if region_size < 1:
        raise ValueError("region_size must be positive")

    final_line = 2 * expansion_depth + platform_lines
    line_coordinates: List[List[Coordinate]] = []
    for line in range(final_line + 1):
        if line <= expansion_depth:
            coordinate_length = line
        elif line <= expansion_depth + platform_lines:
            coordinate_length = expansion_depth
        else:
            coordinate_length = final_line - line
        line_coordinates.append(_coordinates(branch_factor, coordinate_length))

    def node_id(line: int, coordinate: Coordinate) -> str:
        return f"node.l{line:04d}.{_coordinate_key(coordinate)}"

    nodes: List[NodeSpec] = []
    regions: List[RegionSpec] = []
    for line, coordinates in enumerate(line_coordinates):
        if line < expansion_depth:
            phase = "expand"
        elif line <= expansion_depth + platform_lines:
            phase = "plateau"
        else:
            phase = "contract"
        for region_index, start in enumerate(range(0, len(coordinates), region_size)):
            members = coordinates[start : start + region_size]
            region_id = f"region.l{line:04d}.r{region_index:04d}"
            member_ids = [node_id(line, coordinate) for coordinate in members]
            boundary_singleton = line in {0, final_line} and len(member_ids) == 1
            for member_id in member_ids:
                nodes.append(
                    _node(
                        member_id,
                        region_id,
                        d_model,
                        forced_active=boundary_singleton,
                    )
                )
            regions.append(
                _region(
                    region_id,
                    member_ids,
                    k=1,
                    forced=boundary_singleton,
                    line=line,
                    phase=phase,
                )
            )

    edge_records: List[Tuple[str, str, str]] = []
    # Expansion and contraction tree edges.
    for line in range(expansion_depth):
        for parent in line_coordinates[line]:
            for digit in range(branch_factor):
                child = parent + (digit,)
                edge_records.append((node_id(line, parent), node_id(line + 1, child), "tree"))
    contraction_start = expansion_depth + platform_lines
    for line in range(contraction_start, final_line):
        for child_coordinate in line_coordinates[line]:
            parent_coordinate = child_coordinate[:-1]
            edge_records.append(
                (
                    node_id(line, child_coordinate),
                    node_id(line + 1, parent_coordinate),
                    "tree",
                )
            )

    # Peak-width hops: identity local edge and a fixed-degree cyclic neighbor.
    for line in range(expansion_depth, expansion_depth + platform_lines):
        coordinates = line_coordinates[line]
        for index, coordinate in enumerate(coordinates):
            edge_records.append(
                (node_id(line, coordinate), node_id(line + 1, coordinate), "local")
            )
            if include_shortcuts and len(coordinates) > 1:
                neighbor = coordinates[(index + 1) % len(coordinates)]
                edge_records.append(
                    (
                        node_id(line, coordinate),
                        node_id(line + 1, neighbor),
                        "shortcut",
                    )
                )

    if include_mirrors:
        for line in range(1, expansion_depth):
            mirror_line = final_line - line
            for coordinate in line_coordinates[line]:
                edge_records.append(
                    (
                        node_id(line, coordinate),
                        node_id(mirror_line, coordinate),
                        "mirror",
                    )
                )

    # Sort endpoint records first so edge IDs do not depend on generation order.
    edge_records = sorted(set(edge_records))
    edges = [
        _edge(
            f"edge.{index:06d}", source, target, label=label
        )
        for index, (source, target, label) in enumerate(edge_records)
    ]
    config = {
        "branch_factor": branch_factor,
        "expansion_depth": expansion_depth,
        "platform_lines": platform_lines,
        "region_size": region_size,
        "include_shortcuts": include_shortcuts,
        "include_mirrors": include_mirrors,
        "d_model": d_model,
    }
    return _plan(
        f"hb-b{branch_factor}-d{expansion_depth}-p{platform_lines}",
        d_model,
        nodes,
        edges,
        regions,
        topology_kind="hb",
        builder_name="hb-lattice",
        builder_config=config,
    )


def build_small_hb(*, d_model: int = 4) -> Plan:
    """Build the standard small 1--2--4--4--2--1 HB fixture."""

    return build_hb_lattice(
        branch_factor=2,
        expansion_depth=2,
        platform_lines=1,
        region_size=2,
        include_shortcuts=True,
        include_mirrors=True,
        d_model=d_model,
    )


__all__ = [
    "build_chain",
    "build_diamond",
    "build_hb_lattice",
    "build_mixed_regions",
    "build_multi_entry_terminal",
    "build_single_layer",
    "build_singleton",
    "build_small_hb",
    "build_unequal_path",
]

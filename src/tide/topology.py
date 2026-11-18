"""Immutable topology records for the current FC and HB instance bindings.

Definitions are pinned to ObsidianVault commit
4f5b9f20a4030d68ee66ae2d26e34659a5e37234, topologies/ and the new bindings in
instances/backbone-comparison.md and instances/hb-settle-weighted-message.md.
Addresses are zero based; ranks and the first external input time start at one.
This module constructs graph metadata, never an actual message trajectory.
"""
from dataclasses import dataclass, replace

from .topology_hb import exchange_edges, lattice_edges, multiscale_edges, multiscale_widths

HubMask = str | tuple[tuple[bool, ...], ...]


@dataclass(frozen=True, slots=True)
class TopologyConfig:
    """A concrete instance graph, with explicit unsupported configurations.

    ``depth`` means binary expansion h for Lattice, m for Exchange. Exchange R
    defaults to m. Multiscale profiles are 5/5-dd4/7/9/13/15/17/25/33 and
    compact-5-dd4/7/9/9-two-inputs/15. Only Multiscale has a Timed binding,
    arbitrary Hub masks, or the three partition choices. Terminal Hubs require
    the explicit ``terminal_hubs`` option on Exchange/Multiscale.
    ``layers=None`` resolves to FC4, Lattice7, Exchange10 or Multiscale9.
    """

    kind: str = "fc"
    n: int = 8
    layers: int | None = None
    depth: int | None = None
    rounds: int | None = None
    profile: str | None = None
    hubs: int = 0
    partition: str = "group"
    timing: str = "settle"
    hub_mask: HubMask = "all"
    period: int | None = None
    probes: tuple[int, int, int] | None = None
    terminal_hubs: bool = False


@dataclass(frozen=True, slots=True)
class Node:
    id: int
    layer: int
    position: int
    local: int
    role: str
    region: int


@dataclass(frozen=True, slots=True)
class Edge:
    id: int
    src: int
    dst: int
    delay: int
    source_class: str


@dataclass(frozen=True, slots=True)
class Graph:
    config: TopologyConfig
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    regions: tuple[tuple[int, ...], ...]
    input_nodes: tuple[int, ...]
    output_nodes: tuple[int, ...]
    widths: tuple[int, ...]
    period: int
    ranks: tuple[int, ...]
    region_ranks: tuple[int, ...]
    incoming: tuple[tuple[int, ...], ...]
    outgoing: tuple[tuple[int, ...], ...]

    def input_time(self, position: int) -> int:
        _integer("input position", position, 0)
        return self.period * position + 1


def _integer(name, value, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _resolve(config):
    if not isinstance(config, TopologyConfig):
        raise TypeError("build_topology expects TopologyConfig")
    c = config
    if c.kind not in ("fc", "lattice", "exchange", "multiscale"):
        raise ValueError(f"unknown topology kind: {c.kind!r}")
    _integer("n", c.n)
    _integer("hubs", c.hubs, 0)
    if type(c.terminal_hubs) is not bool:
        raise ValueError("terminal_hubs must be boolean")
    if c.terminal_hubs and (c.kind not in ("exchange", "multiscale") or not c.hubs):
        raise ValueError("terminal_hubs requires Exchange/Multiscale with hubs")
    for name in ("layers", "depth", "rounds", "period"):
        if (value := getattr(c, name)) is not None:
            _integer(name, value)
    if c.timing not in ("settle", "timed") or (c.timing == "timed" and c.kind != "multiscale"):
        raise ValueError("Timed is defined only for multiscale")
    if c.partition not in ("group", "singleton", "mixed"):
        raise ValueError(f"unknown partition: {c.partition!r}")
    if c.kind != "multiscale" and c.partition != "group":
        raise ValueError("singleton/mixed partitions are defined only for multiscale")
    if c.kind != "multiscale" and (not isinstance(c.hub_mask, str) or c.hub_mask not in ("all", "none")):
        raise ValueError("custom/checkerboard masks are defined only for multiscale")
    if c.hubs and (c.kind in ("fc", "lattice") or c.n % c.hubs):
        raise ValueError("hubs require Exchange/Multiscale and must divide n")
    if c.profile is not None and (c.kind != "multiscale" or not isinstance(c.profile, str)):
        raise ValueError("profile must be a multiscale profile name")
    if c.depth is not None and c.kind not in ("lattice", "exchange"):
        raise ValueError("depth is the Lattice/Exchange expansion depth")
    if c.rounds is not None and c.kind != "exchange":
        raise ValueError("rounds applies only to Exchange")

    depth, rounds, profile = c.depth, c.rounds, c.profile
    if c.kind == "fc":
        widths = (1,) * (c.layers if c.layers is not None else 4)
    elif c.kind == "lattice":
        depth = depth if depth is not None else (3 if c.layers is None else (c.layers - 1) // 2)
        _integer("Lattice depth", depth)
        widths = tuple(2 ** min(l, 2 * depth - l) for l in range(2 * depth + 1))
    elif c.kind == "exchange":
        if c.n < 2 or c.n & (c.n - 1):
            raise ValueError("Exchange n must be a power of two >= 2")
        if depth is None:
            depth = (3 if c.layers is None else (c.layers - 1) // 3 if rounds is None
                     else (c.layers - rounds - 1) // 2)
        if rounds is None:
            rounds = depth if c.layers is None else c.layers - 2 * depth - 1
        _integer("Exchange depth", depth)
        _integer("Exchange rounds", rounds)
        count = 2 * depth + rounds + 1
        widths = tuple(2 ** min(l, depth, count - 1 - l) for l in range(count))
    else:
        profile = profile if profile is not None else str(c.layers if c.layers is not None else 9)
        widths = multiscale_widths(profile)
    if c.layers is not None and c.layers != len(widths):
        raise ValueError("layers conflicts with the selected depth/rounds/profile")

    default_period = (4 if c.timing == "timed" else
                      len(widths) + (2 if c.kind in ("fc", "lattice") else 1))
    period = c.period if c.period is not None else default_period
    if c.timing == "settle" and period <= len(widths):
        raise ValueError("Settle requires period > maximum region rank")
    if c.kind != "multiscale" and period != default_period:
        raise ValueError("this binding fixes its input period; only multiscale exposes a period choice")
    probes = c.probes
    if probes is not None and (c.kind != "multiscale" or not c.hubs):
        raise ValueError("probes applies only to multiscale with hubs")
    if c.kind == "multiscale" and c.hubs:
        if probes is None:
            probes = ((max(1, c.n // 32), max(1, c.n // 32), max(1, c.n // 64))
                      if c.hubs == 4 and c.n in (8, 16, 32, 64, 128) else (1, 1, 1))
        if not isinstance(probes, tuple) or len(probes) != 3:
            raise ValueError("probes must be a tuple for spans 1, 2, 4")
        for probe in probes:
            _integer("probe count", probe)
            if (c.n // c.hubs) % probe:
                raise ValueError("each probe count must divide n/hubs")
    return replace(c, layers=len(widths), depth=depth, rounds=rounds,
                   profile=profile, period=period, probes=probes), widths


def _hub_presence(config, widths):
    mask = config.hub_mask
    if isinstance(mask, str):
        if mask not in ("all", "none", "checkerboard"):
            raise ValueError(f"unknown hub mask: {mask!r}")
        return tuple(tuple(bool(config.hubs) and (config.terminal_hubs or l < len(widths) - 1) and
                           (mask == "all" or mask == "checkerboard" and (l + p) % 2 == 0)
                           for p in range(width)) for l, width in enumerate(widths))
    if not isinstance(mask, tuple) or len(mask) != len(widths):
        raise ValueError("hub mask needs one tuple per layer")
    if any(not isinstance(row, tuple) or len(row) != width or
           any(type(value) is not bool for value in row) for row, width in zip(mask, widths)):
        raise ValueError("hub mask rows must contain one bool per position")
    if any(mask[-1]) and not config.terminal_hubs:
        raise ValueError("the current binding has no terminal hubs")
    if not config.hubs and any(any(row) for row in mask):
        raise ValueError("a nonempty hub mask requires hubs > 0")
    return mask


def build_topology(config: TopologyConfig) -> Graph:
    """Build stable node/edge identities and complete region/port metadata.

    Graph.config records the resolved defaults. All other fields are immutable.
    Incoming/outgoing contain edge IDs, preserving the edge/source distinction.
    Hub removal is induced deletion, with no redirected collection traffic.
    """
    c, widths = _resolve(config)
    presence = _hub_presence(c, widths)
    nodes, regions, lookup = [], [], {}
    for layer, width in enumerate(widths):
        for position in range(width):
            groups = [[("D", i) for i in range(c.n)]]
            if presence[layer][position]:
                hubs = [("H", i) for i in range(c.hubs)]
                if c.partition == "mixed":
                    groups[0].extend(hubs)
                elif c.partition == "singleton":
                    groups.extend([hub] for hub in hubs)
                else:
                    groups.append(hubs)
            for group in groups:
                members = []
                for role, local in group:
                    node = Node(len(nodes), layer, position, local, role, len(regions))
                    nodes.append(node)
                    members.append(node.id)
                    lookup[layer, position, role, local] = node.id
                regions.append(tuple(members))
    if c.kind == "fc":
        spatial = (((l, 0, "D", i), (l + 1, 0, "D", j), "DD")
                   for l in range(c.layers - 1) for i in range(c.n) for j in range(c.n))
    elif c.kind == "lattice":
        spatial = lattice_edges(widths, c.n)
    elif c.kind == "exchange":
        spatial = exchange_edges(widths, c.n, c.hubs)
    else:
        spatial = multiscale_edges(widths, c.n, c.hubs, c.probes, c.profile.endswith("dd4"))
    identities = set()
    for source, target, source_class in spatial:
        if source in lookup and target in lookup:
            identities.add((lookup[source], lookup[target], source_class))
    edges, incoming, outgoing = [], [[] for _ in nodes], [[] for _ in nodes]
    for source, target, source_class in sorted(identities):
        span = nodes[target].layer - nodes[source].layer
        if span <= 0 or nodes[source].region == nodes[target].region:
            raise ValueError("every edge must strictly increase region rank")
        edge = Edge(len(edges), source, target, 1 if c.timing == "timed" else span, source_class)
        edges.append(edge)
        incoming[target].append(edge.id)
        outgoing[source].append(edge.id)
    return Graph(c, tuple(nodes), tuple(edges), tuple(regions),
                 tuple(node.id for node in nodes if node.layer == 0),
                 tuple(node.id for node in nodes if node.layer == len(widths) - 1 and node.role == "D"),
                 widths, c.period, tuple(node.layer + 1 for node in nodes),
                 tuple(nodes[group[0]].layer + 1 for group in regions),
                 tuple(map(tuple, incoming)), tuple(map(tuple, outgoing)))

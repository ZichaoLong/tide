"""HB endpoint formulas from upstream 4f5b9f20a4030d68ee66ae2d26e34659a5e37234.

Only spatial addresses live here. Partition, port, identity and time binding are
owned by topology.py. Sources: topologies/hb-{lattice,exchange,multiscale}.md.
"""
from hashlib import sha256

Address = tuple[int, int, str, int]
SpatialEdge = tuple[Address, Address, str]

_WIDTHS = {
    "5": (4, 4, 8, 4, 4),
    "5-dd4": (4, 4, 8, 4, 4),
    "7": (4, 4, 8, 8, 8, 4, 4),
    "9": (4, 4, 4, 8, 8, 8, 4, 4, 4),
    "13": (4, 4, 4, 8, 8, 16, 16, 16, 8, 8, 4, 4, 4),
    "15": (4, 4, 4, 8, 8, 8, 16, 16, 16, 8, 8, 8, 4, 4, 4),
    "17": (4, 4, 4, 8, 8, 8, 16, 16, 16, 16, 16, 8, 8, 8, 4, 4, 4),
    "compact-5-dd4": (1, 1, 2, 1, 1),
    "compact-7": (1, 1, 2, 2, 2, 1, 1),
    "compact-9": (1, 1, 2, 2, 4, 2, 2, 1, 1),
    "compact-9-two-inputs": (2, 2, 2, 4, 2, 2, 2, 2, 2),
    "compact-15": (1,) * 4 + (2,) * 7 + (1,) * 4,
}


def multiscale_widths(profile: str) -> tuple[int, ...]:
    if profile in ("25", "33"):
        return _WIDTHS["17"][:8] + (16,) * (int(profile) - 17) + _WIDTHS["17"][8:]
    try:
        return _WIDTHS[profile]
    except KeyError as error:
        raise ValueError(f"unknown multiscale profile: {profile!r}") from error


def _permutation(n: int, tag: str, *coordinates: int) -> tuple[int, ...]:
    # The exact UTF-8 seed serialization is part of the mathematical binding.
    prefix = "|".join(("hb-timed-discussion-20261114-v1", tag,
                       *(str(value) for value in coordinates))) + "|"
    return tuple(sorted(range(n), key=lambda i: (sha256(f"{prefix}{i}".encode()).digest(), i)))


def _targets(widths, layer, position, span, modes):
    source_width, target_width = widths[layer], widths[layer + span]
    coarse_width = min(source_width, target_width)
    coarse = position // (source_width // coarse_width)
    expansion = target_width // coarse_width
    targets = set()
    for mode in modes:
        mapped = coarse
        if mode == "near":
            mapped = (mapped + 1) % coarse_width
        elif mode == "cross" and coarse_width > 1:
            bit = (layer + span.bit_length() - 1) % (coarse_width.bit_length() - 1)
            mapped ^= 1 << bit
        targets.update(mapped * expansion + j for j in range(expansion))
    return tuple(sorted(targets))


def lattice_edges(widths, n):
    """The new comparison binding fixes binary addresses and offsets {0, 1}."""
    for layer, width in enumerate(widths[:-1]):
        for position in range(width):
            targets = ((2 * position, 2 * position + 1) if widths[layer + 1] > width
                       else (position // 2,))
            for target in targets:
                for i in range(n):
                    for j in sorted({i, (i + 1) % n}):
                        yield (layer, position, "D", i), (layer + 1, target, "D", j), "DDL"


def _hub_edges(widths, n, hubs, layer, position, span, collect, exchange, hooks, probes):
    bank = n // hubs
    if span == 1:
        for target in collect:
            permutation = _permutation(n, "collect", layer, position, target)
            for i in range(n):
                yield ((layer, position, "D", i),
                       (layer + 1, target, "H", permutation[i] // bank), "DH1")
    for target in exchange:
        for a in range(hubs):
            local = (a + layer + position + target + span.bit_length() - 1) % hubs
            yield (layer, position, "H", a), (layer + span, target, "H", local), f"HH{span}"
    for target in hooks:
        permutation = _permutation(n, "hook", layer + span, target)
        phase = (7 * layer + 3 * position + target + 5 * span) % bank
        for a in range(hubs):
            group = (a + layer + position + span) % hubs
            for probe in range(probes):
                slot = (phase + probe * (bank // probes)) % bank
                yield ((layer, position, "H", a),
                       (layer + span, target, "D", permutation[group * bank + slot]), f"HD{span}")


def exchange_edges(widths, n, hubs):
    for layer, width in enumerate(widths[:-1]):
        for position in range(width):
            base = _targets(widths, layer, position, 1, ("base",))
            targets = _targets(widths, layer, position, 1, ("base", "cross"))
            bit = layer % (n.bit_length() - 1)
            for target in targets:
                for i in range(n):
                    for j in sorted((i, i ^ (1 << bit))):
                        yield (layer, position, "D", i), (layer + 1, target, "D", j), "DDX"
            if hubs:
                yield from _hub_edges(widths, n, hubs, layer, position, 1,
                                      targets, targets, base, 1)


def multiscale_edges(widths, n, hubs, probes, add_dd4):
    for layer, width in enumerate(widths[:-1]):
        for position in range(width):
            adjacent = _targets(widths, layer, position, 1, ("base", "near"))
            for target in adjacent:
                for i in range(n):
                    for j in sorted({i, (i + 1) % n}):
                        yield (layer, position, "D", i), (layer + 1, target, "D", j), "DD1"
            for span in ((2, 4) if add_dd4 else (2,)):
                if layer + span >= len(widths):
                    continue
                for target in _targets(widths, layer, position, span, ("cross",)):
                    permutation = _permutation(n, "detail" if span == 2 else "detail4",
                                               layer, position, target)
                    for i, j in enumerate(permutation):
                        yield (layer, position, "D", i), (layer + span, target, "D", j), f"DD{span}"
            if not hubs:
                continue
            for span, count in zip((1, 2, 4), probes):
                if layer + span < len(widths):
                    exchange = _targets(widths, layer, position, span, ("base", "cross"))
                    hooks = _targets(widths, layer, position, span,
                                     ("base",) if span == 1 else ("cross",))
                    yield from _hub_edges(widths, n, hubs, layer, position, span,
                                          adjacent, exchange, hooks, count)

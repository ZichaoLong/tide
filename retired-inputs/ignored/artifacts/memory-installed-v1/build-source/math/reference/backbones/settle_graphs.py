"""Original lattice and staged-exchange Settle graphs; no neural execution.

Definitions: topologies/hb-lattice.md and topologies/hb-exchange.md.
Node addresses use zero-based layers, with canonical rank = layer + 1.
"""
from collections import Counter, defaultdict
from .multiscale import permutation


def exchange_widths(m, rounds=None):
    rounds = m if rounds is None else rounds
    if m < 1 or rounds < 1:
        raise ValueError('positive expansion depth and exchange rounds required')
    depth = 2*m + rounds + 1
    return [2 ** min(l, m, depth-1-l) for l in range(depth)]


def targets(widths, layer, position, exchange=True):
    w, v = widths[layer:layer + 2]
    g = min(w, v)
    x = position // (w // g)
    groups = {x}
    if exchange and g > 1:
        groups.add(x ^ (1 << (layer % (g.bit_length() - 1))))
    return tuple(sorted((v // g) * y + j for y in groups for j in range(v // g)))


def construct(widths, n=16, h=0, exchange=True, terminal_hubs=False):
    if (len(widths) < 2 or any(w < 1 or w & (w - 1) for w in widths)
            or any(max(a, b) > 2 * min(a, b) for a, b in zip(widths, widths[1:]))):
        raise ValueError('power-of-two widths with adjacent ratio <= 2 required')
    if n < 2 or n & (n - 1) or h < 0 or (h and n % h):
        raise ValueError('n must be a power of two >= 2; optional positive h divides n')
    if not exchange and h:
        raise ValueError('original lattice has no hub')
    nodes = [(l, p, r, i) for l, w in enumerate(widths) for p in range(w)
             for r, size in [('d', n), ('h', h if terminal_hubs or l < len(widths)-1 else 0)]
             for i in range(size)]
    ids = {x: u for u, x in enumerate(nodes)}
    outs, ins = [[] for _ in nodes], [[] for _ in nodes]
    kinds, connections = Counter(), defaultdict(set)

    def add(src, dst, kind):
        if src not in ids or dst not in ids:
            return  # Delete terminal hubs and their incident edges together.
        u, v = ids[src], ids[dst]
        outs[u].append((v, kind)); ins[v].append((u, kind))
        kinds[kind] += 1
        connections[kind].add((src[0], src[1], dst[0], dst[1]))

    for l, w in enumerate(widths[:-1]):
        c = l % (n.bit_length() - 1)
        for p in range(w):
            for q in targets(widths, l, p, exchange):
                rho = permutation('collect', l, p, q, size=n) if h else None
                for i in range(n):
                    js = (i, i ^ (1 << c)) if exchange else (i, (i + 1) % n)
                    for j in js:
                        add((l, p, 'd', i), (l + 1, q, 'd', j), 'DDX' if exchange else 'DDL')
                    if h:
                        add((l, p, 'd', i), (l + 1, q, 'h', rho[i] // (n // h)), 'DH1')
                for a in range(h):
                    add((l, p, 'h', a), (l + 1, q, 'h', (a + l + p + q) % h), 'HH1')
            # One summary probe per hub per base-mapped destination.
            for q in targets(widths, l, p, exchange=False):
                if not h:
                    continue
                b = n // h
                chi = permutation('hook', l + 1, q, size=n)
                eta = (7 * l + 3 * p + q + 5) % b
                for a in range(h):
                    bank = (a + l + p + 1) % h
                    add((l, p, 'h', a), (l + 1, q, 'd', chi[bank * b + eta]), 'HD1')
    return nodes, ids, outs, ins, kinds, connections


def lattice(depth=7, n=16):
    if depth < 3 or depth % 2 != 1:
        raise ValueError('odd lattice depth >= 3 required')
    h = (depth - 1) // 2
    widths = [2 ** min(l, 2 * h - l) for l in range(depth)]
    return widths, construct(widths, n=n, exchange=False)

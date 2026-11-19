"""HB multiscale spatial graph and potential paths; no neural execution.

The maintained definition is topologies/hb-multiscale.md. Timing is an
independent binding of this exact endpoint graph. No global mutable size.
"""
from collections import Counter, defaultdict
import hashlib

SEED = 'hb-timed-discussion-20261114-v1'
PROFILES = [
    ('5', [4, 4, 8, 4, 4], False),
    ('5-dd4', [4, 4, 8, 4, 4], True),
    ('7', [4, 4, 8, 8, 8, 4, 4], False),
    ('9', [4, 4, 4, 8, 8, 8, 4, 4, 4], False),
    ('13', [4, 4, 4, 8, 8, 16, 16, 16, 8, 8, 4, 4, 4], False),
    ('15', [4, 4, 4, 8, 8, 8, 16, 16, 16, 8, 8, 8, 4, 4, 4], False),
    ('17', [4, 4, 4, 8, 8, 8, 16, 16, 16, 16, 16, 8, 8, 8, 4, 4, 4], False),
]

# Additional width profiles; the original figures and keys remain stable.
COMPACT_PROFILES = [
    ('compact-5-dd4', [1, 1, 2, 1, 1], True),
    ('compact-7', [1, 1, 2, 2, 2, 1, 1], False),
    ('compact-9', [1, 1, 2, 2, 4, 2, 2, 1, 1], False),
    ('compact-9-two-inputs', [2, 2, 2, 4, 2, 2, 2, 2, 2], False),
    ('compact-15', [1]*4 + [2]*7 + [1]*4, False),
]


def extended_widths(depth):
    """Extend the 17-layer plateau without reducing its content dimensions."""
    if depth not in (25, 33):
        raise ValueError('extended depths are 25 and 33')
    seed = next(w for key, w, _ in PROFILES if key == '17')
    return seed[:8] + [16]*(depth-17) + seed[8:]


def probes_for(n):
    return max(1, n//32), max(1, n//32), max(1, n//64)


def permutation(tag, *indices, size):
    prefix = '|'.join(map(str, (SEED, tag, *indices))) + '|'
    return sorted(range(size), key=lambda i: (
        hashlib.sha256((prefix + str(i)).encode()).digest(), i))


def destinations(widths, layer, span, position, mode):
    w, v = widths[layer], widths[layer + span]
    g = min(w, v)
    coarse = position // (w // g)
    if mode == 'near':
        coarse = (coarse + 1) % g
    elif mode == 'cross':
        if g > 1:
            bit = (layer + span.bit_length() - 1) % (g.bit_length() - 1)
            coarse ^= 1 << bit
    elif mode != 'base':
        raise ValueError(mode)
    return tuple((v // g) * coarse + j for j in range(v // g))


def region_targets(widths, layer, span, position, modes):
    return tuple(sorted({q for mode in modes
                         for q in destinations(widths, layer, span, position, mode)}))


def construct(widths, n=16, h=4, probes=(1, 1, 1), placement='all', add_dd4=False,
              terminal_hubs=False):
    if not widths or any(w < 1 or w & (w - 1) for w in widths):
        raise ValueError('widths must be powers of two >= 1')
    if n < 1 or h < 1 or n % h or len(probes) != 3:
        raise ValueError('positive h must divide n; give probes for spans 1,2,4')
    b = n // h
    if any(r < 1 or b % r for r in probes):
        raise ValueError('each positive probe count must divide n/h')
    if placement not in ('all', 'none', 'checkerboard'):
        raise ValueError(placement)
    probe = dict(zip((1, 2, 4), probes))
    nodes, ids = [], {}
    for layer, width in enumerate(widths):
        for p in range(width):
            present = placement == 'all' or (placement == 'checkerboard' and (layer + p) % 2 == 0)
            present = present and (terminal_hubs or layer < len(widths) - 1)
            for role, size in [('d', n), ('h', h if present else 0)]:
                for i in range(size):
                    ids[layer, p, role, i] = len(nodes)
                    nodes.append((layer, p, role, i))
    outgoing, incoming = [[] for _ in nodes], [[] for _ in nodes]
    kinds, connections = Counter(), defaultdict(set)

    def add(src, dst, kind):
        # Hub deletion is induced-subgraph deletion, with no traffic rerouting.
        if src not in ids or dst not in ids:
            return
        u, v = ids[src], ids[dst]
        outgoing[u].append((v, kind))
        incoming[v].append((u, kind))
        kinds[kind] += 1
        connections[kind].add((src[0], src[1], dst[0], dst[1]))

    for layer, width in enumerate(widths[:-1]):
        for p in range(width):
            for q in region_targets(widths, layer, 1, p, ('base', 'near')):
                rho = permutation('collect', layer, p, q, size=n)
                for i in range(n):
                    for j in {i, (i + 1) % n}:
                        add((layer, p, 'd', i), (layer + 1, q, 'd', j), 'DD1')
                    add((layer, p, 'd', i), (layer + 1, q, 'h', rho[i] // b), 'DH1')
            for span in ((2, 4) if add_dd4 else (2,)):
                if layer + span >= len(widths):
                    continue
                for q in region_targets(widths, layer, span, p, ('cross',)):
                    pi = permutation('detail' if span == 2 else 'detail4', layer, p, q, size=n)
                    for i in range(n):
                        add((layer, p, 'd', i), (layer + span, q, 'd', pi[i]), f'DD{span}')
            for span in (1, 2, 4):
                if layer + span >= len(widths):
                    continue
                for q in region_targets(widths, layer, span, p, ('base', 'cross')):
                    for a in range(h):
                        target = (a + layer + p + q + span.bit_length() - 1) % h
                        add((layer, p, 'h', a), (layer + span, q, 'h', target), f'HH{span}')
                for q in region_targets(widths, layer, span, p, ('base',) if span == 1 else ('cross',)):
                    chi = permutation('hook', layer + span, q, size=n)
                    eta = (7 * layer + 3 * p + q + 5 * span) % b
                    for a in range(h):
                        bank = (a + layer + p + span) % h
                        for t in range(probe[span]):
                            slot = (eta + t * (b // probe[span])) % b
                            add((layer, p, 'h', a),
                                (layer + span, q, 'd', chi[bank * b + slot]), f'HD{span}')
    return nodes, ids, outgoing, incoming, kinds, connections


def support(bits):
    return [i for i in range(bits.bit_length()) if bits >> i & 1]


def reachability(nodes, outgoing, widths, only_detail=False, origin_layer=0, timing='timed'):
    if timing not in ('timed', 'settle'):
        raise ValueError(timing)
    source_width = widths[origin_layer]
    delay, origins = [0] * len(nodes), [0] * len(nodes)
    for u, (layer, p, role, _) in enumerate(nodes):
        if layer == origin_layer and (not only_detail or role == 'd'):
            delay[u], origins[u] = 1, 1 << p
        if layer < origin_layer:
            continue
        for v, kind in outgoing[u]:
            if only_detail and not kind.startswith('DD'):
                continue
            delta = 1 if timing == 'timed' else nodes[v][0] - layer
            delay[v] |= delay[u] << delta
            origins[v] |= origins[u] << (delta * source_width)
    return delay, origins


def source_node_coverage(nodes, outgoing, widths, ordinary_only=False, period=4):
    """Potential first-layer D identities per output phase, not content flow."""
    roots = [u for u,x in enumerate(nodes) if x[0] == 0 and x[2] == 'd']
    paths = [{} for _ in nodes]
    for i,u in enumerate(roots):
        paths[u][0] = 1 << i
    for u in range(len(nodes)):
        for v,kind in outgoing[u]:
            if ordinary_only and not kind.startswith('DD'):
                continue
            for delay,mask in paths[u].items():
                paths[v][delay+1] = paths[v].get(delay+1,0) | mask
    rows = []
    for u,x in enumerate(nodes):
        if x[0] != len(widths)-1 or x[2] != 'd':
            continue
        counts = []
        for phase in range(period):
            mask = 0
            for delay,sources in paths[u].items():
                if delay % period == phase:
                    mask |= sources
            counts.append(bin(mask).count('1'))
        rows.append(counts)
    return len(roots), [[min(r[p] for r in rows),max(r[p] for r in rows)]
                        for p in range(period)]

"""Check Settle endpoints, bounded degrees, bit coverage and aligned timing.

These are spatial/timing witnesses, not KV/SSD network execution or training.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reference'))
from backbones import multiscale
from backbones import settle_graphs as settle
from hb_lh_checks import check_output_paths


def check_alignment(widths, graph, period):
    nodes, _, outs, ins, _, _ = graph
    assert period > len(widths)
    delays, _ = multiscale.reachability(nodes, outs, widths, timing='settle')
    for u, (l, p, role, i) in enumerate(nodes):
        assert multiscale.support(delays[u]) == [l]
        assert l == 0 or ins[u]
        assert l == len(widths)-1 or outs[u]
        assert len(outs[u]) == len({v for v, k in outs[u]})
        for v, _ in outs[u]:
            rank, target_rank = l + 1, nodes[v][0] + 1
            assert target_rank > rank
            for t in (0, 1, 7):
                assert period*t+rank+(target_rank-rank) == period*t+target_rank
    # Complete input-position intervals have no in-flight message across them.
    assert max(1 + multiscale.support(d)[-1] for d in delays) < period


def lattice_checks():
    for depth in (5, 7, 9):
        for n in (16, 32, 64):
            widths, graph = settle.lattice(depth, n)
            nodes, _, outs, _, _, _ = graph
            h = (depth-1)//2
            # Independent word-address interpretation of the original definition.
            expected = set()
            for u, (l, p, _, i) in enumerate(nodes):
                if l == depth-1:
                    continue
                word_length = min(l, 2*h-l)
                word = tuple((p >> b) & 1 for b in reversed(range(word_length)))
                targets = [word+(a,) for a in (0, 1)] if l < h else [word[:-1]]
                for target in targets:
                    q = sum(a << b for b, a in enumerate(reversed(target)))
                    for j in (i, (i+1) % n):
                        expected.add(((l, p, 'd', i), (l+1, q, 'd', j)))
            actual = {(nodes[u], nodes[v]) for u, es in enumerate(outs) for v, _ in es}
            assert expected == actual
            assert len(nodes) == n*(2**h+2*(2**h-1))
            assert len(actual) == 8*n*(2**h-1)
            check_alignment(widths, graph, depth+2)
    widths, graph = settle.lattice(7, 64)
    assert len(graph[0]) == 1408 and sum(graph[4].values()) == 3584


def exchange_checks():
    rows = []
    for m in (2, 3, 4):
        widths = settle.exchange_widths(m)
        for n in (16, 32, 64):
            plain = settle.construct(widths, n=n)
            for h in (0, 4):
                graph = settle.construct(widths, n=n, h=h)
                check_output_paths(widths, graph)
                nodes, ids, outs, ins, kinds, _ = graph
                full = settle.construct(widths, n=n, h=h, terminal_hubs=True)
                kept = set(nodes)
                assert {(nodes[u], nodes[v], k) for u, es in enumerate(outs) for v, k in es} == {
                    (full[0][u], full[0][v], k) for u, es in enumerate(full[2]) for v, k in es
                    if full[0][u] in kept and full[0][v] in kept}
                assert len(full[0])-len(nodes) == h*widths[-1]
                actual_d = {(nodes[u], nodes[v]) for u, es in enumerate(outs)
                            for v, kind in es if kind == 'DDX'}
                expected_d = {(plain[0][u], plain[0][v]) for u, es in enumerate(plain[2])
                              for v, _ in es}
                assert actual_d == expected_d  # Hub deletion is exact.
                counts = {'DDX': 0, 'DH1': 0, 'HH1': 0, 'HD1': 0}
                for l, (w, v) in enumerate(zip(widths, widths[1:])):
                    # Each coarse group chooses itself and one different bit group.
                    pairs = max(w, v)*(2 if min(w, v) > 1 else 1)
                    counts['DDX'] += 2*n*pairs
                    if h:
                        if l < len(widths)-2:
                            counts['DH1'] += n*pairs
                            counts['HH1'] += h*pairs
                        counts['HD1'] += h*max(w, v)
                assert dict(kinds) == {k: v for k, v in counts.items() if v}
                assert max(len(es) for es in plain[2]) == 8
                assert max(len(es) for es in plain[3]) == 8
                assert all(nodes[v][0] == nodes[u][0]+1 for u, es in enumerate(outs) for v, _ in es)
                if h:
                    assert max(len(ins[u]) for u, x in enumerate(nodes) if x[2]=='h') == 4*(n//h+1)
                    assert max(len(outs[u]) for u, x in enumerate(nodes) if x[2]=='h') == 6
                check_alignment(widths, graph, len(widths)+1)
                if n == 16:
                    rows.append((len(widths), h, len(nodes), sum(kinds.values())))
            # Independent bit-cube count from every target in every plateau round.
            nodes, ids, outs, _, _, _ = plain
            ancestors = [0]*len(nodes)
            for p in range(2**m):
                for i in range(n):
                    ancestors[ids[m, p, 'd', i]] = 1 << (p*n+i)
            q = n.bit_length()-1
            for l in range(m, 2*m):
                for p in range(2**m):
                    for i in range(n):
                        u = ids[l, p, 'd', i]
                        for v, _ in outs[u]:
                            ancestors[v] |= ancestors[u]
                r = l-m+1
                for p in range(2**m):
                    for i in range(n):
                        mask = ancestors[ids[l+1, p, 'd', i]]
                        assert bin(mask).count('1') == 2**(r+min(r, q))
                        region_count = sum(bool((mask >> (a*n)) & ((1 << n)-1)) for a in range(2**m))
                        assert region_count == 2**r
    assert rows == [(7,0,288,1152),(7,4,356,1912),(10,0,736,3200),
                    (10,4,916,5368),(13,0,1760,7808),(13,4,2196,13144)]
    # g=1 and the smallest legal node set are well-defined, including deduplication.
    check_alignment(settle.exchange_widths(1), settle.construct(settle.exchange_widths(1), n=2), 5)
    print('PASS: 18 exchange configurations; exact counts, hub deletion, output reachability, degrees and every plateau target')


def main():
    lattice_checks()
    print('PASS: 9 original lattice configurations match the word-address definition; original period retained')
    exchange_checks()
    print('PASS: strict rank growth, aligned paths and D > M')


if __name__ == '__main__':
    main()

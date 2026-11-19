#!/usr/bin/env python3
"""Spatial/timing checks and a scalar PDG continuation witness.

This is not a KV/SSD network implementation or a training benchmark.
"""
from collections import defaultdict, deque
from copy import deepcopy
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reference'))
from backbones import multiscale as hb


def lh_graph(depth=3, rows=4, columns=8, feedback=False):
    base = [('h', a, p, 0) for a in range(depth + 1) for p in range(2 ** a)]
    base += [('d', depth, p, i) for p in range(2 ** depth) for i in range(rows * columns)]
    nodes = [('I', *v) for v in base] + [('O', *v) for v in reversed(base)]
    ids = {v: i for i, v in enumerate(nodes)}
    edges = []
    def side_edge(u, v, kind):
        edges.append((ids[('I', *u)], ids[('I', *v)], kind))
        edges.append((ids[('O', *v)], ids[('O', *u)], kind))
    for a in range(depth):
        for p in range(2 ** a):
            for q in (2 * p, 2 * p + 1):
                side_edge(('h', a, p, 0), ('h', a + 1, q, 0), 'hierarchy')
    for p in range(2 ** depth):
        for i in range(rows * columns):
            side_edge(('h', depth, p, 0), ('d', depth, p, i), 'star')
            x, y = divmod(i, columns)
            for j in ([i + 1] if y + 1 < columns else []) + ([i + columns] if x + 1 < rows else []):
                side_edge(('d', depth, p, i), ('d', depth, p, j), 'local')
    for v in base:
        edges.append((ids[('I', *v)], ids[('O', *v)], 'bridge'))
        if feedback and v[0] == 'h':
            edges.append((ids[('O', *v)], ids[('I', *v)], 'feedback'))
    assert len(edges) == len({(u, v) for u, v, _ in edges})
    return nodes, edges, ids['I', 'h', 0, 0, 0], ids['O', 'h', 0, 0, 0]


def topological(nodes, edges):
    degree = [0] * len(nodes)
    out = [[] for _ in nodes]
    for u, v, _ in edges:
        degree[v] += 1
        out[u].append(v)
    ready = deque(i for i, d in enumerate(degree) if not d)
    order = []
    while ready:
        u = ready.popleft()
        order.append(u)
        for v in out[u]:
            degree[v] -= 1
            if not degree[v]:
                ready.append(v)
    return order, out


def scalar_interval(state, pending, out, source, sink, start, stop):
    """All received nodes retain a sum and broadcast it, with unit edge delay.

    This legal scalar specialization makes private state affect future output.
    It exercises event fibers and pending messages, independently of KV/SSD.
    """
    state, pending = list(state), deepcopy(pending)
    records = []
    for theta in range(start, stop):
        fiber = defaultdict(list)
        for v, identity, value in pending.pop(theta, []):
            fiber[v].append((identity, value))
        if theta in (1, 5):
            fiber[source].append((('ext', theta, source), 1 if theta == 1 else 2))
        for v in sorted(fiber):
            before = state[v]
            state[v] += sum(value for _, value in fiber[v])
            records.append((theta, v, tuple(sorted(fiber[v])), before, state[v],
                            state[v] if v == sink else None))
            for w in out[v]:
                pending.setdefault(theta + 1, []).append((w, ('msg', theta, v), state[v]))
    return state, pending, records


def check_output_paths(widths, graph):
    nodes, _, outs, ins, _, _ = graph
    reachable = {u for u, x in enumerate(nodes) if x[0] == len(widths)-1 and x[2] == 'd'}
    pending = list(reachable)
    while pending:
        for u, _ in ins[pending.pop()]:
            if u not in reachable:
                reachable.add(u)
                pending.append(u)
    assert len(reachable) == len(nodes), [x for u, x in enumerate(nodes) if u not in reachable]
    assert all(x[2] == 'd' for x in nodes if x[0] == len(widths)-1)


def main():
    summaries = []
    for key, widths, extra in hb.PROFILES:
        graph = hb.construct(widths, add_dd4=extra)
        check_output_paths(widths, graph)
        nodes, ids, outs, ins, kinds, _ = graph
        full = hb.construct(widths, add_dd4=extra, terminal_hubs=True)
        kept = set(nodes)
        assert {(nodes[u], nodes[v], k) for u, es in enumerate(outs) for v, k in es} == {
            (full[0][u], full[0][v], k) for u, es in enumerate(full[2]) for v, k in es
            if full[0][u] in kept and full[0][v] in kept}
        assert len(full[0])-len(nodes) == 4*widths[-1]
        assert all(nodes[u][0] < nodes[v][0] for u, es in enumerate(outs) for v, _ in es)
        assert all(len(es) == len({v for v, _ in es}) for es in outs)
        assert sum(map(len, outs)) == sum(kinds.values()) == sum(map(len, ins))
        delays, _ = hb.reachability(nodes, outs, widths, only_detail=True)
        settle_delays, _ = hb.reachability(nodes, outs, widths, timing='settle')
        expected = list(range((len(widths) // 2), len(widths))) if not extra else [1, 2, 3, 4]
        for u, node in enumerate(nodes):
            assert hb.support(settle_delays[u]) == [node[0]]
            if node[0] == len(widths) - 1 and node[2] == 'd':
                assert hb.support(delays[u]) == expected
        summaries.append((key, len(nodes), sum(kinds.values())))
    widths = hb.PROFILES[-1][1]
    expected_sizes = [(16, (1, 1, 1), 3024, 23840, (32, 24)),
                      (32, (1, 1, 1), 5456, 41760, (48, 24)),
                      (64, (2, 2, 1), 10320, 78912, (80, 28)),
                      (128, (4, 4, 2), 20048, 153920, (144, 40))]
    for n, probes, count, edges, degrees in expected_sizes:
        nodes, _, outs, ins, kinds, _ = hb.construct(widths, n=n, probes=probes)
        assert (len(nodes), sum(kinds.values())) == (count, edges)
        actual = tuple(max(len(es[u]) for u, x in enumerate(nodes) if x[2] == 'h') for es in (ins, outs))
        assert actual == degrees
        assert tuple(max(len(es[u]) for u, x in enumerate(nodes) if x[2] == 'd') for es in (ins, outs)) == (12, 14)
    # Explicitly verify induced deletion and unchanged worker paths.
    widths = dict((key, w) for key, w, _ in hb.PROFILES)['9']
    all_nodes, _, all_out, _, _, _ = hb.construct(widths)
    ordinary_edges = {(all_nodes[u], all_nodes[v], k) for u, es in enumerate(all_out)
                      for v, k in es if k.startswith('DD')}
    for placement in ('none', 'checkerboard'):
        graph = hb.construct(widths, placement=placement)
        nodes, _, outs, _, _, _ = graph
        edges = {(nodes[u], nodes[v], k) for u, es in enumerate(outs) for v, k in es}
        assert {e for e in edges if e[2].startswith('DD')} == ordinary_edges
        assert len(nodes) == 16*sum(widths)+(2*sum(widths[:-1]) if placement == 'checkerboard' else 0)
        check_output_paths(widths, graph)
        dd, _ = hb.reachability(nodes, outs, widths, only_detail=True)
        assert all(hb.support(dd[u]) == [4, 5, 6, 7, 8]
                   for u, x in enumerate(nodes) if x[0] == 8 and x[2] == 'd')
    # Real HB path support gives old/new co-arrival at 1+4*0+6=1+4*1+2.
    potential, _ = hb.reachability(all_nodes, all_out, widths)
    assert any(x[0] == 8 and x[2] == 'd' and {2, 6} <= set(hb.support(potential[u]))
               for u, x in enumerate(all_nodes))
    assert 1 + 6 == 1 + 4 + 2 and 1 + 8 != 1 + 4 + 8
    lh_summary = []
    for rows, cols, expected in [(4, 4, 286), (4, 8, 542), (8, 8, 1054)]:
        for feedback in (False, True):
            nodes, edges, source, sink = lh_graph(rows=rows, columns=cols, feedback=feedback)
            assert len(nodes) == expected
            order, out = topological(nodes, edges)
            indegree = [0] * len(nodes)
            for _, v, _ in edges:
                indegree[v] += 1
            for role, maximum in [('d', 3), ('h', rows * cols + 1)]:
                assert max(indegree[u] for u, x in enumerate(nodes) if x[1] == role) == maximum
                assert max(len(out[u]) for u, x in enumerate(nodes) if x[1] == role) == maximum
            if feedback:
                assert len(order) < len(nodes)
                assert sink in out[source] and source in out[sink]
                first = scalar_interval([0] * len(nodes), {}, out, source, sink, 0, 3)
                assert any(v == source for v, _, _ in first[1][3])
                continued = scalar_interval(first[0], first[1], out, source, sink, 3, 14)
                whole = scalar_interval([0] * len(nodes), {}, out, source, sink, 0, 14)
                assert whole[:2] == continued[:2]
                assert whole[2] == first[2] + continued[2]
            else:
                assert len(order) == len(nodes)
                paths = [set() for _ in nodes]
                paths[source] = {0}
                for u in order:
                    for v in out[u]:
                        paths[v].update(t + 1 for t in paths[u])
                assert {1, 3, 5, 7, 9, 10, 11, 12} <= paths[sink]
                assert {d % 4 for d in paths[sink]} == set(range(4))
            # The base hub/worker partition has real intra-region messages.
            def region(v):
                side, role, a, p, _ = nodes[v]
                return (side, a, p) if a == 3 else (side, a, p, role)
            assert any(region(u) == region(v) for u, v, _ in edges)
            lh_summary.append({'n': rows * cols, 'feedback': feedback,
                               'nodes': len(nodes), 'edges': len(edges)})
    print(json.dumps({'HB': summaries, 'LH': lh_summary}, ensure_ascii=False))
    print('PASS: 7 depth profiles; 4 degree/size cases; terminal/placement deletion and output reachability; aligned ranks; '
          'LH DAG/cycle/phase witnesses; 3 exact scalar cut continuations.')


if __name__ == '__main__':
    main()

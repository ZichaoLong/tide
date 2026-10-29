"""Static physical topologies; no numeric trajectory or Torch dependency."""
from itertools import product
from .legacy_topology import ranked_graph, graph_counts


def positive(name, x, maximum=200000):
    if type(x) is not int or not 1 <= x <= maximum:
        raise ValueError(f"{name} must be in 1..{maximum}")


def layered(kind, *, depth=1, region_width=2, branches=2, offsets=None):
    positive("depth", depth, 128)
    positive("region_width", region_width, 4096)
    if kind == "fc":
        if offsets is not None:
            raise ValueError("FC does not have lateral offsets")
        if depth*region_width > 200000 or (depth-1)*region_width**2 > 10000000:
            raise ValueError("FC static node/edge limit exceeded")
        ranks = list(range(1, depth+1))
        edges = [[j*region_width+a, (j+1)*region_width+b, 1] for j in range(depth-1)
                 for a in range(region_width) for b in range(region_width)]
    elif kind == "hb":
        positive("branches", branches, 16)
        if branches < 2:
            raise ValueError("HB requires at least two branches")
        regions = 2*sum(branches**i for i in range(depth))+branches**depth
        if regions*region_width > 200000:
            raise ValueError("HB static node limit exceeded")
        words = [tuple(product(range(branches), repeat=depth-abs(j-depth))) for j in range(2*depth+1)]
        owners, ranks = {}, []
        for j, addresses in enumerate(words):
            for word in addresses:
                owners[j, word] = len(ranks)
                ranks.append(j+1)
        offsets = ((0,),)*(2*depth) if offsets is None else tuple(tuple(row) for row in offsets)
        if len(offsets) != 2*depth or any(not row or 0 not in row or len(set(row)) != len(row)
                or any(type(x) is not int or not 0 <= x < region_width for x in row) for row in offsets):
            raise ValueError("HB offsets must be distinct coordinates including zero")
        if 2*sum(branches**i for i in range(1, depth+1))*region_width*max(map(len, offsets)) > 10000000:
            raise ValueError("HB static edge limit exceeded")
        edges = []
        for j in range(2*depth):
            links = ([(a, a+(b,)) for a in words[j] for b in range(branches)] if j < depth else
                     [(a+(b,), a) for a in words[j+1] for b in range(branches)])
            for a, b in sorted(links):
                for s in range(region_width):
                    for delta in sorted(offsets[j]):
                        edges.append([owners[j, a]*region_width+s, owners[j+1, b]*region_width+(s+delta)%region_width, 1])
    else:
        raise ValueError("unknown layered topology")
    n = len(ranks)*region_width
    graph = dict(nodes=n, regions=len(ranks), node_regions=[j for j in range(len(ranks)) for _ in range(region_width)],
                 ranks=ranks, edges=edges, inputs=list(range(region_width)),
                 outputs=list(range(n-region_width, n)), kind="ranked-local")
    graph["counts"] = graph_counts(graph)
    return graph


def topology(kind, *, layers=15, region_width=32, branches=2, depth=1):
    if kind in {"fc", "hb"}:
        return layered(kind, depth=depth, branches=branches, region_width=region_width)
    if kind not in {"ranked", "dag", "pdg"}:
        raise ValueError("unknown graph family")
    positive("region_width", region_width, 4096)
    # Keep the historical width-32 packet exactly. Other widths use complete
    # neighborhoods; this choice is static, recorded in the resulting edges.
    span = max(x for x in range(1, min(8, region_width)+1) if region_width % x == 0)
    g = ranked_graph(layers=layers, region_width=region_width, fanout=min(4, span),
                     local_span=span, cross_every=8, delayed=kind != "ranked")
    if kind == "pdg":
        g["edges"] += [[a, b, 1] for a, b in zip(g["outputs"], g["inputs"])]
        g["kind"] = "positive-delay"
        g["counts"]["body_edges"] = len(g["edges"])
        g["counts"]["encoded_edges"] += len(g["inputs"])
        # The feedforward path length is an input grid choice, not a claimed
        # longest path of the cyclic graph.
        g["counts"]["feedforward_delay"] = g["counts"].pop("longest_body_delay")
    return g


def lh_inspired(*, depth=3, region_width=32, branches=2, fanout=4, cross_every=8, skip=2):
    """HB expansion/contraction with local offsets, cross-region and skip wires.

    This is a named core-5 topology, not LH's local functions or its old fixture.
    Extra physical wires do not collapse atoms or alter regional candidate sets.
    """
    positive("fanout", fanout, region_width)
    if type(cross_every) is not int or cross_every < 0 or type(skip) is not int or skip < 0:
        raise ValueError("cross_every and skip must be nonnegative integers")
    graph = layered("hb", depth=depth, region_width=region_width, branches=branches,
                    offsets=(tuple(range(fanout)),)*(2*depth))
    layers = [[j for j, rank in enumerate(graph["ranks"]) if rank == i+1] for i in range(2*depth+1)]
    existing = {tuple(e) for e in graph["edges"]}
    additions = []
    for i, row in enumerate(layers[:-1]):
        for a, source in enumerate(row):
            if cross_every and len(layers[i+1]) > 1:
                target = layers[i+1][(a+1) % len(layers[i+1])]
                additions.extend((source*region_width+s, target*region_width+s, 1)
                                 for s in range(0, region_width, cross_every))
            if skip and i+skip < len(layers):
                target = layers[i+skip][a % len(layers[i+skip])]
                additions.extend((source*region_width+s, target*region_width+s, skip)
                                 for s in range(region_width))
    graph["edges"].extend([list(e) for e in additions if e not in existing])
    graph["counts"] = graph_counts(graph)
    return graph

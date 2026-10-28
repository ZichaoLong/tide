"""Static physical topologies; no numeric trajectory or Torch dependency."""
from itertools import product
from flow_topology import ranked_graph, graph_counts


def positive(name, x, maximum=200000):
    if type(x) is not int or not 1 <= x <= maximum:
        raise ValueError(f"{name} must be in 1..{maximum}")


def layered(kind, *, depth=1, region_width=2, branches=2):
    positive("depth", depth, 128)
    positive("region_width", region_width, 4096)
    if kind == "fc":
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
        edges = []
        for j in range(2*depth):
            links = ([(a, a+(b,)) for a in words[j] for b in range(branches)] if j < depth else
                     [(a+(b,), a) for a in words[j+1] for b in range(branches)])
            for a, b in sorted(links):
                for s in range(region_width):
                    edges.append([owners[j, a]*region_width+s, owners[j+1, b]*region_width+s, 1])
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

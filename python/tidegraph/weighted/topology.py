"""0770593 topologies/fully-connected-layered and hb-lattice §1–4."""
from itertools import product
from ..graph import Edge, Graph, Node, Region
from ..settle import SettleGraph
from .config import positive_int


def fully_connected(depth, region_width):
    positive_int("depth",depth); positive_int("region_width",region_width)
    n = region_width
    graph = Graph(tuple(Node(layer) for layer in range(depth) for _ in range(n)),
                  tuple(Edge(layer*n+s,(layer+1)*n+t,1) for layer in range(depth-1)
                        for s in range(n) for t in range(n)),
                  tuple(Region(n) for _ in range(depth)),tuple(range(n)),tuple(range((depth-1)*n,depth*n)))
    return SettleGraph(graph,tuple(range(1,depth+1)),binding="ports",time_stride=depth+2)


def hb_lattice(branches, depth, region_width, offsets=None, relations=None):
    """Base recursive relation or explicit supersets; offsets are true sets."""
    positive_int("branches",branches,2); positive_int("depth",depth); positive_int("region_width",region_width)
    layers, n = 2*depth+1, region_width
    addresses = [tuple(product(range(branches),repeat=depth-abs(layer-depth))) for layer in range(layers)]
    offsets = ((0,),)*(layers-1) if offsets is None else tuple(tuple(row) for row in offsets)
    if len(offsets) != layers-1 or any(not row or 0 not in row or len(set(row)) != len(row)
                                      or any(type(x) is not int or not 0 <= x < n for x in row) for row in offsets):
        raise ValueError("HB offsets must be distinct lateral coordinates including zero")
    base = []
    for layer in range(layers-1):
        base.append({(word,word+(a,)) for word in addresses[layer] for a in range(branches)}
                    if layer < depth else {(word+(a,),word) for word in addresses[layer+1] for a in range(branches)})
    relations = base if relations is None else [set(row) for row in relations]
    if len(relations) != layers-1 or any(not base[i] <= row or
            not row <= set(product(addresses[i],addresses[i+1])) for i,row in enumerate(relations)):
        raise ValueError("HB relations must contain the recursive base within adjacent layers")
    owners, ranks = {}, []
    for layer,words in enumerate(addresses):
        for word in words:
            owners[layer,word] = len(ranks); ranks.append(layer+1)
    nodes = tuple(Node(j) for j in range(len(ranks)) for _ in range(n))
    edges = tuple(Edge(owners[layer,a]*n+s,owners[layer+1,b]*n+(s+delta)%n,1)
                  for layer,row in enumerate(relations) for a,b in sorted(row)
                  for s in range(n) for delta in sorted(offsets[layer]))
    graph = Graph(nodes,edges,tuple(Region(n) for _ in ranks),tuple(range(n)),tuple(range(len(nodes)-n,len(nodes))))
    return SettleGraph(graph,tuple(ranks),binding="ports",time_stride=layers+2)

"""Online closure bounds from actual pending events; no tensor or value prepass."""
from collections import defaultdict
import heapq


class ClosurePlan:
    """A region's unfinished event cannot precede any seed-to-region path.

    Seeds are *actual* queued fibers. A future, unpublished message must descend
    from a seed through a positive-delay wire. Region ownership is a conservative
    envelope: it also accounts for selection/history shared by member nodes.
    Bounds below certify no *additional* arrival before a region's ready prefix.
    They do not create candidates or compute their selection.
    """
    def __init__(self, graph):
        self.graph = graph
        edges = {}
        for edge in graph.edges:
            pair = graph.nodes[edge.source].region, graph.nodes[edge.target].region
            edges[pair] = min(edges.get(pair, edge.delay), edge.delay)
        self.outgoing = [[] for _ in graph.regions]
        for (source, target), delay in sorted(edges.items()):
            self.outgoing[source].append((target, delay))

    def ready(self, fibers, stop, max_events):
        # The limit concerns simultaneously materialized fibers, not cumulative
        # work, clock span, or an enumeration of every possible future event.
        if len(fibers) > max_events:
            raise ValueError("greedy live-fiber capacity exceeded")
        frames = defaultdict(set)
        seeds = defaultdict(dict)
        for (batch, node, time), atoms in fibers.items():
            if atoms and time < stop:
                region = self.graph.nodes[node].region
                frames[batch, region, time].add(node)
                owner = seeds[batch]
                owner[region] = min(owner.get(region, time), time)
        bounds, relaxations = {}, 0
        for batch, owner in sorted(seeds.items()):
            earliest = [stop] * len(self.outgoing)
            todo = []
            for region, time in owner.items():
                earliest[region] = time
                heapq.heappush(todo, (time, region))
            while todo:
                time, source = heapq.heappop(todo)
                if time != earliest[source]:
                    continue
                for target, delay in self.outgoing[source]:
                    relaxations += 1
                    arrival = min(stop, time + delay)
                    if arrival < earliest[target]:
                        earliest[target] = arrival
                        heapq.heappush(todo, (arrival, target))
            safe = [stop] * len(self.outgoing)
            for source, wires in enumerate(self.outgoing):
                for target, delay in wires:
                    safe[target] = min(safe[target], earliest[source] + delay)
            bounds[batch] = safe
        blocks = defaultdict(list)
        for (batch, region, time), nodes in sorted(frames.items(), key=lambda x: (x[0][2], x[0][0], x[0][1])):
            if time < bounds[batch][region]:
                blocks[region].append((batch, region, time, nodes))
        if frames and not blocks:
            raise RuntimeError("greedy closure failed to expose the earliest event")
        return blocks, relaxations

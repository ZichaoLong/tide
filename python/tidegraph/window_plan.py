"""PDG §9.5 / time-batch §6.5: region components, sealed-window execution.

Only topology is cached. Every value, candidate and message comes from this
execution. A whole region is indivisible for dependency analysis.
"""
from collections import defaultdict
from .greedy_plan import ClosurePlan


class Components:
    def __init__(self, graph):
        n = len(graph.regions)
        outgoing, incoming = [set() for _ in range(n)], [set() for _ in range(n)]
        for edge in graph.edges:
            a, b = graph.nodes[edge.source].region, graph.nodes[edge.target].region
            outgoing[a].add(b); incoming[b].add(a)
        # Iterative DFS also handles very deep/wide graph representatives.
        order, seen = [], set()
        for root in range(n):
            if root in seen:
                continue
            seen.add(root); stack = [(root, iter(sorted(outgoing[root])))]
            while stack:
                node, children = stack[-1]
                child = next(children, None)
                if child is None:
                    order.append(node); stack.pop()
                elif child not in seen:
                    seen.add(child); stack.append((child, iter(sorted(outgoing[child]))))
        self.groups, seen = [], set()
        for root in reversed(order):
            if root in seen:
                continue
            group, stack = [], [root]; seen.add(root)
            while stack:
                node = stack.pop(); group.append(node)
                for child in sorted(incoming[node], reverse=True):
                    if child not in seen:
                        seen.add(child); stack.append(child)
            self.groups.append(tuple(sorted(group)))
        self.owner = {region: i for i, group in enumerate(self.groups) for region in group}
        self.cyclic = [len(g) > 1 or g[0] in outgoing[g[0]] for g in self.groups]
        internal = [[] for _ in self.groups]
        for edge in graph.edges:
            a, b = (self.owner[graph.nodes[node].region] for node in (edge.source, edge.target))
            if a == b:
                internal[a].append(edge)
        self.closures = [ClosurePlan(graph, g, wires) if cycle else None
                         for g, cycle, wires in zip(self.groups, self.cyclic, internal)]


class ComponentFibers(defaultdict):
    def __init__(self, graph, components):
        super().__init__(list)
        self.graph, self.components = graph, components
        self.parts = [dict() for _ in components.groups]

    def __missing__(self, key):
        value = super().__missing__(key)
        self.parts[self.components.owner[self.graph.nodes[key[1]].region]][key] = value
        return value

    def __delitem__(self, key):
        del self.parts[self.components.owner[self.graph.nodes[key[1]].region]][key]
        super().__delitem__(key)


class WindowPlan:
    def __init__(self, graph):
        self.graph, self.components, self.cursor = graph, graph.region_components, 0
        self.causal_blocks = 0

    def ready(self, fibers, stop, max_events):
        if len(fibers) > max_events:
            raise ValueError("window live-fiber capacity exceeded")
        while self.cursor < len(self.components.groups):
            part = fibers.parts[self.cursor]
            closure = self.components.closures[self.cursor]
            if closure is not None:
                blocks, relaxed = closure.ready(part, stop, max_events)
                if blocks:
                    self.causal_blocks += 1
                    return blocks, relaxed
            else:
                frames = defaultdict(set)
                for (batch, node, time), atoms in part.items():
                    if atoms and time < stop:
                        frames[time, batch].add(node)
                if frames:
                    region = self.components.groups[self.cursor][0]
                    return {region: [(b, region, t, nodes) for (t, b), nodes in sorted(frames.items())]}, 0
            self.cursor += 1
        return {}, 0

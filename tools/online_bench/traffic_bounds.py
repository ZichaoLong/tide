"""Static traffic bounds for validated finite DAG consumer packets, without Torch.

The consumer's boundary injects one vector per sample/position. All legal choices
of at most the declared region budget are included; no numerical execution,
future route, observed event or parameter value enters this accounting.
"""
from collections import Counter, defaultdict, deque

MAX = 2**63-1


def checked(value):
    if not 0 <= value <= MAX:
        raise ValueError("consumer traffic extent overflow")
    return value


def traffic_bounds(packet, owners):
    """Return per-position upper bounds, retaining parallel edge multiplicity.

    Unequal path lengths use enclosing integer intervals, without enumerating
    exponentially many paths or requiring actual activity. This can be loose;
    the online scheduler is unchanged and never consumes these bounds as routes.
    Multiply by physical/logical sample and position counts for capacity phases.
    """
    g, c = packet["graph"], packet["workload"]
    n, regions = g["nodes"], g["regions"]
    if (len(owners) != n+2 or not owners or any(type(v) is not int or v < 0 for v in owners)
            or list(owners[-2:]) != [0,0] or set(owners) != set(range(max(owners)+1))):
        raise ValueError("traffic owners must cover encoded nodes with owner-zero boundaries")
    count = max(owners)+1
    membership = g["node_regions"]
    members = [[] for _ in range(regions)]
    for i,r in enumerate(membership):
        members[r].append(i)
    if c["budget"] < 1 or any(len(row) < c["budget"] for row in members):
        raise ValueError("invalid traffic region budget")
    outgoing, incoming, degree = [[] for _ in range(n)], [[] for _ in range(n)], [0]*n
    for a,b,delay in g["edges"]:
        if not 0 <= a < n or not 0 <= b < n or not 0 < delay <= MAX:
            raise ValueError("invalid traffic edge")
        outgoing[a].append((b,delay));incoming[b].append(a);degree[b] += 1
    external, readout = Counter(g["inputs"]), Counter(g["outputs"])
    low, high = [MAX]*n, [-1]*n
    aligned = g["kind"] == "ranked-local"
    for node in external:
        low[node] = high[node] = g["ranks"][membership[node]] if aligned else 1
    queue = deque(i for i in range(n) if degree[i] == 0)
    visited = 0
    while queue:
        node = queue.popleft();visited += 1
        if high[node] < 0:
            raise ValueError("traffic graph has an input-unreachable node")
        for target,delay in outgoing[node]:
            low[target] = min(low[target], checked(low[node]+delay))
            high[target] = max(high[target], checked(high[node]+delay))
            degree[target] -= 1
            if degree[target] == 0:
                queue.append(target)
    if visited != n:
        raise ValueError("finite DAG traffic accounting cannot admit a feedback packet")
    if aligned and any(low[i] != high[i] or low[i] != g["ranks"][membership[i]] for i in range(n)):
        raise ValueError("rank-aligned traffic packet has incompatible delays")
    output_low = min((max(g["ranks"])+1) if aligned else low[i]+1 for i in readout)
    output_high = max((max(g["ranks"])+1) if aligned else high[i]+1 for i in readout)
    if min(low) < 1 or max(max(high),output_high) >= c["stride"]:
        raise ValueError("traffic position is not sealed before the next injection")
    node_frames = [checked(hi-lo+1) for lo,hi in zip(low,high)]
    region_frames = [checked(max(high[i] for i in row)-min(low[i] for i in row)+1) for row in members]
    top = lambda costs: sum(sorted(costs,reverse=True)[:c["budget"]])
    # For each target, a selected source can use every parallel physical wire.
    node_atoms = [external[i] for i in range(n)]
    for node in range(n):
        by_region = defaultdict(Counter)
        for source in incoming[node]:
            by_region[membership[source]][source] += 1
        node_atoms[node] += sum(region_frames[r]*top(sources.values()) for r,sources in by_region.items())
    # At one region-time frame only budget sources may emit, so summing each
    # target's separate maximum would unnecessarily multiply the same choice.
    owner_atoms = [sum(external[i] for i in range(n) if owners[i] == d) for d in range(count)]
    owner_body_emissions = [0]*count
    output_atoms = 0
    for r,row in enumerate(members):
        target_counts = {i:Counter(owners[target] for target,_ in outgoing[i]) for i in row}
        for d in range(count):
            owner_atoms[d] += region_frames[r]*top(target_counts[i][d] for i in row)
            owner_body_emissions[d] += region_frames[r]*top(len(outgoing[i]) for i in row if owners[i] == d)
        output_atoms += region_frames[r]*top(readout[i] for i in row)
    node_events = [min(frames,atoms) for frames,atoms in zip(node_frames,node_atoms)]
    owner_events = [min(owner_atoms[d],sum(node_events[i] for i in range(n) if owners[i] == d)) for d in range(count)]
    output_frames = min(output_atoms,output_high-output_low+1)
    for values in [node_frames,region_frames,node_atoms,node_events,owner_atoms,owner_events,owner_body_emissions,[output_atoms,output_frames]]:
        for value in values:checked(value)
    return dict(schema="tide-consumer-traffic-bound-v1", scope="per sample/input position; validated DAG packet, empty initial state; all legal selections; not actual events",
        node_offset_intervals=list(map(list,zip(low,high))),region_frame_factors=region_frames,
        node_event_factors=node_events,node_atom_factors=node_atoms,
        owner_body_atom_factors=owner_atoms,owner_body_event_factors=owner_events,
        owner_body_emission_factors=owner_body_emissions,
        output_atom_factor=output_atoms,output_frame_factor=output_frames,
        maximum_position_offset=output_high,positions_are_sealed=True)

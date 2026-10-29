"""Versioned, dependency-free active topology packets for complete-flow clients.

The historical capacity presets are deliberately untouched. A ranked packet
describes one Settle body and its equivalent TimedDAG/PDG encoding. Every body
node lies on an input-to-output path; selection still controls actual work.
"""
import hashlib
import json
from collections import deque


def integer(value, name, minimum=1, maximum=2**63-1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f'{name} requires an integer in [{minimum}, {maximum}]')
    return value


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def ranked_graph(*, layers=15, region_width=32, fanout=4, skip=1,
                 local_span=8, cross_every=8, rank_gap=1, delayed=False):
    """Local neighborhoods plus sparse cross-neighborhood and skip-layer wires.

    Adjacent offset zero guarantees reachability of every node. Physical wires
    have independent identities, even where the skip and local paths reconverge.
    ``delayed`` introduces unequal arrivals and is a TimedDAG/PDG stress only.
    """
    for name, value, maximum in (
        ('layers', layers, 2048), ('region_width', region_width, 4096),
        ('fanout', fanout, region_width), ('local_span', local_span, region_width),
        ('rank_gap', rank_gap, 1024)):
        integer(value, name, maximum=maximum)
    integer(skip, 'skip', 0, region_width)
    integer(cross_every, 'cross_every', 0, 4096)
    if type(delayed) is not bool: raise ValueError('delayed requires bool')
    if layers < 2 or region_width % local_span or fanout > local_span:
        raise ValueError('need at least two layers, complete local groups and fanout <= local_span')
    if layers * region_width > 200000:
        raise ValueError('body-node capacity exceeded')
    ranks = [1 + i*rank_gap for i in range(layers)]
    edges = []
    for layer in range(layers-1):
        for column in range(region_width):
            group = column // local_span * local_span
            for offset in range(fanout):
                target = group + (column + offset) % local_span
                # Keep offset zero. The last other wire occasionally crosses
                # a neighborhood, so locality is configurable and measurable.
                if (offset and offset == fanout-1 and cross_every
                        and (layer*region_width+column) % cross_every == 0):
                    target = (target + local_span) % region_width
                delay = rank_gap + (1 if delayed and offset and column % 3 == 0 else 0)
                edges.append([layer*region_width+column, (layer+1)*region_width+target, delay])
            if layer+2 < layers:
                for offset in range(skip):
                    target = (column+offset) % region_width
                    edges.append([layer*region_width+column, (layer+2)*region_width+target, 2*rank_gap])
    graph = dict(nodes=layers*region_width, regions=layers,
                 node_regions=[i//region_width for i in range(layers*region_width)],
                 ranks=ranks, edges=edges, inputs=list(range(region_width)),
                 outputs=list(range((layers-1)*region_width, layers*region_width)),
                 kind='timed-local' if delayed else 'ranked-local')
    graph['counts'] = graph_counts(graph)
    return graph


def graph_counts(graph):
    n = integer(graph['nodes'], 'nodes', maximum=200000)
    forward, reverse, indegree = [[] for _ in range(n)], [[] for _ in range(n)], [0]*n
    for edge in graph['edges']:
        if len(edge) != 3:
            raise ValueError('wire requires source, target, delay')
        a, b, delay = edge
        integer(a, 'source', 0, n-1); integer(b, 'target', 0, n-1); integer(delay, 'delay')
        forward[a].append((b, delay)); reverse[b].append((a, delay)); indegree[b] += 1
    def reachable(roots, adjacency):
        seen = set(roots); pending = list(roots)
        for node in pending:
            integer(node, 'boundary node', 0, n-1)
            for neighbor, _ in adjacency[node]:
                if neighbor not in seen: seen.add(neighbor); pending.append(neighbor)
        return seen
    touched = reachable(graph['inputs'], forward)
    useful = reachable(graph['outputs'], reverse)
    if len(touched) != n or len(useful) != n:
        raise ValueError('every body node must be reachable from input and reach output')
    queue = deque(i for i, count in enumerate(indegree) if count == 0)
    order = []; distance = [0]*n
    while queue:
        node = queue.popleft(); order.append(node)
        for target, delay in forward[node]:
            distance[target] = max(distance[target], distance[node]+delay)
            indegree[target] -= 1
            if not indegree[target]: queue.append(target)
    if len(order) != n: raise ValueError('ranked/TimedDAG packet contains a cycle')
    return dict(body_nodes=n, body_edges=len(graph['edges']), reachable_nodes=len(touched),
                output_reachable_nodes=len(useful), longest_body_delay=max(distance),
                encoded_nodes=n+2, encoded_edges=len(graph['edges'])+len(graph['inputs'])+len(graph['outputs']))


def make_packet(*, graph, memory='add', width=8, batch=2, tokens=3, vocab=17,
                budget=1, seed=7, clear=True):
    if memory not in ('add', 'attention'): raise ValueError('unknown memory')
    for name, value in (('width', width), ('batch', batch), ('tokens', tokens), ('vocab', vocab)):
        integer(value, name)
    integer(seed, 'seed', 0); integer(budget, 'budget')
    if width % 4 or batch > 512 or tokens > 32 or vocab < 2 or type(clear) is not bool:
        raise ValueError('unsupported width/batch/window/vocabulary/clear configuration')
    counts = graph_counts(graph)
    if graph.get('counts') != counts: raise ValueError('graph count inventory mismatch')
    regions = graph['node_regions']; ranks = graph['ranks']
    if len(regions) != graph['nodes'] or len(ranks) != graph['regions']:
        raise ValueError('region inventory mismatch')
    members = [0]*graph['regions']
    for r in regions: members[integer(r, 'region', 0, len(members)-1)] += 1
    if min(members) < budget: raise ValueError('selection budget exceeds a region')
    if len(set(ranks)) != len(ranks) or any(type(r) is not int or r < 1 for r in ranks):
        raise ValueError('distinct positive region ranks required')
    aligned = all(delay == ranks[regions[b]]-ranks[regions[a]] for a,b,delay in graph['edges'])
    if graph['kind'] == 'ranked-local' and not aligned: raise ValueError('invalid rank-aligned packet')
    # Input/output encoding is part of every flow. Delayed DAG outputs use a
    # separately declared sealed window rather than pretending to be Settle.
    stride = max(ranks)+2 if aligned else counts['longest_body_delay']+3
    integer(tokens*stride, 'window stop')
    d, n, e = width, graph['nodes'], len(graph['edges'])
    parameters = e*d*d+n*d+e+len(graph['inputs'])+2*vocab*d
    if memory == 'attention': parameters += 4*n*d*d
    payload = dict(schema='tide-complete-flow-workload-v1', graph=graph,
                   workload=dict(memory=memory, width=width, batch=batch, tokens=tokens,
                                 vocab=vocab, budget=budget, seed=seed, clear=clear, stride=stride),
                   families=['settle', 'timed-dag', 'pdg'] if aligned else ['timed-dag', 'pdg'],
                   counts=dict(counts, parameters=parameters, parameter_sharing='none'),
                   input_policy='(token*7+batch*3)%vocab; embedding/head included',
                   state_boundary='empty state per window; training parameters and optimizer persist')
    return dict(payload, sha256=hashlib.sha256(canonical_bytes(payload)).hexdigest())


def validate_packet(packet):
    if packet.get('schema') != 'tide-complete-flow-workload-v1': raise ValueError('unknown packet schema')
    payload = {k:v for k,v in packet.items() if k != 'sha256'}
    if packet.get('sha256') != hashlib.sha256(canonical_bytes(payload)).hexdigest():
        raise ValueError('packet content hash mismatch')
    config = {k:v for k,v in packet['workload'].items() if k != 'stride'}
    expected = make_packet(graph=packet['graph'], **config)
    if expected != packet: raise ValueError('inconsistent packet counts or derived fields')
    return packet


def native_text(packet):
    validate_packet(packet)
    g, c = packet['graph'], packet['workload']
    header = [g['nodes'], g['regions'], len(g['edges']), len(g['inputs']), len(g['outputs']),
              c['stride'], c['budget'], int(c['clear'])]
    rows = ['TIDE_COMPLETE_FLOW_1', ' '.join(map(str, header)),
            ' '.join(map(str, g['node_regions'])), ' '.join(map(str, g['ranks'])),
            ' '.join(map(str, g['inputs'])), ' '.join(map(str, g['outputs']))]
    rows.extend(' '.join(map(str, edge)) for edge in g['edges'])
    return '\n'.join(rows)+'\n'

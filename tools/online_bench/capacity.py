"""Shape envelopes for the declared Add/Attention consumer, without Torch.

These estimates are intentionally conservative, not vendor allocation proofs.
Local workspace budgets are ceilings, not allocations, and are never summed as
HBM demand. Only physical row maxima may shrink; logical capacities stay fixed.
"""
from dataclasses import dataclass, asdict, replace
from .head_budget import head_budget

MIB = 1024**2
MAX = 2**63-1


class MemoryRefusal(ValueError):
    """Valid geometry cannot fit even at the minimum operator row counts."""


def checked(value):
    if not 0 <= value <= MAX:
        raise ValueError('consumer memory extent overflow')
    return value


@dataclass
class Geometry:
    width: int
    batch: int
    vocab: int
    windows: int
    payload: int
    attention: bool
    training: bool
    adamw: bool
    diagnostics: bool
    regions: int
    sources: list
    slots: list
    edges: list
    devices: int
    locality: bool = True
    sample_chunks: int = 1
    context_bytes: int = 0


@dataclass
class Chunks:
    full: int = 16
    emission: int = 16
    aggregate: int = 8
    attention: int = 8
    keys: int = 128
    reverse: int = 16
    head: int = 1


@dataclass
class Capacities:
    queue: int = 1024
    arrivals: int = 1024
    outputs: int = 1024
    trace: int = 4096
    kv: int = 128
    kv_trace: int = 4096
    program: int = 64*MIB


def placement(g):
    """Same stable, capacity-preserving static locality heuristic as the backend."""
    n = len(g.sources)
    costs = [checked(1+g.payload*(g.width+s*(g.width*g.width+g.width))) for s in g.slots]+[1, 1]
    order = sorted(range(n+2), key=lambda i: -costs[i])
    owners, loads, members = [0]*(n+2), [0]*g.devices, [0]*g.devices
    for i in order:
        d = min(range(g.devices), key=lambda d: loads[d])
        owners[i] = d; loads[d] = checked(loads[d]+costs[i]); members[d] += 1
    if not g.locality:
        return owners
    adjacent = [[] for _ in owners]
    for a, b in g.edges:
        if a != b:
            adjacent[a].append(b); adjacent[b].append(a)
    limit = max(loads)
    for _ in range(2):
        for i in order:
            old = owners[i]
            if members[old] <= 1:
                continue
            affinity = [0]*g.devices
            for other in adjacent[i]:
                affinity[owners[other]] += 1
            new = old
            for d in range(g.devices):
                if loads[d] <= limit-costs[i] and affinity[d] > affinity[new]:
                    new = d
            if new != old:
                loads[old] -= costs[i]; loads[new] += costs[i]
                members[old] -= 1; members[new] += 1; owners[i] = new
    for _ in range(2):
        for i in order:
            old, gain, best = owners[i], 0, -1
            affinity = [0]*g.devices
            for other in adjacent[i]:
                affinity[owners[other]] += 1
            for j in range(n+2):
                new = owners[j]
                if new == old or loads[old]-costs[i] > limit-costs[j] or loads[new]-costs[j] > limit-costs[i]:
                    continue
                back = sum(owners[x] == old for x in adjacent[j])
                stay = sum(owners[x] == new for x in adjacent[j])
                delta = affinity[new]-affinity[old]+back-stay-2*adjacent[j].count(i)
                if delta > gain:
                    gain, best = delta, j
            if best >= 0:
                new = owners[best]
                loads[old] += costs[best]-costs[i]; loads[new] += costs[i]-costs[best]
                owners[i], owners[best] = new, old
    return owners


def canonical_loads(g):
    w = g.width
    # Only trainable tensors enter the consumer's canonical optimizer. Ties
    # have equal extent, so lexical identity cannot change per-card byte loads.
    n = len(g.sources)
    body_edges = sum(a < n and b < n for a, b in g.edges)
    sizes = [w*w]*body_edges+[w]*n
    sizes += ([x for s in g.sources for x in (3*w*w, w*w, s)] if g.attention else [1]*sum(g.sources))
    loads = [0]*g.devices
    for size in sorted(sizes, reverse=True):
        d = min(range(g.devices), key=lambda d: loads[d]); loads[d] = checked(loads[d]+size)
    return loads


def envelope(g, c, chunks, owners, canonical, state_owners=None):
    """Per-card simultaneous-liveness envelopes; bytes, not user budget sums."""
    w, b, v, p, windows = g.width, g.batch, g.vocab, g.payload, g.windows
    n = len(g.sources); trace = c.trace if g.diagnostics else 0
    state_owners = state_owners or owners
    result = []
    for device in range(g.devices):
        ids = [i for i in range(n) if state_owners[i] == device]
        body, nodes = len(ids), max(owners.count(device),state_owners.count(device))
        slots = sum(g.slots[i] for i in range(n) if owners[i] == device)
        domain = max([g.sources[i] for i in ids]+[1])
        projection = (slots+1)*(w*w+w)*p
        state_parameters = ((body+1)*(4*w*w+4*w)*p+4*(body+1)*domain+body*p) if g.attention and body else 0
        parameters = projection+state_parameters+nodes*(2*w*p+128)
        cache = (b*body*c.kv+1)*(2*w+1) if g.attention and body else 0
        state = b*nodes*(p*w+17)
        # Forward proposals, gathered parameters and packed module scratch are
        # covered by these shape bounds (see fiber_cache/content_flow sources).
        # Append-only KV proposals reuse the invisible live-cache tail. Bias
        # remains separate because decay changes its already-visible prefix.
        kv_proposal_saving = 2*p*(b*body*c.kv+1)*w if g.attention and body else 0
        persistent_state = 24*cache+4*state-kv_proposal_saving
        coordinator = device == 0
        routing = (64*(c.queue+c.arrivals+c.outputs+trace)*(5*w+32)
                   +64*b*((n+2)*(w+4)+g.regions*(g.regions+4))+160*(len(g.edges)+n+4)) if coordinator else 0
        owner_packets = 128*c.queue*(6*w+64) if g.devices > 1 else 0
        journals = 24*c.kv_trace*(2*w+8) if g.attention and g.diagnostics and body else 0
        forward_work = (96*chunks.emission*w*w+128*(chunks.full+chunks.aggregate)*(w+domain))
        if g.attention and body:
            forward_work += 96*chunks.attention*w*w+chunks.attention*chunks.keys*(32*w+192)+512*chunks.attention*(w+1)
        masters = 4*(3 if g.adamw else 2)*canonical[device] if g.training else 0
        consumer = (2*p*v*w+(4*(3 if g.adamw else 2)*2*v*w if g.training else 0)) if coordinator else 0
        # Serial programs reuse their own arena. Different retained-window and
        # peer programs can coexist. Reserve their caps plus 512MiB for backend
        # allocations outside the tensor formulas; calibrate on each platform.
        programs = (1+(4*windows if g.training else 2))*c.program+512*MIB
        snapshot = saved_contexts = accumulation = context_pack = 0
        if g.sample_chunks > 1:
            # Dense continuation handles retain only numerical state, not the
            # active owner's parameters, proposals, journals or operator arenas.
            snapshot = state+4096
            if g.attention and body:
                snapshot += (b*body*c.kv+1)*p*(2*w+1)+8*b*body
            if coordinator:
                snapshot += 9*b*(n+2+g.regions)+c.queue*(p*w+49)+16
            # Drop the restored handle before executing its replacement. At
            # most K saved copies coexist with the separately charged owner.
            saved_contexts = g.sample_chunks*snapshot
            if g.context_bytes:
                saved_contexts = min(saved_contexts,g.context_bytes)
                # One group's mask/workspace plus concurrently retained indices.
                # Vendor workspaces still have the separate allowance above.
                context_pack = 32*(b*body*c.kv if g.attention else 0)+16*MIB
                if coordinator:
                    context_pack += 32*c.queue
            accumulation = 8*canonical[device]+16*MIB if g.training else 0
        # Completed-window prefix extents allocate masks/indices one journal at
        # a time. Charge their scratch even when the conservative policy keeps
        # dense copies; retained payloads below still use the dense envelope.
        retained_pack = (32*(trace+(c.kv_trace if g.attention and body else 0))+16*MIB) if g.training else 0
        base = parameters+persistent_state+routing+owner_packets+journals+forward_work+masters+consumer+programs+saved_contexts+accumulation+context_pack+retained_pack
        construction = base+4*canonical[device]+parameters
        head = head_budget(c.outputs,w,v,p,g.training,MAX,True)
        head_work = head.fixed_bytes+chunks.head*head.row_bytes if coordinator else 0
        retained = gradients = reverse_work = communication = roots = proposal = attention_gradients = 0
        if g.training:
            retained = projection+windows*(state_parameters+2*nodes*w*p+state+p*cache+journals)
            if coordinator:
                retained += windows*32*(trace+c.queue+c.outputs)*(10*w+64)
                roots = 4*windows*c.outputs*w+(12 if g.sample_chunks>1 else 8)*v*w
            # Only the Attention profile owns QKV/output matrices. Add has
            # scalar aggregate logits and vector LH/state/Read adjoints, already
            # covered below; its edge projection matrices remain fully charged.
            attention_gradients = windows*16*body*w*w if g.attention else 0
            gradients = windows*(4*(slots+1)*(w*w+w)+4*body*domain+32*b*nodes*(w+1)+24*cache)+attention_gradients
            if coordinator:
                gradients += windows*64*(trace+c.queue+c.outputs)*(w+32)
            gradients += 4*canonical[device]
            reverse_work = windows*chunks.reverse*(256*w*w+128*c.kv*(w+1)+128*domain+4096)
            communication = (g.devices if coordinator else 1)*128*MIB if g.devices > 1 else 0
            proposal = ((4*(3 if g.adamw else 2)+p)*2*v*w+32*v*w) if coordinator else 0
        phases = dict(construction=construction, forward_loss=base+retained+roots+head_work,
                      backward=base+retained+roots+gradients+reverse_work+communication,
                      optimizer=base+4*canonical[device]+roots+proposal+communication)
        if not g.training:
            phases = {k: phases[k] for k in ('construction','forward_loss')}
        components = dict(parameters=parameters,state_and_kv=persistent_state,routing=routing,owner_packets=owner_packets,
                          journals=journals,forward_workspace=forward_work,graph_optimizer=masters,
                          consumer_parameters_optimizer=consumer,programs_and_vendor_allowance=programs,
                          retained=retained,roots_and_consumer_gradients=roots,physical_and_canonical_gradients=gradients,
                          reverse_workspace=reverse_work,canonical_communication=communication,consumer_proposals=proposal,
                          head_workspace=head_work,continuation_snapshot_bytes=snapshot,
                          saved_contexts=saved_contexts,gradient_accumulation=accumulation,context_pack_workspace=context_pack,
                          retained_pack_workspace=retained_pack,attention_parameter_gradients=attention_gradients)
        if max(*phases.values(),*components.values()) > MAX:
            raise ValueError('consumer memory extent overflow')
        result.append(dict(index=device,estimated_peak_bytes=max(phases.values()),phases=phases,components=components))
    return result


def plan(g, c, requested, budgets, aggressive=False, full_owners=None, state_owners=None):
    ints = [g.width,g.batch,g.vocab,g.windows,g.payload,g.regions,g.devices,*g.sources,*g.slots,
            g.sample_chunks,g.context_bytes,*asdict(c).values(),*asdict(requested).values(),*budgets]
    if (any(type(x) is not int or x < 0 or x > MAX for x in ints)
            or min(g.width,g.batch,g.vocab,g.windows,g.regions,g.sample_chunks,c.queue,c.arrivals,c.outputs,c.kv,c.program) < 1
            or g.payload not in (2,4) or not 1 <= g.devices <= min(16,len(g.sources)+2)
            or len(g.slots) != len(g.sources) or len(budgets) != g.devices
            or min(*asdict(requested).values(),*budgets) < 1
            or any(a < 0 or b < 0 or a >= len(g.sources)+2 or b >= len(g.sources)+2 for a,b in g.edges)):
        raise ValueError('invalid complete-consumer memory geometry/budget')
    owners, canonical = full_owners or placement(g), canonical_loads(g)
    state_owners = state_owners or owners
    for layout in (owners,state_owners):
        if len(layout) != len(g.sources)+2 or set(layout) != set(range(g.devices)):
            raise ValueError('invalid consumer owner map')
    usable = [x-x//(10 if aggressive else 4)-128*MIB for x in budgets]
    chunks = Chunks(**asdict(requested)); reductions = 0
    while True:
        cards = envelope(g,c,chunks,owners,canonical,state_owners)
        if all(card['estimated_peak_bytes'] <= cap for card,cap in zip(cards,usable)):
            break
        values = asdict(chunks)
        if max(values.values()) == 1:
            details = ', '.join(f"device {i}: estimated {d['estimated_peak_bytes']} > usable {usable[i]}" for i,d in enumerate(cards) if d['estimated_peak_bytes'] > usable[i])
            raise MemoryRefusal('complete-consumer memory admission refused at minimum physical rows; '+details)
        selected = None
        if aggressive:
            # Prefer the single halving that most reduces aggregate excess on
            # the constrained cards. Small/nonlimiting operators keep their
            # batching. Static shape envelopes only, never a numerical prepass.
            excess = lambda rows: sum(max(0,d['estimated_peak_bytes']-cap) for d,cap in zip(rows,usable))
            best = excess(cards)
            for key,value in values.items():  # Stable Chunks field order.
                if value == 1:
                    continue
                trial = Chunks(**{**values,key:max(1,value//2)})
                score = excess(envelope(g,c,trial,owners,canonical,state_owners))
                if score < best:
                    selected,best = trial,score
        # Equal peak phases can mask every individual gain. Joint halving
        # crosses that plateau and is also the unchanged conservative policy.
        chunks = selected or Chunks(**{k:max(1,x//2) for k,x in values.items()})
        reductions += 1
    for card, budget, cap in zip(cards,budgets,usable):
        card.update(budget_bytes=budget,usable_bytes=cap,headroom_bytes=budget-cap)
    return dict(schema='tide-consumer-capacity-v1', scope='resident Add/Attention complete consumer; conservative shape estimate, not a vendor allocation guarantee',
                devices=cards,full_owners=owners,state_owners=state_owners,canonical_elements=canonical,
                requested_chunks=asdict(requested),effective_chunks=asdict(chunks),physical_reductions=reductions,
                row_selection='greedy_peak_excess' if aggressive else 'joint_halving',
                policy='aggressive' if aggressive else 'conservative')


def plan_samples(g, c, requested, budgets, aggressive, logical_batch, automatic,
                 full_owners=None, state_owners=None):
    if type(automatic) is not bool:
        raise ValueError('auto-sample-chunks must be boolean')
    if automatic and (type(logical_batch) is not int or not 1 <= g.batch <= logical_batch <= MAX):
        raise ValueError('invalid automatic sample admission geometry')
    attempted = []
    while True:
        if automatic:
            g = replace(g,sample_chunks=(logical_batch-1)//g.batch+1)
            attempted.append(g.batch)
        try:
            result = plan(g,c,requested,budgets,aggressive,full_owners,state_owners)
        except MemoryRefusal:
            if not automatic or g.batch == 1:
                raise
            g = replace(g,batch=(g.batch-1)//2+1)
            continue
        if automatic:
            result['sample_admission'] = dict(logical_batch=logical_batch,
                effective_sample_rows=g.batch,physical_chunks=g.sample_chunks,
                attempted_sample_rows=attempted,policy='halve_on_memory_refusal')
        return result


def packet_geometry(packet, *, windows, payload, training, adamw, diagnostics, devices, locality=True):
    config, graph = packet['workload'], packet['graph']
    n = len(graph['node_regions']); sources, slots = [0]*n,[0]*n
    edges = [(a,b) for a,b,*_ in graph['edges']]
    for a,b in edges:
        sources[b] += 1; slots[a] += 1
    for i in graph['inputs']:
        sources[i] += 1; edges.append((n,i))
    for i in graph['outputs']:
        slots[i] += 1; edges.append((i,n+1))
    return Geometry(config['width'],config['batch'],config['vocab'],windows,payload,config['memory']=='attention',
                    training,adamw,diagnostics,graph['regions']+2,sources,slots,edges,devices,locality)

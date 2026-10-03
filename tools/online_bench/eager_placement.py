"""Static eager consumer ownership; no numerical execution or capacity claim."""


def owner_indices(packet, devices=1, policy="locality", explicit=()):
    g, c = packet["graph"], packet["workload"]
    n, width = len(g["node_regions"]), c["width"]
    if type(devices) is not int or not 1 <= devices <= min(16, n):
        raise ValueError("eager devices must be in 1..min(16, body nodes)")
    if policy not in ("memory", "locality"):
        raise ValueError("invalid eager owner policy")
    out, sources, adjacent = [0]*n, [0]*n, [[] for _ in range(n+2)]
    for a, b, _ in g["edges"]:
        out[a] += 1; sources[b] += 1
        adjacent[a].append(b); adjacent[b].append(a)
    for node in g["inputs"]:
        sources[node] += 1; adjacent[node].append(n)
    for node in g["outputs"]:
        adjacent[node].append(n+1)
    costs = [out[i]*width**2+width+sources[i]+(4*width**2 if c["memory"] == "attention" else 0)
             for i in range(n)]
    owners = list(explicit)
    if owners:
        if (len(owners) != n+2 or any(type(d) is not int or not 0 <= d < devices for d in owners)
                or owners[0] != 0 or owners[-2:] != [0, 0] or set(owners) != set(range(devices))):
            raise ValueError("eager owner map must cover encoded nodes, use every device, and keep node zero/boundaries on owner zero")
    else:
        owners = [-1]*n+[0, 0]; owners[0] = 0
        loads = [2*c["vocab"]*width+costs[0]]+[0]*(devices-1)
        members = [1]+[0]*(devices-1)
        for node in sorted(range(1, n), key=lambda i: (-costs[i], i)):
            available = [i for i in range(devices) if not members[i]] or list(range(devices))
            affinity = [sum(owners[v] == d for v in adjacent[node]) for d in range(devices)]
            def score(d):
                if policy == "memory":
                    return (loads[d], 0, 0, d)
                return (max(max(loads), loads[d]+costs[node]), -affinity[d], loads[d], d)
            owner = min(available, key=score)
            owners[node] = owner; loads[owner] += costs[node]; members[owner] += 1
    elements = [2*c["vocab"]*width]+[0]*(devices-1)
    for node, cost in enumerate(costs):
        elements[owners[node]] += cost
    assert sum(elements) == packet["counts"]["parameters"]
    return owners, elements


def resolve(packet, device, devices=1, policy="locality", explicit=()):
    import torch
    from tidegraph.runtime import resolve_device
    owners, elements = owner_indices(packet, devices, policy, explicit)
    first, _ = resolve_device(str(device))
    if first.type == "cpu" and devices != 1:
        raise ValueError("multiple eager devices require an accelerator")
    logical = [first] if first.type == "cpu" else [torch.device(first.type, first.index+i) for i in range(devices)]
    try:
        for value in logical[1:]:
            resolve_device(str(value))
    finally:
        resolve_device(str(first))
    return logical, dict(devices=[str(d) for d in logical], node_owners=owners,
                        policy=policy, owner_selection="explicit" if explicit else "automatic",
                        parameter_elements=elements,
                        scope="static learned-parameter balance; not total peak-memory admission")

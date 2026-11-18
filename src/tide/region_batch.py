"""Compatible regions share tensor submissions while retaining private budgets."""
from dataclasses import fields, is_dataclass

import torch

from .attention import weighted_attention
from .parameters import NodeOwners, norm
from .region import region_control


def state_signature(value):
    if isinstance(value, torch.Tensor):
        return (tuple(value.shape[1:]), value.dtype, value.device)
    if is_dataclass(value):
        return (type(value), tuple(state_signature(getattr(value, f.name)) for f in fields(value)))
    if isinstance(value, dict):
        return tuple((key, state_signature(item)) for key, item in value.items())
    if isinstance(value, tuple):
        return (type(value), tuple(state_signature(item) for item in value))
    return value


def join_states(values):
    first = values[0]
    if first is None:
        return None
    if isinstance(first, torch.Tensor):
        return torch.cat(values, 0)
    if is_dataclass(first):
        return type(first)(**{f.name: join_states([getattr(v, f.name) for v in values]) for f in fields(first)})
    if isinstance(first, dict):
        return {key: join_states([v[key] for v in values]) for key in first}
    if isinstance(first, tuple) and hasattr(first, "_fields"):
        return type(first)(*(join_states([v[i] for v in values]) for i in range(len(first))))
    raise TypeError("unsupported regional private state")


def split_state(value, sizes):
    if isinstance(value, torch.Tensor):
        return value.split(sizes, 0)
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        columns = [split_state(item, sizes) for item in value]
        return tuple(type(value)(*(items[i] for items in columns)) for i in range(len(sizes)))
    columns = {f.name: split_state(getattr(value, f.name), sizes) for f in fields(value)}
    return tuple(type(value)(**{name: items[i] for name, items in columns.items()}) for i in range(len(sizes)))


def regional_scores(networks, z, score, weight):
    """Region axis is independent of the within-region attention sequence."""
    r, n, b, length, width = z.shape
    x = z.permute(0, 2, 3, 1, 4)
    heads = networks[0].heads

    def linear(values, name):
        layers = [getattr(network, name) for network in networks]
        matrices = torch.stack([layer.weight for layer in layers])
        flat = values.reshape(r, -1, values.shape[-1])
        if layers[0].bias is None:
            result = torch.bmm(flat, matrices.transpose(1, 2))
        else:
            # Keep the bias inside the selected compute dtype; a separate
            # FP32 addition would promote the activation. Layout-dependent
            # Linear decompositions can still round differently (precision.md).
            result = torch.baddbmm(torch.stack([layer.bias for layer in layers])[:, None],
                                  flat, matrices.transpose(1, 2))
        return result.reshape(*values.shape[:-1], matrices.shape[1])

    def project(name):
        return linear(x, name).reshape(r, b, length, n, heads, 32).transpose(-3, -2)

    read = weighted_attention(project("q"), project("k"), project("v"),
                              weights=weight.permute(0, 2, 3, 1), backend=networks[0].backend)
    delta = linear(read.transpose(-3, -2).flatten(-2), "o")
    hidden = torch.nn.functional.silu(linear(norm(x + delta), "first"))
    with torch.autocast(x.device.type, enabled=False):
        readout = torch.stack([network.readout for network in networks])
        second = torch.stack([network.second.weight for network in networks])
        bias = torch.stack([network.second.bias for network in networks])
        projected = torch.bmm(second.transpose(1, 2), readout[..., None])
        correction = torch.bmm(delta.to(readout.dtype).reshape(r, -1, width), readout[..., None])
        correction = correction + torch.bmm(hidden.to(readout.dtype).reshape(r, -1, hidden.shape[-1]), projected)
        correction = correction + (bias * readout).sum(-1)[:, None, None]
    correction = correction.reshape(r, b, length, n).permute(0, 3, 1, 2)
    return score + torch.where(weight > 0, correction, torch.zeros_like(score))


def evaluate_regions(model, regions, contents, masses, classes, time, old=None, record_work=False):
    members = tuple(v for region in regions for v in model.graph.regions[region])
    r, n = len(regions), len(model.graph.regions[regions[0]])
    device, b, length = contents.device, contents.shape[1], contents.shape[2]
    owners = NodeOwners(torch.tensor(members, device=device), members)
    received, u = model._receive(contents, masses, classes, owners)
    mass = (torch.ones_like(received) if model.graph.config.kind == "fc" else
            torch.where(received >= 1, torch.ones_like(received), received.clamp_min(0)))
    times = time[None, None].expand(len(members), b, length)
    desc = [u.new_zeros((len(members), b, length, model.config.d//4)) for _ in range(3)]
    desc += [u.new_zeros((len(members), b, length)) for _ in range(2)]
    contexts, saved = {}, [{} for _ in regions]
    for kind in model.memories:
        indices = [i for i, node in enumerate(members) if model.kinds[node] == kind]
        if not indices:
            continue
        sub = torch.tensor(indices, device=device)
        possible = tuple(model.memory_ids[kind][members[i]] for i in indices)
        mids = NodeOwners(torch.tensor(possible, device=device), possible)
        previous = None if old is None else join_states([item[kind] for item in old])
        parts, context, updated = model.memories[kind].prepare(u[sub], mass[sub], times[sub], mids, previous)
        desc = [a.index_copy(0, sub, z.to(a.dtype)) for a, z in zip(desc, parts)]
        contexts[kind] = sub, context
        for target, state in zip(saved, split_state(updated, [len(indices)//r]*r)):
            target[kind] = state
    roles = torch.tensor([model.graph.nodes[v].role == "H" for v in model.graph.regions[regions[0]]], device=device)
    if model.region_group[regions[0]] < 0:
        control = torch.where(received >= 0.5, torch.ones_like(received), 2*received)
    else:
        possible = tuple(model.description_ids[v] for v in members)
        d_owners = NodeOwners(torch.tensor(possible, device=device), possible)
        z, score = model.describe(u, received, desc, d_owners)
        score = model._score_regions(regions, z.reshape(r, n, b, length, -1),
                                     score.reshape(r, n, b, length), mass.reshape(r, n, b, length))
        # Region becomes a batch dimension, never an allocation/attention axis.
        region_score = score.permute(1, 0, 2, 3).reshape(n, r*b, length)
        region_received = received.reshape(r, n, b, length).permute(1, 0, 2, 3).reshape(n, r*b, length)
        control = region_control(region_score, region_received, roles,
                                 model.config.budget or max(1, model.graph.config.n//32),
                                 model.config.allocation, model.graph.config.partition)
        control = control.reshape(n, r, b, length).permute(1, 0, 2, 3).reshape(r*n, b, length)
    full = torch.zeros_like(u)
    for kind, (sub, context) in contexts.items():
        full = full.index_copy(0, sub, model.memories[kind].finish(context, control[sub]).to(full.dtype))
    measured = None
    if record_work:
        measured = torch.stack(((received > 0).sum(), (control > 0).sum(), (masses > 0).sum(),
                                torch.as_tensor(received.numel(), device=device)))
    return full.split(n, 0), control.split(n, 0), saved, measured

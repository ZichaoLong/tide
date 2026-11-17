"""Chronological K/quad witness for small HB-Timed graphs, fixed math 4f5b9f20.

Graph records are inputs (their formulas have separate structural witnesses).
This oracle never calls production scheduling or neural methods: messages move
through an explicit queue, positive reception updates a private scalar cache,
and region controls use independently enumerated quadratic active faces.
"""
from dataclasses import dataclass
from itertools import product

import torch

from oracle_fc import (Entry, Output, _at_least_one, _at_most_one, _full, _linear,
                       _normal, _regional, _silu)


@dataclass(frozen=True)
class Delivery:
    time: int
    edge: int
    mass: torch.Tensor
    content: torch.Tensor


@dataclass(frozen=True)
class State:
    position: int
    caches: tuple
    pending: tuple


def _source_slots(graph):
    result = {}
    for node in graph.nodes:
        classes = {graph.edges[e].source_class for e in graph.incoming[node.id]}
        if node.id in graph.input_nodes:
            classes.add("ext")
        for source_class in sorted(classes):
            result[node.id, source_class] = len(result)
    return result


def _aggregate(parameters, node, slots, messages):
    total = sum(mass for mass, _, _ in messages)
    values = []
    for mass, content, source_class in messages:
        hidden = _linear(parameters, "receive.a", node, _normal(content))
        hidden = hidden + parameters["receive.embedding.weight"][slots[node, source_class]]
        values.append(mass * (content + _linear(parameters, "receive.b", node, _silu(hidden))))
    return total, sum(values) / total


def _describe(parameters, node, descriptor, content, mass, cache, time, period):
    current = _linear(parameters, "memories.K.summary", node, _normal(content))
    short = [entry.weight * 2 ** (-(time - entry.time) / (8 * period)) for entry in cache]
    long = [entry.weight for entry in cache]
    ws, wl = sum(short), sum(long)
    ms = sum(w * entry.feature for w, entry in zip(short, cache)) / _at_least_one(ws)
    ml = sum(w * entry.feature for w, entry in zip(long, cache)) / _at_least_one(wl)
    feature = torch.cat((_normal(content), torch.log1p(mass).reshape(1), ms, ml,
                         torch.log1p(ws).reshape(1), torch.log1p(wl).reshape(1), current * ms, current * ml))
    z = _normal(_silu(_linear(parameters, "describe.encode", descriptor, feature)))
    score = parameters["describe.offset.value"][descriptor, 0]
    score = score + torch.dot(parameters["describe.readout.value"][descriptor], z)
    return z, score


def _bounded_quad(score, received, roles, budget):
    """Euclidean box/slab projection by active-face enumeration, not bisection."""
    weight = torch.stack([_at_most_one(mass) for mass in received])
    upper = torch.stack([_at_most_one(2 * mass) for mass in received])
    lower = torch.stack([bound / 4 if hub else bound * 0 for bound, hub in zip(upper, roles)])
    total_upper = upper.sum() if bool(upper.sum() < budget) else upper.new_tensor(float(budget))
    received_lower = _at_most_one(weight.sum())
    clipped = received_lower if bool(received_lower <= total_upper) else total_upper
    total_lower = lower.sum() if bool(lower.sum() > clipped) else clipped
    point = torch.maximum(lower, torch.minimum(upper, score))
    if bool(total_lower <= point.sum()) and bool(point.sum() <= total_upper):
        return point
    total = total_lower if bool(point.sum() < total_lower) else total_upper
    sv, lv, uv = (value.detach().tolist() for value in (score, lower, upper))
    target = float(total.detach())
    tolerance = 2e-10
    for face in product((-1, 0, 1), repeat=len(sv)):
        free = [i for i, side in enumerate(face) if side == 0]
        fixed = sum(lv[i] if side == -1 else uv[i] for i, side in enumerate(face) if side)
        if not free:
            if abs(fixed - target) > tolerance:
                continue
            lo = max((sv[i] - lv[i] for i, side in enumerate(face) if side == -1), default=-float("inf"))
            hi = min((sv[i] - uv[i] for i, side in enumerate(face) if side == 1), default=float("inf"))
            if lo > hi + tolerance:
                continue
            result = torch.stack([lower[i] if side == -1 else upper[i] for i, side in enumerate(face)])
        else:
            lagrange = (sum(sv[i] for i in free) + fixed - target) / len(free)
            valid = all((sv[i] - lagrange <= lv[i] + tolerance if side == -1 else
                         sv[i] - lagrange >= uv[i] - tolerance if side == 1 else
                         lv[i] - tolerance <= sv[i] - lagrange <= uv[i] + tolerance)
                        for i, side in enumerate(face))
            if not valid:
                continue
            fixed_tensor = sum(lower[i] if side == -1 else upper[i]
                               for i, side in enumerate(face) if side)
            multiplier = (score[free].sum() + fixed_tensor - total) / len(free)
            result = torch.stack([lower[i] if side == -1 else upper[i] if side == 1
                                  else score[i] - multiplier for i, side in enumerate(face)])
        return result + score * 0
    raise AssertionError("no feasible stationary face for the small role allocation")


def evaluate(parameters, inputs, graph, state=None, *, inject=True):
    """Advance the tensor's duration; inject=False adds no external atoms."""
    assert graph.config.timing == "timed"
    batches, positions, width = inputs.shape
    start = 0 if state is None else state.position
    stop = (start + positions) * graph.period
    slots = _source_slots(graph)
    scored = [v.id for v in graph.nodes if not (v.role == "H" and graph.config.partition == "singleton")]
    descriptor = {node: index for index, node in enumerate(scored)}
    fixed_regions = {r for r, members in enumerate(graph.regions)
                     if graph.config.partition == "singleton" and all(graph.nodes[v].role == "H" for v in members)}
    network = {r: index for index, r in enumerate(r for r in range(len(graph.regions)) if r not in fixed_regions)}
    caches = ([[[] for _ in graph.nodes] for _ in range(batches)] if state is None else
              [[list(entries) for entries in batch] for batch in state.caches])
    queues = [{} for _ in range(batches)]
    if state is not None:
        for batch, deliveries in enumerate(state.pending):
            for delivery in deliveries:
                queues[batch].setdefault(delivery.time, []).append(delivery)
    output_values, output_controls = [], []
    for time in range(start * graph.period + 1, stop + 1):
        batch_values, batch_controls = [], []
        for batch in range(batches):
            inbox = [[] for _ in graph.nodes]
            for delivery in queues[batch].pop(time, []):
                edge = graph.edges[delivery.edge]
                inbox[edge.dst].append((delivery.mass, delivery.content, edge.source_class))
            if inject and (time - 1) % graph.period == 0:
                index = (time - 1) // graph.period - start
                for node in graph.input_nodes:
                    role = graph.nodes[node].role
                    content = parameters[f"inputs.{role}.weight"] @ inputs[batch, index]
                    inbox[node].append((inputs.new_ones(()), content, "ext"))
            values = {node: inputs.new_zeros(width) for node in graph.output_nodes}
            controls_out = {node: inputs.new_zeros(()) for node in graph.output_nodes}
            for region, members in enumerate(graph.regions):
                active = [node for node in members if inbox[node]]
                if not active:
                    continue
                contents, received, encodings, scores = [], [], [], []
                for node in active:
                    mass, content = _aggregate(parameters, node, slots, inbox[node])
                    normal = _normal(content)
                    kv = _linear(parameters, "memories.K.kv", node, normal)
                    feature = _linear(parameters, "memories.K.summary", node, normal)
                    cache = [entry for entry in caches[batch][node] if time - entry.time < 128 * graph.period]
                    cache.append(Entry(time, kv[:width], kv[width:], feature, _at_most_one(mass)))
                    caches[batch][node] = cache
                    contents.append(content); received.append(mass)
                    if region not in fixed_regions:
                        z, score = _describe(parameters, node, descriptor[node], content, mass, cache, time, graph.period)
                        encodings.append(z); scores.append(score)
                if region in fixed_regions:
                    controls = torch.stack([_at_most_one(2 * mass) for mass in received])
                else:
                    weights = [_at_most_one(mass) for mass in received]
                    score = _regional(parameters, network[region], encodings, scores, weights)
                    hubs = sum(graph.nodes[node].role == "H" for node in members)
                    budget = (1 if hubs < len(members) else 0) + (max(1, (hubs + 1) // 2) if hubs else 0)
                    roles = [graph.nodes[node].role == "H" for node in active]
                    controls = _bounded_quad(score, received, roles, budget)
                for node, content, control in zip(active, contents, controls):
                    if bool(control <= 0):
                        continue
                    # Timed position is (theta-1)/period, independent of layer.
                    full = _full(parameters, node, content, caches[batch][node], time, 1, graph.period)
                    for eid in graph.outgoing[node]:
                        edge = graph.edges[eid]
                        delivery = Delivery(time + edge.delay, eid, control, full)
                        queues[batch].setdefault(delivery.time, []).append(delivery)
                    if node in values:
                        values[node], controls_out[node] = full, control
            batch_values.append(torch.stack([values[node] for node in graph.output_nodes]))
            batch_controls.append(torch.stack([controls_out[node] for node in graph.output_nodes]))
        output_values.append(torch.stack(batch_values).transpose(0, 1))
        output_controls.append(torch.stack(batch_controls).transpose(0, 1))
    pending = tuple(tuple(delivery for time in sorted(queue) for delivery in queue[time]) for queue in queues)
    updated = State(start + positions, tuple(tuple(tuple(c) for c in batch) for batch in caches), pending)
    times = torch.arange(start * graph.period + 1, stop + 1, dtype=torch.int64)
    return Output(torch.stack(output_values, 2), torch.stack(output_controls, 2), times, updated)

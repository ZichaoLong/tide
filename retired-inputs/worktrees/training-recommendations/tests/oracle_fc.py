"""Scalar FC/K witness from fixed math revision 4f5b9f20, not model helpers.

The source_residual equations are architecture §2, KV is §§1–4, encoded and
regional scoring are region-allocation §§3.1/3.7, and pool is §4.2.3. The FC
binding is backbone-comparison §1. Parameters are independent mathematical
matrices; state advances input position first, then layer, batch and node.
"""
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class Entry:
    time: int
    key: torch.Tensor
    value: torch.Tensor
    feature: torch.Tensor
    weight: torch.Tensor


@dataclass(frozen=True)
class State:
    position: int
    caches: tuple


@dataclass(frozen=True)
class Output:
    values: torch.Tensor
    strengths: torch.Tensor
    times: torch.Tensor
    state: State

    def readout(self):
        rows = []
        for batch in range(self.values.shape[1]):
            positions = []
            for t in range(self.values.shape[2]):
                mass = self.strengths[:, batch, t]
                denominator = _at_least_one(mass.sum())
                positions.append(sum(mass[i] * self.values[i, batch, t]
                                     for i in range(len(mass))) / denominator)
            rows.append(torch.stack(positions))
        return torch.stack(rows)


def _normal(x):
    return x / torch.sqrt(x.square().sum() / x.numel() + 1e-5)


def _silu(x):
    return x * torch.sigmoid(x)


def _at_most_one(x):
    return torch.ones_like(x) if bool(x >= 1) else x


def _at_least_one(x):
    return x if bool(x > 1) else torch.ones_like(x)


def _linear(parameters, name, node, value):
    result = parameters[name + ".weight"][node] @ value
    bias = parameters.get(name + ".bias")
    return result if bias is None else result + bias[node]


def _receive(parameters, node, messages):
    positive = [(mass, content) for mass, content in messages if bool(mass > 0)]
    assert positive, "this small FC witness has positive reception at every position"
    mass = sum(weight for weight, _ in positive)
    contents = []
    for weight, content in positive:
        hidden = _linear(parameters, "receive.a", node, _normal(content))
        # Independent FC has exactly one source class per node: ext or DD.
        hidden = hidden + parameters["receive.embedding.weight"][node]
        correction = _linear(parameters, "receive.b", node, _silu(hidden))
        contents.append(weight * (content + correction))
    return mass, sum(contents) / mass


def _description(parameters, node, content, received, cache, time, period):
    current = _linear(parameters, "memories.K.summary", node, _normal(content))
    short, long = [], []
    for entry in cache:
        age = time - entry.time
        assert 0 <= age < 128 * period
        long.append(entry.weight)
        short.append(entry.weight * math.exp(-math.log(2) * age / (8 * period)))
    ws, wl = sum(short), sum(long)
    ms = sum(w * entry.feature for w, entry in zip(short, cache)) / _at_least_one(ws)
    ml = sum(w * entry.feature for w, entry in zip(long, cache)) / _at_least_one(wl)
    # This order is the literal f_v in region-allocation §3.1.
    fields = [_normal(content), torch.log1p(received).reshape(1), ms, ml,
              torch.log1p(ws).reshape(1), torch.log1p(wl).reshape(1), current * ms, current * ml]
    encoded = _normal(_silu(_linear(parameters, "describe.encode", node, torch.cat(fields))))
    score = parameters["describe.offset.value"][node, 0]
    score = score + torch.dot(parameters["describe.readout.value"][node], encoded)
    return encoded, score


def _distribution(logits, prior):
    terms = torch.exp(logits - logits.max()) * prior
    return terms / terms.sum()


def _regional(parameters, layer, encodings, local_scores, weights):
    prefix = f"regional.{layer}."
    heads = encodings[0].numel() // 32
    queries = [parameters[prefix + "q.weight"] @ x for x in encodings]
    keys = [parameters[prefix + "k.weight"] @ x for x in encodings]
    values = [parameters[prefix + "v.weight"] @ x for x in encodings]
    result = []
    for query, encoding, local in zip(queries, encodings, local_scores):
        reads = []
        for head in range(heads):
            section = slice(32 * head, 32 * (head + 1))
            logits = torch.stack([torch.dot(query[section], key[section]) / math.sqrt(32)
                                  for key in keys])
            attention = _distribution(logits, torch.stack(weights))
            reads.append(sum(a * value[section] for a, value in zip(attention, values)))
        y = encoding + parameters[prefix + "o.weight"] @ torch.cat(reads)
        hidden = _silu(parameters[prefix + "first.weight"] @ _normal(y)
                       + parameters[prefix + "first.bias"])
        zhat = y + parameters[prefix + "second.weight"] @ hidden + parameters[prefix + "second.bias"]
        result.append(local + torch.dot(parameters[prefix + "readout"], zhat - encoding))
    return torch.stack(result)


def _rotate(vector, position):
    pairs = []
    for pair in range(16):
        angle = position * 10000 ** (-2 * pair / 32)
        cosine, sine = math.cos(angle), math.sin(angle)
        x, y = vector[2 * pair:2 * pair + 2]
        pairs.extend((cosine * x - sine * y, sine * x + cosine * y))
    return torch.stack(pairs)


def _full(parameters, node, content, cache, time, rank, period):
    width = content.numel()
    normal = _normal(content)
    weight = torch.stack([entry.weight for entry in cache])
    amplitude = _at_most_one(weight.sum())
    heads = []
    for head in range(width // 32):
        query = _linear(parameters, f"memories.K.query.heads.{head}", node, normal)
        query = _rotate(query, (time - rank) / period)
        section = slice(32 * head, 32 * (head + 1))
        logits = torch.stack([torch.dot(query, _rotate(entry.key[section], (entry.time - rank) / period))
                              / math.sqrt(32) for entry in cache])
        attention = _distribution(logits, weight)
        heads.append(amplitude * sum(a * entry.value[section] for a, entry in zip(attention, cache)))
    residual = content + _linear(parameters, "memories.K.output", node, torch.cat(heads))
    normal = _normal(residual)
    gate = _silu(_linear(parameters, "memories.K.ffn.gate", node, normal))
    up = _linear(parameters, "memories.K.ffn.up", node, normal)
    return residual + _linear(parameters, "memories.K.ffn.down", node, gate * up)


def evaluate(parameters, inputs, layers=2, n=2, allocation="pool", state=None):
    """Time-major scalar FC/K evaluation; no tested module methods are called."""
    batches, length, width = inputs.shape
    period = layers + 2
    start = 0 if state is None else state.position
    caches = ([[[] for _ in range(layers * n)] for _ in range(batches)] if state is None else
              [[list(entries) for entries in batch] for batch in state.caches])
    output_values, output_controls = [], []
    for t in range(length):
        batch_values, batch_controls = [], []
        for batch in range(batches):
            previous_values, previous_controls = None, None
            for layer in range(layers):
                time = (start + t) * period + layer + 1
                contents, received, encodings, local_scores, weights = [], [], [], [], []
                for local_node in range(n):
                    node = layer * n + local_node
                    if layer == 0:
                        projected = parameters["inputs.D.weight"] @ inputs[batch, t]
                        messages = [(inputs.new_ones(()), projected)]
                    else:
                        messages = list(zip(previous_controls, previous_values))
                    mass, content = _receive(parameters, node, messages)
                    w = _at_most_one(mass)
                    kv = _linear(parameters, "memories.K.kv", node, _normal(content))
                    feature = _linear(parameters, "memories.K.summary", node, _normal(content))
                    caches[batch][node] = [e for e in caches[batch][node] if time - e.time < 128 * period]
                    caches[batch][node].append(Entry(time, kv[:width], kv[width:], feature, w))
                    z, score = _description(parameters, node, content, mass, caches[batch][node], time, period)
                    contents.append(content); received.append(mass); weights.append(w)
                    encodings.append(z); local_scores.append(score)
                scores = _regional(parameters, layer, encodings, local_scores, weights)
                if allocation == "pool":
                    upper = torch.stack([_at_most_one(2 * mass) for mass in received])
                    controls = _at_most_one(upper.sum()) * _distribution(scores, upper)
                elif allocation == "quad":
                    # n=2,k=1 and unit receptions imply c0+c1=1, 0<=ci<=1.
                    assert n == 2 and all(bool(mass > 0.5) for mass in received)
                    first = ((scores[0] - scores[1] + 1) / 2).clamp(0, 1)
                    controls = torch.stack((first, 1 - first))
                else:
                    raise ValueError("the independent witness covers pool and two-node quad")
                values = []
                for local_node, (content, control) in enumerate(zip(contents, controls)):
                    node = layer * n + local_node
                    value = (_full(parameters, node, content, caches[batch][node], time, layer + 1, period)
                             if bool(control > 0) else torch.zeros_like(content))
                    values.append(value)
                previous_values, previous_controls = values, controls
            batch_values.append(torch.stack(previous_values)); batch_controls.append(previous_controls)
        output_values.append(torch.stack(batch_values).transpose(0, 1))
        output_controls.append(torch.stack(batch_controls).transpose(0, 1))
    updated = State(start + length, tuple(tuple(tuple(entries) for entries in batch) for batch in caches))
    times = torch.tensor([(start + t) * period + layers for t in range(length)], dtype=torch.int64)
    return Output(torch.stack(output_values, 2), torch.stack(output_controls, 2), times, updated)

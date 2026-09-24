"""Source-signature groups with per-event normalization before owner reduction."""
from collections import defaultdict
import torch
from .aggregate import AggregateResult
from .isolated_aggregate import aggregate


def evaluate(program, weights, requests):
    groups = defaultdict(list)
    for i, request in enumerate(requests):
        groups[tuple(s.slot for s in request.sources)].append(i)
    results = [None]*len(requests)
    for signature, rows in groups.items():
        request = requests[rows[0]]
        if program.kind == "weighted_mean":
            coe = torch.stack([program.coefficients(weights, requests[i]) for i in rows])
        elif "softmax" in program.kind:
            domain = range(request.slots) if program.kind == "all_softmax" else signature
            logits = torch.stack([weights.extra[f"agg_logit_{s}"] for s in domain])
            coe = logits.unsqueeze(0).expand(len(rows), -1).softmax(1)
            if program.kind == "all_softmax":
                coe = torch.stack([coe[:, s] for s in signature], 1)
        else:
            coe = request.sources[0].atom.value.new_empty(0)
        atoms = [s.atom.value for i in rows for s in requests[i].sources]
        scales = [s.scale for i in rows for s in requests[i].sources]
        values = aggregate(atoms, scales, coe, len(signature), program.kind == "mean")
        for j, i in enumerate(rows):
            offset = j*(len(signature)+1)
            results[i] = AggregateResult(values[offset], dict(sorted(
                (slot, values[offset+1+k]) for k, slot in enumerate(signature))))
    return results

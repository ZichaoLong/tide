"""Architecture §2: source-class projection batches and complete-fiber sums.

Every physical atom has a row. Source class only chooses parameters; it does
not merge atoms. Rows are grouped by class before reducing each actual fiber.
"""
from collections import defaultdict
import torch
from ..aggregate import AggregateResult
from .packing import project
from .tensors import norm
from .mean import weighted_mean


def evaluate(program, requests):
    if not requests:
        return []
    atoms, offsets, sizes, classes = [], [0], defaultdict(list), defaultdict(list)
    for i, request in enumerate(requests):
        if not request.sources:
            raise ValueError("weighted Aggregate requires nonempty fibers")
        sizes[len(request.sources)].append(i)
        for source in request.sources:
            if program.alpha:
                classes[program.classes.classify(source.atom)].append(len(atoms))
            atoms.append(source.atom.value)
        offsets.append(len(atoms))
    payload = torch.stack(atoms)
    mass, original = payload[:, 0].clamp_min(0), payload[:, 1:]
    content = original
    for label, members in sorted(classes.items()):
        ids = torch.tensor(members, dtype=torch.int64, device=payload.device)
        x = original[ids]; transform = program.transforms[str(label)]
        a, b = transform.matrices()
        hidden = torch.nn.functional.silu(project(norm(x, program.epsilon), a)+transform.embedding)
        values = x+program.alpha*project(hidden, b)
        content = content.index_copy(0, ids, values)  # Distinct physical rows; no conflicting scatter.
    result = [None]*len(requests)
    for length, owners in sorted(sizes.items()):
        ids = torch.tensor([[offsets[i]+j for j in range(length)] for i in owners],
                           dtype=torch.int64, device=payload.device)
        total = mass[ids].sum(1)
        values = weighted_mean(mass[ids], content[ids])
        values = torch.where((total > 0)[:, None], values, torch.zeros_like(values))
        encoded = torch.cat((total[:, None], values), -1)
        for row, owner in enumerate(owners):
            result[owner] = AggregateResult(encoded[row], {})
    return result

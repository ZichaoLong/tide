"""Allocation §2–6: complete region rows, actual scores/head/sending batches.

Weighted history is a singleton: each returned timestamp is the request time.
Scheduling readiness and independent public VJP roots belong to other layers.
"""
from collections import defaultdict
import torch
from ..history import History
from ..region import Selection
from .config import HARD
from .allocation_rows import allocate_batch
from .allocation_failure import AllocationFailure


def evaluate(program, requests, *, dependencies=None):
    groups, result = defaultdict(list), [None]*len(requests)
    for i, r in enumerate(requests):
        if not r.candidates:
            raise ValueError("region batch requires nonempty complete candidate sets")
        first = r.candidates[0][1]["w"]
        groups[(len(r.candidates), first.dtype, first.device)].append(i)
    c = program.config
    for (_, _, device), rows in groups.items():
        names = requests[rows[0]].candidates[0][1]
        from .private_rows import join
        fields = {name: torch.stack([join([requests[i].candidates[j][1][name] for i in rows], stack=True)
                  for j in range(len(requests[rows[0]].candidates))], 1) for name in names}
        # Only stable sorted sending consumes node IDs. Continuous allocation
        # and normalized sending operate on the original candidate order.
        mode = c.selection_mode
        nodes = (torch.tensor([[v for v, _ in requests[i].candidates] for i in rows],
                              dtype=torch.int64, device=device)
                 if mode == "regional_total_zero" or mode in HARD else None)
        from .regional_columns import region_columns
        try:
            column = region_columns(program, fields, nodes)
        except AllocationFailure as error:
            error.row = rows[error.row]
            request = requests[error.row]
            error.time = request.time
            error.nodes = [v for v, _ in request.candidates]
            error.members = list(request.layout.members)
            raise
        values = column.controls.unbind()
        if dependencies is not None:
            for i, branch in zip(rows, column.dependencies()):
                dependencies[i] = branch
        mappings = [{v: value for (v, _), value in zip(requests[i].candidates, controls.unbind())}
                    for i, controls in zip(rows, values)]
        flags = None
        if len(rows) > 1 and next(iter(mappings[0].values())).device.type != "cpu":
            from ..region_metadata import positive
            flags = iter(positive([value for mapping in mappings for value in mapping.values()]))
        for i, mapping in zip(rows, mappings):
            active = {v for v, value in mapping.items() if (next(flags) if flags is not None else value.detach() > 0)}
            result[i] = Selection(active, mapping, History(last_time=requests[i].time))
    return result

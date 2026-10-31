"""Allocation §2–6: complete region rows, actual scores/head/sending batches.

Weighted history is a singleton: each returned timestamp is the request time.
Scheduling readiness and independent public VJP roots belong to other layers.
"""
from collections import defaultdict
import math
import torch
from ..history import History
from ..region import Selection
from .config import TOTAL, ATTENTION, HARD
from .allocation_rows import allocate_batch
from .allocation_failure import AllocationFailure
from .sending_batch import attention, proportional, sorted_send


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
        fields = {name: torch.stack([torch.stack([d[name] for _, d in requests[i].candidates]) for i in rows]) for name in names}
        nodes = torch.tensor([[v for v, _ in requests[i].candidates] for i in rows], dtype=torch.int64, device=device)
        recv, caps, scores = fields["w"], fields["mu"], fields["score"]
        if c.region_score == "att":
            weights = fields["w" if c.query_weight == "received" else "mu"]
            x = (weights[..., None]*fields["x"]).sum(1)/weights.sum(1).clamp_min(1)[:, None]
            query = program.query(x)
            scores = c.alpha_loc*scores+c.alpha_sel*(fields["key"]*query[:, None]).sum(-1)/math.sqrt(c.selection_width)
        received, mass = recv.sum(1), caps.sum(1)
        context = None
        if c.region_control_mode == "context":
            z = (recv[..., None]*fields["evidence"]).sum(1)/received.clamp_min(1)[:, None]
            context = program.head(torch.cat((z, received.log1p()[:, None], mass.log1p()[:, None]), 1))
        mode, shift, values = c.selection_mode, None, None
        if mode in TOTAL:
            low, high = received.clamp_max(1), mass.clamp_max(c.k)
            rho = context.sigmoid() if context is not None else received.new_full(received.shape, c.total_fraction)
            low = high = low+rho*(high-low)
        elif mode not in ATTENTION and mode != "proportional":
            shift = context if context is not None else None if c.eta is None else received.new_full(received.shape, c.eta)
            low = (received if c.regional else mass).clamp_max(1)
            high = received.new_full(received.shape, c.k)
        selection = getattr(program, "_selection_device", None)
        if selection is not None:
            from .placement_tensors import move
            recv, caps, scores, nodes, received = (move(t, selection) for t in (recv, caps, scores, nodes, received))
            shift = move(shift, selection)
            if mode not in ATTENTION and mode != "proportional":
                low, high = move(low, selection), move(high, selection)
        if mode == "regional_total_zero":
            values = sorted_send(scores, caps, nodes, target=low)
        elif mode in ATTENTION:
            values = attention(scores, caps, recv, mode, c.allocation_base, received.new_full(received.shape, c.allocation_temperature))
        elif mode == "proportional":
            values = proportional(caps, c.k)
        elif mode in HARD:
            values = sorted_send(scores-shift[:, None], caps, nodes, k=c.k)
        if values is None:
            try:
                allocations = allocate_batch(list(scores.unbind()), list(caps.unbind()), list(low.unbind()), list(high.unbind()),
                    c.allocation_temperature, regularizer=c.regularizer, ent_alpha=c.ent_alpha,
                    score_shift=None if shift is None else list(shift.unbind()))
            except AllocationFailure as error:
                error.row = rows[error.row]
                request = requests[error.row]
                error.time = request.time
                error.nodes = [v for v, _ in request.candidates]
                error.members = list(request.layout.members)
                raise
            values = [a.controls for a in allocations]
            if dependencies is not None:
                for i, allocation in zip(rows, allocations):
                    dependencies[i] = allocation.constraint
        elif dependencies is not None:
            for i in rows:
                dependencies[i] = mode
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

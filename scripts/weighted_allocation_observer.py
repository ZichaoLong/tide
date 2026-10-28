"""Optional Python diagnostic observer; actual allocation calls are unchanged.

This is not a scheduler, fallback, or part of benchmark timings. Native calls
are deliberately outside its scope. Per-row records use the failure codec so
the same independent Decimal tool can inspect a successful but inaccurate row.
"""
from contextlib import contextmanager


@contextmanager
def observe_allocations():
    import torch
    from tidegraph.weighted import programs, region_batch
    from tidegraph.weighted.allocation_failure import AllocationFailure
    scalar, batched = programs.allocate, region_batch.allocate_batch
    records = {"scalar": [], "batch": []}
    def record(kind, s, c, l, h, t, result, options):
        tensor = lambda x: x if isinstance(x, torch.Tensor) else s.new_tensor(x)
        row = AllocationFailure("observation; allocation returned successfully", len(records[kind]),
            dict(scores=s, caps=c, lower=tensor(l), upper=tensor(h), temperature=tensor(t),
                 controls=result.controls, multiplier=result.multiplier, residual=result.residual,
                 shift=None if options.get("score_shift") is None else tensor(options["score_shift"])),
            dict(regularizer=options.get("regularizer", "quad"), alpha=str(options.get("ent_alpha")),
                 branch=result.constraint, solver=kind))
        data = row.record()
        data["observation"] = "successful actual return; not a thrown failure"
        data["free"] = result.free.detach().cpu().tolist()
        records[kind].append(data)
    def one(s, c, l, h, t=1., **options):
        result = scalar(s, c, l, h, t, **options)
        record("scalar", s, c, l, h, t, result, options)
        return result
    def many(s, c, l, h, t=1., **options):
        result = batched(s, c, l, h, t, **options)
        def get(x, i):
            return x[i] if isinstance(x, (tuple, list)) else x
        for i, value in enumerate(result):
            local = dict(options, score_shift=get(options.get("score_shift"), i))
            record("batch", s[i], c[i], get(l, i), get(h, i), get(t, i), value, local)
        return result
    try:
        programs.allocate, region_batch.allocate_batch = one, many
        yield records
    finally:
        programs.allocate, region_batch.allocate_batch = scalar, batched

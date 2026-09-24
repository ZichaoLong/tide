"""Full's affine rows batch independently of per-event nonlinear/Emit roots."""
import torch
from .isolated_linear import linear
from .full import FullResult


def evaluate(program, w, requests, slots, mode, zeta):
    from .ops import emit
    from . import lh_full
    if any(r.control.ndim != 0 for r in requests):
        raise ValueError("projection Emit requires scalar control")
    rows = [r.comparison.value for r in requests]
    if program.identity:
        fresh = [r.content.value for r in requests]
    elif w.full_kind in lh_full.PROFILES:
        fresh = [lh_full.fresh(w, row) for row in rows]
    else:
        if w.full_kind == "swiglu":
            gate, up = linear(rows, w.extra["ffn_gate"].t()), linear(rows, w.extra["ffn_up"].t())
            rows = linear([torch.nn.functional.silu(a)*b for a, b in zip(gate, up)], w.extra["ffn_down"].t())
        else:
            rows = [(v+w.bias).tanh() for v in linear(rows, w.weight.t())]
        fresh = [r.content.value+v for r, v in zip(requests, rows)]
    results = [FullResult(r.content.value if program.identity else emit(r.content.value, f, r.control, mode, zeta), {})
               for r, f in zip(requests, fresh)]
    for slot in range(slots):
        ids = [i for i, r in enumerate(requests) if program.present(slot, r.time)]
        if not ids:
            continue
        if program.kind == "broadcast":
            for i in ids:
                results[i].emitted[slot] = results[i].value
        else:
            weight = w.extra[f"emit_w_{slot}"]
            active = linear([fresh[i] for i in ids], weight.t())
            held = [requests[i].content.value for i in ids]
            if mode != "hard":
                held = linear(held, weight.t())
            for i, a, h in zip(ids, active, held):
                results[i].emitted[slot] = emit(h, a+w.extra[f"emit_b_{slot}"], requests[i].control, mode, zeta)
    return results

"""Post-attention pooling: graph-owned source domain, vector parameter."""
import torch

PROFILES = {f"lh-fiber-attention-{name}-repeat-v1": name
            for name in ("sum", "mean", "linear", "active-softmax", "all-softmax")}
LEARNED = {"linear", "active-softmax", "all-softmax"}


def pool_rows(w, kind, slots, rows):
    if kind == "sum":
        return rows.sum(0)
    if kind == "mean":
        return rows.mean(0)
    if kind not in LEARNED:
        raise ValueError("unknown fiber pooling kind")
    weight = w.extra["fiber_pool"]
    index = torch.tensor(slots, dtype=torch.long, device=rows.device)
    if kind == "all-softmax":
        coefficient = weight.softmax(0).index_select(0, index)
    else:
        coefficient = weight.index_select(0, index)
        if kind == "active-softmax":
            coefficient = coefficient.softmax(0)
    return coefficient @ rows

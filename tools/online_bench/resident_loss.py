"""Packed consumer loss/VJPs at public device-window boundaries.

The graph alone determines output presence and input connectivity. Invalid rows
are removed before arithmetic; they are never evaluated and then multiplied by 0.
"""
import torch
from .head_budget import head_budget


def head_loss(window, head, *, stride, denominator, backward, plan=None, sample_begin=0):
    plan = plan or head_budget(len(window.valid), head.shape[1], head.shape[0], head.element_size(), backward, 4*1024**3)
    indices = window.valid.nonzero().flatten()
    if not indices.numel():
        return None, None, None, 0
    master = head.float() if backward else None
    root = torch.zeros(window.values.shape, dtype=torch.float32, device=head.device) if backward else None
    dh = torch.zeros_like(master) if backward else None
    loss = None
    for selected in indices.split(plan.rows):
        coordinates = window.coordinates.index_select(0, selected)
        rows = window.values.index_select(0, selected)
        targets = ((coordinates[:, 2] // stride + 1)*7 + (coordinates[:, 0]+sample_begin)*3) % head.shape[0]
        logits = rows @ head.t()
        logp = logits.float().log_softmax(1)
        value = -logp.gather(1, targets[:, None]).sum() / denominator
        loss = value if loss is None else loss+value
        if backward:
            dl = logp.exp()
            dl.scatter_add_(1, targets[:, None], -torch.ones_like(targets[:, None], dtype=dl.dtype))
            dl.div_(denominator)
            root.index_copy_(0, selected, dl @ master)
            dh.add_(dl.t() @ rows.float())
    return loss, root, dh, indices.numel()


def embedding_gradient(boundaries, embedding, sample_begin=0):
    gradient = None
    for boundary in boundaries:
        # External inputs are kind 0; previous-cut pending messages are detached
        # leaves, even when their numerical values originated from old tokens.
        index = (boundary.valid & boundary.connected & (boundary.coordinates[:, 3] == 0)).nonzero().flatten()
        if not index.numel():
            continue
        coord = boundary.coordinates.index_select(0, index)
        ids = (coord[:, 5]*7 + (coord[:, 0]+sample_begin)*3) % embedding.shape[0]
        if gradient is None:
            gradient = torch.zeros_like(embedding, dtype=torch.float32)
        gradient.index_add_(0, ids, boundary.values.index_select(0, index))
    return gradient


class ConsumerOptimizer:
    """Two consumer parameters, FP32 masters and staged SGD/AdamW updates.

    Prepare validates gradients, slots and rounded values without changing live
    state. Commit follows a successful graph optimizer transaction. This is a
    consumer policy, not a graph semantic or a replacement general optimizer.
    """
    def __init__(self, embedding, head, kind):
        if kind not in {"sgd", "adamw"}:
            raise ValueError("unknown consumer optimizer")
        self.kind = kind
        self.payloads = (embedding, head)
        self.masters = tuple(x.detach().float().clone() for x in self.payloads)
        self.slots = [(None, None, 0) for _ in self.payloads]
        self.proposal = None

    def prepare(self, gradients):
        if self.proposal is not None or len(gradients) != 2:
            raise ValueError("invalid consumer optimizer transaction")
        proposals, flags = [], []
        for payload, master, (first, second, step), grad in zip(self.payloads, self.masters, self.slots, gradients):
            if grad is None:
                proposals.append(None)
                continue
            if grad.dtype != torch.float32 or grad.shape != master.shape or grad.device != master.device:
                raise ValueError("consumer gradients require matching FP32 matrices")
            flags.extend([torch.isfinite(grad).all(), torch.isfinite(master).all()])
            if self.kind == "sgd":
                direction = grad + .001*master
                nf = direction if step == 0 else .25*first + direction
                ns = None
                value = master - .0001*nf
            else:
                nf = .9*first + .1*grad if step else .1*grad
                ns = .999*second + .001*grad.square() if step else .001*grad.square()
                value = master*(1-.0001*.001) - (.0001/(1-.9**(step+1)))*nf / (ns.sqrt()/((1-.999**(step+1))**.5)+1e-6)
            rounded = value.to(payload.dtype)
            flags.extend(torch.isfinite(v).all() for v in (value, rounded, nf, ns) if v is not None)
            proposals.append((value, rounded, nf, ns, step+1))
        if flags and not torch.stack(flags).all().item():
            raise RuntimeError("nonfinite consumer proposal; graph and consumer optimizer not applied")
        self.proposal = proposals

    def commit(self):
        if self.proposal is None:
            raise ValueError("consumer commit requires a finite proposal")
        for i, proposal in enumerate(self.proposal):
            if proposal is not None:
                value, rounded, first, second, step = proposal
                self.payloads[i].copy_(rounded)
                self.masters[i].copy_(value)
                self.slots[i] = (first, second, step)
        self.proposal = None

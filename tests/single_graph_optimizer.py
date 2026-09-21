"""Training trajectory with stable application owners and optional single-PDG resume."""
from copy import deepcopy
from dataclasses import replace
import torch
from tidegraph.checkpoint import save, load
from tidegraph.compare import equivalent
from tidegraph.records import Result
from single_graph_training import Case, Frame, TwoClock, Encoded
from single_graph_roots import objective


def detached(frame):
    def result(r):
        return Result(r.continuation.detach(), [], [(b, t, p, x.detach()) for b, t, p, x in r.outputs],
                      [replace(a, value=a.value.detach()) for a in r.messages], {})
    return Frame(result(frame.body), result(frame.read), [(b, t, p, x.detach()) for b, t, p, x in frame.buffer])


def make_optimizer(case, optimizer_kind):
    parameters = list(case.owner.parameters())  # Stable application owners, deduplicated aliases.
    if optimizer_kind == "sgd":
        return torch.optim.SGD(parameters, lr=.001, momentum=.8, weight_decay=.01)
    # Default 1e-8 amplifies tiny FP32 attention-gradient roundoff past the
    # parameter tolerance. Keep that separate failed reproducer.
    return torch.optim.AdamW(parameters, lr=.0002, eps=1e-5, weight_decay=.01)



def train(dtype, pool, clear, mode, implementation, optimizer_kind, checkpoints=None):
    case = Case(dtype, pool, clear)
    runner = TwoClock(case, mode) if implementation == "two-clock" else Encoded(case, mode, implementation)
    optimizer = make_optimizer(case, optimizer_kind)
    records = []
    stops = (case.period-1, 3*case.period-1, 4*case.period)
    for stop in stops:
        optimizer.zero_grad(set_to_none=True)
        for x in case.xs:
            x.value.grad = None
        frame = runner.advance(stop)
        objective(case, frame).backward()
        grad = {k: None if p.grad is None else p.grad.clone() for k, p in case.variables.items()}
        # This truncates both clocks and every in-flight body/readout payload.
        runner.detach()
        saved = detached(frame)
        optimizer.step()
        records.append((saved, grad, deepcopy(case.owner.state_dict()), deepcopy(optimizer.state_dict())))
        if checkpoints is not None and stop != stops[-1]:
            assert frame.buffer, "checkpoint must include a partial readout window"
            adapter = runner.adapter
            path = checkpoints/f"cut-{stop}.pt"
            save(path, adapter.graph, adapter.model, runner.q, optimizer)
            previous = runner.q
            case = Case(dtype, pool, clear)
            runner = Encoded(case, mode, implementation)
            optimizer = make_optimizer(case, optimizer_kind)
            # Restore actual values; reconstructed alias and group identities stay fixed.
            with torch.no_grad():
                for p in case.owner.parameters():
                    p.add_(.03)
            runner.q = load(path, runner.adapter.graph, runner.adapter.model, optimizer)
            equivalent(previous, runner.q)
            assert all(not s.value.requires_grad and all(not t.requires_grad for t in s.slots.values())
                       for s in runner.q.states.values())
            assert all(not a.value.requires_grad for a in runner.q.pending)
            if runner.cursor is not None:
                runner.cursor = runner.engine.cursor(runner.q)
            equivalent(records[-1][2], case.owner.state_dict())
            equivalent(records[-1][3], optimizer.state_dict())
    return case, records

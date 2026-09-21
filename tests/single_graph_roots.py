"""Isolated observable training roots and independent parameter connectivity."""
import torch
from tidegraph.compare import equivalent


def roots(case, frame):
    logits = case.logits(frame.read.outputs)
    values = {"output": next(x for b, _, _, x in logits if b == 0),
              "outputs": sum(x.square().sum() for _, _, _, x in logits),
              "body.pending": next(a.value for a in frame.body.continuation.pending if a.batch == 0),
              "buffer": next(x for b, _, _, x in frame.buffer if b == 0)}
    for name, q in (("body", frame.body.continuation), ("read", frame.read.continuation)):
        state = q.states[0, 0]
        values[f"{name}.state.value"] = state.value
        values.update({f"{name}.state.{k}": v for k, v in state.slots.items()})
    return values


def gradients(value, variables, zero=False):
    direction = torch.arange(1, value.numel()+1, dtype=value.dtype).reshape(value.shape)/max(1, value.numel())
    loss = (value*direction).sum()*(0.0 if zero else .7)
    if not loss.requires_grad:
        return dict.fromkeys(variables)
    return dict(zip(variables, torch.autograd.grad(loss, list(variables.values()),
                                                 allow_unused=True, retain_graph=True)))


def check_roots(case, actual, expected, *, zero=False):
    ar, er = roots(case, actual), roots(case, expected)
    equivalent(ar, er)
    for name in er:
        assert ar[name].requires_grad == er[name].requires_grad
        ag, eg = gradients(ar[name], case.variables, zero), gradients(er[name], case.variables, zero)
        equivalent(ag, eg, f"{name}.vjp")
        # A root for sample zero cannot connect an independent sample-one leaf.
        if name != "outputs":
            assert all(v is None for k, v in eg.items() if k.startswith("input.1."))
        if not name.startswith("output"):
            assert eg["readout.nodes.0.extra.token_head"] is None
        if name.startswith("body."):
            assert all(v is None for k, v in eg.items() if k.startswith("readout."))
        if zero:
            assert any(v is not None for v in eg.values())
            assert all(v is None or not v.count_nonzero() for v in eg.values())


def objective(case, frame):
    terms = [x.square().sum()*.7 for _, _, _, x in case.logits(frame.read.outputs)]
    for q in (frame.body.continuation, frame.read.continuation):
        terms.extend(s.value.square().sum()*.3 for s in q.states.values())
        terms.extend(v.square().sum()*.11 for s in q.states.values() for v in s.slots.values())
        terms.extend(a.value.square().sum()*.2 for a in q.pending)
    # These tensors are pending body-to-readout messages in the encoded graph.
    terms.extend(x.square().sum()*.2 for _, _, _, x in frame.buffer)
    return sum(terms)

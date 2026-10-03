"""Shared value codecs and parameter preflight; no live model mutation."""
import torch
from .records import Atom, Continuation, State
from .history import History
from .checkpoint_ownership import parameter_aliases


def encode(continuation):
    q = continuation
    def cpu(value):
        return value.detach().cpu()
    return {
        "identity": q.identity, "batch_size": q.batch_size, "cut": q.cut,
        "states": {k: (cpu(s.value), s.last_time, s.observations, {n: cpu(v) for n, v in s.slots.items()})
                   for k, s in q.states.items()},
        "history": {k: (h.last_time, h.scalars, h.node_maps, {n: cpu(v) for n, v in h.tensors.items()})
                    for k, h in q.history.items()}, "ledger": q.ledger,
        "pending": [(a.batch, a.node, a.time, a.kind, a.source, a.position, cpu(a.value)) for a in q.pending],
    }


def decode(record, device=None, *, graph=None, model=None):
    if model is not None and (graph is None or device is not None):
        raise ValueError("owner-aware decode requires graph/model and no global device")
    from .ownership import region_reference
    def place(value, owner, region=False):
        target = device
        if model is not None:
            if type(owner) is not int:
                raise ValueError("checkpoint tensor owner must be int64")
            if not 0 <= owner < (len(model.regions) if region else len(model.nodes)):
                raise ValueError("invalid checkpoint tensor owner")
            target = (region_reference(graph, model, owner) if region else model.nodes[owner].bias).device
        return value.to(target) if target is not None else value
    return Continuation(record["identity"], record["batch_size"], record["cut"],
                        {k: State(place(s[0], k[1]), s[1], s[2], {n: place(v, k[1]) for n, v in s[3].items()})
                         for k, s in record["states"].items()},
                        {k: History(h[0], h[1], h[2], {n: place(v, k[1], True) for n, v in h[3].items()})
                         for k, h in record["history"].items()},
                        [Atom(a[0], a[1], a[2], a[3], a[4], a[5], place(a[6], a[1])) for a in record["pending"]],
                        record["ledger"])


def validate_weights(model, actual, aliases):
    if aliases != parameter_aliases(model):
        raise ValueError("checkpoint parameter sharing mismatch; reconstruct the same aliases before loading")
    expected = model.state_dict()
    if not isinstance(actual, dict) or expected.keys() != actual.keys():
        raise ValueError("checkpoint parameter keys mismatch")
    for key, value in expected.items():
        if (not isinstance(actual[key], torch.Tensor) or actual[key].shape != value.shape
                or actual[key].dtype != value.dtype or not torch.isfinite(actual[key]).all()):
            raise ValueError(f"checkpoint parameter mismatch: {key}")
    for names in aliases:
        if any(not torch.equal(actual[names[0]], actual[name]) for name in names[1:]):
            raise ValueError("checkpoint shared parameter values disagree")

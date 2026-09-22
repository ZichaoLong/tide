"""Shared value codecs and parameter preflight; no live model mutation."""
import torch
from .records import Atom, Continuation, State
from .history import History
from .checkpoint_ownership import parameter_aliases


def encode(continuation):
    q = continuation
    return {
        "identity": q.identity, "batch_size": q.batch_size, "cut": q.cut,
        "states": {k: (s.value.detach(), s.last_time, s.observations, {n: v.detach() for n, v in s.slots.items()})
                   for k, s in q.states.items()},
        "history": {k: (h.last_time, h.scalars, h.node_maps, {n: v.detach() for n, v in h.tensors.items()})
                    for k, h in q.history.items()}, "ledger": q.ledger,
        "pending": [(a.batch, a.node, a.time, a.kind, a.source, a.position, a.value.detach()) for a in q.pending],
    }


def decode(record):
    return Continuation(record["identity"], record["batch_size"], record["cut"],
                        {k: State(*s) for k, s in record["states"].items()},
                        {k: History(*h) for k, h in record["history"].items()},
                        [Atom(*a) for a in record["pending"]], record["ledger"])


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

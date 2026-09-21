"""Value checkpoints. Loading explicitly starts a new autograd segment."""
from pathlib import Path
import torch
from .records import Atom, Continuation, State
from .history import History
from .validation import validate_window
from .checkpoint_ownership import parameter_aliases, optimizer_record, preflight_optimizer


def save(path, graph, model, continuation, optimizer=None):
    q = continuation
    validate_window(graph, model, q, [], q.cut, q.cut)
    layout, optimizer_state = optimizer_record(model, optimizer)
    record = {
        "schema": "tide-continuation-v5", "identity": graph.identity, "aliases": parameter_aliases(model),
        "weights": model.state_dict(), "optimizer": optimizer_state, "optimizer_layout": layout,
        "batch_size": q.batch_size, "cut": q.cut,
        "states": {k: (s.value.detach(), s.last_time, s.observations, {n: v.detach() for n, v in s.slots.items()})
                   for k, s in q.states.items()},
        "history": {k: (h.last_time, h.scalars, h.node_maps, {n: v.detach() for n, v in h.tensors.items()})
                    for k, h in q.history.items()}, "ledger": q.ledger,
        "pending": [(a.batch, a.node, a.time, a.kind, a.source, a.position, a.value.detach()) for a in q.pending],
    }
    target = Path(path)
    # Exclusive creation keeps evidence/checkpoints from accidental replacement.
    with target.open("xb") as output:
        torch.save(record, output)


def load(path, graph, model, optimizer=None):
    record = torch.load(path, map_location="cpu", weights_only=True)
    if record["schema"] != "tide-continuation-v5" or record["identity"] != graph.identity:
        raise ValueError("checkpoint schema/graph mismatch")
    if record["aliases"] != parameter_aliases(model):
        raise ValueError("checkpoint parameter sharing mismatch; reconstruct the same aliases before loading")
    expected, actual = model.state_dict(), record["weights"]
    if expected.keys() != actual.keys():
        raise ValueError("checkpoint parameter keys mismatch")
    for key, value in expected.items():
        if (not isinstance(actual[key], torch.Tensor) or actual[key].shape != value.shape
                or actual[key].dtype != value.dtype or not torch.isfinite(actual[key]).all()):
            raise ValueError(f"checkpoint parameter mismatch: {key}")
    for names in record["aliases"]:
        if any(not torch.equal(actual[names[0]], actual[name]) for name in names[1:]):
            raise ValueError("checkpoint shared parameter values disagree")
    q = Continuation(record["identity"], record["batch_size"], record["cut"],
                     {k: State(*s) for k, s in record["states"].items()},
                     {k: History(*h) for k, h in record["history"].items()},
                     [Atom(*a) for a in record["pending"]], record["ledger"])
    validate_window(graph, model, q, [], q.cut, q.cut)
    preflight_optimizer(model, optimizer, record["optimizer_layout"], record["optimizer"])
    model.load_state_dict(actual)
    if optimizer is not None:
        optimizer.load_state_dict(record["optimizer"])
    return q

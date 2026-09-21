"""Value checkpoints. Loading explicitly starts a new autograd segment."""
from pathlib import Path
import torch
from .records import Atom, Continuation, State
from .validation import validate_window


def save(path, graph, model, continuation, optimizer=None):
    q = continuation
    validate_window(graph, model, q, [], q.cut, q.cut)
    record = {
        "schema": "tide-continuation-v1", "identity": graph.identity,
        "weights": model.state_dict(), "optimizer": None if optimizer is None else optimizer.state_dict(),
        "batch_size": q.batch_size, "cut": q.cut,
        "states": {k: (s.value.detach(), s.last_time, s.observations) for k, s in q.states.items()},
        "history": q.history, "ledger": q.ledger,
        "pending": [(a.batch, a.node, a.time, a.kind, a.source, a.position, a.value.detach()) for a in q.pending],
    }
    target = Path(path)
    # Exclusive creation keeps evidence/checkpoints from accidental replacement.
    with target.open("xb") as output:
        torch.save(record, output)


def load(path, graph, model, optimizer=None):
    record = torch.load(path, map_location="cpu", weights_only=True)
    if record["schema"] != "tide-continuation-v1" or record["identity"] != graph.identity:
        raise ValueError("checkpoint schema/graph mismatch")
    expected, actual = model.state_dict(), record["weights"]
    if expected.keys() != actual.keys():
        raise ValueError("checkpoint parameter keys mismatch")
    for key, value in expected.items():
        if actual[key].shape != value.shape or actual[key].dtype != value.dtype:
            raise ValueError(f"checkpoint parameter mismatch: {key}")
    q = Continuation(record["identity"], record["batch_size"], record["cut"],
                     {k: State(*s) for k, s in record["states"].items()}, record["history"],
                     [Atom(*a) for a in record["pending"]], record["ledger"])
    validate_window(graph, model, q, [], q.cut, q.cut)
    if optimizer is not None and record["optimizer"] is None:
        raise ValueError("checkpoint has no optimizer state")
    model.load_state_dict(actual)
    if optimizer is not None:
        optimizer.load_state_dict(record["optimizer"])
    return q

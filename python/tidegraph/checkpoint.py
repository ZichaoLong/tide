"""Value checkpoints. Loading explicitly starts a new autograd segment."""
import torch
from .checkpoint_io import publish
from .checkpoint_values import encode, decode, validate_weights
from .validation import validate_window
from .checkpoint_ownership import parameter_aliases, optimizer_record, preflight_optimizer


def save(path, graph, model, continuation, optimizer=None):
    q = continuation
    validate_window(graph, model, q, [], q.cut, q.cut)
    layout, optimizer_state = optimizer_record(model, optimizer)
    record = {
        "schema": "tide-continuation-v5", "aliases": parameter_aliases(model),
        "weights": model.state_dict(), "optimizer": optimizer_state, "optimizer_layout": layout,
        **encode(q),
    }
    publish(path, record)


def load(path, graph, model, optimizer=None):
    record = torch.load(path, map_location="cpu", weights_only=True)
    if record["schema"] != "tide-continuation-v5" or record["identity"] != graph.identity:
        raise ValueError("checkpoint schema/graph mismatch")
    actual = record["weights"]
    validate_weights(model, actual, record["aliases"])
    q = decode(record)
    validate_window(graph, model, q, [], q.cut, q.cut)
    preflight_optimizer(model, optimizer, record["optimizer_layout"], record["optimizer"])
    model.load_state_dict(actual)
    if optimizer is not None:
        optimizer.load_state_dict(record["optimizer"])
    return q

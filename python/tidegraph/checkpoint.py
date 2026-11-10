"""Value checkpoints. Loading explicitly starts a new autograd segment."""
import torch
from .checkpoint_io import publish
from .checkpoint_values import encode, decode, validate_weights
from .validation import validate_window
from .checkpoint_ownership import parameter_aliases, optimizer_record, preflight_optimizer
from .checkpoint_metrics import phase


def _place_optimizer_state(optimizer):
    for parameter, state in optimizer.state.items():
        for name, value in list(state.items()):
            # load_state_dict owns scalar step placement (CPU for ordinary
            # AdamW, device for capturable/fused modes). Slots follow their leaf.
            if name != "step" and isinstance(value, torch.Tensor) and value.device != parameter.device:
                state[name] = value.to(parameter.device)


def save(path, graph, model, continuation, optimizer=None):
    q = continuation
    with phase("save.validate_window"):
        validate_window(graph, model, q, [], q.cut, q.cut)
    with phase("save.optimizer_snapshot"):
        layout, optimizer_state = optimizer_record(model, optimizer)
    with phase("save.weights_to_host"):
        weights = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    with phase("save.encode_values"):
        record = {
            "schema": "tide-continuation-v5", "aliases": parameter_aliases(model),
            "weights": weights, "optimizer": optimizer_state, "optimizer_layout": layout,
            **encode(q),
        }
    publish(path, record)


def load(path, graph, model, optimizer=None, *, validate_continuation=None):
    """Read/decode once; validate every owner before publishing live state.

    ``validate_continuation`` may reject application boundary constraints on the
    decoded continuation. It runs before weights or optimizer slots are changed
    and must not mutate the live owners. The returned continuation is that same
    validated value, detached from any previous autograd segment.
    """
    with phase("load.deserialize_to_host"):
        record = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(record, dict) or record.get("schema") != "tide-continuation-v5":
        raise ValueError("checkpoint schema mismatch")
    if not {"weights", "aliases", "optimizer", "optimizer_layout"} <= record.keys():
        raise ValueError("malformed checkpoint ownership")
    if record.get("identity") != graph.identity:
        raise ValueError("checkpoint schema/graph mismatch")
    actual = record["weights"]
    with phase("load.validate_weights"):
        validate_weights(model, actual, record["aliases"])
    with phase("load.decode_values"):
        try:
            q = decode(record, graph=graph, model=model)
        except (KeyError, TypeError, IndexError, AttributeError) as error:
            raise ValueError("malformed checkpoint continuation") from error
        validate_window(graph, model, q, [], q.cut, q.cut)
        if validate_continuation is not None:
            validate_continuation(q)
    with phase("load.optimizer_preflight"):
        preflight_optimizer(model, optimizer, record["optimizer_layout"], record["optimizer"], actual)
    with phase("load.copy_weights"):
        model.load_state_dict(actual)
    if optimizer is not None:
        optimizer.load_state_dict(record["optimizer"])
        _place_optimizer_state(optimizer)
    return q

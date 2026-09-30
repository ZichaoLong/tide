"""Explicit portable export without uploading a checkpoint merely to validate it."""
import copy
import torch
from .checkpoint import save
from .checkpoint_ownership import parameter_aliases


def cpu_model(model):
    # Substitute CPU tensors during the structural copy, not after copying a
    # second full model on NPU. Memoization preserves repeated parameter owners.
    memo = {}
    for value in model.state_dict(keep_vars=True).values():
        if id(value) not in memo:
            tensor = value.detach().cpu()
            memo[id(value)] = (torch.nn.Parameter(tensor, requires_grad=value.requires_grad)
                              if isinstance(value, torch.nn.Parameter) else tensor)
    reference = copy.deepcopy(model, memo)
    if parameter_aliases(reference) != parameter_aliases(model):
        raise ValueError("resident CPU checkpoint view changed parameter ownership")
    return reference


def save_snapshot(path, graph, model, continuation):
    save(path, graph, cpu_model(model), continuation)


def prepare_restore(path, runtime, batch_size):
    """Validate CPU values and aliases before replacing any live owner/weight."""
    from .checkpoint_values import decode, validate_weights
    from .validation import validate_window
    record = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(record, dict) or record.get("schema") != "tide-continuation-v5"
            or record.get("identity") != runtime.execution_graph.identity):
        raise ValueError("checkpoint graph/schema mismatch")
    validate_weights(runtime.execution_model, record.get("weights"), record.get("aliases"))
    try:
        q = decode(record, device="cpu")
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError("malformed checkpoint continuation") from error
    if q.batch_size != batch_size:
        raise ValueError("checkpoint batch size mismatch")
    if runtime.spec and (q.cut % runtime.spec.stride or q.pending):
        raise ValueError("Settle resident session requires a complete position boundary")
    reference = cpu_model(runtime.execution_model)
    reference.load_state_dict(record["weights"])
    validate_window(runtime.execution_graph, reference, q, [], q.cut, q.cut)
    return q, record["weights"]

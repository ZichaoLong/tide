"""Explicit portable export without uploading a checkpoint merely to validate it."""
import copy
import torch
from .checkpoint import save
from .checkpoint_ownership import parameter_aliases


def save_snapshot(path, graph, model, continuation):
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
    save(path, graph, reference, continuation)

"""Bind positional optimizer state to named, shared model parameters."""
import copy
import math
import torch


def parameter_aliases(model):
    groups = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        groups.setdefault(id(parameter), []).append(name)
    return sorted(sorted(names) for names in groups.values())


def optimizer_layout(model, optimizer):
    if optimizer is None:
        return None
    aliases = parameter_aliases(model)
    named = dict(model.named_parameters(remove_duplicate=False))
    names = {id(named[group[0]]): group[0] for group in aliases}
    seen, groups = set(), []
    for group in optimizer.param_groups:
        row = []
        for parameter in group["params"]:
            key = id(parameter)
            if key not in names:
                raise ValueError("checkpoint optimizer parameter is not owned by the model")
            if key in seen:
                raise ValueError("checkpoint optimizer repeats a shared parameter")
            seen.add(key); row.append(names[key])
        groups.append(row)
    kind = type(optimizer)
    return {"class": f"{kind.__module__}.{kind.__qualname__}", "groups": groups}


def _finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(v) for v in value)
    return True


def validate_optimizer_state(optimizer, state):
    """Validate identity and built-in state layouts before loading live objects."""
    groups = state.get("param_groups")
    slots = state.get("state")
    if not isinstance(groups, list) or len(groups) != len(optimizer.param_groups) or not isinstance(slots, dict):
        raise ValueError("checkpoint optimizer state/group structure mismatch")
    parameters = []
    for saved, live in zip(groups, optimizer.param_groups):
        ids = saved.get("params")
        expected = list(range(len(parameters), len(parameters)+len(live["params"])))
        if not isinstance(ids, list) or any(type(i) is not int for i in ids) or ids != expected:
            raise ValueError("checkpoint optimizer parameter IDs mismatch")
        parameters.extend(live["params"])
    if any(type(i) is not int or not 0 <= i < len(parameters) for i in slots):
        raise ValueError("checkpoint optimizer state has an unowned parameter ID")
    if not _finite(state):
        raise ValueError("checkpoint optimizer state is nonfinite")
    adam = type(optimizer) in (torch.optim.Adam, torch.optim.AdamW)
    sgd = type(optimizer) is torch.optim.SGD
    for index, values in slots.items():
        if not isinstance(values, dict):
            raise ValueError("checkpoint optimizer parameter state must be a mapping")
        if not values or not (adam or sgd):
            continue
        allowed = {"step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"} if adam else {"momentum_buffer"}
        required = {"step", "exp_avg", "exp_avg_sq"} if adam else {"momentum_buffer"}
        if not required <= values.keys() <= allowed:
            raise ValueError("checkpoint optimizer tensor slots mismatch")
        parameter = parameters[index]
        for name, value in values.items():
            if name == "step":
                scalar = value.item() if isinstance(value, torch.Tensor) and value.ndim == 0 else value
                if type(scalar) not in (int, float) or scalar < 0 or scalar != int(scalar):
                    raise ValueError("checkpoint optimizer step mismatch")
            elif not isinstance(value, torch.Tensor) or value.shape != parameter.shape or value.dtype != parameter.dtype:
                raise ValueError("checkpoint optimizer tensor shape/dtype mismatch")
    if adam:
        for group in groups:
            if group.get("amsgrad"):
                for index in group["params"]:
                    if slots.get(index) and "max_exp_avg_sq" not in slots[index]:
                        raise ValueError("checkpoint optimizer is missing AMSGrad state")


def optimizer_record(model, optimizer):
    layout = optimizer_layout(model, optimizer)
    if optimizer is None:
        return layout, None
    state = optimizer.state_dict()
    validate_optimizer_state(optimizer, state)
    return layout, state


def preflight_optimizer(model, optimizer, layout, state):
    if optimizer is None:
        return
    if state is None:
        raise ValueError("checkpoint has no optimizer state")
    if layout != optimizer_layout(model, optimizer):
        raise ValueError("checkpoint optimizer ownership/order/class mismatch")
    validate_optimizer_state(optimizer, state)
    # Standard load hooks and group conversions must succeed before live weights
    # change. Custom hooks with external side effects are outside this contract.
    copy.deepcopy(optimizer).load_state_dict(copy.deepcopy(state))

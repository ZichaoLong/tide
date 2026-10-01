"""FP16 client loss/continuation mechanics; graph VJP oracle lives in the C++ gate."""
from dataclasses import replace
import os
import torch
from tidegraph import ResidentTrainingLimits
from test_resident_library import config, runtime as make_runtime, inputs


def runtime(family, target, schedule="greedy", memory="ema", mode="hard", model_device=None):
    return make_runtime(replace(config(family, memory), dtype="float16"), target, schedule, mode,
                     native_library=os.environ["TIDE_BUILD_DIR"], model_device=model_device,
                        resident_workspace_bytes=1024**3 if model_device == "cpu" else 64*1024**2)


def limits():
    return ResidentTrainingLimits(retained_bytes=256*1024*1024, backward_bytes=2*1024**3)


def roots(session, window, mode="all"):
    if mode == "none":
        return session.cotangents(window)
    factor = 0 if mode == "zero" else 1
    # Loss/head autograd leaves are FP32. Differentiating a half leaf would
    # itself round the consumer's cotangent before the explicit graph VJP.
    with torch.enable_grad():
        leaf = window.outputs.values.detach().float().requires_grad_(True)
        safe = torch.where(window.outputs.valid[:, None], leaf, torch.zeros_like(leaf))
        gradient, = torch.autograd.grad(safe.square().sum() * (factor*.0625), (leaf,))
    full = lambda x, value: torch.full(x.shape, factor*value, dtype=torch.float32, device=x.device)
    def state(values, groups):
        cache = []
        for group in groups:
            item = dict(key=full(group.key, .0078125), value=full(group.value, -.015625))
            if group.log_bias is not None:
                item["log_bias"] = full(group.log_bias, .0234375)
            padding = torch.arange(group.key.shape[1], device=group.key.device)[None, :] >= group.lengths[:, None]
            for name in item:
                item[name].masked_fill_(padding if name == "log_bias" else padding[:, :, None, None], float("nan"))
            cache.append(item)
        return dict(final=full(values, .03125), cache=cache)
    state_roots = (dict(states=[state(s.values, s.cache) for s in window.states])
                   if window.states else state(window.state_values, window.cache))
    return session.cotangents(window, outputs=gradient.detach(),
        pending=full(window.pending_values, .015625), **state_roots)



def update(session, values, start, stop, mode="all"):
    args, kw = inputs(session, values, start, stop)
    window = session.advance_device(*args, **kw)
    gradient = session.backward([roots(session, window, mode)])
    assert all(p.values.dtype == torch.float32 for p in (gradient.parameter_shards or [gradient]))
    assert session.step().applied
    return gradient


def payload_master(checkpoint):
    assert all(x.dtype == torch.float16 for x in checkpoint["parameters"].values())
    for name, value in checkpoint["state"].items():
        assert value.dtype == (torch.int64 if name == "steps" else torch.float32)
    for aliases, offset in zip(
            [a for a in checkpoint["aliases"] if a[0] in checkpoint["trainable"]], checkpoint["offsets"]):
        if offset < 0:
            continue
        value = checkpoint["parameters"][aliases[0]]
        master = checkpoint["state"]["values"].narrow(0, offset, value.numel()).reshape(value.shape)
        assert torch.equal(value, master.half())

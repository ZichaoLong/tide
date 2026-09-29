"""Stateful optimizer trajectories and new-process checkpoint continuation."""
import torch
from .qualification_checks import plain, vjp, probe_loss, assert_placement


def optimizer(runtime, kind):
    parameters = runtime.model.parameters()
    if runtime.config.dtype == "float16":
        from .precision import FP32MasterOptimizer
        if kind == "adamw":
            return FP32MasterOptimizer(parameters, optimizer="adamw", lr=.0002, weight_decay=.01,
                                       eps=1e-5, betas=(.9,.99), amsgrad=True)
        if kind in {"sgd", "momentum"}:
            return FP32MasterOptimizer(parameters, optimizer="sgd", lr=.0002, weight_decay=.01,
                                       momentum=.8 if kind=="momentum" else 0.)
    if kind == "adamw":
        return torch.optim.AdamW(parameters, lr=.0002, weight_decay=.01, eps=1e-5, betas=(.9,.99), amsgrad=True)
    if kind in {"sgd", "momentum"}:
        return torch.optim.SGD(parameters, lr=.0002, weight_decay=.01, momentum=.8 if kind == "momentum" else 0.)
    raise ValueError("optimizer must be adamw, sgd or momentum")


def trajectory(runtime, probe, steps, kind, *, checkpoint=None, resume=None):
    session = runtime.session(probe.batch_size)
    opt = optimizer(runtime, kind)
    if resume is not None:
        session.load(resume, opt)
    owners = {name:id(value) for name,value in runtime.model.named_parameters(remove_duplicate=False)}
    records = []
    for cycle in range(1 if resume is not None else 0, steps):
        opt.zero_grad(set_to_none=True)
        inputs = probe.clone(runtime.device)
        result = inputs.advance(session, cycle=cycle)
        assert_placement(result, runtime.device)
        loss = probe_loss(result)
        leaves = {k:v for k,v in runtime.model.named_parameters() if v.requires_grad} | inputs.leaves()
        gradient = plain(vjp(loss, leaves))
        if runtime.config.dtype == "float16": opt.backward(loss)
        else: loss.backward()
        # Parameter updates never invalidate live sequence autograd graphs.
        session.detach()
        opt.step()
        records.append(dict(result=plain(result), loss=plain(loss), gradients=gradient,
                            weights=plain(runtime.model.state_dict()), optimizer=plain(opt.state_dict())))
        if checkpoint is not None and cycle == 0:
            session.save(checkpoint, opt)
        if owners != {name:id(value) for name,value in runtime.model.named_parameters(remove_duplicate=False)}:
            raise AssertionError("optimizer replaced parameter owners")
    return records

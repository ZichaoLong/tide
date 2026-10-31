"""Public task extension: inputs/objective/optimizer belong to the consumer.

Factories receive the expanded task and plan; constructing one must not execute
the graph. The candidate and independent oracle get separate task instances.
No input callback receives a candidate trace or reference-selected actions.

An optional ``module: torch.nn.Module`` registers downstream parameters/buffers.
The eager runner places it at the first graph parameter's device/dtype before
calling ``optimizer(model)``. That model includes unique graph.* and task.*
owners; the optimizer must cover every declared trainable owner exactly once.
``state_dict/load_state_dict`` retain task data/RNG/scheduler/progress with
weights-only serializable values. Module weights and optimizer slots are saved
by the public checkpoint codec, not duplicated in task data state. Modules are
currently an eager extension; resident and standalone have separate contracts.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import importlib
import torch
from .task_identity import identity


class WeightedOutputTask:
    def __init__(self, task, plan):
        self.config, self.plan = task, plan

    def inputs(self, templates, index, *, device=None, training=False):
        occurrences = Counter((x.batch, x.port) for x in templates)
        copies = {}
        for x in templates:
            if id(x.value) not in copies:
                copies[id(x.value)] = x.value.detach().to(x.value.device if device is None else device).clone().requires_grad_(training)
        return [replace(x, time=x.time+index*self.plan["span"],
                        position=x.position+index*occurrences[x.batch, x.port],
                        value=copies[id(x.value)]) for x in templates]

    def loss(self, result, *, training):
        scale = self.config["objective"]["scale"]
        values = [x.float() if x.dtype == torch.float16 else x for *_, x in result.outputs]
        terms = [(x[0]*x[1:]).square().sum()*scale for x in values]
        if not terms:
            if training:
                raise ValueError("task.objective: window produced no differentiable output")
            return torch.zeros(())
        from ..weighted.placement_tensors import move
        return torch.stack([move(x, terms[0].device) for x in terms]).sum()

    def resident_loss(self, output):
        x = output.values.detach().float().clone().requires_grad_()
        numeric = x[..., :1]*x[..., 1:]
        loss = torch.where(output.present[..., None], numeric.square(), torch.zeros_like(numeric)).sum()*self.config["objective"]["scale"]
        bar, = torch.autograd.grad(loss, x)
        return loss, bar

    def optimizer(self, model, *, storage_reference=False):
        options = {k: v for k, v in self.config["optimizer"].items() if k != "kind"}
        if storage_reference:
            from .storage_optimizer import StorageSGD
            return StorageSGD((p for p in model.parameters() if p.requires_grad), **options)
        if next(model.parameters()).dtype == torch.float16:
            from ..precision import FP32MasterOptimizer
            return FP32MasterOptimizer([p for p in model.parameters() if p.requires_grad],
                                      optimizer="sgd", loss_scale=1., **options)
        return torch.optim.SGD(model.parameters(), **options)

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        if state:
            raise ValueError("built-in task has no mutable task state")


def create(task, plan):
    manifest = identity(task)
    plugin = task.get("plugin")
    if plugin is None:
        return WeightedOutputTask(task, plan), manifest
    module = importlib.import_module(plugin["module"])
    factory = getattr(module, plugin["factory"])
    owner = factory(task=deepcopy(task), plan=deepcopy(plan), options=deepcopy(plugin["options"]))
    required = ("inputs", "loss", "optimizer", "state_dict", "load_state_dict")
    missing = [name for name in required if not callable(getattr(owner, name, None))]
    if missing:
        raise ValueError("task.plugin: missing methods: "+", ".join(missing))
    if getattr(owner, "module", None) is not None and not isinstance(owner.module, torch.nn.Module):
        raise ValueError("task.module: expected torch.nn.Module or None")
    return owner, manifest

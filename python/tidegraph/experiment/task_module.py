"""Explicit graph/task parameter owners, independent of graph scheduling."""
import torch
from ..checkpoint_ownership import optimizer_layout, parameter_aliases


class GraphTaskModel(torch.nn.Module):
    """Value-checkpoint owner: graph.* and task.* keep distinct alias inventories.

    Schedulers continue to consume the original graph model. This owner is used
    for parameters, unified optimizer slots and the public value checkpoint codec.
    Module-free tasks keep their original model and parameter names unchanged.
    """
    def __init__(self, graph, task):
        super().__init__()
        self.graph, self.task = graph, task

    @property
    def nodes(self):
        return self.graph.nodes

    @property
    def regions(self):
        return self.graph.regions

    @property
    def width(self):
        return self.graph.width

    def validate_binding(self, graph):
        self.graph.validate_binding(graph)

    def validate_checkpoint_weights(self, actual):
        self.graph.validate_checkpoint_weights({k[len("graph."):]: v for k, v in actual.items() if k.startswith("graph.")})


def bind(task, graph):
    module = getattr(task, "module", None)
    if module is None:
        return graph
    if not isinstance(module, torch.nn.Module):
        raise ValueError("task.module: expected torch.nn.Module or None")
    graph_parameters = {id(p) for p in graph.parameters()}
    if any(id(x) in graph_parameters for x in module.parameters()):
        raise ValueError("task.module: graph and task parameters must have separate owners; sharing within each owner is preserved")
    aliases = parameter_aliases(module)
    for name, value in module.state_dict().items():
        if not isinstance(value, torch.Tensor) or value.is_complex():
            raise ValueError(f"task.module.{name}: use tensor state with real payloads; put non-tensor task state in state_dict()")
    reference = next(graph.parameters())
    module.to(device=reference.device, dtype=reference.dtype)
    if parameter_aliases(module) != aliases:
        raise ValueError("task.module: device/dtype conversion changed parameter aliases")
    return GraphTaskModel(graph, module)


def inventory(model):
    """Live object identity prevents task code from replacing optimizer owners."""
    return tuple((name, id(p), p.requires_grad, p.device, p.dtype, tuple(p.shape))
                 for name, p in model.named_parameters(remove_duplicate=False))


def validate_optimizer(model, optimizer):
    if isinstance(model, GraphTaskModel):
        for aliases in parameter_aliases(model):
            if {name.split(".", 1)[0] for name in aliases} == {"graph", "task"}:
                raise ValueError("task.module: optimizer callback shared parameters across graph/task owners")
    layout = optimizer_layout(model, optimizer)
    if optimizer is not None:
        actual = {id(p) for group in optimizer.param_groups for p in group["params"]}
        missing = [name for name, p in model.named_parameters() if p.requires_grad and id(p) not in actual]
        if missing:
            raise ValueError("task.optimizer: missing trainable parameter owners: "+", ".join(missing[:8]))
    return layout


def manifest(task, owner):
    module = getattr(task, "module", None)
    if module is None:
        return None
    return dict(schema="tide-task-module-v1", parameter_namespace="graph.* and task.*",
                parameter_aliases=parameter_aliases(owner),
                parameters={n: dict(shape=list(p.shape), dtype=str(p.dtype), device=str(p.device), trainable=p.requires_grad)
                            for n, p in module.named_parameters()},
                buffers={n: dict(shape=list(p.shape), dtype=str(p.dtype), device=str(p.device))
                         for n, p in module.named_buffers()},
                placement="task module follows the first graph parameter; task loss explicitly transports other outputs")

"""Public device-owned windows; no host readiness or action prepass."""
from types import SimpleNamespace
import torch
from ..runtime import synchronize
from .returns import snapshot
from .tasks import create
from .eager import work, finite_tensors


def limits(resolved):
    return dict(resolved["resources"]["resident_limits"])


class Resident:
    def __init__(self, resolved, *, checkpoint=None):
        from ..planning.materialize import materialize
        from ..weighted import WeightedResident, WeightedResidentTraining
        self.resolved, self.spec = resolved, resolved["execution"]["effective"]
        self.training = resolved["measurement"]["mode"] == "training"
        self.devices = tuple(torch.device(x) for x in self.spec["peer_devices"] or (self.spec["device"],))
        self.task, self.task_manifest = create(resolved["task"], resolved["plan"])
        if getattr(self.task, "module", None) is not None:
            raise ValueError("task.module: resident does not register external task parameters/optimizer; use an eager client or supply a separately implemented resident task protocol")
        binding, self.inputs, self.span = materialize(resolved["plan"], dtype=getattr(torch, self.spec["dtype"]),
                max_bytes=resolved["resources"]["max_materialize_bytes"])
        if self.training and not callable(getattr(self.task, "resident_loss", None)):
            raise ValueError("task.plugin: resident training requires resident_loss(output_field) -> (loss, FP32 bars)")
        options = {k: v for k, v in resolved["task"]["optimizer"].items() if k != "kind"}
        from .training_layout import SGDLayout
        self.layout = SGDLayout(binding.model, **options) if self.training else None
        self.last_parameter_gradients = None
        self.last_work = None
        kwargs = dict(device=self.spec["device"], **{k: resolved["site"]["paths"][k] for k in ("native_library", "resident_library")},
            batch_size=resolved["plan"]["samples"], limits=limits(resolved),
            node_devices=resolved["payload_placement"]["node_devices"])
        with torch.no_grad():
            if checkpoint is not None:
                extra = torch.load(checkpoint/"task.pt", map_location="cpu", weights_only=True)
                if extra.get("task_manifest") != self.task_manifest:
                    raise ValueError("resume: task source/configuration identity differs")
                self.task.load_state_dict(extra["task"])
                if self.training:
                    kwargs["checkpoint"] = checkpoint/"state.pt"
                else:
                    from ..checkpoint import load
                    kwargs["continuation"] = load(checkpoint/"state.pt", binding.graph, binding.model)
            elif self.training:
                from ..native_loader import load_native
                from ..native_model import encode_model
                core = load_native(kwargs["native_library"], backend=self.devices[0].type)
                _, model = encode_model(core, binding.graph, binding.model)
                names = [p.canonical for p in model.parameters().owners()]
                kwargs.update(optimizer="sgd", groups=[dict(parameters=names, **options)])
            cls = WeightedResidentTraining if self.training else WeightedResident
            self.owner = cls(binding.graph, binding.model, **kwargs)

    def step(self, index, *, return_mode=None):
        mode = return_mode or self.spec.get("return_mode", "complete")
        inputs = self.task.inputs(self.inputs, index, device="cpu", training=False)
        with torch.no_grad():
            if self.training:
                window = self.owner.advance(inputs, stop=(index+1)*self.span, sealed_until=(index+1)*self.span)
                with torch.enable_grad():
                    loss, bar = self.task.resident_loss(window.fields["outputs"])
                finite_tensors((loss, bar))
                roots = self.owner.cotangents(window, {"outputs": bar})
                result = self.owner.result() if mode == "complete" else None
                gradients = self.owner.backward([roots])
                self.last_parameter_gradients = gradients.parameters
                self.last_backward_statistics = dict(gradients.statistics)
                finite_tensors(gradients.parameters.values)
                if mode == "complete":
                    actual = snapshot(result)
                    self.last_work = work(result)
                else:
                    actual = dict(return_mode=mode, cut=(index+1)*self.span,
                                  outputs=self.owner.outputs() if mode == "outputs" else None)
                    self.last_work = self.owner.metrics()
                step = self.owner.step()
                if not step.applied:
                    raise RuntimeError("resident optimizer refused: "+str(step.refusal_code))
            else:
                actual = self.owner.advance(inputs, stop=(index+1)*self.span, sealed_until=(index+1)*self.span, return_mode=mode)
                if mode == "complete":
                    finite_tensors(actual)
                    self.last_work = work(actual)
                    loss = self.task.loss(actual, training=False)
                    actual = snapshot(actual)
                else:
                    self.last_work = dict(actual["stats"])
                    if mode == "outputs":
                        loss = self.task.loss(SimpleNamespace(outputs=actual["outputs"]), training=False)
                    else:
                        loss = None
            if loss is not None:
                finite_tensors(loss)
            finite_tensors(self.task.state_dict())
            value = None if loss is None else float(loss.detach().cpu())
            for device in self.devices:
                synchronize(device)
            return actual, value

    def parameters(self):
        with torch.no_grad():
            return snapshot(self.layout.parameters(self.owner.checkpoint())) if self.training else snapshot(dict(self.owner.model.named_parameters()))

    def training_state(self):
        with torch.no_grad():
            return None if not self.training else self.layout.decode(self.last_parameter_gradients, self.owner.checkpoint())

    def decoded_state(self):
        with torch.no_grad():
            return dict(checkpoint=snapshot(self.owner.checkpoint()), task=snapshot(self.task.state_dict()))

    def task_state(self):
        if not self.resolved["task"].get("plugin"):
            return None
        return snapshot(dict(module=None, data=self.task.state_dict()))

    def save(self, directory):
        from ..checkpoint_io import publish
        from .task_identity import verify
        verify(self.resolved["task"], self.task_manifest)
        with torch.no_grad():
            if self.training:
                self.owner.save(directory/"state.pt")
            else:
                from ..checkpoint import save
                save(directory/"state.pt", self.owner.graph, self.owner.model, self.owner.checkpoint())
            publish(directory/"task.pt", dict(task=self.task.state_dict(), task_manifest=self.task_manifest))

    def manifest(self):
        with torch.no_grad():
            return dict(self.owner.manifest(), client=self.spec["client"], task=self.task_manifest,
                        return_mode=self.spec.get("return_mode", "complete"),
                        scheduler="public device resident library", last_backward_statistics=getattr(self, "last_backward_statistics", None))

    def close(self):
        with torch.no_grad():
            if self.owner is not None:
                self.owner.close()
            self.owner = self.inputs = self.layout = self.last_parameter_gradients = self.task = None

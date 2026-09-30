"""Python client of the explicit C++/CANN training owner, never a CPU routing pass."""
from dataclasses import replace
from pathlib import Path
import torch
from .native_model import encode_model
from .native_records import from_continuation, to_continuation, window_records
from .records import Continuation, Result
from .resident import parameter_identity
from .resident_inputs import external_window, forward_limits
from .resident_training_options import ResidentTrainingLimits, optimizer_groups


class ResidentTrainingSession:
    def __init__(self, runtime, batch_size, *, continuation=None, optimizer=None,
                 groups=None, limits=None, checkpoint=None):
        from .coordinates import integers
        integers("training batch size", batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if torch.is_grad_enabled():
            raise ValueError("explicit resident VJP requires torch.no_grad(); compute consumer cotangents separately")
        if checkpoint is not None and any(x is not None for x in (continuation, optimizer, groups)):
            raise ValueError("checkpoint resume is exclusive with continuation/optimizer/group initialization")
        if limits is None:
            limits = ResidentTrainingLimits()
        elif not isinstance(limits, ResidentTrainingLimits):
            limits = ResidentTrainingLimits(**limits)
        self.runtime, self.batch_size, self.limits = runtime, batch_size, limits
        core, module = runtime.engine.core, runtime.engine.module
        if not hasattr(module, "TrainingSession"):
            raise ValueError("resident backend was built without the explicit training API")
        self.compiled, weights = encode_model(core, runtime.execution_graph, runtime.execution_model)
        self.parameters = parameter_identity(runtime.execution_model)
        native_limits = module.TrainingLimits()
        native_limits.forward = forward_limits(runtime, training=True)
        for name, value in limits.to_dict().items():
            setattr(native_limits, name, value)
        if checkpoint is not None:
            if isinstance(checkpoint, (str, Path)):
                checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
            from .resident_training_checkpoint import prepare
            saved = prepare(runtime, self.compiled, checkpoint, batch_size)
            self.owner = module.TrainingSession(self.compiled, weights, saved, runtime.device, native_limits)
        else:
            optimizer = "sgd" if optimizer is None else optimizer
            if optimizer not in {"sgd", "adamw"}:
                raise ValueError("resident optimizer must be sgd or adamw")
            q = Continuation(runtime.execution_graph.identity, batch_size) if continuation is None else continuation
            if q.batch_size != batch_size:
                raise ValueError("continuation batch size mismatch")
            if runtime.spec and (q.cut % runtime.spec.stride or q.pending):
                raise ValueError("Settle training requires a complete position boundary")
            self.owner = module.TrainingSession(self.compiled, weights,
                to_continuation(core, runtime.execution_graph, self.compiled, q), runtime.device,
                getattr(module.OptimizerKind, optimizer), optimizer_groups(core, groups), native_limits)

    def _check(self):
        if parameter_identity(self.runtime.execution_model) != self.parameters:
            raise RuntimeError("caller parameters changed after resident training construction")

    @property
    def cut(self):
        return self.owner.cut

    @property
    def position(self):
        if not self.runtime.spec:
            raise ValueError("PDG/TimedDAG use explicit logical cuts")
        return self.cut // self.runtime.spec.stride

    @property
    def generation(self):
        return self.owner.generation

    @property
    def retained_windows(self):
        return self.owner.retained_windows

    def advance_device(self, inputs, *, stop=None, sealed_until=None):
        self._check()
        external, stop, sealed_until = external_window(self.runtime, self.batch_size, self.cut, inputs, stop, sealed_until)
        return self.owner.advance(external, stop, sealed_until)

    def cotangents(self, window, *, outputs=None, outputs_connected=None, pending=None,
                   pending_connected=None, final=None, final_connected=None):
        """Make explicit roots; supplied values default to the window's presence mask.

        Omitted values and masks mean None. All tensors must be detached FP32
        values/boolean masks on the session NPU. No host event lookup is required.
        """
        r = self.runtime.engine.module.Cotangents()
        r.token = window.token
        for name, value, connected, valid in (
                ("outputs", outputs, outputs_connected, window.outputs.valid),
                ("pending", pending, pending_connected, window.pending_valid),
                ("final", final, final_connected, window.state_present)):
            if value is None:
                if connected is not None:
                    raise ValueError("a cotangent connection mask requires values")
            else:
                setattr(r, name, value)
                setattr(r, name + "_connected", valid if connected is None else connected)
        return r

    def backward(self, roots):
        self._check()
        return self.owner.backward(list(roots))

    def step(self):
        self._check()
        return self.owner.step()

    def detach(self):
        self._check()
        self.owner.detach()

    def checkpoint(self):
        self._check()
        from .resident_training_checkpoint import export
        return export(self.runtime, self.owner.checkpoint())

    def save(self, path):
        from .checkpoint_io import publish
        publish(path, self.checkpoint())

    def result(self):
        self._check()
        r = self.runtime
        value = self.owner.result()
        result = Result(from_continuation(r.execution_graph, value.continuation), *window_records(value))
        result = r.spec.project(result) if r.spec else result
        return result if r.options.trace else replace(result, trace=[], messages=[])

    def close(self):
        self.owner.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

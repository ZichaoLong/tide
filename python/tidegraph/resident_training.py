"""Python client of the explicit C++/CANN training owner, never a CPU routing pass."""
from dataclasses import replace
from pathlib import Path
import torch
from .native_model import encode_model
from .native_records import from_continuation, to_continuation, window_records
from .records import Continuation, Result
from .resident import parameter_identity
from .resident_inputs import external_window, forward_limits
from .resident_training_options import ResidentTrainingLimits, ResidentPlacement, optimizer_groups


class ResidentTrainingSession:
    def __init__(self, runtime, batch_size, *, continuation=None, optimizer=None,
                 groups=None, limits=None, checkpoint=None, placement=None):
        from .coordinates import integers
        integers("training batch size", batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if runtime.config.dtype not in {"float32", "float16"}:
            raise ValueError("resident training requires FP32/FP16 payloads with FP32 cotangents")
        if torch.is_grad_enabled():
            raise ValueError("explicit resident VJP requires torch.no_grad(); compute consumer cotangents separately")
        if checkpoint is not None and any(x is not None for x in (continuation, optimizer, groups)):
            raise ValueError("checkpoint resume is exclusive with continuation/optimizer/group initialization")
        if limits is None:
            limits = ResidentTrainingLimits()
        elif not isinstance(limits, ResidentTrainingLimits):
            limits = ResidentTrainingLimits(**limits)
        self.runtime, self.batch_size, self.limits = runtime, batch_size, limits
        self.requested_placement = (placement if isinstance(placement, ResidentPlacement)
                                    else ResidentPlacement(**({} if placement is None else placement)))
        core, module = runtime.engine.core, runtime.engine.module
        if not hasattr(module, "TrainingSession"):
            raise ValueError("resident backend was built without the explicit training API")
        self.compiled, weights = encode_model(core, runtime.execution_graph, runtime.execution_model)
        self.parameters = parameter_identity(runtime.execution_model)
        native_limits = module.TrainingLimits()
        native_limits.forward = forward_limits(runtime, training=True)
        native_limits.placement = self.requested_placement.native(
            module, runtime.device, len(runtime.execution_graph.nodes))
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

    @property
    def accumulated_batches(self):
        return self.owner.accumulated_batches

    def advance_device(self, inputs, *, stop=None, sealed_until=None):
        self._check()
        external, stop, sealed_until = external_window(self.runtime, self.batch_size, self.cut, inputs, stop, sealed_until)
        return self.owner.advance(external, stop, sealed_until)

    @property
    def placement(self):
        value = self.owner.placement
        return dict(devices=[str(d) for d in value.devices], policy=value.policy,
                    full_owners=value.full_owners, state_owners=value.state_owners)

    def manifest(self):
        record = self.runtime.manifest()
        record["resident"].update(devices=len(self.placement["devices"]), training=True,
                                  requested_training_placement=self.requested_placement.to_dict(),
                                  resolved_training_placement=self.placement,
                                  training_limits=self.limits.to_dict())
        return record

    def cotangents(self, window, **values):
        """Detached FP32 roots, including optional owner-local ``states`` dicts.

        Supplied values default to their owner's presence mask. Omitted values
        and masks mean None; connected zero remains distinct. State/cache roots
        stay on the device of the corresponding window tensor.
        """
        from .resident_training_roots import cotangents
        return cotangents(self.runtime.engine.module, window, **values)

    def backward(self, roots):
        self._check()
        return self.owner.backward(list(roots))

    def step(self):
        self._check()
        return self.owner.step()

    def accumulate(self, *, max_bytes=128*1024**2):
        """Sum this backward into a device bank and explicitly detach state.

        Parameters/generation stay fixed. Normalize consumer losses against the
        complete logical batch; this operation only sums. Accumulate the final
        backward too, then call step once. This does not switch sample state.
        """
        from .coordinates import integers
        self._check()
        integers("gradient accumulation budget", max_bytes)
        if max_bytes < 1:
            raise ValueError("gradient accumulation budget must be positive")
        self.owner.accumulate(max_bytes)

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

"""Public explicit weighted VJP/optimizer owner. No Python scheduling or replay.

Contracts and qualification scopes: docs/resident-weighted-training.md.
"""
from pathlib import Path
from copy import deepcopy
import torch
from ..coordinates import integers, window_inputs
from ..native_loader import load_native
from ..native_model import encode_model
from ..native_records import from_continuation, to_continuation, window_records
from ..records import Continuation, Result
from ..resident import parameter_identity
from ..resident_loader import load_resident
from ..resident_training_options import GROUP_FIELDS, optimizer_groups


class WeightedResidentTraining:
    def __init__(self, graph, model, *, device, native_library, resident_library,
                 batch_size=1, continuation=None, optimizer=None, groups=None, limits=None, checkpoint=None):
        integers("weighted training batch size", batch_size)
        if batch_size < 1 or torch.is_grad_enabled():
            raise ValueError("weighted training requires positive batch size and torch.no_grad()")
        if checkpoint is not None and any(x is not None for x in (continuation, optimizer, groups)):
            raise ValueError("weighted checkpoint resume is exclusive with continuation/optimizer/groups")
        from .binding import WeightedModel
        if not isinstance(model, WeightedModel):
            raise ValueError("weighted training requires an installed WeightedModel")
        if str(device).split(":", 1)[0] == "npu":
            import torch_npu
        self.device = torch.device(device)
        if self.device.type not in {"npu", "cuda"} or self.device.index is None:
            raise ValueError("weighted training requires an explicit indexed accelerator")
        if any(x.device.type != "cpu" or x.dtype not in {torch.float32, torch.float16} for x in model.state_dict().values()):
            raise ValueError("weighted training requires CPU FP32/FP16 boundary parameters")
        model.validate_binding(graph)
        self.core = load_native(native_library, backend=self.device.type)
        self.module, self.build_record = load_resident(resident_library, self.core, component="weighted")
        if not hasattr(self.module, "WeightedTrainingSession") or "libtide-weighted-training.so" not in self.build_record["binary_sha256"]:
            raise ValueError("weighted resident build does not provide the public training owner")
        self.graph, self.model, self.batch_size = graph, model, batch_size
        self.dtype = next(iter(model.state_dict().values())).dtype
        self.compiled, weights = encode_model(self.core, graph, model)
        self.identity = parameter_identity(model)
        self.trainable = tuple((name, x.requires_grad) for name, x in model.state_dict(keep_vars=True).items())
        self.requested_limits = dict(limits or {})
        self.requested_optimizer, self.requested_groups = optimizer, deepcopy(groups)
        self.initialization = "resume" if checkpoint is not None else "fresh"
        spec = self.module.WeightedTrainingLimits()
        forward = spec.forward
        forward_names = {"queue", "stages", "cache", "tensor_bytes", "operator_bytes", "timeout_ms", "prefill"}
        training_names = {"windows", "retained_bytes", "backward_bytes", "optimizer_bytes", "total_bytes"}
        for name, value in self.requested_limits.items():
            if name not in forward_names | training_names:
                raise ValueError("unknown weighted training limit: " + name)
            if name == "prefill":
                if type(value) is not bool:
                    raise ValueError("prefill must be bool")
            else:
                integers("weighted training " + name, value)
                if value < 1:
                    raise ValueError("weighted training limits must be positive")
            setattr(forward if name in forward_names else spec, name, value)
        spec.forward = forward
        self.effective_limits = {name: getattr(forward, name) for name in forward_names}
        self.effective_limits.update({name: getattr(spec, name) for name in training_names})
        if checkpoint is not None:
            if isinstance(checkpoint, (str, Path)):
                checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
            from .resident_training_checkpoint import prepare
            self.owner = self.module.WeightedTrainingSession(self.compiled, weights, prepare(self, checkpoint), self.device, spec)
        else:
            optimizer = "sgd" if optimizer is None else optimizer
            if optimizer not in {"sgd", "adamw"}:
                raise ValueError("weighted optimizer must be sgd or adamw")
            q = Continuation(graph.identity, batch_size) if continuation is None else continuation
            if q.batch_size != batch_size:
                raise ValueError("weighted continuation batch size differs")
            self.owner = self.module.WeightedTrainingSession(self.compiled, weights,
                to_continuation(self.core, graph, self.compiled, q), self.device,
                getattr(self.module.OptimizerKind, optimizer), optimizer_groups(self.core, groups), spec)
        self.effective_optimizer = "sgd" if self.owner.optimizer_kind() == self.module.OptimizerKind.sgd else "adamw"
        self.effective_groups = [{name: getattr(g, name) for name in GROUP_FIELDS} for g in self.owner.optimizer_groups()]

    def _check(self):
        if torch.is_grad_enabled():
            raise ValueError("weighted training methods require torch.no_grad()")
        if (parameter_identity(self.model) != self.identity
                or tuple((n, x.requires_grad) for n, x in self.model.state_dict(keep_vars=True).items()) != self.trainable):
            raise RuntimeError("caller parameters changed after weighted training construction")

    @property
    def cut(self):
        self._check(); return self.owner.cut

    @property
    def generation(self):
        self._check(); return self.owner.generation

    def advance(self, inputs, *, stop, sealed_until):
        self._check(); integers("weighted cut/seal", stop, sealed_until)
        if stop != sealed_until:
            raise ValueError("weighted training requires an exact complete input seal")
        xs = window_inputs(inputs, stop, sealed_until)
        return self.owner.advance([self.core.External(x.batch, x.port, x.position, x.time, x.value) for x in xs], stop)

    def cotangents(self, window, fields=None):
        """Field name -> detached values or (values, connected). Omission is None."""
        self._check(); out = self.module.WeightedCotangents(); out.token = window.token
        result = {}
        for name, supplied in (fields or {}).items():
            if name not in window.fields:
                raise ValueError("unknown weighted root field: " + name)
            f = window.fields[name]
            values, flags = supplied if isinstance(supplied, tuple) else (supplied, f.present)
            for value, expected, dtype in ((values, f.values, torch.float32), (flags, f.present, torch.bool)):
                if (not isinstance(value, torch.Tensor) or value.shape != expected.shape
                        or value.device != expected.device or value.dtype != dtype or value.requires_grad):
                    raise ValueError("weighted cotangent shape/device/dtype/gradient differs: " + name)
            r = self.module.WeightedCotangent(); r.values, r.connected = values.contiguous(), flags.contiguous()
            result[name] = r
        out.fields = result
        return out

    def backward(self, roots):
        self._check(); return self.owner.backward(list(roots))

    def accumulate(self):
        self._check(); self.owner.accumulate()

    def step(self):
        self._check(); return self.owner.step()

    @staticmethod
    def step_replicas(sessions, *, communication_bytes=256*1024*1024, timeout_ms=30000):
        """Sum independent samples' gradients once, then update all replicas.

        Every owner has the same graph, parameters and optimizer state on a
        distinct device. Loss normalization belongs to the caller; no implicit
        average is applied. Private raw state and local sample IDs stay private.
        """
        sessions = list(sessions)
        integers("weighted peer budget/timeout", communication_bytes, timeout_ms)
        if not 2 <= len(sessions) <= 16 or communication_bytes < 1 or not 1 <= timeout_ms <= 60000:
            raise ValueError("weighted peer step needs 2..16 replicas and bounded resources")
        for session in sessions:
            if not isinstance(session, WeightedResidentTraining):
                raise ValueError("weighted peer step requires training owners")
            session._check()
            if session.module is not sessions[0].module:
                raise ValueError("weighted peer owners require the same resident library")
        return sessions[0].module.WeightedTrainingSession.step_replicas(
            [session.owner for session in sessions], communication_bytes, timeout_ms)

    def detach(self):
        self._check(); self.owner.detach()

    def result(self):
        self._check(); actual = self.owner.result()
        trace, outputs, messages, stats = window_records(actual)
        return Result(from_continuation(self.graph, actual.continuation), trace, outputs, messages, stats)

    def checkpoint(self):
        self._check()
        from .resident_training_checkpoint import export
        return export(self, self.owner.checkpoint())

    def save(self, path):
        from ..checkpoint_io import publish
        publish(path, self.checkpoint())

    def manifest(self):
        return dict(requested_device=str(self.device), effective_device=str(self.device), dtype=str(self.dtype).removeprefix("torch."),
                    raw_dtype=str(self.dtype).removeprefix("torch."), compute_dtype="float32",
                    scoring_dtype="float32", adjoint_dtype="float32", master_dtype="float32",
                    execution="resident", training=True, requested_limits=self.requested_limits,
                    effective_limits=self.effective_limits, initialization=self.initialization,
                    requested_optimizer=self.requested_optimizer, requested_groups=deepcopy(self.requested_groups),
                    effective_optimizer=self.effective_optimizer, effective_groups=deepcopy(self.effective_groups),
                    build=self.build_record)

    def close(self):
        self.owner.close()

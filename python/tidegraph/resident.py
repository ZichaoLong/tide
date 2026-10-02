"""Public Python client for the optional C++/CANN resident inference backend."""
import torch
from .native_model import encode_model
from .native_records import from_continuation, to_continuation, window_records
from .records import Continuation, Result


def parameter_identity(model):
    # Detect replacements as well as normal in-place updates, including buffers
    # and aliases. This does not support writes through .data or foreign pointers.
    return tuple((name, id(value), value._version, value.data_ptr(), tuple(value.shape), value.dtype, value.device)
                 for name, value in model.state_dict(keep_vars=True).items())


class ResidentBackend:
    def __init__(self, runtime, native_library, resident_library):
        from .native_loader import load_native
        from .resident_loader import load_resident
        self.core = load_native(native_library, backend="npu")
        self.module, self.record = load_resident(resident_library, self.core)
        self.runtime = runtime

    def session(self, batch_size, continuation=None, placement=None):
        return ResidentSession(self, batch_size, continuation, placement)


class ResidentSession:
    def __init__(self, backend, batch_size, continuation=None, placement=None):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if torch.is_grad_enabled():
            raise ValueError("resident inference requires explicit torch.no_grad()")
        self.backend, self.runtime, self.batch_size = backend, backend.runtime, batch_size
        r = self.runtime
        q = Continuation(r.execution_graph.identity, batch_size) if continuation is None else continuation
        if q.batch_size != batch_size:
            raise ValueError("continuation batch size mismatch")
        if r.spec and (q.cut % r.spec.stride or q.pending):
            raise ValueError("Settle resident session requires a complete position boundary")
        self.compiled, weights = encode_model(backend.core, r.execution_graph, r.execution_model)
        self.parameters = parameter_identity(r.execution_model)
        from .resident_training_options import ResidentPlacement
        self.requested_placement = (placement if isinstance(placement, ResidentPlacement)
                                    else ResidentPlacement(**({} if placement is None else placement)))
        native_placement = self.requested_placement.native(backend.module, r.device, len(r.execution_graph.nodes))
        from .resident_inputs import forward_limits
        limits = forward_limits(r)
        self.owner = backend.module.Session(self.compiled, weights,
            to_continuation(backend.core, r.execution_graph, self.compiled, q), r.device, limits, native_placement)

    @property
    def placement(self):
        value = self.owner.placement
        return dict(devices=[str(d) for d in value.devices], policy=value.policy,
                    full_owners=value.full_owners, state_owners=value.state_owners)

    def manifest(self):
        record = self.runtime.manifest()
        record["resident"].update(devices=len(self.placement["devices"]), training=False,
                                  requested_inference_placement=self.requested_placement.to_dict(),
                                  resolved_inference_placement=self.placement)
        return record

    @property
    def cut(self):
        return self.owner.cut

    @property
    def position(self):
        if not self.runtime.spec:
            raise ValueError("PDG/TimedDAG use explicit logical cuts")
        return self.cut // self.runtime.spec.stride

    @property
    def continuation(self):
        """Explicit CPU export; never used automatically by advance_device."""
        return self.snapshot()

    def advance_device(self, inputs, *, stop=None, sealed_until=None):
        """Borrow packed NPU outputs until next advance/close; no state export.

        CPU and NPU input payloads are accepted at this boundary. Returned
        coordinates retain the physical encoded graph, including Settle's times.
        """
        r = self.runtime
        if parameter_identity(r.execution_model) != self.parameters:
            raise RuntimeError("resident parameters changed; construct a new session from an explicit complete cut")
        from .resident_inputs import external_window
        xs, stop, sealed_until = external_window(r, self.batch_size, self.cut, inputs, stop, sealed_until)
        return self.owner.advance(xs, stop, sealed_until)

    def advance(self, inputs, *, stop=None, sealed_until=None):
        """Compatibility result with explicit CPU materialization after execution."""
        self.advance_device(inputs, stop=stop, sealed_until=sealed_until)
        return self.result()

    def snapshot(self):
        return from_continuation(self.runtime.execution_graph, self.owner.snapshot())

    def snapshot_device(self, *, max_bytes, compact=False, device_budgets=None):
        """Save numerical continuation on its original NPU owners, within budget."""
        from .coordinates import integers
        if parameter_identity(self.runtime.execution_model) != self.parameters:
            raise RuntimeError("resident parameters changed")
        integers("device continuation budget", max_bytes)
        if max_bytes < 1:
            raise ValueError("device continuation budget must be positive")
        if type(compact) is not bool:
            raise ValueError("device continuation compact must be bool")
        from .resident_options import continuation_device_budgets
        return self.owner.snapshot_device(max_bytes, compact, continuation_device_budgets(device_budgets))

    def restore_device(self, saved):
        """Restore an opaque snapshot from this session; parameters stay fixed."""
        if parameter_identity(self.runtime.execution_model) != self.parameters:
            raise RuntimeError("resident parameters changed")
        self.owner.restore_device(saved)

    def result(self):
        r = self.runtime
        result = self.owner.result()
        result = Result(from_continuation(r.execution_graph, result.continuation), *window_records(result))
        return r.spec.project(result) if r.spec else result

    def save(self, path, optimizer=None):
        if optimizer is not None:
            raise ValueError("resident inference does not own a training optimizer")
        if parameter_identity(self.runtime.execution_model) != self.parameters:
            raise RuntimeError("cannot checkpoint resident state with changed parameters")
        from .resident_checkpoint import save_snapshot
        save_snapshot(path, self.runtime.execution_graph, self.runtime.execution_model, self.snapshot())

    def close(self):
        self.owner.close()

    def load(self, path, optimizer=None):
        """Restore weights and the complete cut; other frozen sessions expire.

        Invalid checkpoint values leave the live owner unchanged. After CPU
        validation, the old owner closes before allocating its replacement;
        a device/construction failure leaves this session closed.
        """
        if optimizer is not None:
            raise ValueError("resident inference does not own a training optimizer")
        if torch.is_grad_enabled():
            raise ValueError("resident inference requires explicit torch.no_grad()")
        from .resident_checkpoint import prepare_restore
        q, weights = prepare_restore(path, self.runtime, self.batch_size)
        self.close()
        self.runtime.execution_model.load_state_dict(weights)
        self._replace(q)

    def reset(self):
        """Explicit fresh sequence with the current weights and limits."""
        if torch.is_grad_enabled():
            raise ValueError("resident inference requires explicit torch.no_grad()")
        self.close()
        self._replace(None)

    def _replace(self, continuation):
        other = ResidentSession(self.backend, self.batch_size, continuation, self.requested_placement)
        self.compiled, self.parameters, self.owner = other.compiled, other.parameters, other.owner

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

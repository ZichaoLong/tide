"""Explicit state lifecycle for a public GraphRuntime."""
import torch
from .records import Continuation
from .validation import validate_window


class Session:
    def __init__(self, runtime, batch_size, *, continuation=None):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.runtime = runtime
        q = Continuation(runtime.execution_graph.identity, batch_size) if continuation is None else continuation
        if q.batch_size != batch_size:
            raise ValueError("continuation batch size mismatch")
        self._validate(q)
        self.continuation = q.fork()

    def _validate(self, q):
        r = self.runtime
        validate_window(r.execution_graph, r.execution_model, q, [], q.cut, q.cut)
        if r.spec and (q.cut % r.spec.stride or q.pending):
            raise ValueError("Settle session requires a complete position boundary")

    @property
    def position(self):
        if not self.runtime.spec:
            raise ValueError("PDG/TimedDAG use explicit logical cuts and per-port positions")
        return self.continuation.cut // self.runtime.spec.stride

    def advance(self, inputs, *, stop=None, sealed_until=None):
        """Settle: [batch, positions, width]; others: explicit External records.

        Inputs retain their autograd connection. No dtype/device conversion,
        padding, hidden detach, optimizer step or inference context is applied.
        Settle results project body records; session.continuation retains the
        encoded boundary state needed by later scheduling and checkpoints.
        """
        r, q = self.runtime, self.continuation
        if r.spec:
            if stop is not None or sealed_until is not None:
                raise ValueError("Settle advances whole positions; do not supply logical stop/seal")
            if (not isinstance(inputs, torch.Tensor) or inputs.ndim != 3
                    or inputs.shape[0] != q.batch_size or inputs.shape[2] != r.config.width
                    or inputs.device != r.device or inputs.dtype != getattr(torch, r.config.dtype)):
                raise ValueError("Settle inputs require matching [batch, positions, width], dtype and device")
            external = r.spec.external(inputs, self.position, encoded=True)
            stop = sealed_until = (self.position + inputs.shape[1]) * r.spec.stride
        else:
            if stop is None or sealed_until is None:
                raise ValueError("PDG/TimedDAG require explicit stop and sealed_until")
            external = inputs
        result = r._run(q, external, stop, sealed_until)
        projected = r.spec.project(result) if r.spec else result
        self.continuation = result.continuation
        return projected

    def detach(self):
        """Start a new truncated-training segment, including all in-flight data."""
        self.continuation = self.continuation.detach()

    def save(self, path, optimizer=None):
        from .checkpoint import save
        save(path, self.runtime.execution_graph, self.runtime.execution_model, self.continuation, optimizer)

    def load(self, path, optimizer=None):
        """Restore declared graph/optimizer state; no task controller/RNG promise."""
        from .checkpoint import load
        from .checkpoint_values import decode
        record = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(record, dict) or record.get("schema") != "tide-continuation-v5":
            raise ValueError("checkpoint schema mismatch")
        if not {"weights", "aliases", "optimizer", "optimizer_layout"} <= record.keys():
            raise ValueError("malformed checkpoint ownership")
        try:
            candidate = decode(record, graph=self.runtime.execution_graph, model=self.runtime.execution_model)
        except (KeyError, TypeError, IndexError, AttributeError) as error:
            raise ValueError("malformed checkpoint continuation") from error
        if candidate.batch_size != self.continuation.batch_size:
            raise ValueError("checkpoint batch size mismatch")
        self._validate(candidate)  # Reject an incomplete Settle cut before changing weights.
        self.continuation = load(path, self.runtime.execution_graph, self.runtime.execution_model, optimizer)

    def reset(self):
        """Clear sequence state while preserving model parameters and optimizer."""
        self.continuation = Continuation(self.runtime.execution_graph.identity, self.continuation.batch_size)

"""Architecture §3 / allocation §2: explicit weighted execution placement.

Placement changes temporary execution views, never parameter owners or raw
continuation. The independent host schedule still owns complete candidate sets.
"""
from types import SimpleNamespace
import torch
from ..readout import ReadProgram
from ..region import RegionProgram
from .programs import Describe, RegionalTotal
from .parameter_tree import rebind
from .placement_tensors import mapper, controls


def _delegate(target, original):
    object.__setattr__(target, "_original", original)
    target._parameters = dict(original._parameters)
    target._buffers = dict(original._buffers)
    target._modules = dict(original._modules)
    target.profile = original.profile


class PlacedDescribe(ReadProgram):
    joint_batch = True

    def __init__(self, spec, original, dtype, device):
        super().__init__()
        if type(original) is PlacedDescribe:
            original = original._original
        if type(original) is not Describe or original.profile != spec.readout:
            raise ValueError("weighted placement requires matching Describe")
        _delegate(self, original)
        self.memory_config, self.config = original.memory_config, original.config
        self.field_layout = dict(original.field_layout)
        self.precision = str(dtype).split(".")[-1] if dtype in {torch.float32, torch.float64} else "payload"
        self.dtype, self.device = dtype, device

    def descriptor_device(self, payload):
        return self.device

    def _mapped(self, weights, requests):
        from .query_vjp import _rows
        bundle = torch.nn.ModuleDict(dict(memory=weights.kernel, program=self._original))
        values = [*bundle.parameters(), *bundle.buffers(), weights.bias]
        def collect(t):
            values.append(t)
            return t
        _rows(requests, "read", collect)
        mapped = mapper(values, self.device, self.dtype)
        bound = rebind(bundle, mapped)
        rows = _rows(requests, "read", mapped)
        return bound["program"], SimpleNamespace(kernel=bound["memory"], bias=mapped(weights.bias)), rows

    def step(self, weights, request):
        program, bound, rows = self._mapped(weights, [request])
        return program.step(bound, rows[0])

    def batch(self, weights, requests):
        program, bound, rows = self._mapped(weights, requests)
        return program.batch(bound, rows)

    def batch_grad(self, weights, requests):
        from .query_vjp import evaluate
        program, bound, rows = self._mapped(weights, requests)
        return evaluate(program, bound, rows)


class PlacedSending(RegionProgram):
    joint_batch = batched_autograd = history_independent = True

    def __init__(self, spec, original, placement):
        super().__init__()
        if type(original) is PlacedSending:
            original = original._original
        if type(original) is not RegionalTotal or original.profile != spec.selector:
            raise ValueError("weighted placement requires matching Sending")
        _delegate(self, original)
        self.config = original.config
        self.placement = placement

    def initial(self, layout, reference):
        return self._original.initial(layout, reference)

    def validate_weights(self, layout):
        self._original.validate_weights(layout)

    def validate_history(self, history, layout):
        self._original.validate_history(history, layout)

    def _mapped(self, requests):
        from .region_vjp import fields
        p = self.placement
        # Each frame retains its own declared descriptor dtype. Batch groups
        # below avoid changing a different frame's precision.
        dtype = requests[0].candidates[0][1]["w"].dtype
        values = [*self._original.parameters(), *self._original.buffers()]
        values += [t for r in requests for _, d in r.candidates for t in d.values()]
        mapped = mapper(values, p["control"], dtype)
        program = rebind(self._original, mapped)
        program._selection_device = p["selection"]
        rows = fields(requests, mapped)
        return program, rows

    def step(self, request):
        program, rows = self._mapped([request])
        return controls(program.step(rows[0]), request)

    def _batch(self, requests, grad):
        groups, result = {}, [None]*len(requests)
        for i, r in enumerate(requests):
            groups.setdefault(r.candidates[0][1]["w"].dtype, []).append(i)
        for ids in groups.values():
            program, rows = self._mapped([requests[i] for i in ids])
            values = program.batch_grad(rows) if grad else program.batch(rows)
            for i, value in zip(ids, values):
                result[i] = controls(value, requests[i])
        return result

    def batch(self, requests):
        return self._batch(requests, False)

    def batch_grad(self, requests):
        return self._batch(requests, True)

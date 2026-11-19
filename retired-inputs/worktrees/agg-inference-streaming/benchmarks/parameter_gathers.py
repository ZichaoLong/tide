"""Untimed parameter-index materialization witness, separate from normal runs.

Only outputs of advanced indexing whose input aliases a real Parameter count.
This is logical allocated output bytes, not total memory traffic or hardware
bandwidth. Views, casts, fused concatenations and dynamic routing have their
own costs. Caller-thread dispatch instrumentation needs serial_region_audit
for concurrent placed execution; it does not change any model defaults.
"""
from collections import defaultdict

import torch
from torch.utils._python_dispatch import TorchDispatchMode


class ParameterGatherAudit(TorchDispatchMode):
    def __init__(self, model):
        super().__init__()
        self.storages = {(str(parameter.device), parameter.untyped_storage().data_ptr()): name
                         for name, parameter in model.named_parameters() if parameter.numel()}
        self.rows = defaultdict(lambda: dict(calls=0, output_bytes=0))

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if str(func) in {"aten.index.Tensor", "aten.index_select.default"} and args and isinstance(args[0], torch.Tensor):
            value = args[0]
            name = self.storages.get((str(value.device), value.untyped_storage().data_ptr()))
            if name is not None:
                row = self.rows[name]
                row["calls"] += 1
                row["output_bytes"] += result.numel()*result.element_size()
        return result

    def report(self):
        return dict(parameters=dict(self.rows), calls=sum(row["calls"] for row in self.rows.values()),
                    output_bytes=sum(row["output_bytes"] for row in self.rows.values()),
                    definition="logical advanced-index output bytes for operands sharing Parameter storage; not total traffic/FLOPs or measured bandwidth",
                    boundary="intrusive untimed witness; no parameter values retained")

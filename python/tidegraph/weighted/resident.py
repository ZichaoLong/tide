"""Explicit native resident weighted owner; architecture §1–4, core-5 cuts.

CPU FP32/FP16 model/boundary records, an explicit accelerator, and torch.no_grad()
are required. Window-internal scheduling and numerical execution are in C++.
This adapter neither runs a reference trajectory nor changes requested dtype.
"""
import torch

from ..coordinates import integers
from ..native_loader import load_native
from ..native_model import encode_model
from ..native_records import from_continuation, to_continuation, window_records
from ..records import Continuation, Result
from ..resident import parameter_identity
from ..resident_loader import load_resident


class WeightedResident:
    def __init__(self, graph, model, *, device, native_library, resident_library,
                 batch_size=1, continuation=None, limits=None, node_devices=None):
        integers("resident batch size",batch_size)
        if batch_size<1 or torch.is_grad_enabled():
            raise ValueError("weighted resident inference requires positive batch size and torch.no_grad()")
        from .binding import WeightedModel
        if not isinstance(model,WeightedModel):
            raise ValueError("weighted resident requires an installed WeightedModel")
        if str(device).split(":",1)[0]=="npu":
            import torch_npu  # Register the explicitly requested backend only.
        self.device=torch.device(device)
        if self.device.type not in {"cuda","npu"} or self.device.index is None:
            raise ValueError("weighted resident requires an explicit indexed accelerator")
        from .resident_placement import node_placement
        placement=node_placement(node_devices,self.device,len(graph.nodes))
        self.requested_node_devices=None if node_devices is None else list(map(str,placement))
        if any(x.device.type!="cpu" or x.dtype not in {torch.float32,torch.float16} for x in model.state_dict().values()):
            raise ValueError("weighted resident boundary requires CPU FP32/FP16 parameters")
        model.validate_binding(graph)
        self.core=load_native(native_library,backend=self.device.type)
        self.module,self.build_record=load_resident(resident_library,self.core,component="weighted")
        self.graph,self.model=graph,model
        self.dtype=next(iter(model.state_dict().values())).dtype
        self.compiled,weights=encode_model(self.core,graph,model)
        q=Continuation(graph.identity,batch_size) if continuation is None else continuation
        if q.batch_size!=batch_size:
            raise ValueError("weighted resident continuation batch size mismatch")
        self.requested_limits=dict(limits or {})
        spec=self.module.WeightedLimits()
        for name,value in self.requested_limits.items():
            if name not in {"queue","stages","cache","tensor_bytes","operator_bytes","timeout_ms","prefill"}:
                raise ValueError("unknown weighted resident limit: "+name)
            if name=="prefill":
                if type(value) is not bool:raise ValueError("prefill must be bool")
            else:
                integers("weighted resident "+name,value)
                if value<1:raise ValueError("weighted resident limits must be positive")
            setattr(spec,name,value)
        self.effective_limits={name:getattr(spec,name) for name in
            ("queue","stages","cache","tensor_bytes","operator_bytes","timeout_ms","prefill")}
        self.identity=parameter_identity(model)
        self.owner=self.module.WeightedFlow(self.compiled,weights,to_continuation(self.core,graph,self.compiled,q),self.device,spec,placement)
        self.effective_node_devices=list(map(str,self.owner.node_devices))
        self.cut=q.cut

    def _validate(self):
        if torch.is_grad_enabled():raise ValueError("weighted resident inference requires torch.no_grad()")
        if parameter_identity(self.model)!=self.identity:
            raise RuntimeError("weighted parameters changed; create a new owner from an explicit checkpoint")

    def _inputs(self, inputs, stop, sealed_until):
        self._validate();integers("weighted resident cut",stop);integers("weighted input seal",sealed_until)
        if stop!=sealed_until:
            raise ValueError("weighted resident requires an exact complete input seal")
        from ..coordinates import window_inputs
        xs=window_inputs(inputs,stop,sealed_until)
        return [self.core.External(x.batch,x.port,x.position,x.time,x.value) for x in xs]

    def _capability(self, name):
        if not hasattr(self.owner,name):
            raise ValueError("resident build lacks "+name+"; rebuild the resident library from matching source")

    def _result(self, actual):
        trace,outputs,messages,stats=window_records(actual)
        return Result(from_continuation(self.graph,actual.continuation),trace,outputs,messages,stats)

    def advance(self, inputs, *, stop, sealed_until, return_mode="complete"):
        """Execute one window; non-complete modes return a distinct small record.

        The device retains its full raw state and bounded journal. These modes
        only avoid unrequested CPU decoding; checkpoint remains explicit.
        """
        if return_mode not in {"complete","outputs","metrics"}:
            raise ValueError("return_mode must be complete, outputs or metrics")
        if return_mode=="complete":
            actual=self.owner.advance(self._inputs(inputs,stop,sealed_until),stop);self.cut=stop
            return self._result(actual)
        self._capability("advance_device");self._capability("metrics")
        if return_mode=="outputs":self._capability("outputs")
        start=self.cut
        self.advance_device(inputs,stop=stop,sealed_until=sealed_until)
        return dict(schema="tide-resident-return-v1",return_mode=return_mode,start=start,stop=stop,
                    outputs=self.outputs() if return_mode=="outputs" else None,stats=self.metrics())

    def advance_device(self, inputs, *, stop, sealed_until):
        """Advance the actual device owner without any diagnostic record decode."""
        self._capability("advance_device")
        self.owner.advance_device(self._inputs(inputs,stop,sealed_until),stop);self.cut=stop

    def result(self):
        self._validate();self._capability("result")
        return self._result(self.owner.result())

    def outputs(self):
        self._validate();self._capability("outputs")
        return [(o.batch,o.time,o.port,o.value) for o in self.owner.outputs()]

    def metrics(self):
        self._validate();self._capability("metrics")
        return dict(self.owner.metrics())

    def checkpoint(self):
        self._validate();return from_continuation(self.graph,self.owner.checkpoint())

    def manifest(self):
        return dict(requested_device=str(self.device),effective_device=str(self.device),dtype=str(self.dtype).removeprefix("torch."),
                    raw_dtype=str(self.dtype).removeprefix("torch."),compute_dtype="float32",
                    scoring_dtype="float32",adjoint_dtype="float32",
                    execution="resident",requested_limits=self.requested_limits,
                    effective_limits=self.effective_limits,reserved_tensor_bytes=self.owner.reserved_bytes,
                    requested_node_devices=self.requested_node_devices,effective_node_devices=self.effective_node_devices,
                    available_return_modes=["complete", "outputs", "metrics"] if
                        all(hasattr(self.owner,n) for n in ("advance_device","outputs","metrics")) else ["complete"],
                    minimal_return_scope="requested CPU fields only; complete device journal remains allocated",
                    output_return_packing="active rows, 8MiB groups or one indivisible wide row" if
                        hasattr(self.owner,"outputs") else None,
                    build=self.build_record,training=False)

    def close(self):
        self.owner.close()

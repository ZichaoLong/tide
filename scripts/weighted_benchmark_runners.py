"""Public candidates and a separately invoked scalar reference consumer."""
from dataclasses import fields, is_dataclass, replace
import json
from pathlib import Path
import torch
from tidegraph import Continuation
from tidegraph.placement import ExecutionPlacement
from tidegraph.runtime import synchronize
from weighted_benchmark_cases import window, limits


def snapshot(value):
    """Decode only actual computed fields, preserving the complete record type."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if is_dataclass(value):
        return replace(value, **{f.name: snapshot(getattr(value, f.name)) for f in fields(value)})
    if isinstance(value, dict):
        return {key: snapshot(x) for key, x in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(snapshot(x) for x in value)
    return value


def loss_from_outputs(result):
    terms = [(x[0]*x[1:]).square().sum()*.001 for *_, x in result.outputs]
    if not terms:
        raise ValueError("benchmark workload produced no differentiable output")
    return torch.stack(terms).sum()


def work(result):
    return dict(candidate_updates=len(result.trace), full_calls=sum(e["active"] for e in result.trace),
                messages=len(result.messages), outputs=len(result.outputs),
                pending=len(result.continuation.pending), actual_stats=result.stats)


class Eager:
    def __init__(self, binding, inputs, span, config, args, device, reference=False):
        from weighted_device_checks import runner
        self.binding, self.inputs, self.span, self.config, self.args, self.device = binding, inputs, span, config, args, device
        self.binding.model.to(device)
        placement = ExecutionPlacement(preset=args.placement, scoring_dtype="float32")
        if reference:
            from tidegraph.reference import run
            self.execute = lambda *a, **kw: run(binding.graph, binding.model, *a, **kw)
        else:
            self.execute = runner(binding, args.implementation, placement, args.native_library,
                                  streaming=args.schedule == "streaming")
        self.state = Continuation(binding.graph.identity, config.samples)
        self.optimizer = torch.optim.SGD(binding.model.parameters(), lr=.001, momentum=.3, weight_decay=.01)

    def step(self, index):
        xs = window(self.inputs, self.span, self.config.positions, index, self.device)
        with torch.set_grad_enabled(self.args.mode == "training"):
            self.optimizer.zero_grad(set_to_none=True)
            result = self.execute(self.state, xs, (index+1)*self.span, sealed_until=(index+1)*self.span)
            loss = loss_from_outputs(result)
            if self.args.mode == "training":
                loss.backward()
                checks = [torch.isfinite(p.grad).all() for p in self.binding.model.parameters() if p.grad is not None]
                if not checks or not torch.stack(checks).all():
                    raise FloatingPointError("nonfinite or absent benchmark parameter gradients")
                self.optimizer.step()
        self.state = result.continuation.detach()
        actual = snapshot(result)  # Complete CPU return is included for every candidate.
        value = float(loss.detach().cpu())
        synchronize(self.device)
        return actual, value

    def parameters(self):
        return snapshot(dict(self.binding.model.named_parameters()))

    def manifest(self):
        placement = ExecutionPlacement(preset=self.args.placement, scoring_dtype="float32")
        result = dict(execution=self.args.placement, schedule=self.args.schedule, operator="aten",
                      effective_placement={k: str(v) for k, v in placement.resolve(self.device).items()},
                      complete_cpu_records=True, precision="FP32", optimizer="sgd", gradient_boundary="detach per window")
        if self.args.implementation == "native":
            path = Path(self.args.native_library)
            result["build"] = json.loads(((path if path.is_dir() else path.parent)/"build-manifest.json").read_text())
        return result

    def close(self):
        pass


class Resident:
    def __init__(self, binding, inputs, span, config, args, device):
        from tidegraph.weighted import WeightedResident, WeightedResidentTraining
        self.inputs, self.span, self.config, self.args, self.device = inputs, span, config, args, device
        self.training = args.mode == "training"
        spec = limits(binding, config, span); spec["prefill"] = args.schedule == "prefill"
        if not self.training:
            spec = {k: v for k, v in spec.items() if k not in {"retained_bytes", "backward_bytes", "optimizer_bytes", "total_bytes"}}
        kwargs = dict(device=str(device), native_library=args.native_library, resident_library=args.resident_library,
                      batch_size=config.samples, limits=spec)
        if self.training:
            from tidegraph.native_loader import load_native
            from tidegraph.native_model import encode_model
            core = load_native(args.native_library, backend=device.type)
            _, model = encode_model(core, binding.graph, binding.model)
            names = [p.canonical for p in model.parameters().owners()]
            kwargs.update(optimizer="sgd", groups=[dict(parameters=names, lr=.001, momentum=.3, weight_decay=.01)])
        with torch.no_grad():
            cls = WeightedResidentTraining if self.training else WeightedResident
            self.owner = cls(binding.graph, binding.model, **kwargs)

    def step(self, index):
        # Public resident entry owns the CPU-boundary upload and native packing.
        xs = window(self.inputs, self.span, self.config.positions, index, "cpu")
        with torch.no_grad():
            result = self.owner.advance(xs, stop=(index+1)*self.span, sealed_until=(index+1)*self.span)
            if self.training:
                output = result.fields["outputs"]
                with torch.enable_grad():
                    x = output.values.detach().float().clone().requires_grad_()
                    numeric = x[..., :1]*x[..., 1:]
                    loss = torch.where(output.present[..., None], numeric.square(), torch.zeros_like(numeric)).sum()*.001
                    bar, = torch.autograd.grad(loss, x)
                roots = self.owner.cotangents(result, {"outputs": bar})
                actual = self.owner.result()
                gradients = self.owner.backward([roots])
                if not torch.isfinite(gradients.parameters.values).all():
                    raise FloatingPointError("nonfinite resident parameter gradient")
                step = self.owner.step()
                if not step.applied:
                    raise RuntimeError("resident optimizer refused: "+str(step.refusal_code))
            else:
                actual = result; loss = loss_from_outputs(actual)
            actual = snapshot(actual)
            value = float(loss.detach().cpu())
            synchronize(self.device)
            return actual, value

    def parameters(self):
        if not self.training:
            return None
        with torch.no_grad():
            return {n.replace(".extra.", "."): x for n, x in self.owner.checkpoint()["parameters"].items()}

    def manifest(self):
        with torch.no_grad():
            return self.owner.manifest()

    def close(self):
        with torch.no_grad():
            self.owner.close()

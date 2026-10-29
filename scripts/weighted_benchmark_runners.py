"""Public candidates and a separately invoked scalar reference consumer."""
from dataclasses import fields, is_dataclass, replace
from contextlib import nullcontext
import json
from pathlib import Path
import torch
from tidegraph import Continuation
from tidegraph.placement import ExecutionPlacement
from tidegraph.runtime import synchronize
from weighted_benchmark_cases import window, limits


def phase(args, name):
    return torch.profiler.record_function("tide."+name) if args.profile else nullcontext()


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
    values = [x.float() if x.dtype == torch.float16 else x for *_, x in result.outputs]
    terms = [(x[0]*x[1:]).square().sum()*.001 for x in values]
    if not terms:
        raise ValueError("benchmark workload produced no differentiable output")
    return torch.stack([t.to(terms[0].device) for t in terms]).sum()


def work(result):
    return dict(candidate_updates=len(result.trace), full_calls=sum(e["active"] for e in result.trace),
                messages=len(result.messages), outputs=len(result.outputs),
                pending=len(result.continuation.pending), actual_stats=result.stats)


class Eager:
    def __init__(self, binding, inputs, span, config, args, device, reference=False):
        from tidegraph.weighted.execution import eager_runner
        self.storage_reference = reference and args.implementation == "resident" and getattr(args, "dtype", "float32") == "float16"
        if self.storage_reference:
            from dataclasses import replace
            from tidegraph.qualification_weighted_precision import reference_model, widen
            binding = replace(binding, model=reference_model(binding.model))
            inputs = widen(inputs)
        self.binding, self.inputs, self.span, self.config, self.args, self.device = binding, inputs, span, config, args, device
        self.devices = (device,) if reference else tuple(torch.device(d) for d in getattr(args, "peer_devices", None) or (device,))
        self.node_owners = None
        if len(self.devices) > 1:
            from weighted_device_checks import locate
            self.inputs, self.node_owners = locate(binding, inputs, self.devices)
        else:
            self.binding.model.to(device)
        placement = ExecutionPlacement(preset=args.placement, scoring_dtype=getattr(args, "scoring_dtype", None) or ("float64" if getattr(args, "dtype", "float32") == "float64" else "float32"))
        if reference:
            from tidegraph.reference import run
            from tidegraph.placement import place_model
            view = place_model(binding.graph, binding.model, ExecutionPlacement(scoring_dtype=placement.scoring_dtype))
            self.execute = lambda *a, **kw: run(binding.graph, view, *a, **kw)
        else:
            self.execute = eager_runner(binding, args.implementation, placement, args.native_library,
                                  streaming=args.schedule == "streaming", workers=getattr(args, "workers", 1),
                                  packing=getattr(args, "packing", "auto"))
        self.state = Continuation(binding.graph.identity, config.samples)
        self.half = next(binding.model.parameters()).dtype == torch.float16
        if self.storage_reference:
            from weighted_storage_optimizer import StorageSGD
            self.optimizer = StorageSGD(p for p in binding.model.parameters() if p.requires_grad)
        elif self.half:
            from tidegraph.precision import FP32MasterOptimizer
            self.optimizer = FP32MasterOptimizer([p for p in binding.model.parameters() if p.requires_grad],
                optimizer="sgd", loss_scale=1., lr=.001, momentum=.3, weight_decay=.01)
        else:
            self.optimizer = torch.optim.SGD(binding.model.parameters(), lr=.001, momentum=.3, weight_decay=.01)
        self.update_counts = dict.fromkeys(dict(binding.model.named_parameters()), 0)

    def step(self, index):
        with phase(self.args, "prepare"):
            xs = window(self.inputs, self.span, self.config.positions, index, None if self.node_owners else self.device)
        with torch.set_grad_enabled(self.args.mode == "training"):
            self.optimizer.zero_grad(set_to_none=True)
            with phase(self.args, "forward"):
                result = self.execute(self.state, xs, (index+1)*self.span, sealed_until=(index+1)*self.span)
            with phase(self.args, "loss"):
                loss = loss_from_outputs(result)
            if self.args.mode == "training":
                with phase(self.args, "backward_finite"):
                    self.optimizer.backward(loss) if self.half or self.storage_reference else loss.backward()
                    checks = {}
                    for p in self.binding.model.parameters():
                        if p.grad is not None:
                            checks.setdefault(p.grad.device, []).append(torch.isfinite(p.grad).all())
                    if not checks or not all(bool(torch.stack(rows).all()) for rows in checks.values()):
                        raise FloatingPointError("nonfinite or absent benchmark parameter gradients")
                with phase(self.args, "optimizer"):
                    for name, p in self.binding.model.named_parameters():
                        if p.grad is not None:self.update_counts[name] += 1
                    self.optimizer.step()
        with phase(self.args, "detach_return_sync"):
            self.state = result.continuation.detach()
            actual = snapshot(result)  # Complete CPU return is included for every candidate.
            value = float(loss.detach().cpu())
            for device in self.devices:synchronize(device)
        return actual, value

    def parameters(self):
        if self.storage_reference:
            return snapshot({n: p.half() for n, p in self.binding.model.named_parameters()})
        return snapshot(dict(self.binding.model.named_parameters()))

    def training_state(self):
        return snapshot(dict(gradients={n: p.grad.float() if self.half and p.grad is not None else p.grad
                                         for n, p in self.binding.model.named_parameters()},
                             optimizer=self.optimizer.state_dict(), update_counts=self.update_counts))

    def manifest(self):
        dtype = getattr(self.args, "dtype", "float32")
        scoring = getattr(self.args, "scoring_dtype", None) or ("float64" if dtype == "float64" else "float32")
        placement = ExecutionPlacement(preset=self.args.placement, scoring_dtype=scoring)
        result = dict(execution=self.args.placement, schedule=self.args.schedule, operator="aten",
                      effective_placement={k: str(v) for k, v in placement.resolve(self.device).items()},
                      complete_cpu_records=True, precision=dtype, scoring_dtype=scoring, optimizer="sgd",
                      master_dtype="float32" if self.half else dtype, loss_scale=1.,
                      packing=getattr(self.args, "packing", "auto"), workers=getattr(self.args, "workers", 1),
                      gradient_boundary="detach per window")
        result.update(devices=[str(d) for d in self.devices],
                      node_owners=None if self.node_owners is None else [str(d) for d in self.node_owners])
        if self.args.implementation == "native":
            path = Path(self.args.native_library)
            result["build"] = json.loads(((path if path.is_dir() else path.parent)/"build-manifest.json").read_text())
        return result

    def close(self):
        pass


class Resident:
    def __init__(self, binding, inputs, span, config, args, device, *, checkpoint=None):
        from tidegraph.weighted import WeightedResident, WeightedResidentTraining
        self.inputs, self.span, self.config, self.args, self.device = inputs, span, config, args, device
        self.training = args.mode == "training"
        self.devices = tuple(torch.device(d) for d in getattr(args, "peer_devices", None) or (device,))
        from weighted_training_observation import SGDLayout
        self.sgd_layout = SGDLayout(binding.model) if self.training else None
        spec = limits(binding, config, span); spec["prefill"] = args.schedule == "prefill"
        spec.update(getattr(args, "resident_limit_values", None) or {})
        if not self.training:
            spec = {k: v for k, v in spec.items() if k not in {"windows", "retained_bytes", "backward_bytes", "optimizer_bytes", "total_bytes"}}
        kwargs = dict(device=str(device), native_library=args.native_library, resident_library=args.resident_library,
                      batch_size=config.samples, limits=spec)
        if len(self.devices) > 1:
            kwargs["node_devices"] = [str(self.devices[n.region % len(self.devices)]) for n in binding.graph.nodes]
        if checkpoint is not None:
            kwargs["checkpoint"] = checkpoint
        elif self.training:
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
        with phase(self.args, "prepare"):
            xs = window(self.inputs, self.span, self.config.positions, index, "cpu")
        with torch.no_grad():
            with phase(self.args, "forward"):
                result = self.owner.advance(xs, stop=(index+1)*self.span, sealed_until=(index+1)*self.span)
            if self.training:
                output = result.fields["outputs"]
                with phase(self.args, "loss"), torch.enable_grad():
                    x = output.values.detach().float().clone().requires_grad_()
                    numeric = x[..., :1]*x[..., 1:]
                    loss = torch.where(output.present[..., None], numeric.square(), torch.zeros_like(numeric)).sum()*.001
                    bar, = torch.autograd.grad(loss, x)
                with phase(self.args, "decode_roots"):
                    roots = self.owner.cotangents(result, {"outputs": bar})
                    actual = self.owner.result()
                with phase(self.args, "backward_finite"):
                    gradients = self.owner.backward([roots])
                    self.last_backward_statistics = dict(gradients.statistics)
                    self.last_parameter_gradients = gradients.parameters
                    if not torch.isfinite(gradients.parameters.values).all():
                        raise FloatingPointError("nonfinite resident parameter gradient")
                with phase(self.args, "optimizer"):
                    step = self.owner.step()
                    if not step.applied:
                        raise RuntimeError("resident optimizer refused: "+str(step.refusal_code))
            else:
                actual = result; loss = loss_from_outputs(actual)
            with phase(self.args, "detach_return_sync"):
                actual = snapshot(actual)
                value = float(loss.detach().cpu())
                for device in self.devices:synchronize(device)
            return actual, value

    def parameters(self):
        if not self.training:
            return None
        with torch.no_grad():
            return self.sgd_layout.parameters(self.owner.checkpoint())

    def training_state(self):
        if not self.training:return None
        with torch.no_grad():
            return self.sgd_layout.decode(self.last_parameter_gradients, self.owner.checkpoint())

    def manifest(self):
        with torch.no_grad():
            return dict(self.owner.manifest(), last_backward_statistics=getattr(self, "last_backward_statistics", None))

    def close(self):
        with torch.no_grad():
            self.owner.close()
            self.last_parameter_gradients = None

"""One public weighted eager session; the scalar reference owns its schedule."""
from dataclasses import replace
import torch
from .. import Continuation
from ..placement import ExecutionPlacement, place_model
from ..runtime import synchronize
from .returns import snapshot
from .tasks import create
from .finite import finite_tensors
from .record_policy import needs_trace


def work(result):
    from .column_task import WindowResult, work as column_work
    if isinstance(result, WindowResult):
        return column_work(result)
    return dict(candidate_updates=result.stats.get("candidate_events", len(result.trace)),
                full_calls=result.stats.get("selected_events", sum(e["active"] for e in result.trace)),
                messages=result.stats.get("visited_edges", len(result.messages)),
                outputs=len(result.outputs), pending=len(result.continuation.pending),
                actual_stats=dict(result.stats))


class Eager:
    def __init__(self, resolved, *, reference=False, checkpoint=None):
        from ..column_policy import validate_column_position_record
        validate_column_position_record(resolved["execution"])
        from ..planning.materialize import materialize
        spec = resolved["execution"]["effective"]
        self.resolved, self.spec, self.reference = resolved, spec, reference
        # A plugin must explicitly declare output-only observations before the
        # selected public return may omit its intermediate event roots.
        self.trace_records = needs_trace(spec, resolved["task"], reference=reference)
        self._complete_execute = None
        self.check_finite = lambda value, path="runtime": finite_tensors(value, path=path, packing=
            "scalar" if reference else spec.get("finite_check_packing", "scalar"))
        self.training = resolved["measurement"]["mode"] == "training"
        self.device = torch.device("cpu" if reference else spec["device"])
        self.devices = (self.device,) if reference else tuple(torch.device(x) for x in spec["peer_devices"] or (spec["device"],))
        binding, inputs, self.span = materialize(resolved["plan"], dtype=getattr(torch, spec["dtype"]),
                max_bytes=resolved["resources"]["max_materialize_bytes"], include_inputs=False)
        self.storage_reference = reference and spec["placement"] == "resident" and spec["dtype"] == "float16"
        if self.storage_reference:
            from ..qualification_weighted_precision import reference_model, widen
            binding = replace(binding, model=reference_model(binding.model))
        self.binding = binding
        from ..ownership import place_payloads
        owners = ["cpu"]*len(binding.graph.nodes) if reference else resolved["payload_placement"]["node_devices"]
        self.node_devices = tuple(torch.device(x) for x in owners)
        place_payloads(binding.graph, binding.model, self.node_devices)
        self.inputs = inputs
        self.placement = ExecutionPlacement(preset="native" if reference else spec["placement"], scoring_dtype=spec["scoring_dtype"])
        self.settle = None
        if resolved["graph_view"]["family"] == "settle":
            from ..settle import SettleGraph
            self.settle = SettleGraph(binding.graph, tuple(resolved["plan"]["graph"]["ranks"]),
                                      binding="ports", time_stride=resolved["plan"]["stride"])
        if reference:
            from ..reference import run
            view = place_model(binding.graph, binding.model, self.placement)
            self._make_execute = lambda trace: lambda *a, **kw: run(binding.graph, view, *a, trace=trace, **kw)
            self.scheduler = "python-independent-scalar-reference"
        else:
            from ..weighted.execution import eager_runner
            implementation = "python" if spec["client"] == "pytorch" else "native"
            self._make_execute = lambda trace: eager_runner(binding, implementation, self.placement,
                resolved["site"]["paths"].get("native_library"), streaming=spec["schedule"] == "streaming",
                workers=spec["workers"], packing=spec["packing"], query_execution=spec.get("query_execution", "legacy"),
                prefill_schedule=spec.get("prefill_schedule", "auto" if spec["schedule"] == "streaming" else "greedy"),
                trace=trace)
            self.scheduler = implementation+"-"+("streaming" if spec["schedule"] == "streaming" else
                "online-greedy" if spec.get("prefill_schedule", "greedy") == "greedy" else "component-window")
        self.execute = self._make_execute(self.trace_records)
        self.record_scheduler = self.scheduler
        from .column_task import ColumnInputs, reason
        from ..planning.materialize import materialize_inputs
        self.column_reason = reason(self)
        self.column_inputs = None
        if self.column_reason is None:
            self.column_inputs = ColumnInputs(resolved["plan"], getattr(torch, spec["dtype"]))
            self.scheduler = "settle-columns-v1"
        else:
            self.inputs = materialize_inputs(resolved["plan"], dtype=getattr(torch, spec["dtype"]))
            if self.storage_reference:
                self.inputs = widen(self.inputs)
        self.last_input_interface = "tensor-window-v1" if self.column_inputs is not None else "records"
        self.state = Continuation(binding.graph.identity, resolved["plan"]["samples"])
        self.task, self.task_manifest = create(resolved["task"], resolved["plan"])
        from .task_module import bind, inventory, manifest, validate_optimizer
        self.parameter_owner = bind(self.task, binding.model)
        self.optimizer = self.task.optimizer(self.parameter_owner, storage_reference=self.storage_reference) if self.training else None
        self.optimizer_layout = validate_optimizer(self.parameter_owner, self.optimizer)
        self.parameter_inventory = inventory(self.parameter_owner)
        self.module_manifest = manifest(self.task, self.parameter_owner)
        self.update_counts = dict.fromkeys(dict(self.parameter_owner.named_parameters()), 0)
        self.last_work = None
        self.core = None
        if not reference and spec["client"] == "python-native":
            from ..native_loader import load_native
            self.core = load_native(resolved["site"]["paths"]["native_library"], backend=self.device.type)
            if not hasattr(self.core, "communication_statistics"):
                raise ValueError("site.paths.native_library: rebuild this source for the public experiment transport contract")
        if checkpoint is not None:
            self.restore(checkpoint)

    def forward(self, index, *, complete=False):
        if self.column_inputs is not None and not complete:
            layout, ports, self.last_inputs = self.column_inputs.window(index, self.device, training=self.training)
            result = self.execute.run_ports(self.settle, layout, ports, state=self.state,
                stop_position=(index+1)*self.resolved["plan"]["positions"])
            self.last_input_interface, self.scheduler = "tensor-window-v1", result.execution["schedule"]
            return result
        self.last_input_interface, self.scheduler = "records", self.record_scheduler
        if self.inputs is None:
            from ..planning.materialize import materialize_inputs
            self.inputs = materialize_inputs(self.resolved["plan"], dtype=getattr(torch, self.spec["dtype"]))
        inputs = self.task.inputs(self.inputs, index, training=self.training)
        self.last_inputs = inputs
        inputs = self.place_inputs(inputs)
        if self.settle is not None:
            from ..settle_ports import validate_alignment
            validate_alignment(self.settle, self.state, inputs)
        execute = self.execute
        if complete and not self.trace_records:
            if self._complete_execute is None:
                self._complete_execute = self._make_execute(True)
            execute = self._complete_execute
        result = execute(self.state, inputs, (index+1)*self.span, sealed_until=(index+1)*self.span)
        return result

    def place_inputs(self, inputs):
        from ..weighted.placement_tensors import move
        copies, result = {}, []
        for x in inputs:
            device = self.node_devices[self.binding.graph.inputs[x.port]]
            key = id(x.value), device
            if key not in copies:
                copies[key] = move(x.value, device)
            result.append(replace(x, value=copies[key]))
        return result

    def step(self, index, *, return_mode=None):
        from ..communication import statistics
        self.verify_task(check_code=False)
        statistics(reset=True)
        if self.core is not None:
            self.core.communication_statistics(True)
        mode = return_mode or self.spec.get("return_mode", "complete")
        with torch.set_grad_enabled(self.training):
            if self.optimizer is not None:
                self.optimizer.zero_grad(set_to_none=True)
            result = self.forward(index, complete=mode == "complete")
            loss = self.task.loss(result, training=self.training)
            self.check_finite(dict(result=result, loss=loss), "runtime.forward_and_loss")
            if self.training:
                backward = getattr(self.optimizer, "backward", None)
                backward(loss) if backward else loss.backward()
                gradients = {n: p.grad for n, p in self.parameter_owner.named_parameters() if p.grad is not None}
                if not gradients:
                    raise ValueError("task.objective: no connected parameter gradients")
                self.check_finite(gradients, "runtime.gradients")
                for name, p in self.parameter_owner.named_parameters():
                    if p.grad is not None:
                        self.update_counts[name] += 1
                self.optimizer.step()
                self.check_finite(dict(self.parameter_owner.named_parameters()), "runtime.updated_parameters")
            self.check_finite(self.task.state_dict(), "runtime.task_state")
            if self.module_manifest is not None:
                self.check_finite((self.task.module.state_dict(),
                                None if self.optimizer is None else self.optimizer.state_dict()),
                                  "runtime.task_module_and_optimizer")
        self.verify_task(check_code=False)
        self.state = result.continuation.detach()
        self.last_work = work(result)
        from .locations import eager_locations
        from .column_task import WindowResult, output_records, locations
        columns = isinstance(result, WindowResult)
        self.last_work["actual_record_locations"] = locations(result) if columns else eager_locations(result)
        packing = "scalar" if self.reference else self.spec["return_packing"]
        if mode == "complete":
            actual = snapshot(result, packing=packing)
        elif mode == "outputs":
            outputs = output_records(result, self.settle) if columns else snapshot(result.outputs, packing=packing)
            actual = dict(return_mode=mode, cut=self.state.cut, outputs=outputs)
        elif mode == "metrics":
            actual = dict(return_mode=mode, cut=self.state.cut)
        else:
            raise ValueError("execution.return_mode: unknown return contract")
        value = float(loss.detach().cpu())
        for device in self.devices:
            synchronize(device)
        self.last_work["communication_observation"] = statistics()
        if self.core is not None:
            self.last_work["native_communication_observation"] = self.core.communication_statistics()
        return actual, value

    def _parameter_values(self):
        return {n: p.half() if self.storage_reference else p for n, p in self.parameter_owner.named_parameters()}

    def parameters(self):
        return snapshot(self._parameter_values())

    def training_state(self):
        if self.optimizer is None:
            return None
        half = self.spec["dtype"] == "float16"
        result = dict(gradients={n: p.grad.float() if half and p.grad is not None else p.grad
                                for n, p in self.parameter_owner.named_parameters()},
                      optimizer=self.optimizer.state_dict(), update_counts=self.update_counts)
        if self.module_manifest is not None:
            result.update(task=self.task.state_dict(), task_buffers=dict(self.task.module.named_buffers()))
        return snapshot(result)

    def decoded_state(self):
        value = dict(continuation=self.state, parameters=self._parameter_values(),
                     optimizer=None if self.optimizer is None else self.optimizer.state_dict(),
                     update_counts=self.update_counts, task=self.task.state_dict())
        if self.module_manifest is not None:
            value["task_module"] = self.task.module.state_dict()
        return snapshot(value)

    def task_state(self):
        if not self.resolved["task"].get("plugin"):
            return None
        return snapshot(dict(module=None if self.module_manifest is None else self.task.module.state_dict(), data=self.task.state_dict()))

    def verify_task(self, *, check_code=True):
        from .task_identity import verify
        from .task_module import inventory, validate_optimizer
        if check_code:
            verify(self.resolved["task"], self.task_manifest)
        if not self.resolved["task"].get("plugin"):
            return
        if inventory(self.parameter_owner) != self.parameter_inventory:
            raise ValueError("task.module: graph/task parameter ownership changed during the run")
        if validate_optimizer(self.parameter_owner, self.optimizer) != self.optimizer_layout:
            raise ValueError("task.optimizer: parameter ownership/order changed during the run")

    def save(self, directory):
        from ..checkpoint import save
        from ..checkpoint_io import publish
        self.verify_task()
        save(directory/"state.pt", self.binding.graph, self.parameter_owner, self.state, self.optimizer)
        publish(directory/"task.pt", dict(update_counts=self.update_counts, task=self.task.state_dict(),
                                         task_manifest=self.task_manifest, module_schema=None if self.module_manifest is None else self.module_manifest["schema"]))

    def restore(self, directory):
        from ..checkpoint import load
        extra = torch.load(directory/"task.pt", weights_only=True, map_location="cpu")
        if extra.get("task_manifest") != self.task_manifest:
            raise ValueError("resume: task source/configuration identity differs")
        expected = None if self.module_manifest is None else self.module_manifest["schema"]
        if extra.get("module_schema") != expected:
            raise ValueError("resume: task module registry differs")
        if extra["update_counts"].keys() != self.update_counts.keys():
            raise ValueError("checkpoint update owner inventory differs")
        self.state = load(directory/"state.pt", self.binding.graph, self.parameter_owner, self.optimizer)
        self.update_counts = extra["update_counts"]
        self.task.load_state_dict(extra["task"])
        self.check_finite(self.task.state_dict())
        self.verify_task()

    def manifest(self):
        from ..ownership import region_reference
        model, graph = self.binding.model, self.binding.graph
        return dict(client="independent-reference" if self.reference else self.spec["client"],
            finite_check_packing="scalar" if self.reference else self.spec.get("finite_check_packing", "scalar"),
            scheduler=self.scheduler, graph_view=self.resolved["graph_view"]["family"],
            requested_placement=self.spec["placement"], effective_placement={k: str(v) for k, v in self.placement.resolve(self.device).items()},
            node_devices=[str(x) for x in self.node_devices],
            parameter_devices={n: str(p.device) for n, p in model.named_parameters()},
            region_devices=[str(region_reference(graph, model, r).device) for r in range(len(graph.regions))],
            return_mode=self.spec.get("return_mode", "complete"), return_packing=self.spec["return_packing"],
            internal_records="complete" if self.trace_records else "boundary-only; output-observing task",
            input_boundary="fresh CPU columns, shared source rows, differentiable block upload" if self.column_inputs is not None else
                "fresh CPU task tensors, shared source roots, differentiable upload per window",
            effective_input_interface=self.last_input_interface,
            input_fallback_reason=self.column_reason or ("explicit complete observations" if self.last_input_interface == "records" else None),
            operator="aten", packing=self.spec["packing"], task=self.task_manifest, task_module=self.module_manifest,
            communication=self.spec.get("communication", "ordinary"), gradient_boundary="detach per optimizer window")

    def close(self):
        self._make_execute = self._complete_execute = None
        self.execute = self.binding = self.inputs = self.state = self.optimizer = self.last_inputs = self.task = self.parameter_owner = self.column_inputs = None

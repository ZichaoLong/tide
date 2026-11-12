"""Experiment adapter for the independent Model-bound resident block owner."""
import torch
from ..runtime import synchronize
from ..settle import SettleGraph
from .block_inputs import BlockInputs, output_loss
from .block_checkpoint import record as checkpoint_record
from . import block_records
from .returns import snapshot
from .finite import finite_tensors
from .tasks import create
from .training_layout import SGDLayout


class BlockResident:
    def __init__(self, resolved, *, checkpoint=None):
        from ..planning.materialize import materialize
        from ..weighted import WeightedBlockResidentTraining
        self.resolved, self.spec = resolved, resolved["execution"]["effective"]
        self.training = resolved["measurement"]["mode"] == "training"
        self.devices = (torch.device(self.spec["device"]),)
        if resolved["graph_view"]["family"] != "settle" or resolved["task"].get("plugin"):
            raise ValueError("block experiment requires its actual Settle certificate and built-in task")
        self.binding, _, self.span = materialize(resolved["plan"], dtype=torch.float32,
            max_bytes=resolved["resources"]["max_materialize_bytes"], include_inputs=False)
        self.settle = SettleGraph(self.binding.graph, tuple(resolved["plan"]["graph"]["ranks"]),
                                 binding="ports", time_stride=resolved["plan"]["stride"])
        self.inputs = BlockInputs(resolved["plan"])
        self.task, self.task_manifest = create(resolved["task"], resolved["plan"])
        options = {k: v for k, v in resolved["task"]["optimizer"].items() if k != "kind"}
        self.layout = SGDLayout(self.binding.model, **options)
        self.last_parameter_gradients = self.last_work = self.last_backward_statistics = None
        self.last_input_interface = "resident-block-B×L×P-v2"
        kwargs = dict(device=self.spec["device"],
            **{k: resolved["site"]["paths"][k] for k in ("native_library", "resident_library")},
            batch_size=resolved["plan"]["samples"], limits=dict(resolved["resources"]["resident_block_limits"]),
            node_devices=resolved["payload_placement"]["node_devices"])
        with torch.no_grad():
            if checkpoint is not None:
                extra = torch.load(checkpoint/"task.pt", map_location="cpu", weights_only=True)
                if extra.get("task_manifest") != self.task_manifest:
                    raise ValueError("resume: task source/configuration identity differs")
                self.task.load_state_dict(extra["task"])
                kwargs["checkpoint"] = checkpoint/"state.pt"
            else:
                from ..native_loader import load_native
                from ..native_model import encode_model
                core = load_native(kwargs["native_library"], backend=self.devices[0].type)
                _, model = encode_model(core, self.binding.graph, self.binding.model)
                names = [p.canonical for p in model.parameters().owners()]
                kwargs.update(optimizer="sgd", groups=[dict(parameters=names, **options)])
            # An explicit selected owner is constructed once. Device, capacity
            # or arithmetic failures are not reasons to try another executor.
            self.owner = WeightedBlockResidentTraining(self.settle, self.binding.model, **kwargs)

    def advance(self, index):
        inputs = self.inputs.window(index)
        return self.owner.advance(*inputs), inputs

    def step(self, index, *, return_mode=None):
        mode = return_mode or self.spec.get("return_mode", "complete")
        if mode not in {"complete", "outputs", "metrics"}:
            raise ValueError("execution.return_mode: unknown return contract")
        with torch.no_grad():
            window, inputs = self.advance(index)
            loss, bars = output_loss(window.fields, self.task.config["objective"]["scale"], training=self.training)
            finite_tensors((loss, bars))
            self.last_work = block_records.work(window, input_positions=self.inputs.input_positions,
                                                input_atoms=self.inputs.input_atoms)
            values = block_records.window_snapshot(window, complete=mode == "complete") if mode != "metrics" else None
            if self.training:
                if self.last_work["outputs"] == 0:
                    raise ValueError("task.objective: window produced no differentiable output")
                gradients = self.owner.backward([self.owner.cotangents(window, bars)])
                self.last_parameter_gradients = gradients.parameters
                finite_tensors(gradients.parameters.values)
                self.last_backward_statistics = dict(parameter_owners=len(gradients.parameters.names),
                    retained_windows=len(gradients.inputs), scope="public block field VJP; no old event counters")
                self.owner.step()
            else:
                self.owner.detach()
            if mode == "complete":
                actual = block_records.complete(values, self.settle, inputs, self.owner.continuation(),
                                                self.last_work["actual_stats"])
                finite_tensors(actual)
            elif mode == "outputs":
                actual = dict(return_mode=mode, cut=self.owner.cut, outputs=block_records.output_records(values, self.settle))
            else:
                actual = dict(return_mode=mode, cut=self.owner.cut)
            finite_tensors(self.task.state_dict())
            value = float(loss.detach().cpu())
            synchronize(self.devices[0])
            return actual, value

    def parameters(self):
        with torch.no_grad():
            return snapshot(self.layout.parameters(checkpoint_record(self.owner)))

    def training_state(self):
        with torch.no_grad():
            return None if not self.training else self.layout.decode(self.last_parameter_gradients, checkpoint_record(self.owner))

    def decoded_state(self):
        with torch.no_grad():
            # checkpoint_record copies every raw padded block and optimizer
            # field; Continuation additionally checks the public exact decoder.
            return dict(checkpoint=checkpoint_record(self.owner), continuation=snapshot(self.owner.continuation()),
                        task=snapshot(self.task.state_dict()))

    def task_state(self):
        return None

    def save(self, directory):
        from ..checkpoint_io import publish
        from .task_identity import verify
        verify(self.resolved["task"], self.task_manifest)
        with torch.no_grad():
            self.owner.save(directory/"state.pt", max_bytes=self.resolved["resources"]["checkpoint_max_bytes"])
            publish(directory/"task.pt", dict(task=self.task.state_dict(), task_manifest=self.task_manifest))

    def manifest(self):
        with torch.no_grad():
            return dict(self.owner.manifest(), client=self.spec["client"], task=self.task_manifest,
                return_mode=self.spec.get("return_mode", "complete"), input_interface=self.last_input_interface,
                requested_limits=dict(self.resolved["resident_dispatch"]["requested_limits"]),
                resident_dispatch=self.resolved["resident_dispatch"],
                last_backward_statistics=self.last_backward_statistics)

    def close(self):
        with torch.no_grad():
            if self.owner is not None:
                self.owner.close()
            self.owner = self.inputs = self.layout = self.last_parameter_gradients = self.task = None

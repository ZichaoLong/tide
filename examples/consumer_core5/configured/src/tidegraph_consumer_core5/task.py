"""Task-owned input RNG, learnable head and loss; no reference event prepass.

The public runner owns graph execution and checkpoint publication. The explicit
``module`` participates in its parameter/optimizer registry. This task's state
owns data progress; module weights and unified optimizer slots are saved by the
runner rather than duplicated in the data-state record.
"""
from collections import Counter
from dataclasses import replace
import math
import torch
from tidegraph.transfer import copy_rows

__version__ = "0.1.0"
STATE_SCHEMA = "consumer-core5-regression-state-v1"


def _options(options, seed):
    result = dict(data_seed=seed+1, head_seed=seed+2, input_scale=.2,
                  target_scale=.05, loss_scale=.001)
    unknown = set(options)-result.keys()
    if unknown:
        raise ValueError("task.plugin.options."+sorted(unknown)[0]+": unknown regression option")
    result.update(options)
    for key in ("data_seed", "head_seed"):
        if type(result[key]) is not int or not 0 <= result[key] < 2**63:
            raise ValueError(f"task.plugin.options.{key}: expected an integer in [0, 2**63)")
    for key in ("input_scale", "target_scale", "loss_scale"):
        value = result[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"task.plugin.options.{key}: expected a positive finite number")
    return result


class RegressionTask:
    """FP32/FP64 eager example, independent of CUDA/NPU device numbering.

    Data are generated once per distinct external root on CPU. Physical input
    ports can share a source root; this relation is preserved exactly. Original
    logical time is advanced by the declared window span, never an activity count.
    """
    def __init__(self, task, plan, options):
        self.config, self.plan = task, plan
        self.options = _options(options, task["seed"])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.options["head_seed"])
            self.module = torch.nn.Linear(plan["width"], 1, dtype=torch.float32)
        self.data_rng = torch.Generator(device="cpu").manual_seed(self.options["data_seed"])
        self.next_window = 0

    def inputs(self, templates, index, *, device=None, training=False):
        if type(index) is not int or index != self.next_window:
            raise ValueError(f"task input progress: expected window {self.next_window}, got {index}")
        counts = Counter((x.batch, x.port) for x in templates)
        values = {}
        for x in templates:
            if id(x.value) in values:
                continue
            if x.value.ndim != 1 or x.value.numel() != self.plan["width"]+1:
                raise ValueError("task input: expected the public weighted payload [mass, content]")
            if x.value.dtype not in (torch.float32, torch.float64):
                raise ValueError("task input: this regression recipe declares FP32/FP64; FP16 needs its own recipe and checks")
            content = torch.randn(self.plan["width"], generator=self.data_rng)*self.options["input_scale"]
            mass = x.value.detach().cpu()[:1].to(content.dtype)
            payload = torch.cat((mass, content)).to(dtype=x.value.dtype)
            values[id(x.value)] = payload.to("cpu" if device is None else device).requires_grad_(training)
        self.next_window += 1
        return [replace(x, time=x.time+index*self.plan["span"],
                        position=x.position+index*counts[x.batch, x.port],
                        value=values[id(x.value)]) for x in templates]

    def loss(self, result, *, training):
        if not result.outputs:
            raise ValueError("task.loss: this regression window requires at least one actual output")
        self.module.train(training)
        parameter = next(self.module.parameters())
        if parameter.dtype not in (torch.float32, torch.float64):
            raise ValueError("task.module: this recipe declares FP32/FP64")
        features, targets = [], []
        for batch, time, port, payload in result.outputs:
            # Public weighted numerical content is mass*content, exactly once.
            row = payload[0]*payload[1:]
            features.append(copy_rows([row], parameter.device)[0])
            targets.append(self.options["target_scale"]*math.sin(.3*batch+.01*time+.07*port))
        predictions = self.module(torch.stack(features))
        target = torch.tensor(targets, device=parameter.device, dtype=predictions.dtype).unsqueeze(1)
        return (predictions-target).square().mean()*self.options["loss_scale"]

    def optimizer(self, model, *, storage_reference=False):
        if storage_reference:
            raise ValueError("task.optimizer: this eager regression recipe has no resident FP16 storage-reference contract")
        unique = {}
        for parameter in (*model.parameters(), *self.module.parameters()):
            if parameter.requires_grad:
                unique.setdefault(id(parameter), parameter)
        if any(p.dtype not in (torch.float32, torch.float64) for p in unique.values()):
            raise ValueError("task.optimizer: this recipe requires FP32/FP64 graph and task parameters")
        options = {k: v for k, v in self.config["optimizer"].items() if k != "kind"}
        if self.config["optimizer"]["kind"] != "sgd":
            raise ValueError("task.optimizer.kind: this regression recipe uses explicit SGD")
        return torch.optim.SGD(list(unique.values()), **options)

    def state_dict(self):
        return dict(schema=STATE_SCHEMA, next_window=self.next_window,
                    data_rng=self.data_rng.get_state().clone())

    def load_state_dict(self, state):
        if not isinstance(state, dict) or set(state) != {"schema", "next_window", "data_rng"}:
            raise ValueError("task checkpoint: expected complete regression data state")
        if state["schema"] != STATE_SCHEMA or type(state["next_window"]) is not int or state["next_window"] < 0:
            raise ValueError("task checkpoint: incompatible schema or input progress")
        rng = state["data_rng"]
        if not isinstance(rng, torch.Tensor) or rng.dtype != torch.uint8 or rng.ndim != 1:
            raise ValueError("task checkpoint: invalid CPU generator state")
        # Validate in a fresh generator before mutating the live data owner.
        restored = torch.Generator(device="cpu")
        restored.set_state(rng.cpu())
        self.data_rng, self.next_window = restored, state["next_window"]


def create_task(*, task, plan, options):
    return RegressionTask(task, plan, options)

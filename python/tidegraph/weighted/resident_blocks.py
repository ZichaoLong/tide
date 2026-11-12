"""Model-bound B×L resident windows with explicit tensor roots and raw archives.

The Settle certificate is mandatory. Neither capability selection nor Model
lowering executes a reference trajectory. See docs/resident-block-session.md.
"""
from copy import deepcopy
from pathlib import Path
import torch
from ..clocks import StateClock
from ..coordinates import integers
from ..native_loader import load_native
from ..native_model import encode_model
from ..native_records import to_continuation, from_continuation
from ..records import Continuation
from ..resident import parameter_identity
from ..resident_loader import load_resident
from ..resident_training_options import optimizer_groups, GROUP_FIELDS
from ..settle import SettleGraph
from .binding import WeightedModel
from .config import KVConfig


def block_resident_capability(spec, model, *, device=None, node_devices=None):
    """Static dispatch reason. Capacity/numerical/device failures never fallback."""
    if not isinstance(spec, SettleGraph) or spec.binding != "ports":
        return False, "requires an explicit ports SettleGraph certificate"
    if not isinstance(model, WeightedModel):
        return False, "requires installed weighted local modules"
    graph = spec.graph
    if graph.origins or any(n.clear or n.identity or n.next_state != "adopt-v1"
                           or n.state_clock != StateClock() or n.emit_period != 1 or n.emit_phases
                           for n in graph.nodes):
        return False, "requires retained logical-time state and every physical broadcast"
    if any(not r.observe_all for r in graph.regions):
        return False, "requires complete regional observations"
    if any(x.dtype != torch.float32 or x.device.type != "cpu" for x in model.state_dict().values()):
        return False, "requires CPU FP32 boundary parameters"
    if node_devices is not None and (device is None or any(str(x) != str(device) for x in node_devices)):
        return False, "the block owner currently binds one accelerator"
    for r, region in enumerate(model.regions):
        if region.config.support_policy != "strict":
            return False, "requires strict mathematical support policy"
        nodes = [model.nodes[v] for v, n in enumerate(graph.nodes) if n.region == r]
        if any(not isinstance(n.kernel.config, KVConfig) for n in nodes):
            return False, "requires KV sum/query sequence capability"
        if any(n.kernel.config != nodes[0].kernel.config or n.aggregate_program.alpha != nodes[0].aggregate_program.alpha for n in nodes):
            return False, "requires compatible KV geometry and Aggregate alpha within each region"
        factors = {hasattr(t, "a_left") for n in nodes for t in n.aggregate_program.transforms.values()}
        if not factors and nodes[0].aggregate_program.alpha:
            return False, "nonzero Aggregate needs a nonempty source parameter group"
        if len(factors) > 1:
            return False, "source factor presence needs separate compatible groups"
    return True, "whole-region B×L device choose/commit"


class WeightedBlockResidentTraining:
    def __init__(self, spec, model, *, device, native_library, resident_library,
                 batch_size=1, continuation=None, optimizer=None, groups=None, limits=None,
                 checkpoint=None, node_devices=None):
        integers("block sample count", batch_size)
        if batch_size < 1 or torch.is_grad_enabled():
            raise ValueError("block resident requires positive batch size and torch.no_grad()")
        if checkpoint is not None and any(x is not None for x in (continuation, optimizer, groups)):
            raise ValueError("block restore is exclusive with continuation/optimizer/groups")
        supported, reason = block_resident_capability(spec, model, device=device, node_devices=node_devices)
        if not supported:
            raise ValueError("block resident capability: " + reason)
        if str(device).split(":", 1)[0] == "npu":
            import torch_npu
        self.device = torch.device(device)
        if self.device.type not in {"npu", "cuda"} or self.device.index is None:
            raise ValueError("block resident needs an explicit indexed accelerator")
        self.spec, self.graph, self.model, self.batch_size = spec, spec.graph, model, batch_size
        model.validate_binding(self.graph)
        self.core = load_native(native_library, backend=self.device.type)
        self.module, self.build_record = load_resident(resident_library, self.core, component="weighted")
        if not hasattr(self.module, "BlockResidentSession") or "libtide-block-resident.so" not in self.build_record["binary_sha256"]:
            raise ValueError("resident plugin lacks the public block owner; rebuild matching sources")
        self.compiled, weights = encode_model(self.core, self.graph, model)
        certificate = self.core.SettleGraph(self.compiled, list(spec.ranks), spec.binding, spec.stride)
        q = Continuation(self.graph.identity, batch_size) if continuation is None else continuation
        if q.batch_size != batch_size:
            raise ValueError("block continuation sample count differs")
        initial = to_continuation(self.core, self.graph, self.compiled, q)
        self.identity = parameter_identity(model)
        self.trainable = tuple((name, x.requires_grad) for name, x in model.state_dict(keep_vars=True).items())
        self.requested_limits, self.requested_groups = dict(limits or {}), deepcopy(groups)
        self.requested_optimizer, self.initialization = optimizer, "resume" if checkpoint is not None else "fresh"
        options = self.module.BlockResidentLimits()
        names = ("cache", "forward_bytes", "backward_bytes", "optimizer_bytes", "total_bytes", "workspace_bytes", "windows", "timeout_ms")
        for name, value in self.requested_limits.items():
            if name not in names:
                raise ValueError("unknown block capacity: " + name)
            integers("block " + name, value)
            if value < 1:
                raise ValueError("block capacities must be positive")
            setattr(options, name, value)
        self.effective_limits = {name: getattr(options, name) for name in names}
        if checkpoint is not None:
            if isinstance(checkpoint, (str, Path)):
                checkpoint = self.module.load_block_checkpoint(str(checkpoint))
            self.owner = self.module.BlockResidentSession(certificate, weights, initial, checkpoint, self.device, options)
        else:
            optimizer = "sgd" if optimizer is None else optimizer
            if optimizer not in {"sgd", "adamw"}:
                raise ValueError("block optimizer must be sgd or adamw")
            self.owner = self.module.BlockResidentSession(certificate, weights, initial, self.device,
                getattr(self.module.OptimizerKind, optimizer), optimizer_groups(self.core, groups), options)

    def _check(self):
        if torch.is_grad_enabled():
            raise ValueError("block resident methods require torch.no_grad()")
        if (parameter_identity(self.model) != self.identity
                or tuple((n, x.requires_grad) for n, x in self.model.state_dict(keep_vars=True).items()) != self.trainable):
            raise RuntimeError("caller parameters changed after block owner construction")

    @property
    def cut(self):
        self._check(); return self.owner.cut

    @property
    def generation(self):
        self._check(); return self.owner.generation

    def advance(self, positions, payload, present):
        """int64 [B,L], FP32 [B,L,P,d+1] (c,G), bool [B,L,P].

        Missing ports have an explicit false bit. Every sample has the same
        last position, so the returned owner has one exact complete cut.
        """
        self._check(); return self.owner.advance(positions, payload, present)

    def cotangents(self, window, fields=None):
        self._check(); out = self.module.BlockRoots(); out.token = window.token
        result = {}
        for name, supplied in (fields or {}).items():
            if name not in window.fields:
                raise ValueError("unknown block root field: " + name)
            field = window.fields[name]
            values, flags = supplied if isinstance(supplied, tuple) else (supplied, field.present)
            for value, expected, dtype in ((values, field.values, torch.float32), (flags, field.present, torch.bool)):
                if (not isinstance(value, torch.Tensor) or value.shape != expected.shape
                        or value.device != expected.device or value.dtype != dtype or value.requires_grad):
                    raise ValueError("block cotangent shape/device/dtype/gradient differs: " + name)
            root = self.module.BlockRoot(); root.values, root.connected = values.contiguous(), flags.contiguous()
            result[name] = root
        out.fields = result
        return out

    def backward(self, roots):
        self._check(); return self.owner.backward(list(roots))

    def step(self):
        self._check(); return self.owner.step()

    def detach(self):
        self._check(); self.owner.detach()

    def continuation(self):
        self._check(); return from_continuation(self.graph, self.owner.continuation())

    def checkpoint(self):
        self._check(); return self.owner.checkpoint()

    def save(self, path, *, max_bytes=1 << 32):
        self._check(); integers("block archive budget", max_bytes)
        self.module.save_block_checkpoint(self.owner.checkpoint(), str(path), max_bytes)

    def manifest(self):
        self._check()
        return dict(execution="resident", scheduler="settle-block-v2", graph_certificate="SettleGraph/ports",
                    device=str(self.device), payload_dtype="float32", adjoint_dtype="float32", metadata_dtype="int64",
                    initial_state="actual raw", requested_limits=self.requested_limits, effective_limits=self.effective_limits,
                    initialization=self.initialization, requested_optimizer=self.requested_optimizer,
                    requested_groups=deepcopy(self.requested_groups), effective_groups=[{n: getattr(g, n) for n in GROUP_FIELDS} for g in self.owner.optimizer_groups()],
                    build=self.build_record)

    def close(self):
        self.owner.close()

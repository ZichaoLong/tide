"""Explicit two-clock application value checkpoints, independent of single-graph v5."""
from dataclasses import dataclass, field
import copy
import math
import torch
from .records import Continuation
from .history import int64
from .validation import validate_window
from .checkpoint_io import publish
from .checkpoint_values import encode, decode, validate_weights
from .checkpoint_ownership import parameter_aliases, optimizer_record, preflight_optimizer

SCHEMA = "tide-token-application-v1"


@dataclass
class TokenState:
    cut: int
    body: Continuation
    readout: Continuation
    buffer: list = field(default_factory=list)

    def detach(self):
        return TokenState(self.cut, self.body.detach(), self.readout.detach(),
                          [(b, t, p, x.detach()) for b, t, p, x in self.buffer])


class TokenApplication:
    """Named owners and a declared L-body-ticks/one-readout-phase boundary.

    This object validates and checkpoints state; it does not own a scheduler,
    external data cursor, RNG or generation policy. Reconstruct the same model
    aliases and optimizer order before loading.
    """
    def __init__(self, body_graph, body_model, readout_graph, readout_model, layers, *, output_port=0,
                 mode="hard", zeta=1.0):
        if not int64(layers) or not 1 <= layers < 2**63-1:
            raise ValueError("invalid token application layer count")
        if not int64(output_port) or not 0 <= output_port < len(body_graph.outputs):
            raise ValueError("invalid token application output port")
        if len(readout_graph.inputs) != layers:
            raise ValueError("token application requires one readout input port per phase")
        if mode not in {"hard", "softp", "hst"} or type(zeta) not in (int, float) or not math.isfinite(zeta):
            raise ValueError("invalid token application Emit policy")
        a, b = body_model.nodes[0].bias, readout_model.nodes[0].bias
        if (body_model.width != readout_model.width or a.dtype != b.dtype
                or a.device.type != "cpu" or b.device.type != "cpu"):
            raise ValueError("token application boundary requires matching CPU width and dtype")
        self.graphs = {"body": body_graph, "readout": readout_graph}
        self.models = torch.nn.ModuleDict({"body": body_model, "readout": readout_model})
        self.layers, self.output_port = layers, output_port
        self.mode, self.zeta = mode, float(zeta)

    @property
    def identity(self):
        return {"body": self.graphs["body"].identity, "readout": self.graphs["readout"].identity,
                "layers": self.layers, "output_port": self.output_port, "mode": self.mode, "zeta": self.zeta}

    def validate(self, state, *, models=None):
        if not isinstance(state, TokenState) or not int64(state.cut) or state.cut < 0:
            raise ValueError("invalid token application cut")
        models = self.models if models is None else models
        token, phase = divmod(state.cut, self.layers+1)
        expected = {"body": token*self.layers+min(phase, self.layers), "readout": token}
        for name in ("body", "readout"):
            q = getattr(state, name)
            if (not isinstance(q, Continuation) or not int64(q.cut) or q.cut != expected[name]
                    or not int64(q.batch_size) or q.batch_size < 1):
                raise ValueError("token application component clock/batch mismatch")
            validate_window(self.graphs[name], models[name], q, [], q.cut, q.cut)
        if state.body.batch_size != state.readout.batch_size or not isinstance(state.buffer, list):
            raise ValueError("token application batch/buffer mismatch")
        seen = set(); reference = models["body"].nodes[0].bias
        for row in state.buffer:
            if not isinstance(row, (tuple, list)) or len(row) != 4:
                raise ValueError("invalid token application buffer row")
            batch, time, port, value = row
            if (any(not int64(v) for v in (batch, time, port)) or port != self.output_port
                    or not 0 <= batch < state.body.batch_size
                    or not token*self.layers <= time < state.body.cut or (batch, time, port) in seen):
                raise ValueError("invalid token application buffer coordinate")
            seen.add((batch, time, port))
            if (not isinstance(value, torch.Tensor) or value.shape != (models["body"].width,)
                    or value.dtype != reference.dtype or value.device.type != "cpu" or not torch.isfinite(value).all()):
                raise ValueError("invalid token application buffer tensor")

    def save(self, path, state, optimizer=None):
        self.validate(state)
        aliases = parameter_aliases(self.models); weights = self.models.state_dict()
        validate_weights(self.models, weights, aliases)
        layout, slots = optimizer_record(self.models, optimizer)
        publish(path, {"schema": SCHEMA, "identity": self.identity, "aliases": aliases, "weights": weights,
                       "optimizer_layout": layout, "optimizer": slots,
                       "state": {"cut": state.cut, "body": encode(state.body), "readout": encode(state.readout),
                                 "buffer": [(b, t, p, x.detach()) for b, t, p, x in state.buffer]}})

    def load(self, path, optimizer=None):
        record = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(record, dict) or record.get("schema") != SCHEMA or record.get("identity") != self.identity:
            raise ValueError("token application schema/identity mismatch")
        validate_weights(self.models, record["weights"], record["aliases"])
        try:
            payload = record["state"]
            state = TokenState(payload["cut"], decode(payload["body"]), decode(payload["readout"]), payload["buffer"])
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError("malformed token application state") from error
        # Validate every component under saved weights before any live owner changes.
        staged = copy.deepcopy(self.models)
        staged.load_state_dict(record["weights"])
        self.validate(state, models=staged)
        preflight_optimizer(self.models, optimizer, record["optimizer_layout"], record["optimizer"])
        self.models.load_state_dict(record["weights"])
        if optimizer is not None:
            optimizer.load_state_dict(record["optimizer"])
        return state

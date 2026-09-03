"""Reference neural operations for the SettleGraph semantic contract.

This module intentionally contains ordinary eager Torch implementations.  The
executors own ordering and routing; the classes here own only the local
mathematics declared by a normalized :class:`tide.plan.Plan`.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class OperationConfigurationError(ValueError):
    """A Plan names an unsupported or internally inconsistent operation."""


@dataclasses.dataclass(frozen=True)
class AttentionState:
    """Canonical bounded attention history for one receiver and sequence.

    ``positions`` records the global Token position of every observed key/value
    row.  It is semantic state rather than an implementation detail: physical
    ring buffers must normalize to this ordered representation before state,
    trace, or checkpoint comparison.
    """

    positions: Tensor
    keys: Tensor
    values: Tensor

    @property
    def length(self) -> int:
        return int(self.positions.shape[0])


ReceiverState = Optional[Union[Tensor, AttentionState]]


def _config_type(config: Mapping[str, Any], default: str) -> str:
    value = config.get("type", default)
    if not isinstance(value, str):
        raise OperationConfigurationError("operation type must be a string")
    return value.lower().replace("-", "_")


class RMSNorm(nn.Module):
    """Small backend-neutral RMSNorm with an explicit accumulation policy."""

    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = float(eps)

    def forward(self, value: Tensor) -> Tensor:
        # Accumulate FP16/BF16 norms in FP32, then restore the declared dtype.
        source_dtype = value.dtype
        work = value.float() if value.dtype in {torch.float16, torch.bfloat16} else value
        variance = work.square().mean(dim=-1, keepdim=True)
        normalized = work * torch.rsqrt(variance + self.eps)
        return normalized.to(source_dtype) * self.weight.to(source_dtype)


def state_tensor_summary(state: ReceiverState, reference: Tensor) -> Tensor:
    """Return a fixed scalar summary without exposing a variable state shape."""

    if state is None:
        return reference.new_zeros(())
    if isinstance(state, AttentionState):
        if state.length == 0:
            return reference.new_zeros(())
        return torch.cat((state.keys.reshape(-1), state.values.reshape(-1))).square().mean().sqrt()
    if state.numel() == 0:
        return reference.new_zeros(())
    return state.reshape(-1).square().mean().sqrt()


class ReceiverModule(nn.Module):
    """One receiver's Update, two Read operations, and NodeCompute."""

    def __init__(self, d_model: int, spec: Any) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.spec = spec
        self.aggregate_config = dict(spec.aggregate)
        self.update_config = dict(spec.update)
        self.selector_read_config = dict(spec.selector_read)
        self.ffn_read_config = dict(spec.ffn_read)
        self.compute_config = dict(spec.node_compute)
        self.emit_config = dict(spec.emit)
        self.input_norm_config = dict(
            getattr(spec, "input_norm", {"type": "rmsnorm", "eps": 1e-6})
        )
        self.ffn_norm_config = dict(
            getattr(spec, "ffn_norm", {"type": "rmsnorm", "eps": 1e-6})
        )
        if _config_type(self.input_norm_config, "rmsnorm") != "rmsnorm":
            raise OperationConfigurationError("only RMS input normalization is supported")
        if _config_type(self.ffn_norm_config, "rmsnorm") != "rmsnorm":
            raise OperationConfigurationError("only RMS FFN normalization is supported")

        self.input_norm = RMSNorm(
            d_model, float(self.input_norm_config.get("eps", 1e-6))
        )
        self.ffn_norm = RMSNorm(
            d_model, float(self.ffn_norm_config.get("eps", 1e-6))
        )

        update_type = _config_type(self.update_config, "none")
        self.update_type = update_type
        if update_type == "none":
            self.state_shape: Tuple[int, ...] = ()
        elif update_type == "ema":
            state_dim = int(self.update_config.get("state_dim", _first_dim(spec.state_shape)))
            if state_dim <= 0:
                raise OperationConfigurationError("EMA state_dim must be positive")
            self.state_shape = (state_dim,)
            self.ema_observe = nn.Linear(d_model, state_dim)
            decay = float(self.update_config.get("decay", 0.9))
            if not 0.0 <= decay < 1.0:
                raise OperationConfigurationError("EMA decay must satisfy 0 <= decay < 1")
            if self.update_config.get("learnable_decay", False):
                # Store an unconstrained logit while preserving the initial value.
                clipped = min(max(decay, 1e-6), 1.0 - 1e-6)
                self.ema_decay_logit = nn.Parameter(torch.tensor(math.log(clipped / (1.0 - clipped))))
                self.register_buffer("ema_decay", torch.empty(0), persistent=False)
            else:
                self.register_buffer("ema_decay", torch.tensor(decay))
                self.ema_decay_logit = None
        elif update_type == "gdn":
            key_dim = int(self.update_config.get("key_dim", 0))
            value_dim = int(self.update_config.get("value_dim", 0))
            if key_dim <= 0 or value_dim <= 0:
                raise OperationConfigurationError("GDN key_dim and value_dim must be positive")
            self.state_shape = (key_dim, value_dim)
            self.gdn_key = nn.Linear(d_model, key_dim, bias=False)
            self.gdn_value = nn.Linear(d_model, value_dim, bias=False)
            self.gdn_eta = nn.Linear(d_model, 1)
            self.gdn_gamma = nn.Linear(d_model, 1)
            self.gdn_query = nn.Linear(d_model, key_dim, bias=False)
            self.gdn_out = nn.Linear(value_dim, d_model, bias=False)
            self.gdn_beta = nn.Parameter(torch.zeros(()))
            self.state_norm_eps = float(self.update_config.get("norm_eps", 1e-12))
        elif update_type == "attention_window":
            key_dim = int(self.update_config.get("key_dim", 0))
            value_dim = int(self.update_config.get("value_dim", 0))
            window = int(self.update_config.get("window", 0))
            if min(key_dim, value_dim, window) <= 0:
                raise OperationConfigurationError(
                    "attention_window key_dim, value_dim, and window must be positive"
                )
            self.state_shape = (window, key_dim, value_dim)
            self.attn_key_dim = key_dim
            self.attn_value_dim = value_dim
            self.attn_window = window
            self.attn_key = nn.Linear(d_model, key_dim, bias=False)
            self.attn_value = nn.Linear(d_model, value_dim, bias=False)
            self.attn_query = nn.Linear(d_model, key_dim, bias=False)
            self.attn_out = nn.Linear(value_dim, d_model, bias=False)
            self.state_norm_eps = float(self.update_config.get("norm_eps", 1e-12))
        else:
            raise OperationConfigurationError(f"unsupported Update type: {update_type}")

        read_type = _config_type(self.selector_read_config, "content_norm")
        self.selector_read_type = read_type
        read_dim = int(self.selector_read_config.get("out_dim", _first_dim(spec.selector_read_shape, 1)))
        if read_dim <= 0:
            raise OperationConfigurationError("selector read out_dim must be positive")
        self.selector_read_dim = read_dim
        if read_type == "content":
            if read_dim != d_model:
                raise OperationConfigurationError(
                    "content selector read must have out_dim=d_model"
                )
            self.selector_read_linear = None
        elif read_type == "content_norm":
            if read_dim != 1:
                raise OperationConfigurationError("content_norm selector read has out_dim=1")
            self.selector_read_linear = None
        elif read_type == "content_linear":
            self.selector_read_linear = nn.Linear(d_model, read_dim)
        elif read_type == "content_state_linear":
            if update_type == "attention_window":
                raise OperationConfigurationError(
                    "content_state_linear requires fixed-shape Tensor state; "
                    "use content_state_summary_linear for window Attention"
                )
            state_width = math.prod(self.state_shape)
            if state_width <= 0:
                raise OperationConfigurationError(
                    "content_state_linear requires a non-empty receiver state"
                )
            self.selector_read_linear = nn.Linear(d_model + state_width, read_dim)
        elif read_type == "content_state_summary_linear":
            self.selector_read_linear = nn.Linear(d_model + 1, read_dim)
        else:
            raise OperationConfigurationError(
                f"unsupported selector Read type: {read_type}"
            )

        ffn_read_type = _config_type(self.ffn_read_config, "state_default")
        self.ffn_read_type = ffn_read_type
        if ffn_read_type == "zero":
            self.state_out = None
        elif ffn_read_type == "state_default":
            if update_type == "ema":
                self.state_out = nn.Linear(self.state_shape[0], d_model, bias=False)
            elif update_type in {"gdn", "attention_window"}:
                self.state_out = None  # those variants own their query/output projections
            elif update_type == "none":
                self.state_out = None
            else:  # pragma: no cover - protected by update validation above
                raise AssertionError(update_type)
        else:
            raise OperationConfigurationError(
                f"unsupported FFN Read type: {ffn_read_type}"
            )

        compute_type = _config_type(self.compute_config, "double_residual_mlp")
        self.compute_type = compute_type
        if compute_type == "identity":
            self.gate_proj = self.up_proj = self.down_proj = None
        elif compute_type in {"double_residual_mlp", "double_residual_swiglu"}:
            hidden_dim = int(self.compute_config.get("hidden_dim", 4 * d_model))
            if hidden_dim <= 0:
                raise OperationConfigurationError("MLP hidden_dim must be positive")
            use_bias = bool(self.compute_config.get("bias", True))
            self.gate_proj = nn.Linear(d_model, hidden_dim, bias=use_bias)
            self.up_proj = nn.Linear(d_model, hidden_dim, bias=use_bias)
            self.down_proj = nn.Linear(hidden_dim, d_model, bias=use_bias)
        elif compute_type == "affine_residual":
            self.gate_proj = self.up_proj = None
            self.down_proj = nn.Linear(d_model, d_model, bias=bool(self.compute_config.get("bias", True)))
        else:
            raise OperationConfigurationError(
                f"unsupported NodeCompute type: {compute_type}"
            )

        aggregate_type = _config_type(self.aggregate_config, "mean")
        self.aggregate_type = aggregate_type
        if aggregate_type in {"learned_convex", "edge_softmax"}:
            # TEST-AGG-EDGE-SOFTMAX-V1: one trainable scalar per fixed parent
            # edge.  Parameters are materialized after the Plan is bound.
            self.aggregate_score = None
            self.edge_scores = nn.ParameterDict()
            self.edge_transforms = None
        elif aggregate_type in {"mean", "edge_linear_mean"}:
            self.aggregate_score = None
            self.edge_scores = None
            self.edge_transforms = nn.ModuleDict()
        else:
            raise OperationConfigurationError(
                f"unsupported Aggregate type: {aggregate_type}"
            )

    def ensure_edge_transforms(self, edge_ids: Sequence[str]) -> None:
        """Materialize edge-specific transforms after Plan binding."""

        for edge_id in edge_ids:
            # Boundary messages are not semantic graph edges and do not own
            # edge-specific parameters under the fixture contract.
            if edge_id.startswith("boundary:"):
                continue
            key = safe_module_key(edge_id)
            if self.aggregate_type in {"learned_convex", "edge_softmax"}:
                assert self.edge_scores is not None
                if key not in self.edge_scores:
                    self.edge_scores[key] = nn.Parameter(
                        self.input_norm.weight.new_zeros(())
                    )
            elif self.aggregate_type == "edge_linear_mean":
                assert self.edge_transforms is not None
                if key not in self.edge_transforms:
                    transform = nn.Linear(
                        self.d_model,
                        self.d_model,
                        bias=bool(self.aggregate_config.get("bias", True)),
                    )
                    self.edge_transforms[key] = transform.to(
                        device=self.input_norm.weight.device,
                        dtype=self.input_norm.weight.dtype,
                    )

    def initial_state(self, reference: Tensor) -> ReceiverState:
        if self.update_type == "none":
            return None
        if self.update_type in {"ema", "gdn"}:
            return reference.new_zeros(self.state_shape)
        if self.update_type == "attention_window":
            return AttentionState(
                torch.empty((0,), dtype=torch.int64, device=reference.device),
                reference.new_empty((0, self.attn_key_dim)),
                reference.new_empty((0, self.attn_value_dim)),
            )
        raise AssertionError(self.update_type)

    def normalize_input(self, hidden: Tensor) -> Tensor:
        return self.input_norm(hidden)

    def aggregate(self, messages: Sequence[Tensor], edge_ids: Sequence[str]) -> Tensor:
        if not messages:
            raise ValueError("Aggregate requires at least one DATA message")
        if len(messages) != len(edge_ids):
            raise ValueError("messages and edge_ids must have equal lengths")
        # Every entry receiver receives exactly one boundary message.  The
        # boundary is not a fixed parent edge and is returned unchanged even
        # for edge-parameterized Aggregate formulas.
        if len(messages) == 1 and edge_ids[0].startswith("boundary:"):
            return messages[0]
        stacked = torch.stack(tuple(messages), dim=0)
        if self.aggregate_type == "mean":
            return stacked.mean(dim=0)
        if self.aggregate_type in {"learned_convex", "edge_softmax"}:
            assert self.edge_scores is not None
            scores = torch.stack(
                [self.edge_scores[safe_module_key(edge_id)] for edge_id in edge_ids]
            ).to(device=stacked.device, dtype=stacked.dtype)
            weights = torch.softmax(scores, dim=0)
            return (weights.unsqueeze(-1) * stacked).sum(dim=0)
        if self.aggregate_type == "edge_linear_mean":
            assert self.edge_transforms is not None
            transformed = [
                self.edge_transforms[safe_module_key(edge_id)](message)
                for edge_id, message in zip(edge_ids, messages)
            ]
            return torch.stack(transformed, dim=0).mean(dim=0)
        raise AssertionError(self.aggregate_type)

    def proposal(
        self,
        state_before: ReceiverState,
        normalized: Tensor,
        *,
        token_position: Optional[int] = None,
    ) -> ReceiverState:
        if self.update_type == "none":
            return None
        state = state_before if state_before is not None else self.initial_state(normalized)
        if self.update_type == "ema":
            assert isinstance(state, Tensor)
            observation = torch.tanh(self.ema_observe(normalized))
            decay = (
                torch.sigmoid(self.ema_decay_logit)
                if self.ema_decay_logit is not None
                else self.ema_decay.to(device=normalized.device, dtype=normalized.dtype)
            )
            return decay * state + (1.0 - decay) * observation
        if self.update_type == "gdn":
            assert isinstance(state, Tensor)
            key = F.normalize(
                self.gdn_key(normalized), dim=-1, eps=self.state_norm_eps
            )
            value = self.gdn_value(normalized)
            eta = torch.sigmoid(self.gdn_eta(normalized)).squeeze(-1)
            gamma = torch.exp(
                -torch.exp(self.gdn_beta)
                * F.softplus(self.gdn_gamma(normalized).squeeze(-1))
            )
            decayed = gamma * state
            error = value - decayed.transpose(-2, -1).matmul(key)
            return decayed + eta * torch.outer(key, error)
        if self.update_type == "attention_window":
            assert isinstance(state, AttentionState)
            if token_position is None:
                raise ValueError("window Attention Update requires token_position")
            if state.length and token_position <= int(state.positions[-1].item()):
                raise ValueError(
                    "window Attention positions must increase strictly within a sequence"
                )
            key = F.normalize(
                self.attn_key(normalized), dim=-1, eps=self.state_norm_eps
            ).unsqueeze(0)
            value = self.attn_value(normalized).unsqueeze(0)
            position = torch.tensor(
                [token_position], dtype=torch.int64, device=normalized.device
            )
            positions = torch.cat((state.positions, position), dim=0)[-self.attn_window :]
            keys = torch.cat((state.keys, key), dim=0)[-self.attn_window :]
            values = torch.cat((state.values, value), dim=0)[-self.attn_window :]
            return AttentionState(positions, keys, values)
        raise AssertionError(self.update_type)

    def selector_read(self, normalized: Tensor, state: ReceiverState) -> Tensor:
        if self.selector_read_type == "content":
            return normalized
        if self.selector_read_type == "content_norm":
            return normalized.square().mean().sqrt().reshape(1)
        assert self.selector_read_linear is not None
        if self.selector_read_type == "content_linear":
            return self.selector_read_linear(normalized)
        if self.selector_read_type == "content_state_linear":
            if not isinstance(state, Tensor):
                raise ValueError("content_state_linear requires Tensor state")
            return self.selector_read_linear(
                torch.cat((normalized, state.reshape(-1)), dim=-1)
            )
        if self.selector_read_type == "content_state_summary_linear":
            summary = state_tensor_summary(state, normalized).reshape(1)
            return self.selector_read_linear(torch.cat((normalized, summary), dim=-1))
        raise AssertionError(self.selector_read_type)

    def ffn_read(self, state: ReceiverState, normalized: Tensor) -> Tensor:
        if self.ffn_read_type == "zero" or state is None:
            return normalized.new_zeros((self.d_model,))
        if self.update_type == "ema":
            assert isinstance(state, Tensor) and self.state_out is not None
            return self.state_out(state)
        if self.update_type == "gdn":
            assert isinstance(state, Tensor)
            query = F.normalize(
                self.gdn_query(normalized), dim=-1, eps=self.state_norm_eps
            )
            return self.gdn_out(state.transpose(-2, -1).matmul(query))
        if self.update_type == "attention_window":
            assert isinstance(state, AttentionState)
            if state.length == 0:
                return normalized.new_zeros((self.d_model,))
            query = F.normalize(
                self.attn_query(normalized), dim=-1, eps=self.state_norm_eps
            )
            weights = torch.softmax(state.keys.matmul(query) / math.sqrt(self.attn_key_dim), dim=0)
            return self.attn_out(state.values.transpose(0, 1).matmul(weights))
        raise AssertionError(self.update_type)

    def compute(self, hidden: Tensor, normalized: Tensor, state: ReceiverState) -> Tensor:
        if self.compute_type == "identity":
            return hidden
        if self.compute_type == "affine_residual":
            assert self.down_proj is not None
            # TEST-NODE-AFFINE-V1.
            return hidden + self.down_proj(normalized)
        first = hidden + self.ffn_read(state, normalized)
        ffn_input = self.ffn_norm(first)
        if self.compute_type in {"double_residual_mlp", "double_residual_swiglu"}:
            assert self.gate_proj is not None and self.up_proj is not None and self.down_proj is not None
            expansion = F.silu(self.gate_proj(ffn_input)) * self.up_proj(ffn_input)
            return first + self.down_proj(expansion)
        raise AssertionError(self.compute_type)

    def emit(self, hidden: Tensor, computed: Tensor, probability: Tensor) -> Tensor:
        emit_type = _config_type(self.emit_config, "hard")
        if emit_type == "hard":
            return computed
        if emit_type == "hst":
            zeta = float(self.emit_config.get("zeta", 1.0))
            rho = 1.0 + zeta * (probability - probability.detach())
            return hidden + rho * (computed - hidden)
        if emit_type == "softp":
            return hidden + probability * (computed - hidden)
        raise OperationConfigurationError(f"unsupported Emit type: {emit_type}")

    def make_identity(self) -> None:
        """Set this receiver to a structural identity where the config permits."""

        if self.state_out is not None:
            nn.init.zeros_(self.state_out.weight)
            if self.state_out.bias is not None:
                nn.init.zeros_(self.state_out.bias)
        for name in ("gdn_out", "attn_out"):
            module = getattr(self, name, None)
            if module is not None:
                nn.init.zeros_(module.weight)
        if self.compute_type == "identity":
            return
        assert self.down_proj is not None
        nn.init.zeros_(self.down_proj.weight)
        if self.down_proj.bias is not None:
            nn.init.zeros_(self.down_proj.bias)


class RegionSelector(nn.Module):
    """One region's Score operation over stable candidate readouts."""

    def __init__(
        self,
        read_dim: int,
        config: Mapping[str, Any],
        node_ids: Sequence[str],
    ) -> None:
        super().__init__()
        self.config = dict(config)
        self.node_ids = tuple(node_ids)
        if not self.node_ids:
            raise OperationConfigurationError("a selector requires at least one node")
        if len(set(self.node_ids)) != len(self.node_ids):
            raise OperationConfigurationError("selector node IDs must be unique")
        self.score_type = _config_type(config, "read_sum")
        self.context_dim = int(config.get("context_dim", 0))
        if self.context_dim < 0:
            raise OperationConfigurationError("Score context_dim must be nonnegative")
        score_dim = int(read_dim) + self.context_dim
        self.shared_parameters = bool(config.get("shared_parameters", False))
        self.linears = nn.ModuleDict()
        self.hidden_layers = nn.ModuleDict()
        self.output_layers = nn.ModuleDict()
        if self.score_type == "read_sum":
            self.linear = self.hidden = self.out = None
        elif self.score_type == "linear":
            self.linear = (
                nn.Linear(score_dim, 1, bias=bool(config.get("bias", True)))
                if self.shared_parameters
                else None
            )
            if not self.shared_parameters:
                for node_id in self.node_ids:
                    self.linears[safe_module_key(node_id)] = nn.Linear(
                        score_dim, 1, bias=bool(config.get("bias", True))
                    )
            self.hidden = self.out = None
        elif self.score_type == "mlp":
            hidden_dim = int(config.get("hidden_dim", max(4, score_dim)))
            if hidden_dim <= 0:
                raise OperationConfigurationError("selector MLP hidden_dim must be positive")
            self.linear = None
            if self.shared_parameters:
                self.hidden = nn.Linear(score_dim, hidden_dim)
                self.out = nn.Linear(hidden_dim, 1)
            else:
                self.hidden = self.out = None
                for node_id in self.node_ids:
                    key = safe_module_key(node_id)
                    self.hidden_layers[key] = nn.Linear(score_dim, hidden_dim)
                    self.output_layers[key] = nn.Linear(hidden_dim, 1)
        elif self.score_type in {"fixed", "constant"}:
            self.linear = self.hidden = self.out = None
        else:
            raise OperationConfigurationError(f"unsupported Score type: {self.score_type}")

    def forward(
        self,
        readouts: Tensor,
        candidate_node_ids: Optional[Sequence[str]] = None,
        context: Optional[Tensor] = None,
    ) -> Tensor:
        if readouts.ndim != 2:
            raise ValueError("selector readouts must have shape [candidates, read_dim]")
        candidates = (
            tuple(candidate_node_ids)
            if candidate_node_ids is not None
            else self.node_ids
        )
        if len(candidates) != readouts.shape[0]:
            raise ValueError("candidate IDs must match selector readout rows")
        unknown = set(candidates) - set(self.node_ids)
        if unknown:
            raise ValueError(f"unknown candidate node IDs: {sorted(unknown)}")
        if self.context_dim:
            if context is None or context.shape != (self.context_dim,):
                raise ValueError(
                    f"Score context must have shape [{self.context_dim}]"
                )
            score_inputs = torch.cat(
                (readouts, context.unsqueeze(0).expand(readouts.shape[0], -1)),
                dim=-1,
            )
        else:
            if context is not None and context.numel() != 0:
                raise ValueError("this Score formula declares no public context")
            score_inputs = readouts
        if self.score_type == "read_sum":
            return score_inputs.sum(dim=-1)
        if self.score_type == "linear":
            if self.shared_parameters:
                assert self.linear is not None
                return self.linear(score_inputs).squeeze(-1)
            return torch.stack(
                [
                    self.linears[safe_module_key(node_id)](row).squeeze(-1)
                    for node_id, row in zip(candidates, score_inputs)
                ]
            )
        if self.score_type == "mlp":
            if self.shared_parameters:
                assert self.hidden is not None and self.out is not None
                return self.out(F.silu(self.hidden(score_inputs))).squeeze(-1)
            return torch.stack(
                [
                    self.output_layers[safe_module_key(node_id)](
                        F.silu(self.hidden_layers[safe_module_key(node_id)](row))
                    ).squeeze(-1)
                    for node_id, row in zip(candidates, score_inputs)
                ]
            )
        if self.score_type == "constant":
            value = float(self.config.get("value", 0.0))
            return readouts.new_full((len(candidates),), value)
        if self.score_type == "fixed":
            values = self.config.get("values_by_node", self.config.get("values"))
            if isinstance(values, Mapping):
                try:
                    selected = [float(values[node_id]) for node_id in candidates]
                except KeyError as exc:
                    raise OperationConfigurationError(
                        f"fixed Score has no value for candidate {exc.args[0]!r}"
                    ) from exc
                return readouts.new_tensor(selected)
            if not isinstance(values, (tuple, list)) or len(values) != len(self.node_ids):
                raise OperationConfigurationError(
                    "fixed Score values must match the static region node count"
                )
            by_node = dict(zip(self.node_ids, values))
            return readouts.new_tensor([by_node[node_id] for node_id in candidates])
        raise AssertionError(self.score_type)


def deterministic_topk_mask(scores: Tensor, k: int) -> Tensor:
    """Return Top-K membership with ties broken by stable candidate order.

    The bounded-region pairwise rank avoids relying on backend-specific stable
    sort behavior.  Candidate order must already be stable node-ID order.
    """

    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    candidate_count = int(scores.shape[0])
    if not 1 <= int(k) <= candidate_count:
        raise ValueError("k must satisfy 1 <= k <= number of candidates")
    # row i, column j: candidate j outranks candidate i.
    greater = scores.unsqueeze(0) > scores.unsqueeze(1)
    indices = torch.arange(candidate_count, device=scores.device)
    earlier = indices.unsqueeze(0) < indices.unsqueeze(1)
    equal = scores.unsqueeze(0) == scores.unsqueeze(1)
    rank = (greater | (equal & earlier)).sum(dim=1)
    return rank < int(k)


def safe_module_key(identifier: str) -> str:
    """Map an arbitrary stable string ID to a ModuleDict-safe reversible key."""

    return "id_" + identifier.encode("utf-8").hex()


def _first_dim(shape: Sequence[int], default: int = 0) -> int:
    return int(shape[0]) if shape else int(default)

"""Topology-specialized SettleGraph execution.

The executors in this module consume the fully expanded :class:`~tide.plan.Plan`
owned by an existing :class:`~tide.engine.SettleGraph`.  They consequently use
the exact same receiver and selector parameter objects as the generic eager
reference, while owning a separate topology schedule:

* ``single-layer.v1`` packs every valid ``[B,T]`` event and evaluates the one
  flat region without a Python Token or batch-row loop in the numerical path;
* ``hb-line.v1`` groups the expanded HB regions by their declared Line and
  enforces an explicit Line barrier before advancing the wavefront.

Support is deliberately static and versioned.  An explicit specialized
executor never falls back to a generic executor after inspecting runtime
values, which keeps the accepted corpus reproducible and prevents accidental
semantic downgrades.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from .engine import (
    BalanceRegionStats,
    BalanceStats,
    BoundaryEventTrace,
    DynamicReachabilityError,
    EdgeEventTrace,
    ExecutionContractError,
    ExecutionResult,
    ExecutionTrace,
    LocalOperationError,
    NodeEventTrace,
    OutputEventTrace,
    RegionEventTrace,
    SettleGraph,
    StateKey,
    StateStore,
    StateWriteTrace,
    UnsupportedPlanError,
    _validate_detach_at_end,
    _validate_model_and_state,
    _validate_optional_mask,
    _validate_position_matrix,
    _validate_prefill_requested_k,
    _validate_reference_plan_capability,
    _validate_stable_id_sequence,
)
from .ops import OperationExecutionError, ReceiverState, safe_module_key
from .plan import Plan


SINGLE_LAYER_V1 = "single-layer.v1"
HB_LINE_V1 = "hb-line.v1"
SPECIALIZATION_VERSIONS = (SINGLE_LAYER_V1, HB_LINE_V1)


def _operation_type(config: Mapping[str, Any], default: str) -> str:
    return str(config.get("type", default)).lower().replace("-", "_")


def _reference_capability_reasons(plan: Plan) -> Tuple[str, ...]:
    """Return static local-operation gaps for the modules this path reuses."""

    try:
        _validate_reference_plan_capability(plan)
    except UnsupportedPlanError as exc:
        return (f"reference capability rejected the Plan: {exc}",)
    return ()


@dataclasses.dataclass(frozen=True)
class SpecializationSupport:
    """The complete static decision for one versioned specialization."""

    specialization: str
    supported: bool
    reasons: Tuple[str, ...] = ()

    def require(self, plan: Plan) -> None:
        if self.supported:
            return
        detail = "\n- ".join(self.reasons) if self.reasons else "unknown reason"
        raise UnsupportedPlanError(
            f"Plan {plan.plan_id!r} is not supported by "
            f"{self.specialization!r}:\n- {detail}"
        )


def single_layer_v1_support(plan: Plan) -> SpecializationSupport:
    """Return the static support decision for the packed flat-region path.

    Version 1 intentionally accepts the state-free ``N/content`` subset.  It
    covers every current Score, NodeCompute, Emit, and terminal Aggregate
    formula in that subset.  Stateful profiles are left to the generic packed
    executor because they require a segmented causal scan rather than this
    flat MoE-shaped implementation.
    """

    reasons: List[str] = []
    try:
        plan.validate()
    except Exception as exc:  # validation owns its detailed public taxonomy
        reasons.append(f"Plan validation failed: {exc}")
        return SpecializationSupport(SINGLE_LAYER_V1, False, tuple(reasons))

    reasons.extend(_reference_capability_reasons(plan))

    if len(plan.regions) != 1:
        reasons.append("requires exactly one region")
    if plan.edges:
        reasons.append("requires a graph with no receiver edges")
    if len(plan.regions) == 1:
        region = plan.regions[0]
        members = set(region.node_ids)
        all_nodes = {node.node_id for node in plan.nodes}
        if members != all_nodes:
            reasons.append("the sole region must contain every receiver")
        if set(plan.entry_node_ids) != all_nodes:
            reasons.append("every receiver must be an entry receiver")
        if set(plan.terminal_node_ids) != all_nodes:
            reasons.append("every receiver must be a terminal receiver")
        if region.control_dependencies:
            reasons.append("the sole region must not have control dependencies")
        if region.profile != "N" or region.selector_timing != "content":
            reasons.append("version 1 requires profile N with content timing")
        if _operation_type(region.k_requested, "fixed") != "fixed":
            reasons.append("version 1 supports only fixed K")
        if _operation_type(region.selector_context, "none") != "none":
            reasons.append("version 1 supports no selector context")
        if _operation_type(region.selector_history, "none") != "none":
            reasons.append("version 1 supports no selector history")
        if int(region.score.get("context_dim", 0)) != 0:
            reasons.append("version 1 Score must have context_dim=0")

    allowed_reads = {
        "content",
        "content_norm",
        "content_linear",
        "content_state_summary_linear",
    }
    for node in plan.nodes:
        if _operation_type(node.update, "none") != "none" or node.state_shape:
            reasons.append(f"receiver {node.node_id!r} must be stateless")
        read_type = _operation_type(node.selector_read, "content_norm")
        if read_type not in allowed_reads:
            reasons.append(
                f"receiver {node.node_id!r} selector Read {read_type!r} is "
                "not batchable by version 1"
            )
    return SpecializationSupport(SINGLE_LAYER_V1, not reasons, tuple(reasons))


def hb_line_v1_support(plan: Plan) -> SpecializationSupport:
    """Return the static support decision for the expanded HB Line schedule."""

    reasons: List[str] = []
    try:
        plan.validate()
    except Exception as exc:
        reasons.append(f"Plan validation failed: {exc}")
        return SpecializationSupport(HB_LINE_V1, False, tuple(reasons))

    reasons.extend(_reference_capability_reasons(plan))

    if plan.topology_kind != "hb":
        reasons.append("requires topology_kind='hb'")
    lines: Dict[int, List[Any]] = {}
    for region in plan.regions:
        if type(region.line) is not int or region.line < 0:
            reasons.append(
                f"region {region.region_id!r} lacks a nonnegative HB Line"
            )
            continue
        lines.setdefault(region.line, []).append(region)
        if _operation_type(region.k_requested, "fixed") != "fixed":
            reasons.append(
                f"region {region.region_id!r} uses non-fixed K"
            )
        if _operation_type(region.selector_context, "none") != "none":
            reasons.append(
                f"region {region.region_id!r} uses selector context"
            )
        if _operation_type(region.selector_history, "none") != "none":
            reasons.append(
                f"region {region.region_id!r} uses selector history"
            )
        if int(region.score.get("context_dim", 0)) != 0:
            reasons.append(
                f"region {region.region_id!r} Score has nonzero context_dim"
            )
    if lines and set(lines) != set(range(max(lines) + 1)):
        reasons.append("HB Lines must be contiguous from zero")

    region_by_id = {region.region_id: region for region in plan.regions}
    node_line = {
        node.node_id: region_by_id[node.region_id].line
        for node in plan.nodes
        if node.region_id in region_by_id
    }
    for edge in plan.edges:
        source_line = node_line.get(edge.source)
        target_line = node_line.get(edge.target)
        if (
            type(source_line) is not int
            or type(target_line) is not int
            or source_line >= target_line
        ):
            reasons.append(
                f"edge {edge.edge_id!r} does not point to a deeper Line"
            )
    return SpecializationSupport(HB_LINE_V1, not reasons, tuple(reasons))


def specialization_support(
    plan: Plan, specialization: str
) -> SpecializationSupport:
    """Dispatch to a named support predicate without runtime fallback."""

    normalized = str(specialization).lower().replace("_", "-")
    aliases = {
        "single-layer": SINGLE_LAYER_V1,
        "single-layer.v1": SINGLE_LAYER_V1,
        "hb": HB_LINE_V1,
        "hb-line": HB_LINE_V1,
        "hb-line.v1": HB_LINE_V1,
    }
    version = aliases.get(normalized)
    if version is None:
        raise ValueError(
            f"unknown specialization {specialization!r}; expected one of "
            f"{SPECIALIZATION_VERSIONS!r}"
        )
    if version == SINGLE_LAYER_V1:
        return single_layer_v1_support(plan)
    return hb_line_v1_support(plan)


class SpecializedExecutor:
    """Bind a statically accepted topology schedule to an existing model.

    This object is intentionally not an ``nn.Module``: registering the shared
    model as a child module would manufacture a second parameter namespace.
    Parameters remain owned solely by ``model`` and are consumed by reference.
    """

    def __init__(self, model: SettleGraph, specialization: str) -> None:
        if not isinstance(model, SettleGraph):
            raise TypeError("model must be a SettleGraph")
        decision = specialization_support(model.plan, specialization)
        decision.require(model.plan)
        self.model = model
        self._support = decision
        self.specialization = decision.specialization
        self.schedule_identity = (
            f"tide.specialized:{self.specialization}:"
            f"{model.plan.canonical_hash()}"
        )
        if self.specialization == HB_LINE_V1:
            grouped: Dict[int, List[Any]] = {}
            for region in model.plan.regions:
                assert type(region.line) is int
                grouped.setdefault(region.line, []).append(region)
            self._hb_lines = tuple(
                (
                    line,
                    tuple(sorted(regions, key=lambda item: item.region_id)),
                    regions[0].phase,
                )
                for line, regions in sorted(grouped.items())
            )
        else:
            self._hb_lines = ()

    def support_report(self) -> SpecializationSupport:
        """Return the construction-time static decision used by this binding."""

        return self._support

    @property
    def hb_line_schedule(
        self,
    ) -> Tuple[Tuple[int, Tuple[str, ...], Optional[str]], ...]:
        """Expose the compiled Line/region/phase order without runtime data."""

        return tuple(
            (
                line,
                tuple(region.region_id for region in regions),
                phase,
            )
            for line, regions, phase in self._hb_lines
        )

    def prefill(
        self,
        hidden: Tensor,
        execution_mask: Tensor,
        sequence_ids: Sequence[Any],
        token_positions: Tensor,
        *,
        state: Optional[StateStore] = None,
        requested_k: Optional[Mapping[str, Tensor]] = None,
        lm_target_mask: Optional[Tensor] = None,
        routing_stats_mask: Optional[Tensor] = None,
        reset_sequence_ids: Iterable[Any] = (),
        detach_at_end: bool = True,
        record_trace: bool = False,
    ) -> ExecutionResult:
        """Execute one chunk with the bound specialized schedule."""

        _validate_detach_at_end(detach_at_end)
        prepared = _prepare_prefill(
            self.model,
            hidden,
            execution_mask,
            sequence_ids,
            token_positions,
            state=state,
            requested_k=requested_k,
            lm_target_mask=lm_target_mask,
            routing_stats_mask=routing_stats_mask,
            reset_sequence_ids=reset_sequence_ids,
        )
        try:
            if self.specialization == SINGLE_LAYER_V1:
                return _single_layer_prefill(
                    self.model,
                    hidden,
                    execution_mask,
                    token_positions,
                    prepared,
                    detach_at_end=detach_at_end,
                    record_trace=record_trace,
                )
            return _hb_line_prefill(
                self.model,
                hidden,
                execution_mask,
                token_positions,
                prepared,
                self._hb_lines,
                detach_at_end=detach_at_end,
                record_trace=record_trace,
            )
        except OperationExecutionError as exc:
            raise LocalOperationError(str(exc)) from exc

    def decode(
        self,
        hidden: Tensor,
        execution_mask: Tensor,
        sequence_ids: Sequence[Any],
        token_positions: Tensor,
        *,
        state: Optional[StateStore] = None,
        requested_k: Optional[Mapping[str, Sequence[int]]] = None,
        lm_target_mask: Optional[Tensor] = None,
        routing_stats_mask: Optional[Tensor] = None,
        reset_sequence_ids: Iterable[Any] = (),
        detach_at_end: bool = True,
        record_trace: bool = False,
    ) -> ExecutionResult:
        """Execute one decode position through the same specialized path."""

        # Keep composite-invalid calls observationally aligned with eager:
        # configuration validation precedes decode-shape validation.
        _validate_detach_at_end(detach_at_end)
        if hidden.ndim != 2:
            raise ExecutionContractError("decode hidden must have shape [B,d_model]")
        batch = hidden.shape[0]
        packed_k: Optional[Dict[str, Tensor]] = None
        if requested_k is not None:
            packed_k = {}
            for region_id, values in requested_k.items():
                if len(values) != batch:
                    raise ExecutionContractError(
                        f"requested_k[{region_id!r}] must have length {batch}"
                    )
                packed_k[region_id] = torch.as_tensor(
                    values, device=token_positions.device
                ).reshape(batch, 1)
        result = self.prefill(
            hidden.unsqueeze(1),
            execution_mask.unsqueeze(1),
            sequence_ids,
            token_positions.unsqueeze(1),
            state=state,
            requested_k=packed_k,
            lm_target_mask=(
                lm_target_mask.unsqueeze(1) if lm_target_mask is not None else None
            ),
            routing_stats_mask=(
                routing_stats_mask.unsqueeze(1)
                if routing_stats_mask is not None
                else None
            ),
            reset_sequence_ids=reset_sequence_ids,
            detach_at_end=detach_at_end,
            record_trace=record_trace,
        )
        return ExecutionResult(
            result.output.squeeze(1),
            result.state,
            result.balance_stats,
            result.trace,
        )


@dataclasses.dataclass(frozen=True)
class _PreparedPrefill:
    sequence_ids: Tuple[str, ...]
    initial: StateStore
    route_mask: Tensor


def _prepare_prefill(
    model: SettleGraph,
    hidden: Tensor,
    execution_mask: Tensor,
    sequence_ids: Sequence[Any],
    token_positions: Tensor,
    *,
    state: Optional[StateStore],
    requested_k: Optional[Mapping[str, Tensor]],
    lm_target_mask: Optional[Tensor],
    routing_stats_mask: Optional[Tensor],
    reset_sequence_ids: Iterable[Any],
) -> _PreparedPrefill:
    if hidden.ndim != 3:
        raise ExecutionContractError("prefill hidden must have shape [B,T,d_model]")
    batch, length, width = hidden.shape
    if width != model.plan.d_model:
        raise ExecutionContractError("prefill hidden width does not match Plan d_model")
    if execution_mask.shape != (batch, length):
        raise ExecutionContractError("execution_mask must have shape [B,T]")
    if token_positions.shape != (batch, length):
        raise ExecutionContractError("token_positions must have shape [B,T]")
    if len(sequence_ids) != batch:
        raise ExecutionContractError("sequence_ids length must equal batch size")
    if execution_mask.dtype != torch.bool:
        raise ExecutionContractError("execution_mask must have bool dtype")
    if not torch.is_floating_point(hidden):
        raise ExecutionContractError("hidden must have a floating dtype")
    _validate_optional_mask(
        "lm_target_mask", lm_target_mask, execution_mask, (batch, length)
    )
    _validate_optional_mask(
        "routing_stats_mask",
        routing_stats_mask,
        execution_mask,
        (batch, length),
    )
    sequence_keys = _validate_stable_id_sequence("sequence", sequence_ids)
    if len(set(sequence_keys)) != len(sequence_keys):
        raise ExecutionContractError(
            "sequence_id values must be unique within one call"
        )
    reset_ids = _validate_stable_id_sequence(
        "reset sequence", reset_sequence_ids
    )
    if len(set(reset_ids)) != len(reset_ids):
        raise ExecutionContractError("reset sequence IDs must be unique")
    initial = state if state is not None else StateStore()
    _validate_model_and_state(model, hidden, initial)
    initial = initial.reset(reset_ids)
    _validate_prefill_requested_k(model.plan, requested_k, batch, length)
    _validate_position_matrix(initial, execution_mask, sequence_keys, token_positions)
    return _PreparedPrefill(
        sequence_keys,
        initial,
        routing_stats_mask if routing_stats_mask is not None else execution_mask,
    )


def _stable_topk_mask(scores: Tensor, k: int) -> Tensor:
    """Batched stable Top-K membership with node-order tie breaking."""

    if scores.ndim not in {1, 2}:
        raise OperationExecutionError("scores must have shape [R] or [E,R]")
    width = scores.shape[-1]
    if not 1 <= int(k) <= int(width):
        raise OperationExecutionError(
            "k must satisfy 1 <= k <= number of candidates"
        )
    indices = torch.arange(width, device=scores.device)
    # [..., i, j] says candidate j outranks candidate i.
    greater = scores.unsqueeze(-2) > scores.unsqueeze(-1)
    equal = scores.unsqueeze(-2) == scores.unsqueeze(-1)
    earlier = indices.unsqueeze(0) < indices.unsqueeze(1)
    rank = (greater | (equal & earlier)).sum(dim=-1)
    return rank < int(k)


@torch.jit.script
def _rms_norm_records_1d(
    values: Tensor, weight: Tensor, eps: float
) -> Tensor:
    """Apply the eager one-vector RMSNorm boundary to packed events."""

    if values.size(0) == 0:
        return torch.empty_like(values)
    rows = torch.jit.annotate(List[Tensor], [])
    for event_index in range(values.size(0)):
        value = values[event_index]
        variance = value.square().mean(dim=-1, keepdim=True)
        rows.append(value * torch.rsqrt(variance + eps) * weight)
    return torch.stack(rows, dim=0)


@torch.jit.script
def _rms_magnitude_records_1d(values: Tensor) -> Tensor:
    """Match the eager no-epsilon one-vector RMS readout per event."""

    if values.size(0) == 0:
        return values.new_empty((0, 1))
    rows = torch.jit.annotate(List[Tensor], [])
    divisor = math.sqrt(float(values.size(-1)))
    for event_index in range(values.size(0)):
        rows.append(
            (torch.linalg.vector_norm(values[event_index]) / divisor).reshape(1)
        )
    return torch.stack(rows, dim=0)


@torch.jit.script
def _shared_linear_records_1d(
    values: Tensor, weight: Tensor, bias: Tensor
) -> Tensor:
    """Apply one Linear to each eager one-vector logical record."""

    if values.size(0) == 0:
        return values.new_empty((0, weight.size(0)))
    rows = torch.jit.annotate(List[Tensor], [])
    for event_index in range(values.size(0)):
        if bias.numel() > 0:
            rows.append(F.linear(values[event_index], weight, bias))
        else:
            rows.append(F.linear(values[event_index], weight, None))
    return torch.stack(rows, dim=0)


@torch.jit.script
def _read_sum_score_events(readouts: Tensor) -> Tensor:
    """Preserve the eager candidate-matrix reduction boundary per event."""

    rows = torch.jit.annotate(List[Tensor], [])
    for event_index in range(readouts.size(0)):
        rows.append(readouts[event_index].sum(dim=-1))
    return torch.stack(rows, dim=0)


@torch.jit.script
def _shared_linear_score_events(
    readouts: Tensor, weight: Tensor, bias: Tensor
) -> Tensor:
    """Run a shared selector Linear on one candidate matrix per event."""

    rows = torch.jit.annotate(List[Tensor], [])
    for event_index in range(readouts.size(0)):
        if bias.numel() > 0:
            score = F.linear(readouts[event_index], weight, bias)
        else:
            score = F.linear(readouts[event_index], weight, None)
        rows.append(score.squeeze(-1))
    return torch.stack(rows, dim=0)


@torch.jit.script
def _shared_mlp_score_events(
    readouts: Tensor,
    hidden_weight: Tensor,
    hidden_bias: Tensor,
    output_weight: Tensor,
    output_bias: Tensor,
) -> Tensor:
    """Run a shared selector MLP on one candidate matrix per event."""

    rows = torch.jit.annotate(List[Tensor], [])
    for event_index in range(readouts.size(0)):
        values = readouts[event_index]
        if hidden_bias.numel() > 0:
            hidden = F.linear(values, hidden_weight, hidden_bias)
        else:
            hidden = F.linear(values, hidden_weight, None)
        hidden = F.silu(hidden)
        if output_bias.numel() > 0:
            score = F.linear(hidden, output_weight, output_bias)
        else:
            score = F.linear(hidden, output_weight, None)
        rows.append(score.squeeze(-1))
    return torch.stack(rows, dim=0)


@torch.jit.script
def _unshared_linear_score_records(
    readouts: Tensor, weights: Tensor, biases: Tensor
) -> Tensor:
    """Run each candidate's Linear on the eager one-vector boundary."""

    events = torch.jit.annotate(List[Tensor], [])
    for event_index in range(readouts.size(0)):
        scores = torch.jit.annotate(List[Tensor], [])
        for node_index in range(readouts.size(1)):
            if biases.numel() > 0:
                score = F.linear(
                    readouts[event_index, node_index],
                    weights[node_index],
                    biases[node_index],
                )
            else:
                score = F.linear(
                    readouts[event_index, node_index],
                    weights[node_index],
                    None,
                )
            scores.append(score.squeeze(-1))
        events.append(torch.stack(scores, dim=0))
    return torch.stack(events, dim=0)


@torch.jit.script
def _unshared_mlp_score_records(
    readouts: Tensor,
    hidden_weights: Tensor,
    hidden_biases: Tensor,
    output_weights: Tensor,
    output_biases: Tensor,
) -> Tensor:
    """Run each candidate's MLP on the eager one-vector boundary."""

    events = torch.jit.annotate(List[Tensor], [])
    for event_index in range(readouts.size(0)):
        scores = torch.jit.annotate(List[Tensor], [])
        for node_index in range(readouts.size(1)):
            value = readouts[event_index, node_index]
            if hidden_biases.numel() > 0:
                hidden = F.linear(
                    value,
                    hidden_weights[node_index],
                    hidden_biases[node_index],
                )
            else:
                hidden = F.linear(value, hidden_weights[node_index], None)
            hidden = F.silu(hidden)
            if output_biases.numel() > 0:
                score = F.linear(
                    hidden,
                    output_weights[node_index],
                    output_biases[node_index],
                )
            else:
                score = F.linear(hidden, output_weights[node_index], None)
            scores.append(score.squeeze(-1))
        events.append(torch.stack(scores, dim=0))
    return torch.stack(events, dim=0)


@torch.jit.script
def _softmax_score_events_1d(logits: Tensor) -> Tensor:
    """Apply the eager one-vector candidate softmax per event."""

    rows = torch.jit.annotate(List[Tensor], [])
    for event_index in range(logits.size(0)):
        rows.append(torch.softmax(logits[event_index], dim=0))
    return torch.stack(rows, dim=0)


@torch.jit.script
def _affine_compute_records_1d(
    hidden: Tensor,
    normalized: Tensor,
    weight: Tensor,
    bias: Tensor,
) -> Tensor:
    """Apply the eager affine-residual NodeCompute per active event."""

    rows = torch.jit.annotate(List[Tensor], [])
    for event_index in range(hidden.size(0)):
        if bias.numel() > 0:
            projected = F.linear(normalized[event_index], weight, bias)
        else:
            projected = F.linear(normalized[event_index], weight, None)
        rows.append(hidden[event_index] + projected)
    return torch.stack(rows, dim=0)


@torch.jit.script
def _swiglu_compute_records_1d(
    hidden: Tensor,
    normalized: Tensor,
    norm_weight: Tensor,
    norm_eps: float,
    gate_weight: Tensor,
    gate_bias: Tensor,
    up_weight: Tensor,
    up_bias: Tensor,
    down_weight: Tensor,
    down_bias: Tensor,
) -> Tensor:
    """Keep the complete eager SwiGLU NodeCompute inside each record."""

    rows = torch.jit.annotate(List[Tensor], [])
    for event_index in range(hidden.size(0)):
        # The state-free specialization still follows the registered formula:
        # FFN Read produces an explicit zero vector before the first residual.
        first = hidden[event_index] + torch.zeros_like(normalized[event_index])
        variance = first.square().mean(dim=-1, keepdim=True)
        ffn_input = first * torch.rsqrt(variance + norm_eps) * norm_weight
        if gate_bias.numel() > 0:
            gate = F.linear(ffn_input, gate_weight, gate_bias)
        else:
            gate = F.linear(ffn_input, gate_weight, None)
        if up_bias.numel() > 0:
            up = F.linear(ffn_input, up_weight, up_bias)
        else:
            up = F.linear(ffn_input, up_weight, None)
        expansion = F.silu(gate) * up
        if down_bias.numel() > 0:
            down = F.linear(expansion, down_weight, down_bias)
        else:
            down = F.linear(expansion, down_weight, None)
        rows.append(first + down)
    return torch.stack(rows, dim=0)


def _batched_selector_read(receiver: Any, normalized: Tensor) -> Tensor:
    kind = receiver.selector_read_type
    if kind == "content":
        return normalized
    if kind == "content_norm":
        return _rms_magnitude_records_1d(normalized)
    if kind == "content_linear":
        assert receiver.selector_read_linear is not None
        linear = receiver.selector_read_linear
        bias = linear.bias if linear.bias is not None else normalized.new_empty((0,))
        return _shared_linear_records_1d(normalized, linear.weight, bias)
    if kind == "content_state_summary_linear":
        assert receiver.selector_read_linear is not None
        summary = normalized.new_zeros((*normalized.shape[:-1], 1))
        values = torch.cat((normalized, summary), dim=-1)
        linear = receiver.selector_read_linear
        bias = linear.bias if linear.bias is not None else normalized.new_empty((0,))
        return _shared_linear_records_1d(values, linear.weight, bias)
    raise AssertionError(kind)


def _batched_score(selector: Any, readouts: Tensor) -> Tensor:
    """Apply one selector to ``[events,nodes,read_dim]`` readouts."""

    kind = selector.score_type
    event_count, width, _ = readouts.shape
    if kind == "read_sum":
        return _read_sum_score_events(readouts)
    if kind == "constant":
        return readouts.new_full(
            (event_count, width), float(selector.config.get("value", 0.0))
        )
    if kind == "fixed":
        values = selector.config.get(
            "values_by_node", selector.config.get("values")
        )
        if isinstance(values, Mapping):
            ordered = [float(values[node_id]) for node_id in selector.node_ids]
        else:
            if not isinstance(values, (tuple, list)):
                raise OperationExecutionError("fixed Score values are malformed")
            ordered = [float(value) for value in values]
        return readouts.new_tensor(ordered).expand(event_count, -1)
    if kind == "linear":
        if selector.shared_parameters:
            assert selector.linear is not None
            linear = selector.linear
            bias = linear.bias if linear.bias is not None else readouts.new_empty((0,))
            return _shared_linear_score_events(readouts, linear.weight, bias)
        linears = tuple(
            selector.linears[safe_module_key(node_id)]
            for node_id in selector.node_ids
        )
        weights = torch.stack(tuple(linear.weight for linear in linears), dim=0)
        biases = (
            torch.stack(tuple(linear.bias for linear in linears), dim=0)
            if linears[0].bias is not None
            else readouts.new_empty((0,))
        )
        return _unshared_linear_score_records(readouts, weights, biases)
    if kind == "mlp":
        if selector.shared_parameters:
            assert selector.hidden is not None and selector.out is not None
            hidden = selector.hidden
            output = selector.out
            hidden_bias = (
                hidden.bias if hidden.bias is not None else readouts.new_empty((0,))
            )
            output_bias = (
                output.bias if output.bias is not None else readouts.new_empty((0,))
            )
            return _shared_mlp_score_events(
                readouts,
                hidden.weight,
                hidden_bias,
                output.weight,
                output_bias,
            )
        hidden_layers = tuple(
            selector.hidden_layers[safe_module_key(node_id)]
            for node_id in selector.node_ids
        )
        output_layers = tuple(
            selector.output_layers[safe_module_key(node_id)]
            for node_id in selector.node_ids
        )
        hidden_weights = torch.stack(
            tuple(layer.weight for layer in hidden_layers), dim=0
        )
        hidden_biases = (
            torch.stack(tuple(layer.bias for layer in hidden_layers), dim=0)
            if hidden_layers[0].bias is not None
            else readouts.new_empty((0,))
        )
        output_weights = torch.stack(
            tuple(layer.weight for layer in output_layers), dim=0
        )
        output_biases = (
            torch.stack(tuple(layer.bias for layer in output_layers), dim=0)
            if output_layers[0].bias is not None
            else readouts.new_empty((0,))
        )
        return _unshared_mlp_score_records(
            readouts,
            hidden_weights,
            hidden_biases,
            output_weights,
            output_biases,
        )
    raise AssertionError(kind)


def _single_layer_prefill(
    model: SettleGraph,
    hidden: Tensor,
    execution_mask: Tensor,
    token_positions: Tensor,
    prepared: _PreparedPrefill,
    *,
    detach_at_end: bool,
    record_trace: bool,
) -> ExecutionResult:
    region = model.plan.regions[0]
    node_ids = region.node_ids
    event_rows, event_columns = execution_mask.nonzero(as_tuple=True)
    event_hidden = hidden[event_rows, event_columns]
    event_count = int(event_hidden.shape[0])
    node_count = len(node_ids)
    forced = bool(event_count) and all(
        model.plan.node_by_id(node_id).forced_active for node_id in node_ids
    )

    normalized: Dict[str, Tensor] = {}
    readout: Dict[str, Tensor] = {}
    for node_id in node_ids:
        receiver = model.receiver(node_id)
        normalized[node_id] = _rms_norm_records_1d(
            event_hidden,
            receiver.input_norm.weight,
            receiver.input_norm.eps,
        )
        if not forced:
            readout[node_id] = _batched_selector_read(
                receiver, normalized[node_id]
            )

    if event_count == 0:
        logits = hidden.new_empty((0, node_count))
        probabilities = hidden.new_empty((0, node_count))
        active_mask = torch.empty(
            (0, node_count), device=hidden.device, dtype=torch.bool
        )
        requested = None if forced else int(region.k_requested["value"])
    elif forced:
        if node_count != 1:
            raise ExecutionContractError(
                "forced-active regions must be singleton regions"
            )
        logits = hidden.new_empty((event_count, 0))
        probabilities = hidden.new_ones((event_count, 1))
        active_mask = torch.ones_like(probabilities, dtype=torch.bool)
        requested = None
    else:
        packed_readouts = torch.stack(
            [readout[node_id] for node_id in node_ids], dim=1
        )
        logits = _batched_score(model.selector(region.region_id), packed_readouts)
        if not bool(torch.isfinite(logits).all().item()):
            raise ExecutionContractError(
                f"region {region.region_id!r} produced non-finite logits"
            )
        probabilities = _softmax_score_events_1d(logits)
        requested = int(region.k_requested["value"])
        active_mask = _stable_topk_mask(logits, requested)

    computed: Dict[str, Tensor] = {}
    emitted: Dict[str, Tensor] = {}
    packed_emitted: List[Tensor] = []
    for node_index, node_id in enumerate(node_ids):
        receiver = model.receiver(node_id)
        offsets = active_mask[:, node_index].nonzero(as_tuple=False).flatten()
        # Calling a parameterized module with an empty Tensor would create
        # zero-valued gradients for a receiver that the semantic graph never
        # executed.  Skipping it preserves the reference's disconnected/None
        # VJP classification.
        if int(offsets.numel()):
            selected_hidden = event_hidden.index_select(0, offsets)
            selected_normalized = normalized[node_id].index_select(0, offsets)
            if receiver.compute_type == "identity":
                node_computed = selected_hidden
            elif receiver.compute_type == "affine_residual":
                assert receiver.down_proj is not None
                down_bias = (
                    receiver.down_proj.bias
                    if receiver.down_proj.bias is not None
                    else selected_hidden.new_empty((0,))
                )
                node_computed = _affine_compute_records_1d(
                    selected_hidden,
                    selected_normalized,
                    receiver.down_proj.weight,
                    down_bias,
                )
            else:
                assert receiver.compute_type in {
                    "double_residual_mlp",
                    "double_residual_swiglu",
                }
                assert receiver.gate_proj is not None
                assert receiver.up_proj is not None
                assert receiver.down_proj is not None
                gate_bias = (
                    receiver.gate_proj.bias
                    if receiver.gate_proj.bias is not None
                    else selected_hidden.new_empty((0,))
                )
                up_bias = (
                    receiver.up_proj.bias
                    if receiver.up_proj.bias is not None
                    else selected_hidden.new_empty((0,))
                )
                down_bias = (
                    receiver.down_proj.bias
                    if receiver.down_proj.bias is not None
                    else selected_hidden.new_empty((0,))
                )
                node_computed = _swiglu_compute_records_1d(
                    selected_hidden,
                    selected_normalized,
                    receiver.ffn_norm.weight,
                    receiver.ffn_norm.eps,
                    receiver.gate_proj.weight,
                    gate_bias,
                    receiver.up_proj.weight,
                    up_bias,
                    receiver.down_proj.weight,
                    down_bias,
                )
            probability = probabilities[:, node_index].index_select(0, offsets)
            node_emitted = receiver.emit(
                selected_hidden,
                node_computed,
                probability.unsqueeze(-1),
            )
        else:
            node_computed = event_hidden.new_empty((0, model.plan.d_model))
            node_emitted = event_hidden.new_empty((0, model.plan.d_model))
        computed[node_id] = event_hidden.new_zeros(
            (event_count, model.plan.d_model)
        ).index_copy(0, offsets, node_computed)
        emitted[node_id] = event_hidden.new_zeros(
            (event_count, model.plan.d_model)
        ).index_copy(0, offsets, node_emitted)
        packed_emitted.append(emitted[node_id])

    emitted_tensor = (
        torch.stack(packed_emitted, dim=1)
        if packed_emitted
        else hidden.new_empty((event_count, 0, model.plan.d_model))
    )
    if event_count == 0:
        event_output = hidden.new_empty((0, model.plan.d_model))
    elif model.output_aggregate_type == "mean":
        divisor = active_mask.sum(dim=-1, keepdim=True).to(hidden.dtype)
        event_output = emitted_tensor.sum(dim=1) / divisor
    else:
        assert model.output_scores is not None
        # Exclude never-active terminal parameters entirely.  A masked
        # ``-inf`` lane has the same forward value but would incorrectly turn
        # a disconnected semantic parameter into a connected zero VJP.
        ever_active = active_mask.any(dim=0).nonzero(as_tuple=False).flatten()
        selected_node_ids = [node_ids[int(index.item())] for index in ever_active]
        scores = torch.stack(
            [
                model.output_scores[safe_module_key(node_id)]
                for node_id in selected_node_ids
            ]
        ).to(device=hidden.device, dtype=hidden.dtype)
        masked_scores = scores.unsqueeze(0).expand(event_count, -1).masked_fill(
            ~active_mask.index_select(1, ever_active), -torch.inf
        )
        weights = torch.softmax(masked_scores, dim=-1)
        event_output = (
            weights.unsqueeze(-1)
            * emitted_tensor.index_select(1, ever_active)
        ).sum(dim=1)

    output = hidden.clone()
    if event_count:
        output[event_rows, event_columns] = event_output

    next_positions = dict(prepared.initial.next_position)
    for event_index in range(event_count):
        sequence_id = prepared.sequence_ids[int(event_rows[event_index].item())]
        next_positions[sequence_id] = int(
            token_positions[event_rows[event_index], event_columns[event_index]].item()
        ) + 1
    result_state = StateStore(
        dict(prepared.initial.values),
        dict(prepared.initial.selector_history),
        next_positions,
    )
    if detach_at_end:
        result_state = result_state.detached()

    balance = BalanceStats(zero=hidden.new_zeros(()))
    route_events = prepared.route_mask[event_rows, event_columns]
    route_offsets = route_events.nonzero(as_tuple=False).flatten()
    route_count = int(route_offsets.shape[0])
    if route_count:
        route_probabilities = probabilities.index_select(0, route_offsets)
        route_active = active_mask.index_select(0, route_offsets)
        hard_divisor = route_active.sum(dim=-1, keepdim=True).to(hidden.dtype)
        balance.regions[region.region_id] = BalanceRegionStats(
            soft_sum=route_probabilities.sum(dim=0),
            availability_sum=hidden.new_full(
                (node_count,), route_count / node_count
            ).detach(),
            hard_share_sum=(
                route_active.to(hidden.dtype) / hard_divisor
            ).sum(dim=0).detach(),
            event_count=route_count,
            competition_count=route_count if node_count >= 2 else 0,
            forced_active=forced,
        )

    trace = None
    if record_trace:
        node_events: List[NodeEventTrace] = []
        boundary_events: List[BoundaryEventTrace] = []
        region_events: List[RegionEventTrace] = []
        output_events: List[OutputEventTrace] = []
        for event_index in range(event_count):
            row = int(event_rows[event_index].item())
            column = int(event_columns[event_index].item())
            sequence_id = prepared.sequence_ids[row]
            position = int(token_positions[row, column].item())
            active_ids = tuple(
                node_id
                for node_index, node_id in enumerate(node_ids)
                if bool(active_mask[event_index, node_index].item())
            )
            for node_id in node_ids:
                boundary_events.append(
                    BoundaryEventTrace(
                        sequence_id, position, node_id, event_hidden[event_index]
                    )
                )
            region_events.append(
                RegionEventTrace(
                    sequence_id=sequence_id,
                    token_position=position,
                    region_id=region.region_id,
                    candidate_node_ids=tuple(node_ids),
                    logits=None if forced else logits[event_index],
                    probabilities=probabilities[event_index],
                    requested_k=None if forced else requested,
                    effective_k=None if forced else len(active_ids),
                    top_k_node_ids=None if forced else active_ids,
                    active_node_ids=active_ids,
                    forced_active=forced,
                )
            )
            terminal_messages: List[Tuple[str, Tensor]] = []
            for node_index, node_id in enumerate(node_ids):
                active = bool(active_mask[event_index, node_index].item())
                node_computed = computed[node_id][event_index] if active else None
                node_emitted = emitted[node_id][event_index] if active else None
                if active:
                    assert node_emitted is not None
                    terminal_messages.append((node_id, node_emitted))
                node_events.append(
                    NodeEventTrace(
                        sequence_id=sequence_id,
                        token_position=position,
                        region_id=region.region_id,
                        node_id=node_id,
                        reached=True,
                        observed=False,
                        active=active,
                        input_hidden=event_hidden[event_index],
                        normalized_input=normalized[node_id][event_index],
                        state_before=None,
                        proposal=None,
                        state_for_compute=None,
                        selector_read=(
                            None if forced else readout[node_id][event_index]
                        ),
                        logit=None if forced else logits[event_index, node_index],
                        probability=probabilities[event_index, node_index],
                        computed=node_computed,
                        emitted=node_emitted,
                        parent_messages=(
                            (
                                f"boundary:{node_id}",
                                "DATA",
                                event_hidden[event_index],
                            ),
                        ),
                    )
                )
            output_events.append(
                OutputEventTrace(
                    sequence_id,
                    position,
                    tuple(terminal_messages),
                    _aggregate_output(
                        model,
                        [payload for _, payload in terminal_messages],
                        [node_id for node_id, _ in terminal_messages],
                    ),
                )
            )
        trace = _canonical_trace(
            model.plan,
            node_events,
            (),
            boundary_events,
            region_events,
            (),
            output_events,
        )
    return ExecutionResult(output, result_state, balance, trace)


def _hb_line_prefill(
    model: SettleGraph,
    hidden: Tensor,
    execution_mask: Tensor,
    token_positions: Tensor,
    prepared: _PreparedPrefill,
    lines: Sequence[Tuple[int, Sequence[Any], Optional[str]]],
    *,
    detach_at_end: bool,
    record_trace: bool,
) -> ExecutionResult:
    batch, length, _ = hidden.shape
    plan = model.plan
    incoming = plan.incoming_edges
    outgoing = plan.outgoing_edges
    entries = set(plan.entry_node_ids)
    terminals = set(plan.terminal_node_ids)

    staged_values: Dict[StateKey, ReceiverState] = dict(prepared.initial.values)
    next_positions = dict(prepared.initial.next_position)
    edge_values: Dict[Tuple[int, int, str], Optional[Tensor]] = {}
    terminal_messages: Dict[Tuple[int, int], List[Tuple[str, Tensor]]] = {}
    balance = BalanceStats(zero=hidden.new_zeros(()))
    node_events: List[NodeEventTrace] = []
    boundary_events: List[BoundaryEventTrace] = []
    region_events: List[RegionEventTrace] = []
    state_writes: List[StateWriteTrace] = []

    # The phase value is compiled into ``lines`` and intentionally traversed:
    # it is not used to recreate topology or alter local semantics.
    for _line, regions, _phase in lines:
        # Token is outside region here, making the same-Token Line barrier
        # explicit while allowing independent regions on a Line to settle in
        # stable order.
        for token_index in range(length):
            for region in regions:
                timing = region.selector_timing.lower().replace("-", "_")
                profile = region.profile.upper()
                for batch_index, sequence_id in enumerate(prepared.sequence_ids):
                    if not bool(execution_mask[batch_index, token_index].item()):
                        continue
                    graph_input = hidden[batch_index, token_index]
                    position = int(token_positions[batch_index, token_index].item())
                    event_key = (batch_index, token_index)
                    reached_ids: List[str] = []
                    inputs: Dict[str, Tensor] = {}
                    normalized: Dict[str, Tensor] = {}
                    state_before: Dict[str, ReceiverState] = {}
                    proposals: Dict[str, ReceiverState] = {}
                    readouts: Dict[str, Tensor] = {}
                    parents_by_node: Dict[
                        str, Tuple[Tuple[str, str, Optional[Tensor]], ...]
                    ] = {}

                    for node_id in region.node_ids:
                        receiver = model.receiver(node_id)
                        if node_id in entries:
                            messages = [graph_input]
                            edge_ids = [f"boundary:{node_id}"]
                            parents_by_node[node_id] = (
                                (f"boundary:{node_id}", "DATA", graph_input),
                            )
                            if record_trace:
                                boundary_events.append(
                                    BoundaryEventTrace(
                                        sequence_id, position, node_id, graph_input
                                    )
                                )
                        else:
                            parent_edges = incoming[node_id]
                            parent_keys = [
                                (batch_index, token_index, edge.edge_id)
                                for edge in parent_edges
                            ]
                            if any(key not in edge_values for key in parent_keys):
                                raise AssertionError(
                                    f"Line schedule reached {node_id!r} before "
                                    "all parent edges settled"
                                )
                            data = [
                                (edge.edge_id, edge_values[key])
                                for edge, key in zip(parent_edges, parent_keys)
                                if edge_values[key] is not None
                            ]
                            edge_ids = [edge_id for edge_id, _ in data]
                            messages = [value for _, value in data]
                            parents_by_node[node_id] = tuple(
                                (
                                    edge.edge_id,
                                    "DATA" if edge_values[key] is not None else "CLOSED",
                                    edge_values[key],
                                )
                                for edge, key in zip(parent_edges, parent_keys)
                            )
                        if not messages:
                            continue
                        aggregate = receiver.aggregate(messages, edge_ids)
                        node_normalized = receiver.normalize_input(aggregate)
                        reached_ids.append(node_id)
                        inputs[node_id] = aggregate
                        normalized[node_id] = node_normalized
                        state_before[node_id] = staged_values.get(
                            (sequence_id, node_id),
                            receiver.initial_state(node_normalized),
                        )

                    forced = bool(reached_ids) and all(
                        plan.node_by_id(node_id).forced_active
                        for node_id in region.node_ids
                    )
                    requested: Optional[int] = None
                    if reached_ids and not forced:
                        requested = int(region.k_requested["value"])
                        if not 1 <= requested <= int(region.k_max):
                            raise ExecutionContractError(
                                f"requested K={requested} for region "
                                f"{region.region_id!r} is outside [1,{region.k_max}]"
                            )
                        for node_id in reached_ids:
                            receiver = model.receiver(node_id)
                            if timing == "post":
                                proposal = receiver.proposal(
                                    state_before[node_id],
                                    normalized[node_id],
                                    token_position=position,
                                )
                                proposals[node_id] = proposal
                                selector_state = proposal
                            elif timing == "pre":
                                selector_state = state_before[node_id]
                            else:
                                selector_state = None
                            readouts[node_id] = receiver.selector_read(
                                normalized[node_id], selector_state
                            )

                    active_ids: List[str] = []
                    logits: Optional[Tensor] = None
                    probabilities: Optional[Tensor] = None
                    top_k_ids: Optional[Tuple[str, ...]] = None
                    effective: Optional[int] = None
                    if reached_ids:
                        if forced:
                            if len(region.node_ids) != 1:
                                raise ExecutionContractError(
                                    "forced-active regions must be singleton regions"
                                )
                            probabilities = graph_input.new_ones((1,))
                            active_ids = list(reached_ids)
                        else:
                            logits = model.selector(region.region_id)(
                                torch.stack(
                                    [readouts[node_id] for node_id in reached_ids]
                                ),
                                reached_ids,
                            )
                            if not bool(torch.isfinite(logits).all().item()):
                                raise ExecutionContractError(
                                    f"region {region.region_id!r} produced "
                                    "non-finite logits"
                                )
                            probabilities = torch.softmax(logits, dim=0)
                            assert requested is not None
                            effective = min(requested, len(reached_ids))
                            active_membership = _stable_topk_mask(logits, effective)
                            active_ids = [
                                node_id
                                for index, node_id in enumerate(reached_ids)
                                if bool(active_membership[index].item())
                            ]
                            top_k_ids = tuple(active_ids)

                    if record_trace:
                        region_events.append(
                            RegionEventTrace(
                                sequence_id=sequence_id,
                                token_position=position,
                                region_id=region.region_id,
                                candidate_node_ids=tuple(reached_ids),
                                logits=logits,
                                probabilities=probabilities,
                                requested_k=None if forced else requested,
                                effective_k=None if forced else effective,
                                top_k_node_ids=None if forced else top_k_ids,
                                active_node_ids=tuple(active_ids),
                                forced_active=forced,
                            )
                        )

                    active_set = set(active_ids)
                    if profile == "N":
                        observe_set: set[str] = set()
                    elif profile == "SD":
                        observe_set = active_set
                    elif profile == "BO":
                        observe_set = set(reached_ids)
                    else:
                        raise ExecutionContractError(
                            f"unsupported profile {region.profile!r}"
                        )
                    probability_by_node = {
                        node_id: probabilities[index]
                        for index, node_id in enumerate(reached_ids)
                    } if probabilities is not None else {}
                    logit_by_node = {
                        node_id: logits[index]
                        for index, node_id in enumerate(reached_ids)
                    } if logits is not None else {}

                    for node_id in region.node_ids:
                        reached = node_id in inputs
                        observed = node_id in observe_set
                        active = node_id in active_set
                        proposal = proposals.get(node_id)
                        compute_state = state_before.get(node_id)
                        computed: Optional[Tensor] = None
                        emitted: Optional[Tensor] = None
                        receiver = model.receiver(node_id)
                        if reached and observed:
                            if node_id not in proposals:
                                proposal = receiver.proposal(
                                    state_before[node_id],
                                    normalized[node_id],
                                    token_position=position,
                                )
                                proposals[node_id] = proposal
                            compute_state = proposal
                            if proposal is not None:
                                staged_values[(sequence_id, node_id)] = proposal
                                if record_trace:
                                    state_writes.append(
                                        StateWriteTrace(
                                            sequence_id,
                                            position,
                                            "receiver",
                                            node_id,
                                            proposal,
                                        )
                                    )
                        if active:
                            computed = receiver.compute(
                                inputs[node_id], normalized[node_id], compute_state
                            )
                            emitted = receiver.emit(
                                inputs[node_id],
                                computed,
                                probability_by_node[node_id],
                            )
                            if node_id in terminals:
                                terminal_messages.setdefault(event_key, []).append(
                                    (node_id, emitted)
                                )
                        for edge in outgoing[node_id]:
                            key = (batch_index, token_index, edge.edge_id)
                            if key in edge_values:
                                raise AssertionError(
                                    f"fixed edge {edge.edge_id!r} settled more than once"
                                )
                            edge_values[key] = emitted if active else None
                        if record_trace:
                            node_events.append(
                                NodeEventTrace(
                                    sequence_id=sequence_id,
                                    token_position=position,
                                    region_id=region.region_id,
                                    node_id=node_id,
                                    reached=reached,
                                    observed=observed,
                                    active=active,
                                    input_hidden=inputs.get(node_id),
                                    normalized_input=normalized.get(node_id),
                                    state_before=state_before.get(node_id),
                                    proposal=proposal,
                                    state_for_compute=compute_state,
                                    selector_read=readouts.get(node_id),
                                    logit=logit_by_node.get(node_id),
                                    probability=probability_by_node.get(node_id),
                                    computed=computed,
                                    emitted=emitted,
                                    parent_messages=parents_by_node.get(node_id, ()),
                                )
                            )
                    if bool(prepared.route_mask[batch_index, token_index].item()):
                        balance.record(
                            region.region_id,
                            region.node_ids,
                            reached_ids,
                            probabilities
                            if probabilities is not None
                            else graph_input.new_empty((0,)),
                            active_ids,
                            forced_active=forced,
                        )

    output = hidden.clone()
    edge_events: List[EdgeEventTrace] = []
    output_events: List[OutputEventTrace] = []
    for batch_index, sequence_id in enumerate(prepared.sequence_ids):
        for token_index in range(length):
            if not bool(execution_mask[batch_index, token_index].item()):
                continue
            position = int(token_positions[batch_index, token_index].item())
            messages = terminal_messages.get((batch_index, token_index), [])
            if not messages:
                raise DynamicReachabilityError(
                    f"sequence {sequence_id!r} token {position} produced no "
                    "terminal message"
                )
            messages.sort(key=lambda pair: pair[0])
            event_output = _aggregate_output(
                model,
                [payload for _, payload in messages],
                [node_id for node_id, _ in messages],
            )
            output[batch_index, token_index] = event_output
            next_positions[sequence_id] = position + 1
            for edge in plan.edges:
                key = (batch_index, token_index, edge.edge_id)
                if key not in edge_values:
                    raise AssertionError(
                        f"fixed edge {edge.edge_id!r} was not settled"
                    )
                if record_trace:
                    payload = edge_values[key]
                    edge_events.append(
                        EdgeEventTrace(
                            sequence_id,
                            position,
                            edge.edge_id,
                            "DATA" if payload is not None else "CLOSED",
                            payload,
                        )
                    )
            if record_trace:
                output_events.append(
                    OutputEventTrace(
                        sequence_id, position, tuple(messages), event_output
                    )
                )

    result_state = StateStore(
        staged_values,
        dict(prepared.initial.selector_history),
        next_positions,
    )
    if detach_at_end:
        result_state = result_state.detached()
    trace = (
        _canonical_trace(
            plan,
            node_events,
            edge_events,
            boundary_events,
            region_events,
            state_writes,
            output_events,
        )
        if record_trace
        else None
    )
    return ExecutionResult(output, result_state, balance, trace)


def _aggregate_output(
    model: SettleGraph, messages: Sequence[Tensor], node_ids: Sequence[str]
) -> Tensor:
    stacked = torch.stack(tuple(messages), dim=0)
    if model.output_aggregate_type == "mean":
        return stacked.mean(dim=0)
    assert model.output_scores is not None
    scores = torch.stack(
        [model.output_scores[safe_module_key(node_id)] for node_id in node_ids]
    ).to(device=stacked.device, dtype=stacked.dtype)
    weights = torch.softmax(scores, dim=0)
    return (weights.unsqueeze(-1) * stacked).sum(dim=0)


def _canonical_trace(
    plan: Plan,
    node_events: Sequence[NodeEventTrace],
    edge_events: Sequence[EdgeEventTrace],
    boundary_events: Sequence[BoundaryEventTrace],
    region_events: Sequence[RegionEventTrace],
    state_writes: Sequence[StateWriteTrace],
    output_events: Sequence[OutputEventTrace],
) -> ExecutionTrace:
    """Normalize diagnostic order without using an eager executor helper."""

    rank = {
        region.region_id: index
        for index, region in enumerate(plan.topological_regions)
    }
    return ExecutionTrace(
        tuple(
            sorted(
                node_events,
                key=lambda event: (
                    event.sequence_id,
                    event.token_position,
                    rank[event.region_id],
                    event.region_id,
                    event.node_id,
                ),
            )
        ),
        tuple(
            sorted(
                edge_events,
                key=lambda event: (
                    event.sequence_id,
                    event.token_position,
                    event.edge_id,
                ),
            )
        ),
        tuple(
            sorted(
                boundary_events,
                key=lambda event: (
                    event.sequence_id,
                    event.token_position,
                    event.node_id,
                ),
            )
        ),
        tuple(
            sorted(
                region_events,
                key=lambda event: (
                    event.sequence_id,
                    event.token_position,
                    rank[event.region_id],
                    event.region_id,
                ),
            )
        ),
        tuple(
            sorted(
                state_writes,
                key=lambda event: (
                    event.sequence_id,
                    event.token_position,
                    event.owner_kind,
                    event.owner_id,
                ),
            )
        ),
        tuple(
            sorted(
                output_events,
                key=lambda event: (event.sequence_id, event.token_position),
            )
        ),
    )


__all__ = [
    "HB_LINE_V1",
    "SINGLE_LAYER_V1",
    "SPECIALIZATION_VERSIONS",
    "SpecializationSupport",
    "SpecializedExecutor",
    "hb_line_v1_support",
    "single_layer_v1_support",
    "specialization_support",
]

"""Eager reference execution for normalized SettleGraph Plans.

The interpreter is deliberately explicit: every fixed edge receives exactly
one logical settlement per valid token, and state changes are constructed in a
new :class:`StateStore`.  A failed token therefore cannot leak a partial state
commit to its caller.
"""

from __future__ import annotations

import dataclasses
import functools
import numbers
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from .ops import (
    AttentionState,
    OperationExecutionError,
    ReceiverModule,
    ReceiverState,
    RegionSelector,
    deterministic_topk_mask,
    safe_module_key,
)
from .plan import (
    Plan,
    ReferenceOperationConfigError,
    TypedPlan,
    validate_reference_operation_config,
    validate_stable_id,
)


StateKey = Tuple[str, str]


class ExecutionContractError(ValueError):
    """Runtime input contradicts the Plan or sequence-state contract."""


class DynamicReachabilityError(RuntimeError):
    """A valid token produced no active terminal message."""


class UnsupportedPlanError(ExecutionContractError):
    """A valid logical Plan is outside this executor's declared capability."""


class LocalOperationError(RuntimeError):
    """A validated local formula explicitly failed during an event."""


def _translate_local_operation_errors(operation: Any) -> Any:
    """Keep the local-formula failure boundary narrow and executor-owned."""

    @functools.wraps(operation)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            return operation(*args, **kwargs)
        except OperationExecutionError as exc:
            raise LocalOperationError(str(exc)) from exc

    return wrapped


@dataclasses.dataclass(frozen=True)
class StateStore:
    """Persistent receiver state and each sequence's next required position."""

    values: Mapping[StateKey, ReceiverState] = dataclasses.field(default_factory=dict)
    selector_history: Mapping[Tuple[str, str], Any] = dataclasses.field(default_factory=dict)
    next_position: Mapping[str, int] = dataclasses.field(default_factory=dict)

    def detached(self) -> "StateStore":
        return StateStore(
            values={key: _detach_state(value) for key, value in self.values.items()},
            selector_history={
                key: _detach_value(value) for key, value in self.selector_history.items()
            },
            next_position=dict(self.next_position),
        )

    def reset(self, sequence_ids: Iterable[str]) -> "StateStore":
        reset_ids = set(
            _validate_stable_id_sequence("reset sequence", sequence_ids)
        )
        return StateStore(
            values={key: value for key, value in self.values.items() if key[0] not in reset_ids},
            selector_history={
                key: value
                for key, value in self.selector_history.items()
                if key[0] not in reset_ids
            },
            next_position={
                key: value for key, value in self.next_position.items() if key not in reset_ids
            },
        )


@dataclasses.dataclass(frozen=True)
class NodeEventTrace:
    sequence_id: str
    token_position: int
    region_id: str
    node_id: str
    reached: bool
    observed: bool
    active: bool
    input_hidden: Optional[Tensor]
    normalized_input: Optional[Tensor]
    state_before: ReceiverState
    proposal: ReceiverState
    state_for_compute: ReceiverState
    selector_read: Optional[Tensor]
    logit: Optional[Tensor]
    probability: Optional[Tensor]
    computed: Optional[Tensor]
    emitted: Optional[Tensor]
    parent_messages: Tuple[Tuple[str, str, Optional[Tensor]], ...] = ()


@dataclasses.dataclass(frozen=True)
class EdgeEventTrace:
    sequence_id: str
    token_position: int
    edge_id: str
    status: str
    payload: Optional[Tensor]


@dataclasses.dataclass(frozen=True)
class BoundaryEventTrace:
    sequence_id: str
    token_position: int
    node_id: str
    payload: Tensor


@dataclasses.dataclass(frozen=True)
class RegionEventTrace:
    sequence_id: str
    token_position: int
    region_id: str
    candidate_node_ids: Tuple[str, ...]
    logits: Optional[Tensor]
    probabilities: Optional[Tensor]
    requested_k: Optional[int]
    effective_k: Optional[int]
    active_node_ids: Tuple[str, ...]
    forced_active: bool
    top_k_node_ids: Optional[Tuple[str, ...]]


@dataclasses.dataclass(frozen=True)
class StateWriteTrace:
    sequence_id: str
    token_position: int
    owner_kind: str
    owner_id: str
    value: Any


@dataclasses.dataclass(frozen=True)
class OutputEventTrace:
    sequence_id: str
    token_position: int
    terminal_messages: Tuple[Tuple[str, Tensor], ...]
    output: Tensor


@dataclasses.dataclass
class BalanceRegionStats:
    """Mergeable sufficient statistics for one site/region window."""

    soft_sum: Tensor
    availability_sum: Tensor
    hard_share_sum: Tensor
    event_count: int = 0
    competition_count: int = 0
    forced_active: bool = False

    @property
    def had_competition(self) -> bool:
        return self.competition_count > 0

    def merge(self, other: "BalanceRegionStats") -> "BalanceRegionStats":
        if self.soft_sum.shape != other.soft_sum.shape:
            raise ExecutionContractError("cannot merge balance stats with different node sets")
        return BalanceRegionStats(
            self.soft_sum + other.soft_sum,
            self.availability_sum + other.availability_sum,
            self.hard_share_sum + other.hard_share_sum,
            self.event_count + other.event_count,
            self.competition_count + other.competition_count,
            self.forced_active or other.forced_active,
        )


@dataclasses.dataclass
class BalanceStats:
    """Differentiable soft sums plus detached routing reference sums."""

    regions: MutableMapping[str, BalanceRegionStats] = dataclasses.field(default_factory=dict)
    zero: Optional[Tensor] = None

    def record(
        self,
        region_id: str,
        static_node_ids: Sequence[str],
        candidate_node_ids: Sequence[str],
        probabilities: Tensor,
        active_node_ids: Sequence[str],
        *,
        forced_active: bool,
    ) -> None:
        if not candidate_node_ids:
            return
        index = {node_id: offset for offset, node_id in enumerate(candidate_node_ids)}
        active = set(active_node_ids)
        zero = probabilities.new_zeros(())
        soft = torch.stack(
            [probabilities[index[node_id]] if node_id in index else zero for node_id in static_node_ids]
        )
        candidate_count = len(candidate_node_ids)
        availability = probabilities.new_tensor(
            [1.0 / candidate_count if node_id in index else 0.0 for node_id in static_node_ids]
        ).detach()
        active_count = len(active_node_ids)
        hard = probabilities.new_tensor(
            [1.0 / active_count if node_id in active else 0.0 for node_id in static_node_ids]
        ).detach()
        event = BalanceRegionStats(
            soft,
            availability,
            hard,
            event_count=1,
            competition_count=int(candidate_count >= 2),
            forced_active=forced_active,
        )
        previous = self.regions.get(region_id)
        self.regions[region_id] = event if previous is None else previous.merge(event)

    def merge(self, other: "BalanceStats") -> "BalanceStats":
        zero = self.zero if self.zero is not None else other.zero
        result = BalanceStats(dict(self.regions), zero)
        for region_id, stats in other.regions.items():
            previous = result.regions.get(region_id)
            result.regions[region_id] = stats if previous is None else previous.merge(stats)
        return result

    def loss(self) -> Tensor:
        competitive = [
            stats
            for stats in self.regions.values()
            if stats.event_count > 0 and stats.had_competition and not stats.forced_active
        ]
        if not competitive:
            if self.regions:
                reference = next(iter(self.regions.values())).soft_sum
                return reference.sum() * 0.0
            if self.zero is not None:
                return self.zero
            # A caller that constructs an entirely empty standalone accumulator
            # has supplied no dtype/device reference.  CPU default is the only
            # meaningful representation in that degenerate API case.
            return torch.tensor(0.0)
        losses = []
        for stats in competitive:
            mean_soft = stats.soft_sum / stats.event_count
            mean_availability = stats.availability_sum / stats.event_count
            losses.append((mean_soft - mean_availability).square().mean())
        return torch.stack(losses).mean()


@dataclasses.dataclass(frozen=True)
class ExecutionTrace:
    node_events: Tuple[NodeEventTrace, ...]
    edge_events: Tuple[EdgeEventTrace, ...]
    boundary_events: Tuple[BoundaryEventTrace, ...] = ()
    region_events: Tuple[RegionEventTrace, ...] = ()
    state_writes: Tuple[StateWriteTrace, ...] = ()
    output_events: Tuple[OutputEventTrace, ...] = ()


@dataclasses.dataclass(frozen=True)
class ExecutionResult:
    output: Tensor
    state: StateStore
    balance_stats: BalanceStats
    trace: Optional[ExecutionTrace]

    @property
    def balance_loss(self) -> Tensor:
        return self.balance_stats.loss()


class SettleGraph(nn.Module):
    """A Plan-bound collection of local operations and reference executors."""

    def __init__(self, plan: Union[Plan, TypedPlan]) -> None:
        super().__init__()
        if isinstance(plan, TypedPlan):
            self.typed_plan: Optional[TypedPlan] = plan.validate()
            logical_plan = plan.logical_plan
        else:
            self.typed_plan = None
            logical_plan = plan
        self.plan = logical_plan.validate()
        _validate_reference_plan_capability(self.plan)
        if self.typed_plan is not None:
            core_dtypes = {
                self.typed_plan.binding.dtype_roles[role]
                for role in ("hidden", "parameter", "state", "readout")
            }
            if len(core_dtypes) != 1:
                raise UnsupportedPlanError(
                    "the eager reference executor currently requires hidden, "
                    "parameter, state, and readout roles to share one dtype"
                )
            concrete_dtype = next(iter(core_dtypes))
            if concrete_dtype not in {"float32", "float64"}:
                raise UnsupportedPlanError(
                    "the eager reference executor has no closed accumulation "
                    f"contract for dtype {concrete_dtype!r}; only float32 and "
                    "float64 are implemented"
                )
        self.receivers = nn.ModuleDict()
        for node in self.plan.nodes:
            self.receivers[safe_module_key(node.node_id)] = ReceiverModule(
                self.plan.d_model, node
            )

        self.selectors = nn.ModuleDict()
        for region in self.plan.regions:
            dims = {
                self.receiver(node_id).selector_read_dim for node_id in region.node_ids
            }
            if len(dims) != 1:
                raise ExecutionContractError(
                    f"region {region.region_id!r} has incompatible selector read dimensions"
                )
            self.selectors[safe_module_key(region.region_id)] = RegionSelector(
                next(iter(dims)), region.score, region.node_ids
            )

        for node in self.plan.nodes:
            parent_ids = [edge.edge_id for edge in self.plan.edges if edge.target == node.node_id]
            if node.node_id in self.plan.entry_node_ids:
                parent_ids = [f"boundary:{node.node_id}"]
            self.receiver(node.node_id).ensure_edge_transforms(parent_ids)

        output_type = _operation_type(self.plan.output_aggregate, "mean")
        self.output_aggregate_type = output_type
        if output_type in {"learned_convex", "node_softmax"}:
            self.output_score = None
            self.output_scores = nn.ParameterDict(
                {
                    safe_module_key(node_id): nn.Parameter(torch.zeros(()))
                    for node_id in self.plan.terminal_node_ids
                }
            )
        elif output_type == "mean":
            self.output_score = None
            self.output_scores = None
        else:
            raise ExecutionContractError(
                f"unsupported output Aggregate type: {output_type}"
            )

    def receiver(self, node_id: str) -> ReceiverModule:
        return self.receivers[safe_module_key(node_id)]

    def selector(self, region_id: str) -> RegionSelector:
        return self.selectors[safe_module_key(region_id)]

    def make_identity(self) -> None:
        """Apply the standard constant-preserving identity construction."""

        for receiver in self.receivers.values():
            receiver.make_identity()
            if receiver.aggregate_type == "edge_linear_mean":
                assert receiver.edge_transforms is not None
                for transform in receiver.edge_transforms.values():
                    nn.init.eye_(transform.weight)
                    if transform.bias is not None:
                        nn.init.zeros_(transform.bias)

    @_translate_local_operation_errors
    def interpret_token(
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
        record_trace: bool = False,
    ) -> ExecutionResult:
        """Interpret one position for a batch of independent stable sequences."""

        _validate_token_inputs(
            self.plan,
            hidden,
            execution_mask,
            sequence_ids,
            token_positions,
            lm_target_mask=lm_target_mask,
            routing_stats_mask=routing_stats_mask,
        )
        sequence_keys = _validate_stable_id_sequence(
            "sequence", sequence_ids
        )
        if len(set(sequence_keys)) != len(sequence_keys):
            raise ExecutionContractError(
                "sequence_id values must be unique within one call"
            )
        reset_ids = _validate_stable_id_sequence(
            "reset sequence", reset_sequence_ids
        )
        if len(set(reset_ids)) != len(reset_ids):
            raise ExecutionContractError("reset sequence IDs must be unique")
        store = state if state is not None else StateStore()
        _validate_model_and_state(self, hidden, store)
        store = store.reset(reset_ids)
        _validate_requested_k_token(self.plan, requested_k, len(sequence_keys))
        _validate_position_rows(store, execution_mask, sequence_keys, token_positions)
        route_mask = (
            routing_stats_mask if routing_stats_mask is not None else execution_mask
        )

        next_values: Dict[StateKey, ReceiverState] = dict(store.values)
        next_history = dict(store.selector_history)
        next_positions = dict(store.next_position)
        outputs: List[Tensor] = []
        balance = BalanceStats(zero=hidden.new_zeros(()))
        node_traces: List[NodeEventTrace] = []
        edge_traces: List[EdgeEventTrace] = []
        boundary_traces: List[BoundaryEventTrace] = []
        region_traces: List[RegionEventTrace] = []
        state_write_traces: List[StateWriteTrace] = []
        output_traces: List[OutputEventTrace] = []

        for batch_index, sequence_id in enumerate(sequence_keys):
            position = int(token_positions[batch_index].item())
            if not bool(execution_mask[batch_index].item()):
                outputs.append(hidden[batch_index])
                continue
            sample_values = dict(next_values)
            (
                output,
                sample_balance,
                sample_nodes,
                sample_edges,
                sample_boundaries,
                sample_regions,
                sample_state_writes,
                sample_outputs,
            ) = self._interpret_sample(
                hidden[batch_index],
                sequence_id,
                position,
                sample_values,
                requested_k=requested_k,
                batch_index=batch_index,
                collect_balance=bool(route_mask[batch_index].item()),
                record_trace=record_trace,
            )
            # State and next-position commits happen only after a terminal output exists.
            next_values = sample_values
            next_positions[sequence_id] = position + 1
            outputs.append(output)
            balance = balance.merge(sample_balance)
            node_traces.extend(sample_nodes)
            edge_traces.extend(sample_edges)
            boundary_traces.extend(sample_boundaries)
            region_traces.extend(sample_regions)
            state_write_traces.extend(sample_state_writes)
            output_traces.extend(sample_outputs)

        result_state = StateStore(next_values, next_history, next_positions)
        trace = (
            _build_trace(
                self.plan,
                node_traces,
                edge_traces,
                boundary_traces,
                region_traces,
                state_write_traces,
                output_traces,
            )
            if record_trace
            else None
        )
        return ExecutionResult(torch.stack(outputs, dim=0), result_state, balance, trace)

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
        """Reference token-major prefill with mergeable auxiliary statistics."""

        if hidden.ndim != 3:
            raise ExecutionContractError("prefill hidden must have shape [B,T,d_model]")
        batch, length, width = hidden.shape
        if width != self.plan.d_model:
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
            "routing_stats_mask", routing_stats_mask, execution_mask, (batch, length)
        )
        sequence_keys = _validate_stable_id_sequence(
            "sequence", sequence_ids
        )
        if len(set(sequence_keys)) != len(sequence_keys):
            raise ExecutionContractError("sequence_id values must be unique within one call")
        reset_ids = _validate_stable_id_sequence(
            "reset sequence", reset_sequence_ids
        )
        if len(set(reset_ids)) != len(reset_ids):
            raise ExecutionContractError("reset sequence IDs must be unique")
        initial = state if state is not None else StateStore()
        _validate_model_and_state(self, hidden, initial)
        initial = initial.reset(reset_ids)
        _validate_prefill_requested_k(self.plan, requested_k, batch, length)
        _validate_position_matrix(
            initial, execution_mask, sequence_keys, token_positions
        )

        current = initial
        outputs: List[Tensor] = []
        balance = BalanceStats(zero=hidden.new_zeros(()))
        node_traces: List[NodeEventTrace] = []
        edge_traces: List[EdgeEventTrace] = []
        boundary_traces: List[BoundaryEventTrace] = []
        region_traces: List[RegionEventTrace] = []
        state_write_traces: List[StateWriteTrace] = []
        output_traces: List[OutputEventTrace] = []
        for token_index in range(length):
            token_k: Dict[str, Sequence[int]] = {}
            if requested_k is not None:
                for region_id, values in requested_k.items():
                    if values.shape != (batch, length):
                        raise ExecutionContractError(
                            f"requested_k[{region_id!r}] must have shape [B,T]"
                        )
                    token_k[region_id] = [int(v) for v in values[:, token_index].tolist()]
            token_result = self.interpret_token(
                hidden[:, token_index],
                execution_mask[:, token_index],
                sequence_ids,
                token_positions[:, token_index],
                state=current,
                requested_k=token_k,
                lm_target_mask=(
                    lm_target_mask[:, token_index]
                    if lm_target_mask is not None
                    else None
                ),
                routing_stats_mask=(
                    routing_stats_mask[:, token_index]
                    if routing_stats_mask is not None
                    else None
                ),
                record_trace=record_trace,
            )
            current = token_result.state
            outputs.append(token_result.output)
            balance = balance.merge(token_result.balance_stats)
            if token_result.trace is not None:
                node_traces.extend(token_result.trace.node_events)
                edge_traces.extend(token_result.trace.edge_events)
                boundary_traces.extend(token_result.trace.boundary_events)
                region_traces.extend(token_result.trace.region_events)
                state_write_traces.extend(token_result.trace.state_writes)
                output_traces.extend(token_result.trace.output_events)
        if detach_at_end:
            current = current.detached()
        trace = (
            _build_trace(
                self.plan,
                node_traces,
                edge_traces,
                boundary_traces,
                region_traces,
                state_write_traces,
                output_traces,
            )
            if record_trace
            else None
        )
        output = torch.stack(outputs, dim=1) if outputs else hidden
        return ExecutionResult(output, current, balance, trace)

    @_translate_local_operation_errors
    def prefill_region_major(
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
        """Execute a generic eager prefill in canonical region-major order.

        This is an independent scheduling reference, not a call-through to
        :meth:`interpret_token`.  Each region is settled for the complete
        ``[B,T]`` input before its dependent regions begin.  Within one region,
        Token positions are visited in increasing input order so every
        ``(sequence_id, node_id)`` state observes the same causal history as the
        token-major interpreter.

        All mutable values are staged in private mappings.  Outputs, routing
        statistics, state, and positions become observable only if every
        execution position produces a terminal message.
        """

        if hidden.ndim != 3:
            raise ExecutionContractError("prefill hidden must have shape [B,T,d_model]")
        batch, length, width = hidden.shape
        if width != self.plan.d_model:
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
            "routing_stats_mask", routing_stats_mask, execution_mask, (batch, length)
        )

        sequence_keys = _validate_stable_id_sequence(
            "sequence", sequence_ids
        )
        if len(set(sequence_keys)) != len(sequence_keys):
            raise ExecutionContractError("sequence_id values must be unique within one call")
        reset_ids = _validate_stable_id_sequence(
            "reset sequence", reset_sequence_ids
        )
        if len(set(reset_ids)) != len(reset_ids):
            raise ExecutionContractError("reset sequence IDs must be unique")

        public_state = state if state is not None else StateStore()
        _validate_model_and_state(self, hidden, public_state)
        initial = public_state.reset(reset_ids)
        _validate_prefill_requested_k(self.plan, requested_k, batch, length)
        _validate_position_matrix(initial, execution_mask, sequence_keys, token_positions)

        route_mask = (
            routing_stats_mask if routing_stats_mask is not None else execution_mask
        )
        staged_values: Dict[StateKey, ReceiverState] = dict(initial.values)
        staged_history = dict(initial.selector_history)
        staged_next_positions = dict(initial.next_position)

        incoming = _incoming_edges(self.plan)
        outgoing = _outgoing_edges(self.plan)
        entry_ids = set(self.plan.entry_node_ids)
        terminal_ids = set(self.plan.terminal_node_ids)

        # Edge settlements and terminal messages are indexed by their public
        # batch row and Token column.  They are private until the complete call
        # has passed its terminal-reachability checks.
        edge_values: Dict[Tuple[int, int, str], Optional[Tensor]] = {}
        terminal_messages: Dict[Tuple[int, int], List[Tuple[str, Tensor]]] = {}
        balance = BalanceStats(zero=hidden.new_zeros(()))

        node_traces: List[NodeEventTrace] = []
        edge_traces: List[EdgeEventTrace] = []
        boundary_traces: List[BoundaryEventTrace] = []
        region_traces: List[RegionEventTrace] = []
        state_write_traces: List[StateWriteTrace] = []
        output_traces: List[OutputEventTrace] = []

        # The region loop is intentionally outermost.  Independent rows share
        # parameters but never mutable receiver state because sequence IDs are
        # unique within the call.
        for region in self.plan.topological_regions:
            timing = region.selector_timing.lower().replace("-", "_")
            if timing not in {"content", "pre", "post"}:
                raise ExecutionContractError(
                    f"unsupported selector timing {region.selector_timing!r}"
                )
            profile = region.profile.upper()
            if profile not in {"N", "SD", "BO"}:
                raise ExecutionContractError(f"unsupported profile {region.profile!r}")

            for token_index in range(length):
                for batch_index, sequence_id in enumerate(sequence_keys):
                    if not bool(execution_mask[batch_index, token_index].item()):
                        continue

                    graph_input = hidden[batch_index, token_index]
                    token_position = int(
                        token_positions[batch_index, token_index].item()
                    )
                    event_key = (batch_index, token_index)
                    reached_ids: List[str] = []
                    hidden_by_node: Dict[str, Tensor] = {}
                    normalized_by_node: Dict[str, Tensor] = {}
                    state_before_by_node: Dict[str, ReceiverState] = {}
                    proposal_by_node: Dict[str, ReceiverState] = {}
                    readout_by_node: Dict[str, Tensor] = {}
                    parents_by_node: Dict[
                        str, Tuple[Tuple[str, str, Optional[Tensor]], ...]
                    ] = {}

                    for node_id in region.node_ids:
                        node = self.receiver(node_id)
                        if node_id in entry_ids:
                            messages = [graph_input]
                            edge_ids = [f"boundary:{node_id}"]
                            parents_by_node[node_id] = (
                                (f"boundary:{node_id}", "DATA", graph_input),
                            )
                            if record_trace:
                                boundary_traces.append(
                                    BoundaryEventTrace(
                                        sequence_id,
                                        token_position,
                                        node_id,
                                        graph_input,
                                    )
                                )
                        else:
                            parents = incoming[node_id]
                            parent_keys = [
                                (batch_index, token_index, edge.edge_id)
                                for edge in parents
                            ]
                            if any(key not in edge_values for key in parent_keys):
                                raise AssertionError(
                                    f"region order reached {node_id!r} before all "
                                    "parent edges settled"
                                )
                            data = [
                                (edge.edge_id, edge_values[key])
                                for edge, key in zip(parents, parent_keys)
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
                                for edge, key in zip(parents, parent_keys)
                            )

                        if not messages:
                            continue
                        reached_ids.append(node_id)
                        aggregate = node.aggregate(messages, edge_ids)
                        normalized = node.normalize_input(aggregate)
                        state_before = staged_values.get(
                            (sequence_id, node_id), node.initial_state(normalized)
                        )
                        hidden_by_node[node_id] = aggregate
                        normalized_by_node[node_id] = normalized
                        state_before_by_node[node_id] = state_before

                    forced = bool(reached_ids) and all(
                        self.plan.node_by_id(node_id).forced_active
                        for node_id in region.node_ids
                    )
                    requested_for_event: Optional[int] = None
                    if reached_ids and not forced:
                        # Resolve this non-differentiable control before any
                        # selector Read or Score for the event.
                        requested_for_event = _resolve_prefill_requested_k(
                            region,
                            requested_k,
                            batch_index,
                            token_index,
                        )

                    active_ids: List[str] = []
                    probabilities = graph_input.new_empty((0,))
                    logits: Optional[Tensor] = None
                    effective_k: Optional[int] = None
                    top_k_node_ids: Optional[Tuple[str, ...]] = None
                    if reached_ids:
                        if forced:
                            if len(region.node_ids) != 1:
                                raise ExecutionContractError(
                                    "forced-active regions must be singleton regions"
                                )
                            probabilities = graph_input.new_ones((1,))
                            active_ids = list(reached_ids)
                        else:
                            for node_id in reached_ids:
                                node = self.receiver(node_id)
                                state_for_selector: ReceiverState
                                if timing == "post":
                                    proposal = node.proposal(
                                        state_before_by_node[node_id],
                                        normalized_by_node[node_id],
                                        token_position=token_position,
                                    )
                                    proposal_by_node[node_id] = proposal
                                    state_for_selector = proposal
                                elif timing == "pre":
                                    state_for_selector = state_before_by_node[node_id]
                                else:
                                    state_for_selector = None
                                readout_by_node[node_id] = node.selector_read(
                                    normalized_by_node[node_id], state_for_selector
                                )
                            readouts = torch.stack(
                                [readout_by_node[node_id] for node_id in reached_ids]
                            )
                            logits = self.selector(region.region_id)(
                                readouts, reached_ids
                            )
                            if not bool(torch.isfinite(logits).all().item()):
                                raise ExecutionContractError(
                                    f"region {region.region_id!r} produced non-finite logits"
                                )
                            probabilities = torch.softmax(logits, dim=0)
                            assert requested_for_event is not None
                            effective_k = min(
                                requested_for_event, len(reached_ids)
                            )
                            active_mask = deterministic_topk_mask(
                                logits, effective_k
                            )
                            active_ids = [
                                node_id
                                for offset, node_id in enumerate(reached_ids)
                                if bool(active_mask[offset].item())
                            ]
                            top_k_node_ids = tuple(active_ids)

                    if record_trace:
                        region_traces.append(
                            RegionEventTrace(
                                sequence_id=sequence_id,
                                token_position=token_position,
                                region_id=region.region_id,
                                candidate_node_ids=tuple(reached_ids),
                                logits=logits,
                                probabilities=(
                                    probabilities if reached_ids else None
                                ),
                                requested_k=requested_for_event,
                                effective_k=effective_k,
                                active_node_ids=tuple(active_ids),
                                forced_active=forced,
                                top_k_node_ids=top_k_node_ids,
                            )
                        )

                    if profile == "N":
                        observe_ids: set[str] = set()
                    elif profile == "SD":
                        observe_ids = set(active_ids)
                    else:
                        observe_ids = set(reached_ids)

                    probability_by_node = {
                        node_id: probabilities[index]
                        for index, node_id in enumerate(reached_ids)
                    }
                    logit_by_node = (
                        {
                            node_id: logits[index]
                            for index, node_id in enumerate(reached_ids)
                        }
                        if logits is not None
                        else {}
                    )
                    active_set = set(active_ids)
                    for node_id in region.node_ids:
                        reached = node_id in hidden_by_node
                        observed = node_id in observe_ids
                        active = node_id in active_set
                        proposal: ReceiverState = proposal_by_node.get(node_id)
                        state_for_compute: ReceiverState = state_before_by_node.get(
                            node_id
                        )
                        computed: Optional[Tensor] = None
                        emitted: Optional[Tensor] = None

                        if reached and observed:
                            if node_id not in proposal_by_node:
                                proposal = self.receiver(node_id).proposal(
                                    state_before_by_node[node_id],
                                    normalized_by_node[node_id],
                                    token_position=token_position,
                                )
                                proposal_by_node[node_id] = proposal
                            state_for_compute = proposal
                            if proposal is not None:
                                staged_values[(sequence_id, node_id)] = proposal
                                if record_trace:
                                    state_write_traces.append(
                                        StateWriteTrace(
                                            sequence_id,
                                            token_position,
                                            "receiver",
                                            node_id,
                                            proposal,
                                        )
                                    )

                        if active:
                            computed = self.receiver(node_id).compute(
                                hidden_by_node[node_id],
                                normalized_by_node[node_id],
                                state_for_compute,
                            )
                            emitted = self.receiver(node_id).emit(
                                hidden_by_node[node_id],
                                computed,
                                probability_by_node[node_id],
                            )
                            if node_id in terminal_ids:
                                terminal_messages.setdefault(event_key, []).append(
                                    (node_id, emitted)
                                )

                        for edge in outgoing[node_id]:
                            edge_key = (batch_index, token_index, edge.edge_id)
                            if edge_key in edge_values:
                                raise AssertionError(
                                    f"fixed edge {edge.edge_id!r} settled more than once"
                                )
                            edge_values[edge_key] = emitted if active else None

                        if record_trace:
                            node_traces.append(
                                NodeEventTrace(
                                    sequence_id,
                                    token_position,
                                    region.region_id,
                                    node_id,
                                    reached,
                                    observed,
                                    active,
                                    hidden_by_node.get(node_id),
                                    normalized_by_node.get(node_id),
                                    state_before_by_node.get(node_id),
                                    proposal,
                                    state_for_compute,
                                    readout_by_node.get(node_id),
                                    logit_by_node.get(node_id),
                                    probability_by_node.get(node_id),
                                    computed,
                                    emitted,
                                    parents_by_node.get(node_id, ()),
                                )
                            )

                    if bool(route_mask[batch_index, token_index].item()):
                        balance.record(
                            region.region_id,
                            region.node_ids,
                            reached_ids,
                            probabilities,
                            active_ids,
                            forced_active=forced,
                        )

        output_rows: List[Tensor] = []
        for batch_index, sequence_id in enumerate(sequence_keys):
            row_outputs: List[Tensor] = []
            for token_index in range(length):
                if not bool(execution_mask[batch_index, token_index].item()):
                    row_outputs.append(hidden[batch_index, token_index])
                    continue

                token_position = int(
                    token_positions[batch_index, token_index].item()
                )
                event_key = (batch_index, token_index)
                messages = terminal_messages.get(event_key, [])
                if not messages:
                    raise DynamicReachabilityError(
                        f"sequence {sequence_id!r} token {token_position} "
                        "produced no terminal message"
                    )
                messages.sort(key=lambda pair: pair[0])
                output = self._aggregate_output(
                    [message for _, message in messages],
                    [node_id for node_id, _ in messages],
                )
                row_outputs.append(output)
                staged_next_positions[sequence_id] = token_position + 1

                event_edge_keys = {
                    (batch_index, token_index, edge.edge_id)
                    for edge in self.plan.edges
                }
                missing = sorted(
                    key[2] for key in event_edge_keys if key not in edge_values
                )
                if missing:
                    raise AssertionError(f"fixed edges were not settled: {missing}")

                if record_trace:
                    edge_traces.extend(
                        EdgeEventTrace(
                            sequence_id,
                            token_position,
                            edge.edge_id,
                            "DATA"
                            if edge_values[
                                (batch_index, token_index, edge.edge_id)
                            ]
                            is not None
                            else "CLOSED",
                            edge_values[(batch_index, token_index, edge.edge_id)],
                        )
                        for edge in self.plan.edges
                    )
                    output_traces.append(
                        OutputEventTrace(
                            sequence_id,
                            token_position,
                            tuple(messages),
                            output,
                        )
                    )
            output_rows.append(
                torch.stack(row_outputs, dim=0) if row_outputs else hidden[batch_index]
            )

        result_state = StateStore(
            staged_values, staged_history, staged_next_positions
        )
        if detach_at_end:
            result_state = result_state.detached()
        trace = (
            _build_trace(
                self.plan,
                node_traces,
                edge_traces,
                boundary_traces,
                region_traces,
                state_write_traces,
                output_traces,
            )
            if record_trace
            else None
        )
        output = torch.stack(output_rows, dim=0) if output_rows else hidden
        return ExecutionResult(output, result_state, balance, trace)

    def _interpret_sample(
        self,
        graph_input: Tensor,
        sequence_id: str,
        token_position: int,
        states: MutableMapping[StateKey, ReceiverState],
        *,
        requested_k: Optional[Mapping[str, Sequence[int]]],
        batch_index: int,
        collect_balance: bool,
        record_trace: bool,
    ) -> Tuple[
        Tensor,
        BalanceStats,
        List[NodeEventTrace],
        List[EdgeEventTrace],
        List[BoundaryEventTrace],
        List[RegionEventTrace],
        List[StateWriteTrace],
        List[OutputEventTrace],
    ]:
        edge_values: Dict[str, Optional[Tensor]] = {}
        terminal_messages: List[Tuple[str, Tensor]] = []
        balance = BalanceStats(zero=graph_input.new_zeros(()))
        node_traces: List[NodeEventTrace] = []
        boundary_traces: List[BoundaryEventTrace] = []
        region_traces: List[RegionEventTrace] = []
        state_write_traces: List[StateWriteTrace] = []
        output_traces: List[OutputEventTrace] = []

        incoming = _incoming_edges(self.plan)
        outgoing = _outgoing_edges(self.plan)
        terminal_ids = set(self.plan.terminal_node_ids)
        entry_ids = set(self.plan.entry_node_ids)

        if record_trace:
            boundary_traces.extend(
                BoundaryEventTrace(
                    sequence_id, token_position, node_id, graph_input
                )
                for node_id in self.plan.entry_node_ids
            )

        for region in self.plan.topological_regions:
            reached_ids: List[str] = []
            hidden_by_node: Dict[str, Tensor] = {}
            normalized_by_node: Dict[str, Tensor] = {}
            state_before_by_node: Dict[str, ReceiverState] = {}
            proposal_by_node: Dict[str, ReceiverState] = {}
            readout_by_node: Dict[str, Tensor] = {}
            parents_by_node: Dict[
                str, Tuple[Tuple[str, str, Optional[Tensor]], ...]
            ] = {}

            for node_id in region.node_ids:
                node = self.receiver(node_id)
                if node_id in entry_ids:
                    messages = [graph_input]
                    edge_ids = [f"boundary:{node_id}"]
                    parents_by_node[node_id] = (
                        (f"boundary:{node_id}", "DATA", graph_input),
                    )
                else:
                    parents = incoming[node_id]
                    if any(edge.edge_id not in edge_values for edge in parents):
                        raise AssertionError(
                            f"region order reached {node_id!r} before all parent edges settled"
                        )
                    data = [
                        (edge.edge_id, edge_values[edge.edge_id])
                        for edge in parents
                        if edge_values[edge.edge_id] is not None
                    ]
                    edge_ids = [edge_id for edge_id, _ in data]
                    messages = [value for _, value in data]
                    parents_by_node[node_id] = tuple(
                        (
                            edge.edge_id,
                            "DATA"
                            if edge_values[edge.edge_id] is not None
                            else "CLOSED",
                            edge_values[edge.edge_id],
                        )
                        for edge in parents
                    )
                if not messages:
                    continue
                reached_ids.append(node_id)
                aggregate = node.aggregate(messages, edge_ids)
                normalized = node.normalize_input(aggregate)
                state_before = states.get(
                    (sequence_id, node_id), node.initial_state(normalized)
                )
                hidden_by_node[node_id] = aggregate
                normalized_by_node[node_id] = normalized
                state_before_by_node[node_id] = state_before

            timing = region.selector_timing.lower().replace("-", "_")
            if timing not in {"content", "pre", "post"}:
                raise ExecutionContractError(
                    f"unsupported selector timing {region.selector_timing!r}"
                )
            forced = bool(reached_ids) and all(
                self.plan.node_by_id(node_id).forced_active
                for node_id in region.node_ids
            )
            requested_for_event: Optional[int] = None
            if reached_ids and not forced:
                # A runtime K is a non-differentiable control input.  Resolve
                # and validate it before selector Read, Score, or Top-K.
                requested_for_event = _resolve_requested_k(
                    region, requested_k, batch_index
                )
            active_ids: List[str] = []
            probabilities = graph_input.new_empty((0,))
            logits: Optional[Tensor] = None
            effective_k: Optional[int] = None
            top_k_node_ids: Optional[Tuple[str, ...]] = None
            if reached_ids:
                if forced:
                    if len(region.node_ids) != 1:
                        raise ExecutionContractError(
                            "forced-active regions must be singleton regions"
                        )
                    probabilities = graph_input.new_ones((1,))
                    active_ids = list(reached_ids)
                else:
                    for node_id in reached_ids:
                        node = self.receiver(node_id)
                        state_for_selector: ReceiverState
                        if timing == "post":
                            proposal = node.proposal(
                                state_before_by_node[node_id],
                                normalized_by_node[node_id],
                                token_position=token_position,
                            )
                            proposal_by_node[node_id] = proposal
                            state_for_selector = proposal
                        elif timing == "pre":
                            state_for_selector = state_before_by_node[node_id]
                        else:
                            state_for_selector = None
                        readout_by_node[node_id] = node.selector_read(
                            normalized_by_node[node_id], state_for_selector
                        )
                    readouts = torch.stack(
                        [readout_by_node[node_id] for node_id in reached_ids]
                    )
                    logits = self.selector(region.region_id)(readouts, reached_ids)
                    if not bool(torch.isfinite(logits).all().item()):
                        raise ExecutionContractError(
                            f"region {region.region_id!r} produced non-finite logits"
                        )
                    probabilities = torch.softmax(logits, dim=0)
                    assert requested_for_event is not None
                    effective_k = min(requested_for_event, len(reached_ids))
                    active_mask = deterministic_topk_mask(logits, effective_k)
                    active_ids = [
                        node_id
                        for offset, node_id in enumerate(reached_ids)
                        if bool(active_mask[offset].item())
                    ]
                    top_k_node_ids = tuple(active_ids)

            if record_trace:
                region_traces.append(
                    RegionEventTrace(
                        sequence_id=sequence_id,
                        token_position=token_position,
                        region_id=region.region_id,
                        candidate_node_ids=tuple(reached_ids),
                        logits=logits,
                        probabilities=probabilities if reached_ids else None,
                        requested_k=requested_for_event,
                        effective_k=effective_k,
                        active_node_ids=tuple(active_ids),
                        forced_active=forced,
                        top_k_node_ids=top_k_node_ids,
                    )
                )

            profile = region.profile.upper()
            if profile == "N":
                observe_ids: set[str] = set()
            elif profile == "SD":
                observe_ids = set(active_ids)
            elif profile == "BO":
                observe_ids = set(reached_ids)
            else:
                raise ExecutionContractError(f"unsupported profile {region.profile!r}")

            probability_by_node = {
                node_id: probabilities[index] for index, node_id in enumerate(reached_ids)
            }
            logit_by_node = (
                {
                    node_id: logits[index]
                    for index, node_id in enumerate(reached_ids)
                }
                if logits is not None
                else {}
            )
            active_set = set(active_ids)
            for node_id in region.node_ids:
                reached = node_id in hidden_by_node
                observed = node_id in observe_ids
                active = node_id in active_set
                proposal: ReceiverState = proposal_by_node.get(node_id)
                state_for_compute: ReceiverState = state_before_by_node.get(node_id)
                computed: Optional[Tensor] = None
                emitted: Optional[Tensor] = None
                if reached and observed:
                    if node_id not in proposal_by_node:
                        proposal = self.receiver(node_id).proposal(
                            state_before_by_node[node_id],
                            normalized_by_node[node_id],
                            token_position=token_position,
                        )
                        proposal_by_node[node_id] = proposal
                    state_for_compute = proposal
                    if proposal is not None:
                        states[(sequence_id, node_id)] = proposal
                        if record_trace:
                            state_write_traces.append(
                                StateWriteTrace(
                                    sequence_id,
                                    token_position,
                                    "receiver",
                                    node_id,
                                    proposal,
                                )
                            )
                if active:
                    computed = self.receiver(node_id).compute(
                        hidden_by_node[node_id],
                        normalized_by_node[node_id],
                        state_for_compute,
                    )
                    emitted = self.receiver(node_id).emit(
                        hidden_by_node[node_id],
                        computed,
                        probability_by_node[node_id],
                    )
                    if node_id in terminal_ids:
                        terminal_messages.append((node_id, emitted))
                for edge in outgoing[node_id]:
                    edge_values[edge.edge_id] = emitted if active else None
                if record_trace:
                    node_traces.append(
                        NodeEventTrace(
                            sequence_id,
                            token_position,
                            region.region_id,
                            node_id,
                            reached,
                            observed,
                            active,
                            hidden_by_node.get(node_id),
                            normalized_by_node.get(node_id),
                            state_before_by_node.get(node_id),
                            proposal,
                            state_for_compute,
                            readout_by_node.get(node_id),
                            logit_by_node.get(node_id),
                            probability_by_node.get(node_id),
                            computed,
                            emitted,
                            parents_by_node.get(node_id, ()),
                        )
                    )

            if collect_balance:
                balance.record(
                    region.region_id,
                    region.node_ids,
                    reached_ids,
                    probabilities,
                    active_ids,
                    forced_active=forced,
                )

        if not terminal_messages:
            raise DynamicReachabilityError(
                f"sequence {sequence_id!r} token {token_position} produced no terminal message"
            )
        terminal_messages.sort(key=lambda pair: pair[0])
        output = self._aggregate_output(
            [message for _, message in terminal_messages],
            [node_id for node_id, _ in terminal_messages],
        )
        if record_trace:
            output_traces.append(
                OutputEventTrace(
                    sequence_id,
                    token_position,
                    tuple(terminal_messages),
                    output,
                )
            )
        edge_traces = [
            EdgeEventTrace(
                sequence_id,
                token_position,
                edge.edge_id,
                "DATA" if edge_values.get(edge.edge_id) is not None else "CLOSED",
                edge_values.get(edge.edge_id),
            )
            for edge in self.plan.edges
        ] if record_trace else []
        if len(edge_values) != len(self.plan.edges):
            missing = sorted({edge.edge_id for edge in self.plan.edges} - set(edge_values))
            raise AssertionError(f"fixed edges were not settled: {missing}")
        return (
            output,
            balance,
            node_traces,
            edge_traces,
            boundary_traces,
            region_traces,
            state_write_traces,
            output_traces,
        )

    def _aggregate_output(
        self, messages: Sequence[Tensor], node_ids: Sequence[str]
    ) -> Tensor:
        stacked = torch.stack(tuple(messages), dim=0)
        if self.output_aggregate_type == "mean":
            return stacked.mean(dim=0)
        assert self.output_scores is not None
        scores = torch.stack(
            [self.output_scores[safe_module_key(node_id)] for node_id in node_ids]
        ).to(device=stacked.device, dtype=stacked.dtype)
        weights = torch.softmax(scores, dim=0)
        return (weights.unsqueeze(-1) * stacked).sum(dim=0)


def _build_trace(
    plan: Plan,
    node_events: Sequence[NodeEventTrace],
    edge_events: Sequence[EdgeEventTrace],
    boundary_events: Sequence[BoundaryEventTrace],
    region_events: Sequence[RegionEventTrace],
    state_writes: Sequence[StateWriteTrace],
    output_events: Sequence[OutputEventTrace],
) -> ExecutionTrace:
    """Normalize trace order independently of executor completion order."""

    region_rank = {
        region.region_id: rank
        for rank, region in enumerate(plan.topological_regions)
    }
    boundaries = sorted(
        boundary_events,
        key=lambda event: (
            event.sequence_id,
            event.token_position,
            event.node_id,
        ),
    )
    regions = sorted(
        region_events,
        key=lambda event: (
            event.sequence_id,
            event.token_position,
            region_rank[event.region_id],
            event.region_id,
        ),
    )
    nodes = sorted(
        node_events,
        key=lambda event: (
            event.sequence_id,
            event.token_position,
            region_rank[event.region_id],
            event.region_id,
            event.node_id,
        ),
    )
    edges = sorted(
        edge_events,
        key=lambda event: (
            event.sequence_id,
            event.token_position,
            event.edge_id,
        ),
    )
    writes = sorted(
        state_writes,
        key=lambda event: (
            event.sequence_id,
            event.token_position,
            event.owner_kind,
            event.owner_id,
        ),
    )
    outputs = sorted(
        output_events,
        key=lambda event: (event.sequence_id, event.token_position),
    )
    return ExecutionTrace(
        tuple(nodes),
        tuple(edges),
        tuple(boundaries),
        tuple(regions),
        tuple(writes),
        tuple(outputs),
    )


def _incoming_edges(plan: Plan) -> Dict[str, List[Any]]:
    result = {node.node_id: [] for node in plan.nodes}
    for edge in plan.edges:
        result[edge.target].append(edge)
    for edges in result.values():
        edges.sort(key=lambda edge: edge.edge_id)
    return result


def _outgoing_edges(plan: Plan) -> Dict[str, List[Any]]:
    result = {node.node_id: [] for node in plan.nodes}
    for edge in plan.edges:
        result[edge.source].append(edge)
    for edges in result.values():
        edges.sort(key=lambda edge: edge.edge_id)
    return result


def _resolve_requested_k(
    region: Any,
    supplied: Optional[Mapping[str, Sequence[int]]],
    batch_index: int,
) -> int:
    config = region.k_requested
    kind = _operation_type(config, "fixed")
    if kind == "fixed":
        value = int(config.get("value", 1))
    elif kind == "input":
        if supplied is None or region.region_id not in supplied:
            raise ExecutionContractError(
                f"region {region.region_id!r} requires an explicit requested_k input"
            )
        values = supplied[region.region_id]
        if batch_index >= len(values):
            raise ExecutionContractError("requested_k input is shorter than the batch")
        value = int(values[batch_index])
    else:
        raise ExecutionContractError(f"unsupported requested K rule: {kind}")
    if not 1 <= value <= int(region.k_max):
        raise ExecutionContractError(
            f"requested K={value} for region {region.region_id!r} is outside [1,{region.k_max}]"
        )
    return value


def _resolve_prefill_requested_k(
    region: Any,
    supplied: Optional[Mapping[str, Tensor]],
    batch_index: int,
    token_index: int,
) -> int:
    """Resolve one region event's K from a validated prefill control."""

    config = region.k_requested
    kind = _operation_type(config, "fixed")
    if kind == "fixed":
        value = int(config.get("value", 1))
    elif kind == "input":
        if supplied is None or region.region_id not in supplied:
            raise ExecutionContractError(
                f"region {region.region_id!r} requires an explicit requested_k input"
            )
        value = int(supplied[region.region_id][batch_index, token_index].item())
    else:
        raise ExecutionContractError(f"unsupported requested K rule: {kind}")
    if not 1 <= value <= int(region.k_max):
        raise ExecutionContractError(
            f"requested K={value} for region {region.region_id!r} is outside "
            f"[1,{region.k_max}]"
        )
    return value


def _validate_token_inputs(
    plan: Plan,
    hidden: Tensor,
    execution_mask: Tensor,
    sequence_ids: Sequence[Any],
    token_positions: Tensor,
    *,
    lm_target_mask: Optional[Tensor],
    routing_stats_mask: Optional[Tensor],
) -> None:
    if hidden.ndim != 2 or hidden.shape[1] != plan.d_model:
        raise ExecutionContractError("token hidden must have shape [B,d_model]")
    batch = hidden.shape[0]
    if execution_mask.shape != (batch,):
        raise ExecutionContractError("token execution_mask must have shape [B]")
    if token_positions.shape != (batch,):
        raise ExecutionContractError("token_positions must have shape [B]")
    if len(sequence_ids) != batch:
        raise ExecutionContractError("sequence_ids length must equal batch size")
    if execution_mask.dtype != torch.bool:
        raise ExecutionContractError("execution_mask must have bool dtype")
    if token_positions.dtype not in _INTEGER_DTYPES:
        raise ExecutionContractError("token_positions must have an integer dtype")
    if not torch.is_floating_point(hidden):
        raise ExecutionContractError("hidden must have a floating dtype")
    _validate_optional_mask(
        "lm_target_mask", lm_target_mask, execution_mask, (batch,)
    )
    _validate_optional_mask(
        "routing_stats_mask", routing_stats_mask, execution_mask, (batch,)
    )


_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


def _validate_stable_id_sequence(
    kind: str, values: Iterable[Any]
) -> Tuple[str, ...]:
    """Validate runtime IDs without coercing caller-supplied objects."""

    validated: List[str] = []
    for value in values:
        try:
            validated.append(validate_stable_id(value, kind=kind))
        except ValueError as exc:
            raise ExecutionContractError(str(exc)) from exc
    return tuple(validated)


def _validate_optional_mask(
    name: str,
    mask: Optional[Tensor],
    execution_mask: Tensor,
    shape: Tuple[int, ...],
) -> None:
    if mask is None:
        return
    if mask.shape != shape:
        raise ExecutionContractError(f"{name} must have shape {list(shape)}")
    if mask.dtype != torch.bool:
        raise ExecutionContractError(f"{name} must have bool dtype")
    if bool((mask & ~execution_mask).any().item()):
        raise ExecutionContractError(f"{name} must be a subset of execution_mask")


def _validate_position_rows(
    store: StateStore,
    execution_mask: Tensor,
    sequence_ids: Sequence[str],
    token_positions: Tensor,
) -> None:
    for row, sequence_id in enumerate(sequence_ids):
        if not bool(execution_mask[row].item()):
            continue
        position = int(token_positions[row].item())
        expected = store.next_position.get(sequence_id, 0)
        if position != expected:
            raise ExecutionContractError(
                f"token_position {position} for sequence {sequence_id!r} must equal "
                f"the next required position {expected}"
            )


def _validate_position_matrix(
    store: StateStore,
    execution_mask: Tensor,
    sequence_ids: Sequence[str],
    token_positions: Tensor,
) -> None:
    if token_positions.dtype not in _INTEGER_DTYPES:
        raise ExecutionContractError("token_positions must have an integer dtype")
    expected = {
        sequence_id: store.next_position.get(sequence_id, 0)
        for sequence_id in sequence_ids
    }
    for token_index in range(execution_mask.shape[1]):
        for row, sequence_id in enumerate(sequence_ids):
            if not bool(execution_mask[row, token_index].item()):
                continue
            position = int(token_positions[row, token_index].item())
            if position != expected[sequence_id]:
                raise ExecutionContractError(
                    f"token_position {position} for sequence {sequence_id!r} must equal "
                    f"the next required position {expected[sequence_id]}"
                )
            expected[sequence_id] += 1


def _input_k_regions(plan: Plan) -> set[str]:
    return {
        region.region_id
        for region in plan.regions
        if _operation_type(region.k_requested, "fixed") == "input"
    }


def _validate_requested_k_token(
    plan: Plan,
    supplied: Optional[Mapping[str, Sequence[int]]],
    batch: int,
) -> None:
    if supplied is None:
        return
    allowed = _input_k_regions(plan)
    unknown = set(supplied) - allowed
    if unknown:
        raise ExecutionContractError(
            f"requested_k was supplied for regions without input K: {sorted(unknown)}"
        )
    for region_id, values in supplied.items():
        if len(values) != batch:
            raise ExecutionContractError(
                f"requested_k[{region_id!r}] must have length {batch}"
            )
        for value in values:
            if isinstance(value, Tensor):
                if value.numel() != 1 or value.dtype not in _INTEGER_DTYPES:
                    raise ExecutionContractError(
                        f"requested_k[{region_id!r}] values must be integer scalars"
                    )
            elif isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise ExecutionContractError(
                    f"requested_k[{region_id!r}] values must be integers"
                )


def _validate_prefill_requested_k(
    plan: Plan,
    supplied: Optional[Mapping[str, Tensor]],
    batch: int,
    length: int,
) -> None:
    if supplied is None:
        return
    allowed = _input_k_regions(plan)
    unknown = set(supplied) - allowed
    if unknown:
        raise ExecutionContractError(
            f"requested_k was supplied for regions without input K: {sorted(unknown)}"
        )
    for region_id, values in supplied.items():
        if not isinstance(values, Tensor):
            raise ExecutionContractError(
                f"requested_k[{region_id!r}] must be a Tensor"
            )
        if values.shape != (batch, length):
            raise ExecutionContractError(
                f"requested_k[{region_id!r}] must have shape [B,T]"
            )
        if values.dtype not in _INTEGER_DTYPES:
            raise ExecutionContractError(
                f"requested_k[{region_id!r}] must have an integer dtype"
            )


def _operation_type(config: Mapping[str, Any], default: str) -> str:
    return str(config.get("type", default)).lower().replace("-", "_")


def _validate_reference_plan_capability(plan: Plan) -> None:
    errors: List[str] = []
    for node in plan.nodes:
        if node.parameter_group is not None:
            errors.append(
                f"node {node.node_id!r} requests parameter_group "
                f"{node.parameter_group!r}; reference parameter aliases are "
                "not implemented"
            )
        update_type = _operation_type(node.update, "")
        for field in (
            "input_norm",
            "ffn_norm",
            "aggregate",
            "update",
            "selector_read",
            "ffn_read",
            "node_compute",
            "emit",
        ):
            try:
                validate_reference_operation_config(
                    field,
                    getattr(node, field),
                    state_update_type=(
                        update_type if field == "ffn_read" else None
                    ),
                )
            except ReferenceOperationConfigError as exc:
                errors.append(f"node {node.node_id!r} {exc}")
    for region in plan.regions:
        for field in (
            "score",
            "selector_context",
            "selector_history",
            "k_requested",
        ):
            try:
                validate_reference_operation_config(
                    field, getattr(region, field)
                )
            except ReferenceOperationConfigError as exc:
                errors.append(f"region {region.region_id!r} {exc}")
    try:
        validate_reference_operation_config(
            "output_aggregate", plan.output_aggregate
        )
    except ReferenceOperationConfigError as exc:
        errors.append(f"Plan {exc}")
    if errors:
        raise UnsupportedPlanError(
            "unsupported Plan capability:\n- " + "\n- ".join(errors)
        )


def _validate_model_and_state(
    model: SettleGraph, hidden: Tensor, store: StateStore
) -> None:
    if model.typed_plan is not None:
        dtype_name = model.typed_plan.binding.dtype_roles["hidden"]
        expected_dtype = getattr(torch, dtype_name)
        if hidden.dtype != expected_dtype:
            raise ExecutionContractError(
                f"hidden dtype {hidden.dtype} does not match typed Plan "
                f"binding {dtype_name}"
            )
    for name, parameter in model.named_parameters():
        if parameter.device != hidden.device:
            raise ExecutionContractError(
                f"parameter {name!r} is on {parameter.device}, but hidden is "
                f"on {hidden.device}"
            )
        if parameter.is_floating_point() and parameter.dtype != hidden.dtype:
            raise ExecutionContractError(
                f"parameter {name!r} has dtype {parameter.dtype}, but hidden "
                f"has dtype {hidden.dtype}"
            )
    _validate_state_store(model, hidden, store)


def _validate_state_store(
    model: SettleGraph, hidden: Tensor, store: StateStore
) -> None:
    for key in store.selector_history:
        if not isinstance(key, tuple) or len(key) != 2:
            raise ExecutionContractError(
                "selector-history keys must be (sequence_id, owner_id) pairs"
            )
        _validate_stable_id_sequence("state sequence", (key[0],))
        _validate_stable_id_sequence("selector-history owner", (key[1],))
    if store.selector_history:
        raise ExecutionContractError(
            "selector_history is non-empty, but this executor only accepts "
            "Plans whose selector history type is none"
        )
    for sequence_id, position in store.next_position.items():
        _validate_stable_id_sequence("state sequence", (sequence_id,))
        if type(position) is not int or position < 0:
            raise ExecutionContractError(
                "next Token positions must be nonnegative integers"
            )

    # Tensor object identity is too weak here: two distinct views can mutate
    # the same backing allocation.  The untyped StorageImpl identity is
    # stable across views (including disjoint views) and remains distinct for
    # independent empty allocations, whose data pointers are both zero.
    storage_owners: Dict[Tuple[str, Optional[int], int], StateKey] = {}
    storage_ranges: List[
        Tuple[str, Optional[int], int, int, StateKey]
    ] = []
    known_nodes = {node.node_id for node in model.plan.nodes}
    for key, state in store.values.items():
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or not all(isinstance(item, str) for item in key)
        ):
            raise ExecutionContractError(
                "receiver state keys must be (sequence_id, node_id) strings"
            )
        sequence_id, node_id = key
        _validate_stable_id_sequence("state sequence", (sequence_id,))
        _validate_stable_id_sequence("state node", (node_id,))
        if node_id not in known_nodes:
            raise ExecutionContractError(
                f"receiver state names unknown node {node_id!r}"
            )
        if sequence_id not in store.next_position:
            raise ExecutionContractError(
                f"receiver state for sequence {sequence_id!r} has no next "
                "Token position"
            )
        receiver = model.receiver(node_id)
        if receiver.update_type == "none":
            raise ExecutionContractError(
                f"stateless receiver {node_id!r} must not have stored state"
            )
        tensors: Tuple[Tensor, ...]
        if receiver.update_type == "attention_window":
            if not isinstance(state, AttentionState):
                raise ExecutionContractError(
                    f"receiver {node_id!r} requires canonical AttentionState"
                )
            length = state.length
            if state.positions.dtype != torch.int64:
                raise ExecutionContractError("Attention positions must use int64")
            if state.positions.shape != (length,):
                raise ExecutionContractError("Attention positions must be one-dimensional")
            if state.keys.shape != (length, receiver.attn_key_dim):
                raise ExecutionContractError("Attention key state has the wrong shape")
            if state.values.shape != (length, receiver.attn_value_dim):
                raise ExecutionContractError("Attention value state has the wrong shape")
            if length > receiver.attn_window:
                raise ExecutionContractError("Attention state exceeds its Plan window")
            if length > 1 and bool(
                (state.positions[1:] <= state.positions[:-1]).any().item()
            ):
                raise ExecutionContractError(
                    "Attention state positions must increase strictly"
                )
            if length and int(state.positions[0].item()) < 0:
                raise ExecutionContractError(
                    "Attention state positions must be nonnegative"
                )
            if length and int(state.positions[-1].item()) >= store.next_position[
                sequence_id
            ]:
                raise ExecutionContractError(
                    "Attention state positions must precede the sequence next "
                    "Token position"
                )
            tensors = (state.positions, state.keys, state.values)
            floating = (state.keys, state.values)
        else:
            if not isinstance(state, Tensor):
                raise ExecutionContractError(
                    f"receiver {node_id!r} requires Tensor state"
                )
            if tuple(state.shape) != receiver.state_shape:
                raise ExecutionContractError(
                    f"receiver {node_id!r} state shape {tuple(state.shape)} "
                    f"does not match {receiver.state_shape}"
                )
            tensors = (state,)
            floating = (state,)
        for tensor in tensors:
            if tensor.device != hidden.device:
                raise ExecutionContractError(
                    f"receiver {node_id!r} state is on {tensor.device}, but "
                    f"hidden is on {hidden.device}"
                )
            storage = tensor.untyped_storage()
            storage_id = (
                tensor.device.type,
                tensor.device.index,
                storage._cdata,
            )
            owner = storage_owners.get(storage_id)
            if owner is not None and owner != key:
                raise ExecutionContractError(
                    f"mutable state storage is shared by {owner!r} and {key!r}"
                )
            start = storage.data_ptr()
            end = start + storage.nbytes()
            if start and end > start:
                for (
                    other_type,
                    other_index,
                    other_start,
                    other_end,
                    other_owner,
                ) in storage_ranges:
                    if (
                        other_owner != key
                        and other_type == tensor.device.type
                        and other_index == tensor.device.index
                        and start < other_end
                        and other_start < end
                    ):
                        raise ExecutionContractError(
                            "mutable state storage is shared by "
                            f"{other_owner!r} and {key!r}"
                        )
                storage_ranges.append(
                    (
                        tensor.device.type,
                        tensor.device.index,
                        start,
                        end,
                        key,
                    )
                )
            storage_owners[storage_id] = key
        for tensor in floating:
            if not tensor.is_floating_point() or tensor.dtype != hidden.dtype:
                raise ExecutionContractError(
                    f"receiver {node_id!r} floating state dtype must match hidden"
                )


def _detach_state(state: ReceiverState) -> ReceiverState:
    if state is None:
        return None
    if isinstance(state, AttentionState):
        return AttentionState(
            state.positions.detach(), state.keys.detach(), state.values.detach()
        )
    return state.detach()


def _detach_value(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach()
    if isinstance(value, AttentionState):
        return _detach_state(value)
    return value

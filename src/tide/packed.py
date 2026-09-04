"""Tensorized generic prefill execution for core-v1 SettleGraph Plans.

The implementation in this module is deliberately independent of the eager
schedulers in :mod:`tide.engine`.  It consumes the same normalized Plan and
the same parameter objects, but it does not call ``prefill``,
``prefill_region_major``, ``interpret_token``, or ``_interpret_sample``.

The hot path advances one static region at a time.  Within a region, all
``[B, T]`` events are handled by tensor operations.  Only actual DATA messages
are retained between regions; bounded dense tensors are temporary region-local
views used for fan-in, selector, and packed active-node dispatch.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
import weakref
from collections import defaultdict
from contextvars import ContextVar
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

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
    NodeEventTrace,
    OutputEventTrace,
    RegionEventTrace,
    SettleGraph,
    StateStore,
    StateWriteTrace,
    UnsupportedPlanError,
    _build_trace,
    _validate_detach_at_end,
    _validate_model_and_state,
    _validate_optional_mask,
    _validate_position_matrix,
    _validate_reference_plan_capability,
    _validate_stable_id_sequence,
)
from .ops import AttentionState, ReceiverState, safe_module_key


PACKED_EXECUTOR_ID = "tide.generic-packed.torch.v1"


def _kind(config: Mapping[str, Any], default: str = "") -> str:
    return str(config.get("type", default)).lower().replace("-", "_")


@dataclasses.dataclass(frozen=True)
class PackedSupportIssue:
    """One stable reason a Plan cannot use this packed implementation."""

    code: str
    path: str
    detail: str


@dataclasses.dataclass(frozen=True)
class PackedSupportReport:
    """Static support decision made before any runtime Tensor is consumed."""

    executor_id: str
    logical_plan_hash: str
    supported: bool
    issues: Tuple[PackedSupportIssue, ...]

    @property
    def accepted(self) -> bool:
        """Compatibility spelling for support-predicate consumers."""

        return self.supported

    def require_supported(self) -> None:
        if self.supported:
            return
        lines = [
            f"{issue.code} at {issue.path}: {issue.detail}"
            for issue in self.issues
        ]
        raise UnsupportedPlanError(
            "unsupported generic packed Plan capability:\n- " + "\n- ".join(lines)
        )


@dataclasses.dataclass(frozen=True)
class PackedExecutionProfile:
    """Inspectable scheduling facts for the most recent successful call.

    The counters describe selected Python dispatch boundaries, not device
    kernels.  A trace request necessarily materializes semantic records in
    Python after tensor execution.  Grad-enabled result publication also runs
    a source-liveness analysis that these counters do not measure, so zero hot
    loop fields are not by themselves a performance or synchronization claim.
    """

    executor_id: str
    schedule_identity: str
    region_batches: int
    formula_group_batches: int
    state_scan_batches: int
    fanout_batches: int
    data_message_rows: int
    active_compute_rows: int
    python_token_hot_loops: int = 0
    python_batch_row_hot_loops: int = 0
    python_node_event_hot_loops: int = 0
    eager_scheduler_calls: int = 0
    trace_materialized: bool = False


@dataclasses.dataclass(frozen=True)
class _FormulaGroup:
    signature: Tuple[Any, ...]
    node_locals: Tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class _StateGroup:
    update_type: str
    state_shape: Tuple[int, ...]
    node_locals: Tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class _FanoutGroup:
    target_region_rank: int
    source_locals: Tuple[int, ...]
    target_locals: Tuple[int, ...]
    target_parent_slots: Tuple[int, ...]
    edge_indices: Tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class _RegionSchedule:
    rank: int
    region_id: str
    node_ids: Tuple[str, ...]
    global_node_indices: Tuple[int, ...]
    parent_edge_ids: Tuple[Tuple[str, ...], ...]
    max_fanin: int
    entry_locals: Tuple[int, ...]
    terminal_locals: Tuple[int, ...]
    terminal_ordinals: Tuple[int, ...]
    forced_singleton: bool
    aggregate_groups: Tuple[_FormulaGroup, ...]
    state_groups: Tuple[_StateGroup, ...]
    selector_read_groups: Tuple[_FormulaGroup, ...]
    compute_groups: Tuple[_FormulaGroup, ...]
    fanout_groups: Tuple[_FanoutGroup, ...]


@dataclasses.dataclass(frozen=True)
class _MessageBatch:
    hidden: Tensor
    event: Tensor
    target_local: Tensor
    parent_slot: Tensor
    edge_index: Tensor


@dataclasses.dataclass(frozen=True)
class _ComputeBatch:
    """Per-formula results before independent event lanes share dense storage."""

    event: Tensor
    local: Tensor
    computed: Tensor
    emitted: Tensor


@dataclasses.dataclass(frozen=True)
class _TensorStateResult:
    before: Tensor
    proposal: Tensor
    after: Tensor
    final: Tensor
    observed: Tensor


@dataclasses.dataclass(frozen=True)
class _AttentionStateResult:
    before_positions: Tensor
    before_keys: Tensor
    before_values: Tensor
    before_mask: Tensor
    after_positions: Tensor
    after_keys: Tensor
    after_values: Tensor
    after_mask: Tensor
    final_positions: Tensor
    final_keys: Tensor
    final_values: Tensor
    final_mask: Tensor
    observed: Tensor


_StateResult = Union[_TensorStateResult, _AttentionStateResult]


@dataclasses.dataclass
class _RegionRuntime:
    schedule: _RegionSchedule
    message_hidden: Tensor
    message_mask: Tensor
    reached: Tensor
    hidden: Tensor
    normalized: Tensor
    observed: Tensor
    active: Tensor
    selector_read: Optional[Tensor]
    logits: Optional[Tensor]
    probabilities: Optional[Tensor]
    computed: Tensor
    emitted: Tensor
    state_results: Tuple[Optional[_StateResult], ...]
    compute_batches: Tuple[_ComputeBatch, ...]


@dataclasses.dataclass(frozen=True)
class _ScanScoreInputs:
    score_type: int
    shared: bool
    fixed: Tensor
    linear_weight: Tensor
    linear_bias: Tensor
    hidden_weight: Tensor
    hidden_bias: Tensor
    output_weight: Tensor
    output_bias: Tensor


def inspect_packed_support(model_or_plan: Any) -> PackedSupportReport:
    """Return the generic packed executor's static core-v1 support decision."""

    plan = model_or_plan.plan if isinstance(model_or_plan, SettleGraph) else model_or_plan
    plan = plan.validate()
    issues: List[PackedSupportIssue] = []
    try:
        _validate_reference_plan_capability(plan)
    except UnsupportedPlanError as exc:
        issues.append(
            PackedSupportIssue(
                "packed.reference-capability",
                "plan",
                f"reference local-operation capability rejected the Plan: {exc}",
            )
        )

    supported_node_ops = {
        "aggregate": {"mean", "learned_convex", "edge_softmax", "edge_linear_mean"},
        "update": {"none", "ema", "gdn", "attention_window"},
        "selector_read": {
            "content",
            "content_norm",
            "content_linear",
            "content_state_linear",
            "content_state_summary_linear",
        },
        "ffn_read": {"zero", "state_default"},
        "node_compute": {
            "identity",
            "affine_residual",
            "double_residual_mlp",
            "double_residual_swiglu",
        },
        "emit": {"hard", "hst", "softp"},
    }
    for node in plan.nodes:
        for field, supported in supported_node_ops.items():
            value = _kind(getattr(node, field))
            if value not in supported:
                issues.append(
                    PackedSupportIssue(
                        "packed.unsupported-node-formula",
                        f"nodes[{node.node_id!r}].{field}",
                        f"formula type {value!r} has no packed implementation",
                    )
                )

    for region in plan.regions:
        history_type = _kind(region.selector_history, "none")
        if history_type != "none":
            issues.append(
                PackedSupportIssue(
                    "packed.selector-history",
                    f"regions[{region.region_id!r}].selector_history",
                    "selector-history ownership/schema is outside core-v1",
                )
            )
        context_type = _kind(region.selector_context, "none")
        if context_type != "none":
            issues.append(
                PackedSupportIssue(
                    "packed.selector-context",
                    f"regions[{region.region_id!r}].selector_context",
                    "public selector context has no packed binding",
                )
            )
        k_type = _kind(region.k_requested, "fixed")
        if k_type != "fixed":
            issues.append(
                PackedSupportIssue(
                    "packed.input-k-extension",
                    f"regions[{region.region_id!r}].k_requested",
                    "core-v1 packed execution requires a fixed K",
                )
            )
        elif int(region.k_requested.get("value", -1)) != int(region.k_max):
            issues.append(
                PackedSupportIssue(
                    "packed.non-core-fixed-k",
                    f"regions[{region.region_id!r}].k_requested.value",
                    "core-v1 requires fixed requested K to equal k_max",
                )
            )

        forced = any(plan.node_by_id(node_id).forced_active for node_id in region.node_ids)
        if forced and not (
            len(region.node_ids) == 1
            and all(plan.node_by_id(node_id).forced_active for node_id in region.node_ids)
        ):
            issues.append(
                PackedSupportIssue(
                    "packed.mixed-forced-region",
                    f"regions[{region.region_id!r}]",
                    "forced-active packed regions must be singleton regions",
                )
            )
        score_type = _kind(region.score)
        if score_type not in {"fixed", "constant", "read_sum", "linear", "mlp"}:
            issues.append(
                PackedSupportIssue(
                    "packed.unsupported-score",
                    f"regions[{region.region_id!r}].score",
                    f"score type {score_type!r} has no packed implementation",
                )
            )

    output_type = _kind(plan.output_aggregate, "mean")
    if output_type not in {"mean", "learned_convex", "node_softmax"}:
        issues.append(
            PackedSupportIssue(
                "packed.unsupported-output-aggregate",
                "output_aggregate",
                f"output aggregate type {output_type!r} has no packed implementation",
            )
        )

    return PackedSupportReport(
        PACKED_EXECUTOR_ID,
        plan.logical_hash(),
        not issues,
        tuple(issues),
    )


def _group_locals(values: Sequence[Tuple[Any, ...]]) -> Tuple[_FormulaGroup, ...]:
    grouped: Dict[Tuple[Any, ...], List[int]] = defaultdict(list)
    for local, signature in enumerate(values):
        grouped[signature].append(local)
    return tuple(
        _FormulaGroup(signature, tuple(locals_))
        for signature, locals_ in sorted(grouped.items(), key=lambda item: repr(item[0]))
    )


def _node_state_signature(module: Any) -> Tuple[Any, ...]:
    return (module.update_type, tuple(module.state_shape))


def _node_read_signature(module: Any) -> Tuple[Any, ...]:
    return (
        module.selector_read_type,
        module.selector_read_dim,
        *_node_state_signature(module),
    )


def _node_compute_signature(module: Any) -> Tuple[Any, ...]:
    hidden_dim = (
        int(module.gate_proj.out_features)
        if module.gate_proj is not None
        else 0
    )
    return (
        module.compute_type,
        hidden_dim,
        module.ffn_read_type,
        *_node_state_signature(module),
        _kind(module.emit_config, "hard"),
    )


def _compile_schedule(model: SettleGraph) -> Tuple[Tuple[_RegionSchedule, ...], str]:
    plan = model.plan
    node_ids = tuple(node.node_id for node in plan.nodes)
    node_global = {node_id: index for index, node_id in enumerate(node_ids)}
    edge_global = {edge.edge_id: index for index, edge in enumerate(plan.edges)}
    terminal_ordinal = {
        node_id: index for index, node_id in enumerate(plan.terminal_node_ids)
    }
    regions = plan.topological_regions
    region_rank = {region.region_id: index for index, region in enumerate(regions)}
    node_region_local: Dict[str, Tuple[int, int]] = {}
    parent_slot: Dict[str, int] = {}
    partial: List[Dict[str, Any]] = []

    for rank, region in enumerate(regions):
        local_by_node = {node_id: index for index, node_id in enumerate(region.node_ids)}
        for node_id, local in local_by_node.items():
            node_region_local[node_id] = (rank, local)
        parents: List[Tuple[str, ...]] = []
        for node_id in region.node_ids:
            edge_ids = tuple(edge.edge_id for edge in plan.incoming_edges[node_id])
            parents.append(edge_ids)
            for slot, edge_id in enumerate(edge_ids):
                parent_slot[edge_id] = slot
        modules = tuple(model.receivers[safe_module_key(node_id)] for node_id in region.node_ids)
        state_map: Dict[Tuple[Any, ...], List[int]] = defaultdict(list)
        for local, module in enumerate(modules):
            if module.update_type != "none":
                # A state batch is also a selector-Read batch.  Keeping the
                # read kind homogeneous lets the compiled SD/pre recurrence
                # operate on one fixed feature width per group while still
                # allowing arbitrary mixtures across a region.
                state_map[
                    (*_node_state_signature(module), module.selector_read_type)
                ].append(local)
        state_groups = tuple(
            _StateGroup(str(signature[0]), tuple(signature[1]), tuple(locals_))
            for signature, locals_ in sorted(state_map.items(), key=lambda item: repr(item[0]))
        )
        partial.append(
            {
                "rank": rank,
                "region": region,
                "modules": modules,
                "parents": tuple(parents),
                "max_fanin": max(1, *(len(item) for item in parents)),
                "entry_locals": tuple(
                    local for local, node_id in enumerate(region.node_ids)
                    if node_id in set(plan.entry_node_ids)
                ),
                "terminal_locals": tuple(
                    local for local, node_id in enumerate(region.node_ids)
                    if node_id in terminal_ordinal
                ),
                "state_groups": state_groups,
            }
        )

    fanout_maps: List[Dict[int, List[Tuple[int, int, int, int]]]] = [
        defaultdict(list) for _ in regions
    ]
    for edge in plan.edges:
        source_rank, source_local = node_region_local[edge.source]
        target_rank, target_local = node_region_local[edge.target]
        fanout_maps[source_rank][target_rank].append(
            (source_local, target_local, parent_slot[edge.edge_id], edge_global[edge.edge_id])
        )

    schedules: List[_RegionSchedule] = []
    for item in partial:
        region = item["region"]
        modules = item["modules"]
        fanouts = []
        for target_rank, records in sorted(fanout_maps[item["rank"]].items()):
            records.sort(key=lambda record: record[3])
            fanouts.append(
                _FanoutGroup(
                    target_rank,
                    tuple(record[0] for record in records),
                    tuple(record[1] for record in records),
                    tuple(record[2] for record in records),
                    tuple(record[3] for record in records),
                )
            )
        terminal_locals = item["terminal_locals"]
        schedules.append(
            _RegionSchedule(
                item["rank"],
                region.region_id,
                tuple(region.node_ids),
                tuple(node_global[node_id] for node_id in region.node_ids),
                item["parents"],
                item["max_fanin"],
                item["entry_locals"],
                terminal_locals,
                tuple(terminal_ordinal[region.node_ids[local]] for local in terminal_locals),
                bool(len(region.node_ids) == 1 and model.plan.node_by_id(region.node_ids[0]).forced_active),
                _group_locals(tuple((module.aggregate_type,) for module in modules)),
                item["state_groups"],
                _group_locals(tuple(_node_read_signature(module) for module in modules)),
                _group_locals(tuple(_node_compute_signature(module) for module in modules)),
                tuple(fanouts),
            )
        )

    identity_payload = {
        "executor": PACKED_EXECUTOR_ID,
        "plan": plan.logical_hash(),
        "regions": [
            {
                "id": schedule.region_id,
                "nodes": list(schedule.node_ids),
                "parents": [list(item) for item in schedule.parent_edge_ids],
                "state_groups": [
                    [group.update_type, list(group.state_shape), list(group.node_locals)]
                    for group in schedule.state_groups
                ],
            }
            for schedule in schedules
        ],
    }
    identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return tuple(schedules), identity


def _device_index(values: Sequence[int], reference: Tensor) -> Tensor:
    return torch.tensor(tuple(values), dtype=torch.int64, device=reference.device)


class _ConnectivityTracker:
    def __init__(self) -> None:
        self.connected_source_tokens: set[Hashable] = set()
        self._resolver: Optional[
            Callable[[frozenset["_ObservableSeed"]], set[Hashable]]
        ] = None
        self._resolved_sources: Dict[
            frozenset["_ObservableSeed"], frozenset[Hashable]
        ] = {}
        self.finalized = False

    def finalize(
        self,
        resolver: Callable[[frozenset["_ObservableSeed"]], set[Hashable]],
    ) -> None:
        self._resolver = resolver
        self._resolved_sources.clear()
        self.finalized = True

    def resolve(
        self, roots: frozenset["_ObservableSeed"]
    ) -> frozenset[Hashable]:
        if self._resolver is None:
            raise AssertionError("packed connectivity was resolved before finalization")
        if roots not in self._resolved_sources:
            self._resolved_sources[roots] = frozenset(self._resolver(roots))
        return self._resolved_sources[roots]

    def select(self, roots: frozenset["_ObservableSeed"]) -> None:
        # Replace rather than accumulate: repeated autograd.grad/backward calls
        # with retain_graph=True must reflect the roots of the current call.
        self.connected_source_tokens = set(self.resolve(roots))


_ACTIVE_CONNECTIVITY: ContextVar[Optional[_ConnectivityTracker]] = ContextVar(
    "tide_packed_connectivity", default=None
)


def _connectivity_tracker_for_backward(ctx: Any) -> Optional[_ConnectivityTracker]:
    """Resolve a selective op's non-owning tracker reference."""

    tracker_ref = ctx.connectivity_tracker_ref
    if tracker_ref is None:
        # This op was created outside a tracked packed call.  Preserve the
        # ordinary usage-only fallback used by the internal tensor helpers.
        return None
    tracker = tracker_ref()
    if tracker is None:
        raise RuntimeError(
            "packed connectivity tracker was released before selective backward; "
            "a differentiable public result must retain its result boundary "
            "throughout backward"
        )
    if not tracker.finalized:
        raise RuntimeError(
            "packed connectivity tracker was not finalized before selective backward"
        )
    return tracker


def _track_connectivity(function: Any) -> Any:
    """Give every packed call an exception-safe structural-liveness scope."""

    @functools.wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if not torch.is_grad_enabled():
            return function(*args, **kwargs)
        tracker = _ConnectivityTracker()
        token = _ACTIVE_CONNECTIVITY.set(tracker)
        try:
            return function(*args, **kwargs)
        finally:
            _ACTIVE_CONNECTIVITY.reset(token)

    return wrapped


def _leaf_tensor_ids(value: Tensor) -> Tuple[int, ...]:
    if not value.requires_grad:
        return ()
    if value.is_leaf:
        return (id(value),)
    pending = [value.grad_fn]
    # Retain the Python autograd-node wrappers themselves while walking.
    # Remembering only ``id(function)`` lets a wrapper be released after it
    # is popped, after which CPython may reuse that id for another upstream
    # node and make the traversal silently skip a real branch.
    seen: set[Any] = set()
    leaves: set[int] = set()
    while pending:
        function = pending.pop()
        if function is None or function in seen:
            continue
        seen.add(function)
        variable = getattr(function, "variable", None)
        if isinstance(variable, Tensor) and variable.requires_grad:
            leaves.add(id(variable))
        pending.extend(next_function for next_function, _ in function.next_functions)
    return tuple(leaves)


class _SelectiveStack(torch.autograd.Function):
    """Stack independent parameters while retaining eager ``None`` VJPs.

    Grouped kernels naturally mention every packed row, which would give a
    never-executed logical parameter a connected all-zero gradient.  The
    semantic contract instead distinguishes that case from an executed
    formula whose derivative happens to be zero.  This narrow packing
    boundary returns ``None`` for rows whose formula never executed.  Its
    forward is ordinary ``torch.stack`` and has no device synchronization;
    the small usage mask is inspected only if backward is requested.
    """

    @staticmethod
    def forward(ctx: Any, usage: Tensor, *values: Tensor) -> Tensor:
        ctx.save_for_backward(usage)
        tracker = _ACTIVE_CONNECTIVITY.get()
        # The public result boundary is the sole strong lifetime owner.  A
        # strong reference here would close a cycle through the resolver's
        # captured runtimes and retain the complete packed autograd graph.
        ctx.connectivity_tracker_ref = (
            weakref.ref(tracker) if tracker is not None else None
        )
        ctx.source_tokens = tuple(
            _leaf_tensor_ids(value) for value in values
        )
        return torch.stack(tuple(values), dim=0)

    @staticmethod
    def backward(ctx: Any, gradient: Tensor) -> Tuple[Any, ...]:
        (usage,) = ctx.saved_tensors
        flags = tuple(bool(item) for item in usage.detach().to(device="cpu").tolist())
        tracker = _connectivity_tracker_for_backward(ctx)
        if tracker is None:
            return (
                None,
                *(gradient[index] if flag else None for index, flag in enumerate(flags)),
            )
        connected = tracker.connected_source_tokens
        semantic = tuple(
            any(source_token in connected for source_token in source_tokens)
            for source_tokens in ctx.source_tokens
        )
        return (
            None,
            *(
                gradient[index] if flag and semantic[index] else None
                for index, flag in enumerate(flags)
            ),
        )


def _stack(values: Sequence[Tensor], usage: Optional[Tensor] = None) -> Tensor:
    if not values:
        raise AssertionError("cannot stack an empty parameter group")
    if usage is None or not any(value.requires_grad for value in values):
        return torch.stack(tuple(values), dim=0)
    if usage.shape != (len(values),) or usage.dtype != torch.bool:
        raise AssertionError("selective parameter usage must be bool[len(values)]")
    return _SelectiveStack.apply(usage, *tuple(values))


_HIDDEN_SOURCE_TOKEN = ("input", "hidden")


def _state_source_token(
    sequence_id: str, node_id: str, component: str
) -> Tuple[str, str, str, str]:
    return ("receiver-state", sequence_id, node_id, component)


def _differentiable_source_tokens(
    model: SettleGraph,
    initial: StateStore,
    hidden: Tensor,
) -> set[Hashable]:
    """Return installed packed source tokens that can own an autograd edge."""

    result: set[Hashable] = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    if hidden.requires_grad:
        result.add(_HIDDEN_SOURCE_TOKEN)
    for (sequence_id, node_id), state in initial.values.items():
        if isinstance(state, Tensor):
            if state.requires_grad:
                result.add(
                    _state_source_token(sequence_id, node_id, "tensor")
                )
        elif isinstance(state, AttentionState):
            if state.keys.requires_grad:
                result.add(_state_source_token(sequence_id, node_id, "keys"))
            if state.values.requires_grad:
                result.add(
                    _state_source_token(sequence_id, node_id, "values")
                )
    return result


@dataclasses.dataclass(frozen=True)
class _ObservableSeed:
    kind: str
    event: Optional[int] = None
    sequence_id: Optional[str] = None
    node_id: Optional[str] = None
    component: Optional[str] = None
    region_id: Optional[str] = None
    edge_id: Optional[str] = None


class _SelectiveInput(torch.autograd.Function):
    """Identity whose VJP follows eager liveness for one packed input owner."""

    @staticmethod
    def forward(ctx: Any, value: Tensor, source_token: Hashable) -> Tensor:
        tracker = _ACTIVE_CONNECTIVITY.get()
        ctx.connectivity_tracker_ref = (
            weakref.ref(tracker) if tracker is not None else None
        )
        ctx.source_token = source_token
        return value

    @staticmethod
    def backward(ctx: Any, gradient: Tensor) -> Tuple[Any, ...]:
        tracker = _connectivity_tracker_for_backward(ctx)
        if tracker is None:
            return gradient, None
        if ctx.source_token not in tracker.connected_source_tokens:
            return None, None
        return gradient, None


def _selective_input(value: Tensor, source_token: Hashable) -> Tensor:
    """Install a liveness boundary only when this call can need a VJP."""

    if _ACTIVE_CONNECTIVITY.get() is None or not value.requires_grad:
        return value
    return _SelectiveInput.apply(value, source_token)


class _ResultConnectivityBoundary(torch.autograd.Function):
    """Observe exactly which public result Tensors seed one backward call.

    All differentiable public observables of one packed call pass through this
    single multi-output identity.  ``set_materialize_grads(False)`` preserves
    the distinction between an unused result and a used result whose incoming
    cotangent happens to be numerically zero.
    """

    @staticmethod
    def forward(
        ctx: Any,
        tracker: _ConnectivityTracker,
        seeds: Tuple[_ObservableSeed, ...],
        *values: Tensor,
    ) -> Tuple[Tensor, ...]:
        ctx.tracker = tracker
        ctx.seeds = seeds
        ctx.set_materialize_grads(False)
        return tuple(values)

    @staticmethod
    def backward(ctx: Any, *gradients: Optional[Tensor]) -> Tuple[Any, ...]:
        selected = frozenset(
            seed
            for seed, gradient in zip(ctx.seeds, gradients)
            if gradient is not None
        )
        ctx.tracker.select(selected)
        return None, None, *gradients


def _record_linear(
    x: Tensor,
    modules: Sequence[Any],
    group_column: Tensor,
    attribute: str,
    usage: Optional[Tensor] = None,
) -> Tensor:
    linears = tuple(getattr(module, attribute) for module in modules)
    if any(linear is None for linear in linears):
        raise AssertionError(f"missing record linear {attribute}")
    weights = _stack(tuple(linear.weight for linear in linears), usage)
    biases = (
        _stack(tuple(linear.bias for linear in linears), usage)
        if linears[0].bias is not None
        else x.new_empty((0,))
    )
    return _record_group_linear_1d(x, weights, biases, group_column)


def _column_usage(columns: Tensor, count: int) -> Tensor:
    usage = torch.zeros((count,), dtype=torch.bool, device=columns.device)
    return usage.index_fill(0, columns, True) if columns.numel() else usage


def _safe_masked_softmax(logits: Tensor, mask: Tensor, dim: int) -> Tensor:
    masked = logits.masked_fill(~mask, -torch.inf)
    any_value = mask.any(dim=dim, keepdim=True)
    maximum = masked.amax(dim=dim, keepdim=True)
    maximum = torch.where(any_value, maximum, torch.zeros_like(maximum))
    exponent = torch.exp(masked - maximum).masked_fill(~mask, 0.0)
    denominator = exponent.sum(dim=dim, keepdim=True).clamp_min(1.0)
    return exponent / denominator


@torch.jit.script
def _record_group_linear_1d(
    values: Tensor,
    weights: Tensor,
    biases: Tensor,
    group_column: Tensor,
) -> Tensor:
    """Apply eager-compatible 1-D Linear per logical edge record.

    On the validated aarch64 Torch build, ``F.linear`` on a 1-D input and an
    explicit ``addmv`` differ by one FP32 ulp.  Edge transforms use the former
    in the eager semantic reference, so retain that exact operator boundary
    inside a compiled record loop.
    """

    rows = torch.jit.annotate(List[Tensor], [])
    for row_index in range(values.size(0)):
        column = group_column[row_index]
        if biases.numel() > 0:
            rows.append(
                torch.nn.functional.linear(
                    values[row_index], weights[column], biases[column]
                )
            )
        else:
            rows.append(
                torch.nn.functional.linear(values[row_index], weights[column], None)
            )
    return torch.stack(rows, dim=0)


def _group_linear_records_1d(
    values: Tensor,
    modules: Sequence[Any],
    attribute: str,
    usage: Optional[Tensor] = None,
) -> Tensor:
    """Apply eager's 1-D Linear boundary to every packed logical record."""

    linears = tuple(getattr(module, attribute) for module in modules)
    if any(linear is None for linear in linears):
        raise AssertionError(f"missing grouped record linear {attribute}")
    weights = _stack(tuple(linear.weight for linear in linears), usage)
    biases = (
        _stack(tuple(linear.bias for linear in linears), usage)
        if linears[0].bias is not None
        else values.new_empty((0,))
    )
    group_width = values.shape[-2]
    columns = torch.arange(group_width, device=values.device).repeat(
        values.numel() // (group_width * values.shape[-1])
    )
    projected = _record_group_linear_1d(
        values.reshape(-1, values.shape[-1]), weights, biases, columns
    )
    return projected.reshape(*values.shape[:-1], projected.shape[-1])


@torch.jit.script
def _normalize_group_records_1d(values: Tensor, eps: Tensor) -> Tensor:
    """Match eager 1-D ``F.normalize`` for every [..., group, width] row."""

    width = values.size(-1)
    group_width = values.size(-2)
    flat = values.reshape(-1, width)
    rows = torch.jit.annotate(List[Tensor], [])
    for row_index in range(flat.size(0)):
        group_index = row_index % group_width
        norm = torch.linalg.vector_norm(flat[row_index])
        rows.append(flat[row_index] / torch.clamp_min(norm, eps[group_index]))
    return torch.stack(rows, dim=0).reshape(values.shape)


@torch.jit.script
def _normalize_selected_records_1d(values: Tensor, eps: Tensor) -> Tensor:
    """Match eager 1-D ``F.normalize`` for an irregular record selection."""

    rows = torch.jit.annotate(List[Tensor], [])
    for row_index in range(values.size(0)):
        norm = torch.linalg.vector_norm(values[row_index])
        rows.append(values[row_index] / torch.clamp_min(norm, eps[row_index]))
    return torch.stack(rows, dim=0)


@torch.jit.script
def _rms_norm_selected_records_1d(
    values: Tensor,
    weights: Tensor,
    eps: Tensor,
    group_column: Tensor,
) -> Tensor:
    """Apply the eager 1-D RMSNorm formula per selected logical record."""

    rows = torch.jit.annotate(List[Tensor], [])
    for row_index in range(values.size(0)):
        column = group_column[row_index]
        value = values[row_index]
        variance = value.square().mean(dim=-1, keepdim=True)
        normalized = value * torch.rsqrt(variance + eps[column])
        rows.append(normalized * weights[column])
    return torch.stack(rows, dim=0)


@torch.jit.script
def _gdn_read_records_1d(state: Tensor, query: Tensor) -> Tensor:
    """Preserve eager's per-record matrix-vector GDN read boundary."""

    rows = torch.jit.annotate(List[Tensor], [])
    for row_index in range(state.size(0)):
        rows.append(
            torch.matmul(
                state[row_index].transpose(-2, -1), query[row_index]
            )
        )
    return torch.stack(rows, dim=0)


@torch.jit.script
def _attention_read_records_1d(
    keys: Tensor,
    values: Tensor,
    mask: Tensor,
    query: Tensor,
) -> Tensor:
    """Preserve variable-length eager Attention score/read reductions."""

    rows = torch.jit.annotate(List[Tensor], [])
    divisor = math.sqrt(float(keys.size(-1)))
    for row_index in range(keys.size(0)):
        valid = mask[row_index].nonzero().squeeze(-1)
        selected_keys = keys[row_index].index_select(0, valid)
        selected_values = values[row_index].index_select(0, valid)
        if selected_keys.size(0) == 0:
            rows.append(values.new_zeros((values.size(-1),)))
        else:
            scores = torch.matmul(selected_keys, query[row_index]) / divisor
            weights = torch.softmax(scores, dim=0)
            rows.append(
                torch.matmul(selected_values.transpose(0, 1), weights)
            )
    return torch.stack(rows, dim=0)


@torch.jit.script
def _mean_aggregate_records_1d(values: Tensor, mask: Tensor) -> Tensor:
    """Match eager's stack-then-mean Aggregate for each logical receiver."""

    rows = torch.jit.annotate(List[Tensor], [])
    for row_index in range(values.size(0)):
        valid = mask[row_index].nonzero().squeeze(-1)
        if valid.numel() == 0:
            rows.append(values.new_zeros((values.size(-1),)))
        else:
            rows.append(values[row_index].index_select(0, valid).mean(dim=0))
    return torch.stack(rows, dim=0)


@torch.jit.script
def _weighted_aggregate_records_1d(
    values: Tensor,
    mask: Tensor,
    scores: Tensor,
    group_width: int,
) -> Tensor:
    """Match eager candidate-length softmax and weighted Aggregate order."""

    rows = torch.jit.annotate(List[Tensor], [])
    for row_index in range(values.size(0)):
        valid = mask[row_index].nonzero().squeeze(-1)
        if valid.numel() == 0:
            rows.append(values.new_zeros((values.size(-1),)))
        else:
            selected = values[row_index].index_select(0, valid)
            logits = scores[row_index % group_width].index_select(0, valid)
            weights = torch.softmax(logits, dim=0)
            rows.append((weights.unsqueeze(-1) * selected).sum(dim=0))
    return torch.stack(rows, dim=0)


@torch.jit.script
def _candidate_softmax_records_1d(logits: Tensor, mask: Tensor) -> Tensor:
    """Run eager's variable-candidate softmax independently per event."""

    rows = torch.jit.annotate(List[Tensor], [])
    for row_index in range(logits.size(0)):
        valid = mask[row_index].nonzero().squeeze(-1)
        row = logits.new_zeros((logits.size(1),))
        if valid.numel() > 0:
            probabilities = torch.softmax(
                logits[row_index].index_select(0, valid), dim=0
            )
            row = row.index_copy(0, valid, probabilities)
        rows.append(row)
    return torch.stack(rows, dim=0)


@torch.jit.script
def _shared_linear_scores_by_event(
    readouts: Tensor,
    reached: Tensor,
    weight: Tensor,
    bias: Tensor,
) -> Tensor:
    """Apply one shared selector Linear per eager logical event."""

    rows = torch.jit.annotate(List[Tensor], [])
    for event_index in range(readouts.size(0)):
        valid = reached[event_index].nonzero().squeeze(-1)
        row = readouts.new_zeros((readouts.size(1),))
        if valid.numel() > 0:
            selected = readouts[event_index].index_select(0, valid)
            if bias.numel() > 0:
                scores = torch.nn.functional.linear(
                    selected, weight, bias
                ).squeeze(-1)
            else:
                scores = torch.nn.functional.linear(
                    selected, weight, None
                ).squeeze(-1)
            row = row.index_copy(0, valid, scores)
        rows.append(row)
    return torch.stack(rows, dim=0)


@torch.jit.script
def _shared_mlp_scores_by_event(
    readouts: Tensor,
    reached: Tensor,
    hidden_weight: Tensor,
    hidden_bias: Tensor,
    output_weight: Tensor,
    output_bias: Tensor,
) -> Tensor:
    """Apply one shared selector MLP per eager logical event."""

    rows = torch.jit.annotate(List[Tensor], [])
    for event_index in range(readouts.size(0)):
        valid = reached[event_index].nonzero().squeeze(-1)
        row = readouts.new_zeros((readouts.size(1),))
        if valid.numel() > 0:
            selected = readouts[event_index].index_select(0, valid)
            if hidden_bias.numel() > 0:
                hidden = torch.nn.functional.linear(
                    selected, hidden_weight, hidden_bias
                )
            else:
                hidden = torch.nn.functional.linear(
                    selected, hidden_weight, None
                )
            hidden = torch.nn.functional.silu(hidden)
            if output_bias.numel() > 0:
                scores = torch.nn.functional.linear(
                    hidden, output_weight, output_bias
                ).squeeze(-1)
            else:
                scores = torch.nn.functional.linear(
                    hidden, output_weight, None
                ).squeeze(-1)
            row = row.index_copy(0, valid, scores)
        rows.append(row)
    return torch.stack(rows, dim=0)


@torch.jit.script
def _unshared_mlp_scores_by_record(
    values: Tensor,
    hidden_weight: Tensor,
    hidden_bias: Tensor,
    output_weight: Tensor,
    output_bias: Tensor,
    group_column: Tensor,
) -> Tensor:
    """Keep each eager nonshared selector MLP inside one record boundary."""

    rows = torch.jit.annotate(List[Tensor], [])
    for row_index in range(values.size(0)):
        column = group_column[row_index]
        if hidden_bias.numel() > 0:
            hidden = F.linear(
                values[row_index], hidden_weight[column], hidden_bias[column]
            )
        else:
            hidden = F.linear(values[row_index], hidden_weight[column], None)
        hidden = F.silu(hidden)
        if output_bias.numel() > 0:
            score = F.linear(
                hidden, output_weight[column], output_bias[column]
            )
        else:
            score = F.linear(hidden, output_weight[column], None)
        rows.append(score.squeeze(-1))
    return torch.stack(rows, dim=0)


@torch.jit.script
def _swiglu_compute_records_1d(
    hidden: Tensor,
    ffn_read: Tensor,
    norm_weight: Tensor,
    norm_eps: Tensor,
    gate_weight: Tensor,
    gate_bias: Tensor,
    up_weight: Tensor,
    up_bias: Tensor,
    down_weight: Tensor,
    down_bias: Tensor,
    group_column: Tensor,
) -> Tensor:
    """Keep each eager double-residual SwiGLU inside one event boundary."""

    rows = torch.jit.annotate(List[Tensor], [])
    for row_index in range(hidden.size(0)):
        column = group_column[row_index]
        first = hidden[row_index] + ffn_read[row_index]
        variance = first.square().mean(dim=-1, keepdim=True)
        ffn_input = (
            first
            * torch.rsqrt(variance + norm_eps[column])
            * norm_weight[column]
        )
        if gate_bias.numel() > 0:
            gate = F.linear(
                ffn_input, gate_weight[column], gate_bias[column]
            )
        else:
            gate = F.linear(ffn_input, gate_weight[column], None)
        if up_bias.numel() > 0:
            up = F.linear(ffn_input, up_weight[column], up_bias[column])
        else:
            up = F.linear(ffn_input, up_weight[column], None)
        expansion = F.silu(gate) * up
        if down_bias.numel() > 0:
            down = F.linear(
                expansion, down_weight[column], down_bias[column]
            )
        else:
            down = F.linear(expansion, down_weight[column], None)
        rows.append(first + down)
    return torch.stack(rows, dim=0)


@torch.jit.script
def _rms_group_records_1d(values: Tensor) -> Tensor:
    """Match eager's per-record no-epsilon RMS reduction order."""

    width = values.size(-1)
    flat = values.reshape(-1, width)
    rows = torch.jit.annotate(List[Tensor], [])
    divisor = math.sqrt(float(width))
    for row_index in range(flat.size(0)):
        rows.append(torch.linalg.vector_norm(flat[row_index]) / divisor)
    return torch.stack(rows, dim=0).reshape(values.shape[:-1])


@torch.jit.script
def _attention_summary_records_1d(
    keys: Tensor, values: Tensor, mask: Tensor
) -> Tensor:
    """Evaluate the canonical variable-length Attention summary per record."""

    key_dim = keys.size(-1)
    value_dim = values.size(-1)
    window = keys.size(-2)
    flat_keys = keys.reshape(-1, window, key_dim)
    flat_values = values.reshape(-1, window, value_dim)
    flat_mask = mask.reshape(-1, window)
    rows = torch.jit.annotate(List[Tensor], [])
    for row_index in range(flat_keys.size(0)):
        valid = flat_mask[row_index]
        selected_keys = flat_keys[row_index][valid]
        selected_values = flat_values[row_index][valid]
        components = torch.cat(
            (selected_keys.reshape(-1), selected_values.reshape(-1)), dim=0
        )
        if components.numel() == 0:
            rows.append(flat_keys.new_zeros(()))
        else:
            rows.append(
                torch.linalg.vector_norm(components)
                / math.sqrt(float(components.numel()))
            )
    return torch.stack(rows, dim=0).reshape(mask.shape[:-1])


@torch.jit.script
def _gdn_proposal_records_1d(
    state: Tensor,
    key: Tensor,
    value: Tensor,
    eta: Tensor,
    gamma: Tensor,
) -> Tensor:
    """Evaluate GDN with eager's scalar-record operator boundaries."""

    batch_rows = torch.jit.annotate(List[Tensor], [])
    for batch_index in range(state.size(0)):
        group_rows = torch.jit.annotate(List[Tensor], [])
        for group_index in range(state.size(1)):
            old = state[batch_index, group_index]
            token_key = key[batch_index, group_index]
            decayed = gamma[batch_index, group_index] * old
            prediction = torch.matmul(decayed.transpose(-2, -1), token_key)
            error = value[batch_index, group_index] - prediction
            proposal = decayed + eta[batch_index, group_index] * torch.outer(
                token_key, error
            )
            group_rows.append(proposal)
        batch_rows.append(torch.stack(group_rows, dim=0))
    return torch.stack(batch_rows, dim=0)


@torch.jit.script
def _sd_pre_active_scan(
    normalized: Tensor,
    reached: Tensor,
    base_readouts: Tensor,
    initial_states: List[Tensor],
    state_types: List[int],
    group_locals: List[Tensor],
    update_a: List[Tensor],
    update_b: List[Tensor],
    gdn_keys: List[Tensor],
    gdn_values: List[Tensor],
    gdn_etas: List[Tensor],
    gdn_gammas: List[Tensor],
    attention_initial_keys: List[Tensor],
    attention_initial_values: List[Tensor],
    attention_initial_mask: List[Tensor],
    attention_event_keys: List[Tensor],
    attention_event_values: List[Tensor],
    read_weights: List[Tensor],
    read_biases: List[Tensor],
    read_is_summary: List[int],
    score_type: int,
    score_shared: bool,
    fixed_scores: Tensor,
    score_linear_weight: Tensor,
    score_linear_bias: Tensor,
    score_hidden_weight: Tensor,
    score_hidden_bias: Tensor,
    score_output_weight: Tensor,
    score_output_bias: Tensor,
    requested_k: int,
) -> Tensor:
    """Discover the causal hard route for an SD/pre region.

    The discrete route is deliberately evaluated without autograd by the
    caller.  This scripted ``prim::Loop`` is the irreducibly sequential part
    of pre-state routing; it is not a Python Token loop.  Once the route is
    known, the differentiable execution uses the ordinary packed affine and
    bounded-window scans, because Top-K itself has no VJP.

    State type codes are 0=EMA, 1=GDN, and 2=window Attention.
    """

    states = torch.jit.annotate(List[Tensor], [])
    attention_keys = torch.jit.annotate(List[Tensor], [])
    attention_values = torch.jit.annotate(List[Tensor], [])
    attention_masks = torch.jit.annotate(List[Tensor], [])
    for group_index in range(len(initial_states)):
        states.append(initial_states[group_index].clone())
        attention_keys.append(attention_initial_keys[group_index].clone())
        attention_values.append(attention_initial_values[group_index].clone())
        attention_masks.append(attention_initial_mask[group_index].clone())

    active_by_token = torch.jit.annotate(List[Tensor], [])
    batch = normalized.size(0)
    node_count = normalized.size(2)
    for token_index in range(normalized.size(1)):
        readout = base_readouts[:, token_index].clone()
        normalized_token = normalized[:, token_index]

        for group_index in range(len(states)):
            locals_index = group_locals[group_index]
            state_type = state_types[group_index]
            if state_type == 2:
                keys = attention_keys[group_index]
                values = attention_values[group_index]
                history_mask = attention_masks[group_index]
                state_feature = _attention_summary_records_1d(
                    keys, values, history_mask
                ).unsqueeze(-1)
            else:
                flat_state = states[group_index].flatten(start_dim=2)
                if read_is_summary[group_index] != 0:
                    state_feature = _rms_group_records_1d(flat_state).unsqueeze(-1)
                else:
                    state_feature = flat_state

            content = normalized_token.index_select(1, locals_index)
            read_input = torch.cat((content, state_feature), dim=-1)
            group_width = read_input.size(1)
            read_dim = read_weights[group_index].size(1)
            columns = torch.arange(
                group_width, device=read_input.device
            ).repeat(batch)
            projected = _record_group_linear_1d(
                read_input.reshape(batch * group_width, read_input.size(-1)),
                read_weights[group_index],
                read_biases[group_index],
                columns,
            ).reshape(batch, group_width, read_dim)
            readout = readout.index_copy(1, locals_index, projected)

        candidates = reached[:, token_index]
        pairs = candidates.nonzero()
        logits_flat = readout.new_zeros((batch * node_count,))
        if pairs.numel() > 0:
            event = pairs[:, 0]
            local = pairs[:, 1]
            values = readout[event, local]
            if score_type == 0:
                scores = fixed_scores[local]
            elif score_type == 1:
                scores = values.sum(dim=-1)
            elif score_type == 2:
                if score_shared:
                    dense_scores = _shared_linear_scores_by_event(
                        readout,
                        candidates,
                        score_linear_weight,
                        score_linear_bias,
                    )
                    scores = dense_scores[event, local]
                else:
                    scores = _record_group_linear_1d(
                        values,
                        score_linear_weight,
                        score_linear_bias,
                        local,
                    ).squeeze(-1)
            else:
                if score_shared:
                    dense_scores = _shared_mlp_scores_by_event(
                        readout,
                        candidates,
                        score_hidden_weight,
                        score_hidden_bias,
                        score_output_weight,
                        score_output_bias,
                    )
                    scores = dense_scores[event, local]
                else:
                    scores = _unshared_mlp_scores_by_record(
                        values,
                        score_hidden_weight,
                        score_hidden_bias,
                        score_output_weight,
                        score_output_bias,
                        local,
                    )
            logits_flat = logits_flat.index_copy(
                0, event * node_count + local, scores
            )

        logits = logits_flat.reshape(batch, node_count)
        candidate_index = torch.arange(node_count, device=reached.device)
        greater = logits.unsqueeze(1) > logits.unsqueeze(2)
        earlier = candidate_index.unsqueeze(0) < candidate_index.unsqueeze(1)
        equal = logits.unsqueeze(1) == logits.unsqueeze(2)
        both_candidates = candidates.unsqueeze(1) & candidates.unsqueeze(2)
        rank = (both_candidates & (greater | (equal & earlier))).sum(dim=2)
        active = candidates & (rank < requested_k)
        active_by_token.append(active)

        for group_index in range(len(states)):
            locals_index = group_locals[group_index]
            selected = active.index_select(1, locals_index)
            state_type = state_types[group_index]
            if state_type == 2:
                old_keys = attention_keys[group_index]
                old_values = attention_values[group_index]
                old_mask = attention_masks[group_index]
                shifted_keys = torch.cat(
                    (
                        old_keys[:, :, 1:],
                        attention_event_keys[group_index][:, token_index].unsqueeze(2),
                    ),
                    dim=2,
                )
                shifted_values = torch.cat(
                    (
                        old_values[:, :, 1:],
                        attention_event_values[group_index][:, token_index].unsqueeze(2),
                    ),
                    dim=2,
                )
                shifted_mask = torch.cat(
                    (old_mask[:, :, 1:], torch.ones_like(selected).unsqueeze(-1)),
                    dim=2,
                )
                history_count = old_mask.sum(dim=-1)
                slot = torch.arange(old_mask.size(-1), device=old_mask.device)
                insertion = slot.reshape(1, 1, -1) == history_count.unsqueeze(-1)
                inserted_keys = torch.where(
                    insertion.unsqueeze(-1),
                    attention_event_keys[group_index][:, token_index].unsqueeze(2),
                    old_keys,
                )
                inserted_values = torch.where(
                    insertion.unsqueeze(-1),
                    attention_event_values[group_index][:, token_index].unsqueeze(2),
                    old_values,
                )
                inserted_mask = old_mask | insertion
                full = (history_count >= old_mask.size(-1)).unsqueeze(-1)
                proposed_keys = torch.where(
                    full.unsqueeze(-1), shifted_keys, inserted_keys
                )
                proposed_values = torch.where(
                    full.unsqueeze(-1), shifted_values, inserted_values
                )
                proposed_mask = torch.where(full, shifted_mask, inserted_mask)
                selected_value = selected.unsqueeze(-1).unsqueeze(-1)
                attention_keys[group_index] = torch.where(
                    selected_value, proposed_keys, old_keys
                )
                attention_values[group_index] = torch.where(
                    selected_value, proposed_values, old_values
                )
                attention_masks[group_index] = torch.where(
                    selected.unsqueeze(-1), proposed_mask, old_mask
                )
            else:
                old_state = states[group_index]
                if state_type == 0:
                    proposal = (
                        update_a[group_index][:, token_index] * old_state
                        + update_b[group_index][:, token_index]
                    )
                else:
                    proposal = _gdn_proposal_records_1d(
                        old_state,
                        gdn_keys[group_index][:, token_index],
                        gdn_values[group_index][:, token_index],
                        gdn_etas[group_index][:, token_index],
                        gdn_gammas[group_index][:, token_index],
                    )
                selected_value = selected
                while selected_value.dim() < proposal.dim():
                    selected_value = selected_value.unsqueeze(-1)
                states[group_index] = torch.where(
                    selected_value, proposal, old_state
                )

    return torch.stack(active_by_token, dim=1)


@torch.jit.script
def _masked_affine_state_scan(
    initial: Tensor,
    update_a: Tensor,
    update_b: Tensor,
    observed: Tensor,
    matrix_update: bool,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Differentiable compiled causal scan for a fixed hard route."""

    before_rows = torch.jit.annotate(List[Tensor], [])
    proposal_rows = torch.jit.annotate(List[Tensor], [])
    after_rows = torch.jit.annotate(List[Tensor], [])
    state = initial
    for token_index in range(observed.size(1)):
        before_rows.append(state)
        if matrix_update:
            proposal = torch.matmul(update_a[:, token_index], state) + update_b[:, token_index]
        else:
            proposal = update_a[:, token_index] * state + update_b[:, token_index]
        proposal_rows.append(proposal)
        selected = observed[:, token_index]
        while selected.dim() < proposal.dim():
            selected = selected.unsqueeze(-1)
        state = torch.where(selected, proposal, state)
        after_rows.append(state)
    return (
        torch.stack(before_rows, dim=1),
        torch.stack(proposal_rows, dim=1),
        torch.stack(after_rows, dim=1),
        state,
    )


@torch.jit.script
def _masked_gdn_state_scan(
    initial: Tensor,
    key: Tensor,
    value: Tensor,
    eta: Tensor,
    gamma: Tensor,
    observed: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Evaluate the specified GDN recurrence in causal token order.

    Rewriting the update as one affine matrix and composing those matrices is
    algebraically valid, but it changes FP32 rounding relative to the public
    GDN equation.  A compiled scan keeps batch/node execution tensorized while
    preserving the same operation order as token decode.
    """

    before_rows = torch.jit.annotate(List[Tensor], [])
    proposal_rows = torch.jit.annotate(List[Tensor], [])
    after_rows = torch.jit.annotate(List[Tensor], [])
    state = initial
    for token_index in range(observed.size(1)):
        before_rows.append(state)
        proposal = _gdn_proposal_records_1d(
            state,
            key[:, token_index],
            value[:, token_index],
            eta[:, token_index],
            gamma[:, token_index],
        )
        proposal_rows.append(proposal)
        selected = observed[:, token_index]
        while selected.dim() < proposal.dim():
            selected = selected.unsqueeze(-1)
        state = torch.where(selected, proposal, state)
        after_rows.append(state)
    return (
        torch.stack(before_rows, dim=1),
        torch.stack(proposal_rows, dim=1),
        torch.stack(after_rows, dim=1),
        state,
    )


def _scalar_affine_scan(a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
    """Inclusive associative scan of ``x <- a*x+b`` along dimension 1."""

    prefix_a, prefix_b = a, b
    offset = 1
    length = int(a.shape[1])
    while offset < length:
        composed_a = prefix_a[:, offset:] * prefix_a[:, :-offset]
        composed_b = prefix_b[:, offset:] + prefix_a[:, offset:] * prefix_b[:, :-offset]
        prefix_a = torch.cat((prefix_a[:, :offset], composed_a), dim=1)
        prefix_b = torch.cat((prefix_b[:, :offset], composed_b), dim=1)
        offset <<= 1
    return prefix_a, prefix_b


def _matrix_affine_scan(a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
    """Inclusive associative scan of ``X <- A@X+B`` along dimension 1."""

    prefix_a, prefix_b = a, b
    offset = 1
    length = int(a.shape[1])
    while offset < length:
        current_a = prefix_a[:, offset:]
        composed_a = current_a.matmul(prefix_a[:, :-offset])
        composed_b = current_a.matmul(prefix_b[:, :-offset]) + prefix_b[:, offset:]
        prefix_a = torch.cat((prefix_a[:, :offset], composed_a), dim=1)
        prefix_b = torch.cat((prefix_b[:, :offset], composed_b), dim=1)
        offset <<= 1
    return prefix_a, prefix_b


def _merge_message_batches(
    batches: Sequence[_MessageBatch], reference: Tensor
) -> _MessageBatch:
    if batches:
        return _MessageBatch(
            torch.cat(tuple(batch.hidden for batch in batches), dim=0),
            torch.cat(tuple(batch.event for batch in batches), dim=0),
            torch.cat(tuple(batch.target_local for batch in batches), dim=0),
            torch.cat(tuple(batch.parent_slot for batch in batches), dim=0),
            torch.cat(tuple(batch.edge_index for batch in batches), dim=0),
        )
    empty_index = torch.empty((0,), dtype=torch.int64, device=reference.device)
    return _MessageBatch(
        reference.new_empty((0, reference.shape[-1])),
        empty_index,
        empty_index,
        empty_index,
        empty_index,
    )


def _region_message_view(
    records: _MessageBatch,
    *,
    event_count: int,
    node_count: int,
    max_fanin: int,
    reference: Tensor,
) -> Tuple[Tensor, Tensor]:
    flat_count = event_count * node_count * max_fanin
    flat_hidden = reference.new_zeros((flat_count, reference.shape[-1]))
    flat_mask = torch.zeros((flat_count,), dtype=torch.bool, device=reference.device)
    if records.event.numel():
        destination = (
            (records.event * node_count + records.target_local) * max_fanin
            + records.parent_slot
        )
        flat_hidden = flat_hidden.index_copy(0, destination, records.hidden)
        flat_mask = flat_mask.index_fill(0, destination, True)
    return (
        flat_hidden.reshape(event_count, node_count, max_fanin, reference.shape[-1]),
        flat_mask.reshape(event_count, node_count, max_fanin),
    )


def _aggregate_region(
    model: SettleGraph,
    schedule: _RegionSchedule,
    messages: Tensor,
    message_mask: Tensor,
    *,
    batch: int,
    length: int,
) -> Tensor:
    event_count, node_count, max_fanin, width = messages.shape
    result = messages.new_zeros((event_count, node_count, width))
    for group in schedule.aggregate_groups:
        locals_index = _device_index(group.node_locals, messages)
        grouped_messages = messages[:, locals_index]
        grouped_mask = message_mask[:, locals_index]
        grouped_flat = grouped_messages.reshape(
            event_count * len(group.node_locals), max_fanin, width
        )
        mask_flat = grouped_mask.reshape(
            event_count * len(group.node_locals), max_fanin
        )
        aggregate_type = str(group.signature[0])
        if aggregate_type == "mean":
            value = _mean_aggregate_records_1d(
                grouped_flat, mask_flat
            ).reshape(event_count, len(group.node_locals), width)
        elif aggregate_type in {"learned_convex", "edge_softmax"}:
            score_values: List[Tensor] = []
            for local in group.node_locals:
                module = model.receivers[safe_module_key(schedule.node_ids[local])]
                zero = module.input_norm.weight.new_zeros(())
                for slot in range(max_fanin):
                    if slot < len(schedule.parent_edge_ids[local]):
                        edge_id = schedule.parent_edge_ids[local][slot]
                        assert module.edge_scores is not None
                        score_values.append(module.edge_scores[safe_module_key(edge_id)])
                    else:
                        score_values.append(zero)
                # Entry boundary Aggregate is an identity and owns no score.
            score_usage = grouped_mask.any(dim=0).reshape(-1)
            scores = _stack(tuple(score_values), score_usage).reshape(
                len(group.node_locals), max_fanin
            ).to(
                device=messages.device, dtype=messages.dtype
            )
            value = _weighted_aggregate_records_1d(
                grouped_flat,
                mask_flat,
                scores,
                len(group.node_locals),
            ).reshape(event_count, len(group.node_locals), width)
        elif aggregate_type == "edge_linear_mean":
            weight_values: List[Tensor] = []
            bias_values: List[Tensor] = []
            for local in group.node_locals:
                module = model.receivers[safe_module_key(schedule.node_ids[local])]
                identity = torch.eye(width, device=messages.device, dtype=messages.dtype)
                zero_bias = messages.new_zeros((width,))
                for slot in range(max_fanin):
                    if slot < len(schedule.parent_edge_ids[local]):
                        edge_id = schedule.parent_edge_ids[local][slot]
                        assert module.edge_transforms is not None
                        linear = module.edge_transforms[safe_module_key(edge_id)]
                        weight_values.append(linear.weight)
                        bias_values.append(linear.bias if linear.bias is not None else zero_bias)
                    else:
                        # This covers padding and the entry-boundary identity.
                        weight_values.append(identity)
                        bias_values.append(zero_bias)
            transform_usage = grouped_mask.any(dim=0).reshape(-1)
            packed_weight = _stack(tuple(weight_values), transform_usage).reshape(
                len(group.node_locals), max_fanin, width, width
            )
            packed_bias = _stack(tuple(bias_values), transform_usage).reshape(
                len(group.node_locals), max_fanin, width
            )
            edge_columns = torch.arange(
                len(group.node_locals) * max_fanin,
                device=grouped_messages.device,
            ).repeat(event_count)
            event_order = torch.arange(
                event_count, device=grouped_messages.device
            ).reshape(batch, length).transpose(0, 1).reshape(-1)
            ordered_messages = grouped_messages.index_select(0, event_order)
            transformed_ordered = _record_group_linear_1d(
                ordered_messages.reshape(-1, width),
                packed_weight.reshape(
                    len(group.node_locals) * max_fanin, width, width
                ),
                packed_bias.reshape(len(group.node_locals) * max_fanin, width),
                edge_columns,
            ).reshape(event_count, len(group.node_locals), max_fanin, width)
            transformed = torch.empty_like(transformed_ordered).index_copy(
                0, event_order, transformed_ordered
            )
            value = _mean_aggregate_records_1d(
                transformed.reshape(
                    event_count * len(group.node_locals), max_fanin, width
                ),
                mask_flat,
            ).reshape(event_count, len(group.node_locals), width)
        else:  # pragma: no cover - protected by support inspection
            raise AssertionError(aggregate_type)
        result = result.index_copy(1, locals_index, value)
    return result


def _aggregate_trace_occurrence(
    model: SettleGraph,
    node_id: str,
    messages: Sequence[Tensor],
    edge_ids: Sequence[str],
) -> Tensor:
    """Rebuild one semantic Aggregate without sharing another node's graph.

    The packed hot path necessarily combines independent node/event lanes in
    dense Tensors.  A slice of such a Tensor inherits the union of the lanes'
    autograd metadata, even when this particular Aggregate only consumed
    non-differentiable messages.  Exact trace materialization is diagnostic,
    so reconstruct its per-occurrence value before exposing it publicly.
    """

    if not messages or len(messages) != len(edge_ids):  # pragma: no cover
        raise AssertionError("invalid packed trace Aggregate occurrence")
    if len(messages) == 1 and edge_ids[0].startswith("boundary:"):
        return messages[0]

    module = model.receivers[safe_module_key(node_id)]
    stacked = torch.stack(tuple(messages), dim=0)
    if module.aggregate_type == "mean":
        return stacked.mean(dim=0)
    if module.aggregate_type in {"learned_convex", "edge_softmax"}:
        assert module.edge_scores is not None
        scores = torch.stack(
            tuple(
                _selective_input(
                    module.edge_scores[safe_module_key(edge_id)],
                    id(module.edge_scores[safe_module_key(edge_id)]),
                )
                for edge_id in edge_ids
            )
        ).to(device=stacked.device, dtype=stacked.dtype)
        weights = torch.softmax(scores, dim=0)
        return (weights.unsqueeze(-1) * stacked).sum(dim=0)
    if module.aggregate_type == "edge_linear_mean":
        assert module.edge_transforms is not None
        transformed = []
        for edge_id, message in zip(edge_ids, messages):
            linear = module.edge_transforms[safe_module_key(edge_id)]
            weight = _selective_input(linear.weight, id(linear.weight))
            bias = (
                _selective_input(linear.bias, id(linear.bias))
                if linear.bias is not None
                else None
            )
            transformed.append(F.linear(message, weight, bias))
        return torch.stack(tuple(transformed), dim=0).mean(dim=0)
    raise AssertionError(module.aggregate_type)  # pragma: no cover


def _normalize_region(
    model: SettleGraph, schedule: _RegionSchedule, hidden: Tensor, reached: Tensor
) -> Tensor:
    modules = tuple(
        model.receivers[safe_module_key(node_id)] for node_id in schedule.node_ids
    )
    weights = _stack(
        tuple(module.input_norm.weight for module in modules), reached.any(dim=0)
    )
    eps = hidden.new_tensor(tuple(module.input_norm.eps for module in modules))
    node_count = hidden.shape[1]
    columns = torch.arange(node_count, device=hidden.device).repeat(
        hidden.shape[0]
    )
    normalized = _rms_norm_selected_records_1d(
        hidden.reshape(-1, hidden.shape[-1]), weights, eps, columns
    )
    return normalized.reshape_as(hidden)


def _state_group_lookup(
    schedule: _RegionSchedule,
) -> Mapping[int, Tuple[int, int]]:
    result: Dict[int, Tuple[int, int]] = {}
    for group_index, group in enumerate(schedule.state_groups):
        for column, local in enumerate(group.node_locals):
            result[local] = (group_index, column)
    return result


def _selector_state_tensor(
    state_result: _TensorStateResult,
    column: int,
    timing: str,
) -> Tensor:
    source = state_result.before if timing == "pre" else state_result.proposal
    return source[:, :, column].reshape(source.shape[0] * source.shape[1], *source.shape[3:])


def _attention_state_summary(
    state_result: _AttentionStateResult,
    column: int,
    timing: str,
) -> Tensor:
    if timing == "pre":
        keys, values, mask = (
            state_result.before_keys[:, :, column],
            state_result.before_values[:, :, column],
            state_result.before_mask[:, :, column],
        )
    else:
        keys, values, mask = (
            state_result.after_keys[:, :, column],
            state_result.after_values[:, :, column],
            state_result.after_mask[:, :, column],
        )
    return _attention_summary_records_1d(keys, values, mask).reshape(-1, 1)


def _selector_readouts(
    model: SettleGraph,
    schedule: _RegionSchedule,
    normalized: Tensor,
    reached: Tensor,
    state_results: Sequence[Optional[_StateResult]],
    timing: str,
) -> Tensor:
    read_dim = model.receivers[
        safe_module_key(schedule.node_ids[0])
    ].selector_read_dim
    event_count, node_count, width = normalized.shape
    result = normalized.new_zeros((event_count * node_count, read_dim))
    state_lookup = _state_group_lookup(schedule)
    for group in schedule.selector_read_groups:
        locals_index = _device_index(group.node_locals, normalized)
        pairs = reached[:, locals_index].nonzero(as_tuple=False)
        if not pairs.numel():
            continue
        event = pairs[:, 0]
        group_column = pairs[:, 1]
        local = locals_index[group_column]
        value = normalized[event, local]
        modules = tuple(
            model.receivers[safe_module_key(schedule.node_ids[item])]
            for item in group.node_locals
        )
        read_type = str(group.signature[0])
        usage = _column_usage(group_column, len(group.node_locals))
        if read_type == "content":
            readout = value
        elif read_type == "content_norm":
            readout = _rms_group_records_1d(value).unsqueeze(-1)
        elif read_type == "content_linear":
            readout = _record_linear(
                value, modules, group_column, "selector_read_linear", usage
            )
        elif read_type in {"content_state_linear", "content_state_summary_linear"}:
            state_rows: List[Tensor] = []
            for local_item in group.node_locals:
                location = state_lookup.get(local_item)
                if location is None:
                    if read_type != "content_state_summary_linear":  # pragma: no cover
                        raise AssertionError("fixed-shape selector Read requires Tensor state")
                    state_rows.append(
                        normalized.new_zeros((event_count, 1))
                    )
                    continue
                state_group_index, state_column = location
                state_result = state_results[state_group_index]
                assert state_result is not None
                if isinstance(state_result, _TensorStateResult):
                    tensor = _selector_state_tensor(state_result, state_column, timing)
                    if read_type == "content_state_linear":
                        tensor = tensor.reshape(event_count, -1)
                    else:
                        tensor = _rms_group_records_1d(
                            tensor.reshape(event_count, -1)
                        ).unsqueeze(-1)
                else:
                    if read_type != "content_state_summary_linear":
                        raise AssertionError("Attention requires a state summary selector Read")
                    tensor = _attention_state_summary(state_result, state_column, timing)
                state_rows.append(tensor)
            state_dense = torch.stack(tuple(state_rows), dim=1)
            state_value = state_dense[event, group_column]
            read_input = torch.cat((value, state_value), dim=-1)
            readout = _record_linear(
                read_input, modules, group_column, "selector_read_linear", usage
            )
        else:  # pragma: no cover
            raise AssertionError(read_type)
        destination = event * node_count + local
        result = result.index_copy(0, destination, readout)
    return result.reshape(event_count, node_count, read_dim)


def _score_and_route(
    model: SettleGraph,
    schedule: _RegionSchedule,
    readouts: Optional[Tensor],
    reached: Tensor,
) -> Tuple[Optional[Tensor], Tensor, Tensor]:
    event_count, node_count = reached.shape
    if schedule.forced_singleton:
        probabilities = reached.to(dtype=model.receivers[
            safe_module_key(schedule.node_ids[0])
        ].input_norm.weight.dtype)
        return None, probabilities, reached

    assert readouts is not None
    selector = model.selectors[safe_module_key(schedule.region_id)]
    pairs = reached.nonzero(as_tuple=False)
    logits_flat = readouts.new_zeros((event_count * node_count,))
    if pairs.numel():
        event, local = pairs[:, 0], pairs[:, 1]
        values = readouts[event, local]
        score_type = selector.score_type
        if score_type == "read_sum":
            scores = values.sum(dim=-1)
        elif score_type in {"fixed", "constant"}:
            if score_type == "constant":
                by_node = values.new_full(
                    (node_count,), float(selector.config.get("value", 0.0))
                )
            else:
                configured = selector.config.get(
                    "values_by_node", selector.config.get("values")
                )
                if isinstance(configured, Mapping):
                    by_node = values.new_tensor(
                        tuple(float(configured[node_id]) for node_id in schedule.node_ids)
                    )
                else:
                    by_node = values.new_tensor(tuple(float(item) for item in configured))
            scores = by_node[local]
        elif score_type == "linear":
            if selector.shared_parameters:
                assert selector.linear is not None
                bias = (
                    selector.linear.bias
                    if selector.linear.bias is not None
                    else values.new_empty((0,))
                )
                dense_scores = _shared_linear_scores_by_event(
                    readouts,
                    reached,
                    selector.linear.weight,
                    bias,
                )
                scores = dense_scores[event, local]
            else:
                usage = _column_usage(local, node_count)
                modules = tuple(
                    selector.linears[safe_module_key(node_id)]
                    for node_id in schedule.node_ids
                )
                weights = _stack(tuple(module.weight for module in modules), usage)
                biases = (
                    _stack(tuple(module.bias for module in modules), usage)
                    if modules[0].bias is not None
                    else values.new_empty((0,))
                )
                scores = _record_group_linear_1d(
                    values, weights, biases, local
                ).squeeze(-1)
        elif score_type == "mlp":
            if selector.shared_parameters:
                assert selector.hidden is not None and selector.out is not None
                hidden_bias = (
                    selector.hidden.bias
                    if selector.hidden.bias is not None
                    else values.new_empty((0,))
                )
                output_bias = (
                    selector.out.bias
                    if selector.out.bias is not None
                    else values.new_empty((0,))
                )
                dense_scores = _shared_mlp_scores_by_event(
                    readouts,
                    reached,
                    selector.hidden.weight,
                    hidden_bias,
                    selector.out.weight,
                    output_bias,
                )
                scores = dense_scores[event, local]
            else:
                usage = _column_usage(local, node_count)
                hidden_modules = tuple(
                    selector.hidden_layers[safe_module_key(node_id)]
                    for node_id in schedule.node_ids
                )
                output_modules = tuple(
                    selector.output_layers[safe_module_key(node_id)]
                    for node_id in schedule.node_ids
                )
                hidden_weight = _stack(
                    tuple(module.weight for module in hidden_modules), usage
                )
                hidden_bias = (
                    _stack(tuple(module.bias for module in hidden_modules), usage)
                    if hidden_modules[0].bias is not None
                    else values.new_empty((0,))
                )
                output_weight = _stack(
                    tuple(module.weight for module in output_modules), usage
                )
                output_bias = (
                    _stack(tuple(module.bias for module in output_modules), usage)
                    if output_modules[0].bias is not None
                    else values.new_empty((0,))
                )
                scores = _unshared_mlp_scores_by_record(
                    values,
                    hidden_weight,
                    hidden_bias,
                    output_weight,
                    output_bias,
                    local,
                )
        else:  # pragma: no cover
            raise AssertionError(score_type)
        if not bool(torch.isfinite(scores).all().item()):
            raise ExecutionContractError(
                f"region {schedule.region_id!r} produced non-finite logits"
            )
        logits_flat = logits_flat.index_copy(0, event * node_count + local, scores)
    logits = logits_flat.reshape(event_count, node_count)
    probabilities = _candidate_softmax_records_1d(logits, reached)
    candidate_index = torch.arange(node_count, device=reached.device)
    # [event, candidate i, candidate j]: j outranks i.
    greater = logits.unsqueeze(1) > logits.unsqueeze(2)
    earlier = candidate_index.unsqueeze(0) < candidate_index.unsqueeze(1)
    equal = logits.unsqueeze(1) == logits.unsqueeze(2)
    both_candidates = reached.unsqueeze(1) & reached.unsqueeze(2)
    rank = (both_candidates & (greater | (equal & earlier))).sum(dim=2)
    requested_k = int(model.plan.region_by_id(schedule.region_id).k_requested["value"])
    active = reached & (rank < requested_k)
    return logits, probabilities, active


def _state_for_compute_tensor(
    result: _TensorStateResult, column: int
) -> Tensor:
    before = result.before[:, :, column]
    proposal = result.proposal[:, :, column]
    observed = result.observed[:, :, column]
    while observed.ndim < before.ndim:
        observed = observed.unsqueeze(-1)
    return torch.where(observed, proposal, before).reshape(
        before.shape[0] * before.shape[1], *before.shape[2:]
    )


def _attention_components(
    result: _AttentionStateResult, column: int, *, after: bool
) -> Tuple[Tensor, Tensor, Tensor]:
    if after:
        batch, length, _, window, key_dim = result.after_keys.shape
        value_dim = result.after_values.shape[-1]
        return (
            result.after_keys[:, :, column].reshape(batch * length, window, key_dim),
            result.after_values[:, :, column].reshape(batch * length, window, value_dim),
            result.after_mask[:, :, column].reshape(batch * length, window),
        )
    batch, length, _, window, key_dim = result.before_keys.shape
    value_dim = result.before_values.shape[-1]
    return (
        result.before_keys[:, :, column].reshape(batch * length, window, key_dim),
        result.before_values[:, :, column].reshape(batch * length, window, value_dim),
        result.before_mask[:, :, column].reshape(batch * length, window),
    )


def _compute_active(
    model: SettleGraph,
    schedule: _RegionSchedule,
    hidden: Tensor,
    normalized: Tensor,
    active: Tensor,
    probabilities: Tensor,
    state_results: Sequence[Optional[_StateResult]],
    *,
    record_occurrences: bool = False,
) -> Tuple[Tensor, Tensor, int, Tuple[_ComputeBatch, ...]]:
    event_count, node_count, width = hidden.shape
    computed_flat = hidden.new_zeros((event_count * node_count, width))
    emitted_flat = hidden.new_zeros((event_count * node_count, width))
    state_lookup = _state_group_lookup(schedule)
    active_rows = 0
    occurrence_batches: List[_ComputeBatch] = []
    for group in schedule.compute_groups:
        locals_index = _device_index(group.node_locals, hidden)
        pairs = active[:, locals_index].nonzero(as_tuple=False)
        if not pairs.numel():
            continue
        event, group_column = pairs[:, 0], pairs[:, 1]
        local = locals_index[group_column]
        active_rows += int(pairs.shape[0])
        modules = tuple(
            model.receivers[safe_module_key(schedule.node_ids[item])]
            for item in group.node_locals
        )
        usage = _column_usage(group_column, len(group.node_locals))
        input_hidden = hidden[event, local]
        input_normalized = normalized[event, local]
        compute_type = str(group.signature[0])
        ffn_read_type = str(group.signature[2])
        if ffn_read_type == "zero" or modules[0].update_type == "none":
            ffn_read = input_hidden.new_zeros(input_hidden.shape)
        else:
            state_rows: List[Any] = []
            for local_item in group.node_locals:
                state_group_index, state_column = state_lookup[local_item]
                state_result = state_results[state_group_index]
                assert state_result is not None
                if isinstance(state_result, _TensorStateResult):
                    state_rows.append(_state_for_compute_tensor(state_result, state_column))
                else:
                    state_rows.append(_attention_components(state_result, state_column, after=True))
            if modules[0].update_type == "ema":
                dense_state = torch.stack(tuple(state_rows), dim=1)
                state = dense_state[event, group_column]
                ffn_read = _record_linear(
                    state, modules, group_column, "state_out", usage
                )
            elif modules[0].update_type == "gdn":
                dense_state = torch.stack(tuple(state_rows), dim=1)
                state = dense_state[event, group_column]
                query = _record_linear(
                    input_normalized, modules, group_column, "gdn_query", usage
                )
                eps = input_hidden.new_tensor(tuple(module.state_norm_eps for module in modules))[group_column]
                query = _normalize_selected_records_1d(query, eps)
                read = _gdn_read_records_1d(state, query)
                ffn_read = _record_linear(
                    read, modules, group_column, "gdn_out", usage
                )
            elif modules[0].update_type == "attention_window":
                key_dense = torch.stack(tuple(item[0] for item in state_rows), dim=1)
                value_dense = torch.stack(tuple(item[1] for item in state_rows), dim=1)
                mask_dense = torch.stack(tuple(item[2] for item in state_rows), dim=1)
                keys = key_dense[event, group_column]
                values = value_dense[event, group_column]
                mask = mask_dense[event, group_column]
                query = _record_linear(
                    input_normalized, modules, group_column, "attn_query", usage
                )
                eps = input_hidden.new_tensor(tuple(module.state_norm_eps for module in modules))[group_column]
                query = _normalize_selected_records_1d(query, eps)
                read = _attention_read_records_1d(
                    keys, values, mask, query
                )
                ffn_read = _record_linear(
                    read, modules, group_column, "attn_out", usage
                )
            else:  # pragma: no cover
                raise AssertionError(modules[0].update_type)

        if compute_type == "identity":
            computed = input_hidden
        elif compute_type == "affine_residual":
            computed = input_hidden + _record_linear(
                input_normalized, modules, group_column, "down_proj", usage
            )
        else:
            ffn_weights = _stack(
                tuple(module.ffn_norm.weight for module in modules), usage
            )
            ffn_eps = input_hidden.new_tensor(
                tuple(module.ffn_norm.eps for module in modules)
            )
            gate_modules = tuple(module.gate_proj for module in modules)
            up_modules = tuple(module.up_proj for module in modules)
            down_modules = tuple(module.down_proj for module in modules)
            if any(module is None for module in gate_modules):
                raise AssertionError("missing gate projection")
            if any(module is None for module in up_modules):
                raise AssertionError("missing up projection")
            if any(module is None for module in down_modules):
                raise AssertionError("missing down projection")
            gate_weights = _stack(
                tuple(module.weight for module in gate_modules), usage
            )
            gate_biases = (
                _stack(tuple(module.bias for module in gate_modules), usage)
                if gate_modules[0].bias is not None
                else input_hidden.new_empty((0,))
            )
            up_weights = _stack(
                tuple(module.weight for module in up_modules), usage
            )
            up_biases = (
                _stack(tuple(module.bias for module in up_modules), usage)
                if up_modules[0].bias is not None
                else input_hidden.new_empty((0,))
            )
            down_weights = _stack(
                tuple(module.weight for module in down_modules), usage
            )
            down_biases = (
                _stack(tuple(module.bias for module in down_modules), usage)
                if down_modules[0].bias is not None
                else input_hidden.new_empty((0,))
            )
            # NodeCompute is allowed to feed a later hard route, so even a
            # sub-tolerance reassociation is observable.  Keep the full eager
            # one-record RMSNorm/Linear/SiLU/residual sequence together.
            computed = _swiglu_compute_records_1d(
                input_hidden,
                ffn_read,
                ffn_weights,
                ffn_eps,
                gate_weights,
                gate_biases,
                up_weights,
                up_biases,
                down_weights,
                down_biases,
                group_column,
            )

        emit_type = str(group.signature[-1])
        probability = probabilities[event, local]
        if emit_type == "hard":
            emitted = computed
        elif emit_type == "hst":
            zeta_by_node = input_hidden.new_tensor(
                tuple(float(module.emit_config.get("zeta", 1.0)) for module in modules)
            )
            rho = 1.0 + zeta_by_node[group_column] * (probability - probability.detach())
            emitted = input_hidden + rho.unsqueeze(-1) * (computed - input_hidden)
        elif emit_type == "softp":
            emitted = input_hidden + probability.unsqueeze(-1) * (computed - input_hidden)
        else:  # pragma: no cover
            raise AssertionError(emit_type)
        if record_occurrences:
            # Keep the formula-batch values before index_copy combines logical
            # lanes with different autograd provenance into one dense Tensor.
            # Exact trace materialization can then expose the same per-event
            # requires_grad metadata as the eager interpreter without changing
            # the non-trace execution path.
            occurrence_batches.append(
                _ComputeBatch(event, local, computed, emitted)
            )
        destination = event * node_count + local
        computed_flat = computed_flat.index_copy(0, destination, computed)
        emitted_flat = emitted_flat.index_copy(0, destination, emitted)
    return (
        computed_flat.reshape(event_count, node_count, width),
        emitted_flat.reshape(event_count, node_count, width),
        active_rows,
        tuple(occurrence_batches),
    )


def _balance_for_region(
    schedule: _RegionSchedule,
    reached: Tensor,
    active: Tensor,
    probabilities: Tensor,
    route_mask: Tensor,
) -> Optional[BalanceRegionStats]:
    selected_events = route_mask & reached.any(dim=1)
    if not bool(selected_events.any().item()):
        return None
    candidate_count = reached.sum(dim=1, keepdim=True).clamp_min(1)
    active_count = active.sum(dim=1, keepdim=True).clamp_min(1)
    soft = probabilities[selected_events].sum(dim=0)
    availability = (
        reached.to(probabilities.dtype) / candidate_count.to(probabilities.dtype)
    )[selected_events].sum(dim=0).detach()
    hard = (
        active.to(probabilities.dtype) / active_count.to(probabilities.dtype)
    )[selected_events].sum(dim=0).detach()
    competition = int((selected_events & (reached.sum(dim=1) >= 2)).sum().item())
    return BalanceRegionStats(
        soft,
        availability,
        hard,
        event_count=int(selected_events.sum().item()),
        competition_count=competition,
        forced_active=schedule.forced_singleton,
    )


def _pack_tensor_initial(
    initial: StateStore,
    sequence_ids: Sequence[str],
    node_ids: Sequence[str],
    state_shape: Tuple[int, ...],
    reference: Tensor,
) -> Tensor:
    rows: List[Tensor] = []
    for sequence_id in sequence_ids:
        node_values: List[Tensor] = []
        for node_id in node_ids:
            value = initial.values.get((sequence_id, node_id))
            if value is None:
                node_values.append(reference.new_zeros(state_shape))
            elif isinstance(value, Tensor):
                node_values.append(
                    _selective_input(
                        value,
                        _state_source_token(sequence_id, node_id, "tensor"),
                    )
                )
            else:  # pragma: no cover - state validation owns the public error
                raise AssertionError("expected Tensor receiver state")
        rows.append(torch.stack(tuple(node_values), dim=0))
    return torch.stack(tuple(rows), dim=0)


def _run_tensor_state_group(
    model: SettleGraph,
    schedule: _RegionSchedule,
    group: _StateGroup,
    normalized_bt: Tensor,
    observed_bt: Tensor,
    initial: StateStore,
    sequence_ids: Sequence[str],
    *,
    sequential: bool = False,
) -> _TensorStateResult:
    node_ids = tuple(schedule.node_ids[local] for local in group.node_locals)
    modules = tuple(model.receivers[safe_module_key(node_id)] for node_id in node_ids)
    locals_index = _device_index(group.node_locals, normalized_bt)
    normalized = normalized_bt[:, :, locals_index]
    observed = observed_bt[:, :, locals_index]
    usage = observed.any(dim=(0, 1))
    before_initial = _pack_tensor_initial(
        initial, sequence_ids, node_ids, group.state_shape, normalized_bt
    )
    batch, length, group_width = observed.shape

    if group.update_type == "ema":
        observation = torch.tanh(
            _group_linear_records_1d(
                normalized, modules, "ema_observe", usage
            )
        )
        decay_values: List[Tensor] = []
        for module in modules:
            if module.ema_decay_logit is not None:
                decay_values.append(torch.sigmoid(module.ema_decay_logit))
            else:
                decay_values.append(module.ema_decay.to(normalized))
        decay = _stack(tuple(decay_values), usage).reshape(1, 1, group_width, 1)
        raw_a = decay.expand(batch, length, -1, -1)
        raw_b = (1.0 - decay) * observation
        before, proposal, after, final = _masked_affine_state_scan(
            before_initial, raw_a, raw_b, observed, False
        )
        return _TensorStateResult(before, proposal, after, final, observed)

    if group.update_type == "gdn":
        key = _group_linear_records_1d(
            normalized, modules, "gdn_key", usage
        )
        eps = normalized.new_tensor(tuple(module.state_norm_eps for module in modules))
        key = _normalize_group_records_1d(key, eps)
        value = _group_linear_records_1d(
            normalized, modules, "gdn_value", usage
        )
        eta = torch.sigmoid(
            _group_linear_records_1d(
                normalized, modules, "gdn_eta", usage
            ).squeeze(-1)
        )
        gamma_input = _group_linear_records_1d(
            normalized, modules, "gdn_gamma", usage
        ).squeeze(-1)
        beta = _stack(tuple(module.gdn_beta for module in modules), usage).reshape(
            1, 1, -1
        )
        gamma = torch.exp(-torch.exp(beta) * F.softplus(gamma_input))
        before, proposal, after, final = _masked_gdn_state_scan(
            before_initial, key, value, eta, gamma, observed
        )
        return _TensorStateResult(before, proposal, after, final, observed)

    raise AssertionError(group.update_type)


def _pack_attention_initial(
    initial: StateStore,
    sequence_ids: Sequence[str],
    node_ids: Sequence[str],
    *,
    window: int,
    key_dim: int,
    value_dim: int,
    reference: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    position_rows: List[Tensor] = []
    key_rows: List[Tensor] = []
    value_rows: List[Tensor] = []
    mask_rows: List[Tensor] = []
    for sequence_id in sequence_ids:
        positions_by_node: List[Tensor] = []
        keys_by_node: List[Tensor] = []
        values_by_node: List[Tensor] = []
        masks_by_node: List[Tensor] = []
        for node_id in node_ids:
            value = initial.values.get((sequence_id, node_id))
            positions = torch.zeros((window,), dtype=torch.int64, device=reference.device)
            keys = reference.new_zeros((window, key_dim))
            values = reference.new_zeros((window, value_dim))
            mask = torch.zeros((window,), dtype=torch.bool, device=reference.device)
            if value is not None:
                if not isinstance(value, AttentionState):  # pragma: no cover
                    raise AssertionError("expected AttentionState")
                length = value.length
                positions = positions.index_copy(0, torch.arange(length, device=reference.device), value.positions)
                keys = keys.index_copy(
                    0,
                    torch.arange(length, device=reference.device),
                    _selective_input(
                        value.keys,
                        _state_source_token(sequence_id, node_id, "keys"),
                    ),
                )
                values = values.index_copy(
                    0,
                    torch.arange(length, device=reference.device),
                    _selective_input(
                        value.values,
                        _state_source_token(sequence_id, node_id, "values"),
                    ),
                )
                mask = mask.index_fill(0, torch.arange(length, device=reference.device), True)
            positions_by_node.append(positions)
            keys_by_node.append(keys)
            values_by_node.append(values)
            masks_by_node.append(mask)
        position_rows.append(torch.stack(tuple(positions_by_node)))
        key_rows.append(torch.stack(tuple(keys_by_node)))
        value_rows.append(torch.stack(tuple(values_by_node)))
        mask_rows.append(torch.stack(tuple(masks_by_node)))
    return (
        torch.stack(tuple(position_rows)),
        torch.stack(tuple(key_rows)),
        torch.stack(tuple(value_rows)),
        torch.stack(tuple(mask_rows)),
    )


def _gather_attention_history(
    source_positions: Tensor,
    source_keys: Tensor,
    source_values: Tensor,
    source_mask: Tensor,
    totals: Tensor,
    window: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # Sources are [B,G,S,*], totals are [B,T,G].  Desired history order is
    # oldest-to-newest among the last W valid sources.
    batch, groups, source_count = source_mask.shape
    length = totals.shape[1]
    ordinal = torch.arange(window, device=source_mask.device).reshape(1, 1, 1, window)
    requested = totals.permute(0, 2, 1).unsqueeze(-1) - window + 1 + ordinal
    valid = requested > 0
    cumulative = source_mask.to(torch.int64).cumsum(dim=-1)
    indices = torch.searchsorted(
        cumulative.contiguous(),
        requested.clamp_min(1).reshape(batch, groups, length * window).contiguous(),
        right=False,
    ).reshape(batch, groups, length, window).clamp_max(source_count - 1)
    positions = torch.gather(
        source_positions.unsqueeze(2).expand(batch, groups, length, source_count),
        3,
        indices,
    )
    keys = torch.gather(
        source_keys.unsqueeze(2).expand(batch, groups, length, source_count, source_keys.shape[-1]),
        3,
        indices.unsqueeze(-1).expand(-1, -1, -1, -1, source_keys.shape[-1]),
    )
    values = torch.gather(
        source_values.unsqueeze(2).expand(batch, groups, length, source_count, source_values.shape[-1]),
        3,
        indices.unsqueeze(-1).expand(-1, -1, -1, -1, source_values.shape[-1]),
    )
    selected_source_mask = torch.gather(
        source_mask.unsqueeze(2).expand(batch, groups, length, source_count), 3, indices
    )
    valid = valid & selected_source_mask
    positions = positions.masked_fill(~valid, 0)
    keys = keys.masked_fill(~valid.unsqueeze(-1), 0.0)
    values = values.masked_fill(~valid.unsqueeze(-1), 0.0)
    return (
        positions.permute(0, 2, 1, 3),
        keys.permute(0, 2, 1, 3, 4),
        values.permute(0, 2, 1, 3, 4),
        valid.permute(0, 2, 1, 3),
    )


def _gather_attention_final(
    source_positions: Tensor,
    source_keys: Tensor,
    source_values: Tensor,
    source_mask: Tensor,
    totals: Tensor,
    window: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # Reuse the event gather with a synthetic length-one target.
    positions, keys, values, mask = _gather_attention_history(
        source_positions,
        source_keys,
        source_values,
        source_mask,
        totals.unsqueeze(1),
        window,
    )
    return positions[:, 0], keys[:, 0], values[:, 0], mask[:, 0]


def _run_attention_state_group(
    model: SettleGraph,
    schedule: _RegionSchedule,
    group: _StateGroup,
    normalized_bt: Tensor,
    observed_bt: Tensor,
    token_positions: Tensor,
    initial: StateStore,
    sequence_ids: Sequence[str],
) -> _AttentionStateResult:
    node_ids = tuple(schedule.node_ids[local] for local in group.node_locals)
    modules = tuple(model.receivers[safe_module_key(node_id)] for node_id in node_ids)
    locals_index = _device_index(group.node_locals, normalized_bt)
    normalized = normalized_bt[:, :, locals_index]
    observed = observed_bt[:, :, locals_index]
    usage = observed.any(dim=(0, 1))
    window = modules[0].attn_window
    key_dim = modules[0].attn_key_dim
    value_dim = modules[0].attn_value_dim
    initial_positions, initial_keys, initial_values, initial_mask = _pack_attention_initial(
        initial,
        sequence_ids,
        node_ids,
        window=window,
        key_dim=key_dim,
        value_dim=value_dim,
        reference=normalized_bt,
    )
    event_keys = _group_linear_records_1d(
        normalized, modules, "attn_key", usage
    )
    eps = normalized.new_tensor(tuple(module.state_norm_eps for module in modules))
    event_keys = _normalize_group_records_1d(event_keys, eps)
    event_values = _group_linear_records_1d(
        normalized, modules, "attn_value", usage
    )
    # Source order is fixed initial history, then input Token order.  False
    # execution/Observe positions stay as holes and never enter a history.
    source_positions = torch.cat(
        (
            initial_positions,
            token_positions.unsqueeze(1).expand(-1, len(node_ids), -1),
        ),
        dim=-1,
    )
    source_keys = torch.cat((initial_keys, event_keys.permute(0, 2, 1, 3)), dim=2)
    source_values = torch.cat((initial_values, event_values.permute(0, 2, 1, 3)), dim=2)
    source_mask = torch.cat((initial_mask, observed.permute(0, 2, 1)), dim=-1)
    initial_count = initial_mask.sum(dim=-1)
    cumulative_observe = observed.to(torch.int64).cumsum(dim=1)
    after_total = initial_count.unsqueeze(1) + cumulative_observe
    before_total = after_total - observed.to(torch.int64)
    before = _gather_attention_history(
        source_positions, source_keys, source_values, source_mask, before_total, window
    )
    after = _gather_attention_history(
        source_positions, source_keys, source_values, source_mask, after_total, window
    )
    final = _gather_attention_final(
        source_positions,
        source_keys,
        source_values,
        source_mask,
        initial_count + observed.to(torch.int64).sum(dim=1),
        window,
    )
    return _AttentionStateResult(
        *before,
        *after,
        *final,
        observed,
    )


def _sd_pre_score_inputs(
    model: SettleGraph,
    schedule: _RegionSchedule,
    reference: Tensor,
) -> _ScanScoreInputs:
    """Pack one selector for the no-grad scripted route discovery."""

    selector = model.selectors[safe_module_key(schedule.region_id)]
    node_count = len(schedule.node_ids)
    empty = reference.new_empty((0,))
    zeros = reference.new_zeros((node_count,))
    score_type = selector.score_type
    if score_type in {"fixed", "constant"}:
        if score_type == "constant":
            fixed = reference.new_full(
                (node_count,), float(selector.config.get("value", 0.0))
            )
        else:
            configured = selector.config.get(
                "values_by_node", selector.config.get("values")
            )
            if isinstance(configured, Mapping):
                fixed = reference.new_tensor(
                    tuple(float(configured[node_id]) for node_id in schedule.node_ids)
                )
            else:
                fixed = reference.new_tensor(tuple(float(item) for item in configured))
        return _ScanScoreInputs(0, selector.shared_parameters, fixed, empty, empty, empty, empty, empty, empty)

    if score_type == "read_sum":
        return _ScanScoreInputs(1, selector.shared_parameters, zeros, empty, empty, empty, empty, empty, empty)

    if score_type == "linear":
        if selector.shared_parameters:
            assert selector.linear is not None
            weight = selector.linear.weight
            bias = selector.linear.bias if selector.linear.bias is not None else empty
        else:
            modules = tuple(
                selector.linears[safe_module_key(node_id)]
                for node_id in schedule.node_ids
            )
            weight = torch.stack(tuple(module.weight for module in modules))
            bias = (
                torch.stack(tuple(module.bias for module in modules))
                if modules[0].bias is not None
                else empty
            )
        return _ScanScoreInputs(2, selector.shared_parameters, zeros, weight, bias, empty, empty, empty, empty)

    if score_type == "mlp":
        if selector.shared_parameters:
            assert selector.hidden is not None and selector.out is not None
            hidden_weight = selector.hidden.weight
            hidden_bias = selector.hidden.bias if selector.hidden.bias is not None else empty
            output_weight = selector.out.weight
            output_bias = selector.out.bias if selector.out.bias is not None else empty
        else:
            hidden_modules = tuple(
                selector.hidden_layers[safe_module_key(node_id)]
                for node_id in schedule.node_ids
            )
            output_modules = tuple(
                selector.output_layers[safe_module_key(node_id)]
                for node_id in schedule.node_ids
            )
            hidden_weight = torch.stack(tuple(module.weight for module in hidden_modules))
            hidden_bias = (
                torch.stack(tuple(module.bias for module in hidden_modules))
                if hidden_modules[0].bias is not None
                else empty
            )
            output_weight = torch.stack(tuple(module.weight for module in output_modules))
            output_bias = (
                torch.stack(tuple(module.bias for module in output_modules))
                if output_modules[0].bias is not None
                else empty
            )
        return _ScanScoreInputs(
            3,
            selector.shared_parameters,
            zeros,
            empty,
            empty,
            hidden_weight,
            hidden_bias,
            output_weight,
            output_bias,
        )
    raise AssertionError(score_type)  # pragma: no cover - support inspection owns this


def _sd_pre_base_readouts(
    model: SettleGraph,
    schedule: _RegionSchedule,
    normalized_bt: Tensor,
) -> Tensor:
    """Build valid zero-state readouts; stateful rows are replaced in scan."""

    rows: List[Tensor] = []
    for node_id in schedule.node_ids:
        module = model.receivers[safe_module_key(node_id)]
        linear = module.selector_read_linear
        if linear is None:  # pragma: no cover - pre timing is Plan-validated
            raise AssertionError("pre-state selector Read must own a linear projection")
        if module.selector_read_type == "content_state_linear":
            state_width = math.prod(module.state_shape)
        else:
            state_width = 1
        content = normalized_bt[:, :, len(rows)]
        zero_state = content.new_zeros((*content.shape[:-1], state_width))
        read_input = torch.cat((content, zero_state), dim=-1)
        columns = torch.zeros(
            (read_input.shape[0] * read_input.shape[1],),
            dtype=torch.int64,
            device=read_input.device,
        )
        bias = (
            linear.bias.unsqueeze(0)
            if linear.bias is not None
            else read_input.new_empty((0,))
        )
        rows.append(
            _record_group_linear_1d(
                read_input.reshape(-1, read_input.shape[-1]),
                linear.weight.unsqueeze(0),
                bias,
                columns,
            ).reshape(*read_input.shape[:-1], linear.out_features)
        )
    return torch.stack(tuple(rows), dim=2)


def _discover_sd_pre_active(
    model: SettleGraph,
    schedule: _RegionSchedule,
    normalized: Tensor,
    reached: Tensor,
    *,
    batch: int,
    length: int,
    initial: StateStore,
    sequence_ids: Sequence[str],
) -> Tensor:
    """Return the causal hard route with one compiled Token recurrence."""

    node_count = len(schedule.node_ids)
    normalized_bt = normalized.reshape(batch, length, node_count, normalized.shape[-1])
    reached_bt = reached.reshape(batch, length, node_count)
    base_readouts = _sd_pre_base_readouts(model, schedule, normalized_bt)

    initial_states: List[Tensor] = []
    state_types: List[int] = []
    group_locals: List[Tensor] = []
    update_a: List[Tensor] = []
    update_b: List[Tensor] = []
    gdn_keys: List[Tensor] = []
    gdn_values: List[Tensor] = []
    gdn_etas: List[Tensor] = []
    gdn_gammas: List[Tensor] = []
    attention_initial_keys: List[Tensor] = []
    attention_initial_values: List[Tensor] = []
    attention_initial_mask: List[Tensor] = []
    attention_event_keys: List[Tensor] = []
    attention_event_values: List[Tensor] = []
    read_weights: List[Tensor] = []
    read_biases: List[Tensor] = []
    read_is_summary: List[int] = []

    for group in schedule.state_groups:
        node_ids = tuple(schedule.node_ids[local] for local in group.node_locals)
        modules = tuple(model.receivers[safe_module_key(node_id)] for node_id in node_ids)
        locals_index = _device_index(group.node_locals, normalized)
        group_locals.append(locals_index)
        group_normalized = normalized_bt[:, :, locals_index]
        group_reached = reached_bt[:, :, locals_index]
        usage = group_reached.any(dim=(0, 1))
        linear_modules = tuple(module.selector_read_linear for module in modules)
        if any(linear is None for linear in linear_modules):  # pragma: no cover
            raise AssertionError("pre-state selector Read must own a projection")
        read_weights.append(torch.stack(tuple(linear.weight for linear in linear_modules)))
        read_biases.append(
            torch.stack(tuple(linear.bias for linear in linear_modules))
            if linear_modules[0].bias is not None
            else normalized.new_zeros((len(modules), modules[0].selector_read_dim))
        )
        read_is_summary.append(
            int(modules[0].selector_read_type == "content_state_summary_linear")
        )

        group_width = len(group.node_locals)
        if group.update_type == "ema":
            state_types.append(0)
            state = _pack_tensor_initial(
                initial, sequence_ids, node_ids, group.state_shape, normalized
            )
            observation = torch.tanh(
                _group_linear_records_1d(
                    group_normalized, modules, "ema_observe", usage
                )
            )
            decay_values: List[Tensor] = []
            for module in modules:
                if module.ema_decay_logit is not None:
                    decay_values.append(torch.sigmoid(module.ema_decay_logit))
                else:
                    decay_values.append(module.ema_decay.to(normalized))
            decay = _stack(tuple(decay_values), usage).reshape(1, 1, group_width, 1)
            initial_states.append(state)
            update_a.append(decay.expand(batch, length, -1, -1))
            update_b.append((1.0 - decay) * observation)
            gdn_keys.append(normalized.new_empty((0,)))
            gdn_values.append(normalized.new_empty((0,)))
            gdn_etas.append(normalized.new_empty((0,)))
            gdn_gammas.append(normalized.new_empty((0,)))
            attention_initial_keys.append(normalized.new_empty((batch, group_width, 0, 0)))
            attention_initial_values.append(normalized.new_empty((batch, group_width, 0, 0)))
            attention_initial_mask.append(
                torch.empty((batch, group_width, 0), dtype=torch.bool, device=normalized.device)
            )
            attention_event_keys.append(normalized.new_empty((batch, length, group_width, 0)))
            attention_event_values.append(normalized.new_empty((batch, length, group_width, 0)))
        elif group.update_type == "gdn":
            state_types.append(1)
            state = _pack_tensor_initial(
                initial, sequence_ids, node_ids, group.state_shape, normalized
            )
            key = _group_linear_records_1d(
                group_normalized, modules, "gdn_key", usage
            )
            eps = normalized.new_tensor(tuple(module.state_norm_eps for module in modules))
            key = _normalize_group_records_1d(key, eps)
            value = _group_linear_records_1d(
                group_normalized, modules, "gdn_value", usage
            )
            eta = torch.sigmoid(
                _group_linear_records_1d(
                    group_normalized, modules, "gdn_eta", usage
                ).squeeze(-1)
            )
            gamma_input = _group_linear_records_1d(
                group_normalized, modules, "gdn_gamma", usage
            ).squeeze(-1)
            beta = _stack(tuple(module.gdn_beta for module in modules), usage).reshape(
                1, 1, -1
            )
            gamma = torch.exp(-torch.exp(beta) * F.softplus(gamma_input))
            initial_states.append(state)
            update_a.append(normalized.new_empty((0,)))
            update_b.append(normalized.new_empty((0,)))
            gdn_keys.append(key)
            gdn_values.append(value)
            gdn_etas.append(eta)
            gdn_gammas.append(gamma)
            attention_initial_keys.append(normalized.new_empty((batch, group_width, 0, 0)))
            attention_initial_values.append(normalized.new_empty((batch, group_width, 0, 0)))
            attention_initial_mask.append(
                torch.empty((batch, group_width, 0), dtype=torch.bool, device=normalized.device)
            )
            attention_event_keys.append(normalized.new_empty((batch, length, group_width, 0)))
            attention_event_values.append(normalized.new_empty((batch, length, group_width, 0)))
        elif group.update_type == "attention_window":
            state_types.append(2)
            window = modules[0].attn_window
            key_dim = modules[0].attn_key_dim
            value_dim = modules[0].attn_value_dim
            _, keys, values, history_mask = _pack_attention_initial(
                initial,
                sequence_ids,
                node_ids,
                window=window,
                key_dim=key_dim,
                value_dim=value_dim,
                reference=normalized,
            )
            event_keys = _group_linear_records_1d(
                group_normalized, modules, "attn_key", usage
            )
            eps = normalized.new_tensor(tuple(module.state_norm_eps for module in modules))
            event_keys = _normalize_group_records_1d(event_keys, eps)
            event_values = _group_linear_records_1d(
                group_normalized, modules, "attn_value", usage
            )
            initial_states.append(normalized.new_empty((batch, group_width, 0)))
            update_a.append(normalized.new_empty((batch, length, group_width, 0)))
            update_b.append(normalized.new_empty((batch, length, group_width, 0)))
            gdn_keys.append(normalized.new_empty((0,)))
            gdn_values.append(normalized.new_empty((0,)))
            gdn_etas.append(normalized.new_empty((0,)))
            gdn_gammas.append(normalized.new_empty((0,)))
            attention_initial_keys.append(keys)
            attention_initial_values.append(values)
            attention_initial_mask.append(history_mask)
            attention_event_keys.append(event_keys)
            attention_event_values.append(event_values)
        else:  # pragma: no cover
            raise AssertionError(group.update_type)

    score = _sd_pre_score_inputs(model, schedule, normalized)
    region = model.plan.region_by_id(schedule.region_id)
    return _sd_pre_active_scan(
        normalized_bt,
        reached_bt,
        base_readouts,
        initial_states,
        state_types,
        group_locals,
        update_a,
        update_b,
        gdn_keys,
        gdn_values,
        gdn_etas,
        gdn_gammas,
        attention_initial_keys,
        attention_initial_values,
        attention_initial_mask,
        attention_event_keys,
        attention_event_values,
        read_weights,
        read_biases,
        read_is_summary,
        score.score_type,
        score.shared,
        score.fixed,
        score.linear_weight,
        score.linear_bias,
        score.hidden_weight,
        score.hidden_bias,
        score.output_weight,
        score.output_bias,
        int(region.k_requested["value"]),
    ).reshape(batch * length, node_count)


def _run_state_groups(
    model: SettleGraph,
    schedule: _RegionSchedule,
    normalized: Tensor,
    observed: Tensor,
    *,
    batch: int,
    length: int,
    token_positions: Tensor,
    initial: StateStore,
    sequence_ids: Sequence[str],
    sequential_tensor_scan: bool = False,
) -> Tuple[Optional[_StateResult], ...]:
    normalized_bt = normalized.reshape(batch, length, normalized.shape[1], normalized.shape[2])
    observed_bt = observed.reshape(batch, length, observed.shape[1])
    results: List[Optional[_StateResult]] = []
    for group in schedule.state_groups:
        if group.update_type in {"ema", "gdn"}:
            results.append(
                _run_tensor_state_group(
                    model,
                    schedule,
                    group,
                    normalized_bt,
                    observed_bt,
                    initial,
                    sequence_ids,
                    sequential=sequential_tensor_scan,
                )
            )
        elif group.update_type == "attention_window":
            results.append(
                _run_attention_state_group(
                    model,
                    schedule,
                    group,
                    normalized_bt,
                    observed_bt,
                    token_positions,
                    initial,
                    sequence_ids,
                )
            )
        else:  # pragma: no cover
            raise AssertionError(group.update_type)
    return tuple(results)


def _publish_receiver_states(
    initial: StateStore,
    runtimes: Sequence[_RegionRuntime],
    sequence_ids: Sequence[str],
) -> Dict[Tuple[str, str], ReceiverState]:
    published: Dict[Tuple[str, str], ReceiverState] = dict(initial.values)
    for runtime in runtimes:
        schedule = runtime.schedule
        for group_index, group in enumerate(schedule.state_groups):
            result = runtime.state_results[group_index]
            assert result is not None
            for column, local in enumerate(group.node_locals):
                node_id = schedule.node_ids[local]
                observed_by_sequence = result.observed[:, :, column].any(dim=1)
                for batch_index, sequence_id in enumerate(sequence_ids):
                    key = (sequence_id, node_id)
                    if not bool(observed_by_sequence[batch_index].item()):
                        continue
                    if isinstance(result, _TensorStateResult):
                        # Public owners may not expose even disjoint views of
                        # one packed backing storage.  clone() preserves the
                        # autograd edge while restoring unique logical owner
                        # storage for chunk continuation/checkpointing.
                        published[key] = result.final[batch_index, column].clone()
                    else:
                        mask = result.final_mask[batch_index, column]
                        published[key] = AttentionState(
                            result.final_positions[batch_index, column][mask].clone(),
                            result.final_keys[batch_index, column][mask].clone(),
                            result.final_values[batch_index, column][mask].clone(),
                        )
    return published


def _aggregate_terminals(
    model: SettleGraph,
    terminal_batches: Sequence[_MessageBatch],
    hidden_flat: Tensor,
    execution_flat: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    event_count, width = hidden_flat.shape
    terminal_count = len(model.plan.terminal_node_ids)
    records = _merge_message_batches(terminal_batches, hidden_flat)
    dense = hidden_flat.new_zeros((event_count * terminal_count, width))
    mask = torch.zeros(
        (event_count * terminal_count,), dtype=torch.bool, device=hidden_flat.device
    )
    if records.event.numel():
        destination = records.event * terminal_count + records.target_local
        dense = dense.index_copy(0, destination, records.hidden)
        mask = mask.index_fill(0, destination, True)
    dense = dense.reshape(event_count, terminal_count, width)
    mask = mask.reshape(event_count, terminal_count)
    missing = execution_flat & ~mask.any(dim=1)
    if bool(missing.any().item()):
        raise DynamicReachabilityError(
            "generic packed execution produced no terminal message for at least one "
            "graph-execution position"
        )
    output_type = model.output_aggregate_type
    if output_type == "mean":
        count = mask.sum(dim=1, keepdim=True).clamp_min(1)
        aggregate = (dense * mask.unsqueeze(-1)).sum(dim=1) / count
    else:
        assert model.output_scores is not None
        scores = _stack(
            tuple(
                model.output_scores[safe_module_key(node_id)]
                for node_id in model.plan.terminal_node_ids
            ),
            mask.any(dim=0),
        ).to(hidden_flat)
        weights = _safe_masked_softmax(
            scores.unsqueeze(0).expand(event_count, -1), mask, 1
        )
        aggregate = (weights.unsqueeze(-1) * dense).sum(dim=1)
    output = torch.where(execution_flat.unsqueeze(-1), aggregate, hidden_flat)
    return output, dense, mask


def _add_parameter_owner(source_tokens: set[Hashable], owner: Any) -> None:
    """Add every trainable leaf owned by a formula implementation."""

    if owner is None:
        return
    if isinstance(owner, Tensor):
        if owner.requires_grad:
            source_tokens.add(id(owner))
        return
    for parameter in owner.parameters(recurse=True):
        if parameter.requires_grad:
            source_tokens.add(id(parameter))


def _reverse_cumulative_any(mask: Tensor) -> Tensor:
    return torch.flip(
        torch.flip(mask.to(torch.int64), dims=(1,)).cumsum(dim=1) > 0,
        dims=(1,),
    )


def _semantic_connected_source_tokens(
    model: SettleGraph,
    runtimes: Sequence[_RegionRuntime],
    initial: StateStore,
    route_mask: Tensor,
    execution_mask: Tensor,
    sequence_ids: Sequence[str],
    selected_roots: frozenset[_ObservableSeed],
    *,
    batch: int,
    length: int,
    detach_at_end: bool,
) -> set[Hashable]:
    """Resolve eager structural liveness for one selected public objective.

    Packed rows share physical Tensors, whereas the eager interpreter builds
    one graph branch per logical event and state owner.  This pass follows the
    fixed Plan and the already-discovered hard route backwards from exactly
    the public result Tensors used by the current backward call.  Cotangent
    values are never inspected: a structurally connected all-zero cotangent
    remains connected.
    """

    connected: set[Hashable] = set()
    if not selected_roots or not runtimes:
        return connected

    runtime_by_node: Dict[str, Tuple[_RegionRuntime, int]] = {}
    runtime_by_region: Dict[str, _RegionRuntime] = {}
    target_by_edge: Dict[str, Tuple[_RegionRuntime, int, int]] = {}
    edges_by_source: Dict[str, List[Any]] = defaultdict(list)
    for runtime in runtimes:
        runtime_by_region[runtime.schedule.region_id] = runtime
        for local, node_id in enumerate(runtime.schedule.node_ids):
            runtime_by_node[node_id] = (runtime, local)
            for slot, edge_id in enumerate(runtime.schedule.parent_edge_ids[local]):
                target_by_edge[edge_id] = (runtime, local, slot)
    for edge in model.plan.edges:
        edges_by_source[edge.source].append(edge)

    def node_masks() -> Dict[str, Tensor]:
        return {
            node_id: torch.zeros_like(runtime.active[:, local])
            for node_id, (runtime, local) in runtime_by_node.items()
        }

    direct_aggregate = node_masks()
    direct_normalized = node_masks()
    direct_selector_read = node_masks()
    direct_score = node_masks()
    direct_computed = node_masks()
    direct_emitted = node_masks()
    probability_events = {
        region_id: torch.zeros(
            (runtime.active.shape[0],),
            dtype=torch.bool,
            device=runtime.active.device,
        )
        for region_id, runtime in runtime_by_region.items()
    }

    state_before: Dict[Tuple[str, str], Tensor] = {}
    state_proposal: Dict[Tuple[str, str], Tensor] = {}
    final_state: Dict[Tuple[str, str], Tensor] = {}
    for node_id, (runtime, local) in runtime_by_node.items():
        module = model.receivers[safe_module_key(node_id)]
        components = (
            ("keys", "values")
            if module.update_type == "attention_window"
            else (("tensor",) if module.update_type != "none" else ())
        )
        for component in components:
            state_before[(node_id, component)] = torch.zeros_like(
                runtime.active[:, local]
            )
            state_proposal[(node_id, component)] = torch.zeros_like(
                runtime.active[:, local]
            )
            final_state[(node_id, component)] = torch.zeros(
                (batch,), dtype=torch.bool, device=runtime.active.device
            )

    output_events = torch.zeros_like(execution_mask.reshape(-1))
    hidden_direct = False
    sequence_row = {
        sequence_id: row for row, sequence_id in enumerate(sequence_ids)
    }

    def mark_node(mask: Dict[str, Tensor], seed: _ObservableSeed) -> None:
        if seed.node_id not in mask or seed.event is None:
            return
        mask[seed.node_id][seed.event] = True

    for seed in selected_roots:
        if seed.kind == "output":
            output_events[:] = True
        elif seed.kind == "event-output" and seed.event is not None:
            output_events[seed.event] = True
        elif seed.kind == "hidden":
            hidden_direct = True
        elif seed.kind == "aggregate":
            mark_node(direct_aggregate, seed)
        elif seed.kind == "normalized":
            mark_node(direct_normalized, seed)
        elif seed.kind == "selector-read":
            mark_node(direct_selector_read, seed)
        elif seed.kind == "node-logit":
            # Eager constructs one stacked logits Tensor for the whole
            # candidate set before exposing an individual NodeEvent logit.
            # Indexing one element therefore leaves every reached candidate
            # structurally connected through StackBackward (with zero
            # cotangents for the unselected elements).  Preserve that exact
            # None-vs-zero connectivity at the packed public boundary.
            if seed.node_id in runtime_by_node and seed.event is not None:
                runtime, _ = runtime_by_node[seed.node_id]
                for local, node_id in enumerate(runtime.schedule.node_ids):
                    if bool(runtime.reached[seed.event, local].item()):
                        direct_score[node_id][seed.event] = True
        elif seed.kind == "region-logits":
            if seed.region_id in runtime_by_region and seed.event is not None:
                runtime = runtime_by_region[seed.region_id]
                for local, node_id in enumerate(runtime.schedule.node_ids):
                    if bool(runtime.reached[seed.event, local].item()):
                        direct_score[node_id][seed.event] = True
        elif seed.kind == "probability":
            if seed.region_id in probability_events and seed.event is not None:
                probability_events[seed.region_id][seed.event] = True
        elif seed.kind == "balance":
            if seed.region_id in probability_events:
                runtime = runtime_by_region[seed.region_id]
                probability_events[seed.region_id] |= (
                    route_mask & runtime.reached.any(dim=1)
                )
        elif seed.kind == "computed":
            mark_node(direct_computed, seed)
        elif seed.kind == "emitted":
            mark_node(direct_emitted, seed)
        elif seed.kind in {"state-before", "state-proposal", "state-compute"}:
            key = (str(seed.node_id), str(seed.component))
            if key not in state_before or seed.event is None:
                continue
            if seed.kind == "state-before":
                state_before[key][seed.event] = True
            elif seed.kind == "state-proposal":
                state_proposal[key][seed.event] = True
            else:
                runtime, local = runtime_by_node[key[0]]
                destination = (
                    state_proposal
                    if bool(runtime.observed[seed.event, local].item())
                    else state_before
                )
                destination[key][seed.event] = True
        elif seed.kind == "state":
            if detach_at_end:
                continue
            key = (str(seed.node_id), str(seed.component))
            row = sequence_row.get(str(seed.sequence_id))
            if row is None:
                # A StateStore may carry owners for sequences that are not in
                # this call's batch.  Eager leaves those values untouched when
                # detach_at_end=False, so their public roots remain connected
                # directly to the corresponding incoming state component.
                initial_key = (str(seed.sequence_id), str(seed.node_id))
                if initial_key in initial.values:
                    connected.add(
                        _state_source_token(
                            initial_key[0], initial_key[1], str(seed.component)
                        )
                    )
            elif key in final_state:
                final_state[key][row] = True
        else:  # pragma: no cover - the internal boundary owns this vocabulary
            raise AssertionError(f"unknown packed observable seed {seed.kind!r}")

    if hidden_direct or bool(
        (output_events & ~execution_mask.reshape(-1)).any().item()
    ):
        connected.add(_HIDDEN_SOURCE_TOKEN)

    # Terminal aggregation is a distinct stage.  The public output Tensor is
    # a stack of all events, while an OutputEvent trace Tensor seeds one event.
    for node_id in model.plan.terminal_node_ids:
        runtime, local = runtime_by_node[node_id]
        live = output_events & runtime.active[:, local]
        direct_emitted[node_id] |= live
        if (
            model.output_aggregate_type != "mean"
            and model.output_scores is not None
            and bool(live.any().item())
        ):
            _add_parameter_owner(
                connected, model.output_scores[safe_module_key(node_id)]
            )

    aggregate_live_by_node: Dict[str, Tensor] = {}
    for runtime in reversed(tuple(runtimes)):
        schedule = runtime.schedule
        region = model.plan.region_by_id(schedule.region_id)
        event_count = runtime.active.shape[0]

        emitted_live = {
            node_id: direct_emitted[node_id].clone()
            for node_id in schedule.node_ids
        }
        for local, node_id in enumerate(schedule.node_ids):
            for edge in edges_by_source[node_id]:
                target_runtime, target_local, target_slot = target_by_edge[
                    edge.edge_id
                ]
                emitted_live[node_id] |= (
                    target_runtime.message_mask[:, target_local, target_slot]
                    & aggregate_live_by_node[edge.target]
                )

        computed_live = {
            node_id: direct_computed[node_id] | emitted_live[node_id]
            for node_id in schedule.node_ids
        }
        probability_event_live = probability_events[schedule.region_id].clone()
        for node_id in schedule.node_ids:
            module = model.receivers[safe_module_key(node_id)]
            if _kind(module.emit_config, "hard") != "hard":
                probability_event_live |= emitted_live[node_id]

        score_uses_readout = _kind(region.score, "read_sum") not in {
            "fixed",
            "constant",
        }
        score_live: Dict[str, Tensor] = {}
        selector_read_live: Dict[str, Tensor] = {}
        normalized_live: Dict[str, Tensor] = {}
        aggregate_live: Dict[str, Tensor] = {}
        for local, node_id in enumerate(schedule.node_ids):
            score_live[node_id] = direct_score[node_id]
            if not schedule.forced_singleton:
                score_live[node_id] |= (
                    runtime.reached[:, local] & probability_event_live
                )
            selector_read_live[node_id] = direct_selector_read[node_id].clone()
            if score_uses_readout:
                selector_read_live[node_id] |= score_live[node_id]
            normalized_live[node_id] = (
                direct_normalized[node_id] | selector_read_live[node_id]
            )
            aggregate_live[node_id] = (
                direct_aggregate[node_id] | computed_live[node_id]
            )

        state_lookup = _state_group_lookup(schedule)
        for local, node_id in enumerate(schedule.node_ids):
            module = model.receivers[safe_module_key(node_id)]
            node_compute_live = computed_live[node_id]
            if module.compute_type == "affine_residual":
                normalized_live[node_id] |= node_compute_live
                if bool(node_compute_live.any().item()):
                    _add_parameter_owner(connected, module.down_proj)
            elif module.compute_type in {
                "double_residual_mlp",
                "double_residual_swiglu",
            }:
                if bool(node_compute_live.any().item()):
                    for owner in (
                        module.ffn_norm,
                        module.gate_proj,
                        module.up_proj,
                        module.down_proj,
                    ):
                        _add_parameter_owner(connected, owner)

            if (
                module.compute_type
                in {"double_residual_mlp", "double_residual_swiglu"}
                and module.ffn_read_type == "state_default"
                and module.update_type != "none"
            ):
                read_live = node_compute_live
                if module.update_type == "attention_window":
                    group_index, column = state_lookup[local]
                    state_result = runtime.state_results[group_index]
                    assert isinstance(state_result, _AttentionStateResult)
                    read_live = read_live & state_result.after_mask[
                        :, :, column
                    ].reshape(event_count, -1).any(dim=1)
                components = (
                    ("keys", "values")
                    if module.update_type == "attention_window"
                    else ("tensor",)
                )
                for component in components:
                    key = (node_id, component)
                    state_proposal[key] |= (
                        read_live & runtime.observed[:, local]
                    )
                    state_before[key] |= (
                        read_live & ~runtime.observed[:, local]
                    )
                if module.update_type in {"gdn", "attention_window"}:
                    normalized_live[node_id] |= read_live
                if bool(read_live.any().item()):
                    if module.update_type == "ema":
                        _add_parameter_owner(connected, module.state_out)
                    elif module.update_type == "gdn":
                        _add_parameter_owner(connected, module.gdn_query)
                        _add_parameter_owner(connected, module.gdn_out)
                    else:
                        _add_parameter_owner(connected, module.attn_query)
                        _add_parameter_owner(connected, module.attn_out)

            selector_uses_state = module.selector_read_type in {
                "content_state_linear",
                "content_state_summary_linear",
            }
            if selector_uses_state and region.selector_timing in {"pre", "post"}:
                state_live = selector_read_live[node_id]
                components = (
                    ("keys", "values")
                    if module.update_type == "attention_window"
                    else ("tensor",)
                )
                if module.update_type == "attention_window":
                    group_index, column = state_lookup[local]
                    state_result = runtime.state_results[group_index]
                    assert isinstance(state_result, _AttentionStateResult)
                    history = (
                        state_result.before_mask[:, :, column]
                        if region.selector_timing == "pre"
                        else state_result.after_mask[:, :, column]
                    ).reshape(event_count, -1).any(dim=1)
                    state_live = state_live & history
                destination = (
                    state_proposal
                    if region.selector_timing == "post"
                    else state_before
                )
                for component in components:
                    destination[(node_id, component)] |= state_live

            if (
                module.selector_read_linear is not None
                and bool(selector_read_live[node_id].any().item())
            ):
                _add_parameter_owner(connected, module.selector_read_linear)

        # State consumers reach all earlier committed updates.  Attention key
        # and value components remain separate throughout this propagation.
        for local, node_id in enumerate(schedule.node_ids):
            module = model.receivers[safe_module_key(node_id)]
            if module.update_type == "none":
                continue
            components = (
                ("keys", "values")
                if module.update_type == "attention_window"
                else ("tensor",)
            )
            observed = runtime.observed[:, local].reshape(batch, length)
            for component in components:
                key = (node_id, component)
                proposal = state_proposal[key].reshape(batch, length)
                before = state_before[key].reshape(batch, length)
                future_proposal = _reverse_cumulative_any(proposal)
                future_before_inclusive = _reverse_cumulative_any(before)
                future_before = torch.cat(
                    (
                        future_before_inclusive[:, 1:],
                        torch.zeros(
                            (batch, 1), dtype=torch.bool, device=before.device
                        ),
                    ),
                    dim=1,
                )
                formula_live = proposal | (
                    observed
                    & (
                        future_proposal
                        | future_before
                        | final_state[key].unsqueeze(1)
                    )
                )
                normalized_live[node_id] |= formula_live.reshape(-1)

                source_rows = (
                    final_state[key]
                    | proposal.any(dim=1)
                    | before.any(dim=1)
                )
                for row, source_live in enumerate(
                    source_rows.detach().to(device="cpu").tolist()
                ):
                    state_key = (sequence_ids[row], node_id)
                    if source_live and state_key in initial.values:
                        connected.add(
                            _state_source_token(
                                sequence_ids[row], node_id, component
                            )
                        )

                if not bool(formula_live.any().item()):
                    continue
                if module.update_type == "ema":
                    _add_parameter_owner(connected, module.ema_observe)
                    _add_parameter_owner(connected, module.ema_decay_logit)
                elif module.update_type == "gdn":
                    for owner in (
                        module.gdn_key,
                        module.gdn_value,
                        module.gdn_eta,
                        module.gdn_gamma,
                        module.gdn_beta,
                    ):
                        _add_parameter_owner(connected, owner)
                elif component == "keys":
                    _add_parameter_owner(connected, module.attn_key)
                else:
                    _add_parameter_owner(connected, module.attn_value)

        selector = model.selectors[safe_module_key(schedule.region_id)]
        if selector.score_type in {"linear", "mlp"}:
            if selector.shared_parameters:
                if any(
                    bool(score_live[node_id].any().item())
                    for node_id in schedule.node_ids
                ):
                    for owner in (selector.linear, selector.hidden, selector.out):
                        _add_parameter_owner(connected, owner)
            else:
                for node_id in schedule.node_ids:
                    if not bool(score_live[node_id].any().item()):
                        continue
                    key = safe_module_key(node_id)
                    owners = (
                        (selector.linears[key],)
                        if selector.score_type == "linear"
                        else (
                            selector.hidden_layers[key],
                            selector.output_layers[key],
                        )
                    )
                    for owner in owners:
                        _add_parameter_owner(connected, owner)

        for local, node_id in enumerate(schedule.node_ids):
            aggregate_live[node_id] |= normalized_live[node_id]
            aggregate_live_by_node[node_id] = aggregate_live[node_id]
            module = model.receivers[safe_module_key(node_id)]
            if bool(normalized_live[node_id].any().item()):
                _add_parameter_owner(connected, module.input_norm)
            if (
                node_id in model.plan.entry_node_ids
                and bool(aggregate_live[node_id].any().item())
            ):
                connected.add(_HIDDEN_SOURCE_TOKEN)
            for slot, edge_id in enumerate(schedule.parent_edge_ids[local]):
                edge_live = (
                    runtime.message_mask[:, local, slot]
                    & aggregate_live[node_id]
                )
                if not bool(edge_live.any().item()):
                    continue
                key = safe_module_key(edge_id)
                if module.edge_scores is not None and key in module.edge_scores:
                    _add_parameter_owner(connected, module.edge_scores[key])
                if (
                    module.edge_transforms is not None
                    and key in module.edge_transforms
                ):
                    _add_parameter_owner(connected, module.edge_transforms[key])

    return connected


class _BoundaryValues:
    """Collect and replace differentiable public result Tensor occurrences."""

    def __init__(
        self,
        tracker: _ConnectivityTracker,
        differentiable_sources: set[Hashable],
    ) -> None:
        self.tracker = tracker
        self.differentiable_sources = differentiable_sources
        self.values: List[Tensor] = []
        self.seeds: List[_ObservableSeed] = []
        self.paths: List[Tuple[Hashable, ...]] = []
        self.replacements: Dict[Tuple[Hashable, ...], Tensor] = {}

    def add(
        self,
        path: Tuple[Hashable, ...],
        value: Optional[Tensor],
        seed: _ObservableSeed,
    ) -> None:
        if value is None:
            return
        semantic_requires_grad = not self.tracker.resolve(
            frozenset((seed,))
        ).isdisjoint(self.differentiable_sources)
        if not value.requires_grad:
            if semantic_requires_grad:  # pragma: no cover - executor invariant
                raise AssertionError(
                    "packed public occurrence lost a semantic autograd source "
                    f"at {path!r} for {seed!r}"
                )
            return
        if not semantic_requires_grad:
            # Dense packing unions autograd metadata across independent lanes.
            # The eager public occurrence has no differentiable source at all,
            # so expose an actual non-differentiable Tensor, not merely a VJP
            # boundary that later turns its false-positive edges into None.
            self.replacements[path] = value.detach()
            return
        self.paths.append(path)
        self.values.append(value)
        self.seeds.append(seed)

    def apply(self, tracker: _ConnectivityTracker) -> None:
        if not self.values:
            return
        result = _ResultConnectivityBoundary.apply(
            tracker, tuple(self.seeds), *tuple(self.values)
        )
        outputs = (result,) if isinstance(result, Tensor) else tuple(result)
        if len(outputs) != len(self.paths):  # pragma: no cover - autograd invariant
            raise AssertionError("packed result boundary changed output arity")
        self.replacements.update(zip(self.paths, outputs))

    def get(self, path: Tuple[Hashable, ...], value: Tensor) -> Tensor:
        return self.replacements.get(path, value)


def _apply_result_connectivity_boundary(
    model: SettleGraph,
    result: ExecutionResult,
    tracker: _ConnectivityTracker,
    initial: StateStore,
    hidden: Tensor,
    sequence_ids: Sequence[str],
    token_positions: Tensor,
    execution_mask: Tensor,
) -> ExecutionResult:
    """Put every differentiable public result occurrence behind one boundary.

    Trace Tensors are included because the qualification contract uses
    selector-logit/probability-only objectives.  Including the remaining
    differentiable trace fields keeps the public diagnostic object honest as
    well: choosing a node proposal, edge payload, or event output cannot
    accidentally inherit liveness from an unrelated packed lane.
    """

    values = _BoundaryValues(
        tracker,
        _differentiable_source_tokens(model, initial, hidden),
    )
    values.add(("output",), result.output, _ObservableSeed("output"))

    def collect_state(
        path: Tuple[Hashable, ...],
        state: ReceiverState,
        seed: _ObservableSeed,
    ) -> None:
        if isinstance(state, Tensor):
            values.add(
                (*path, "tensor"),
                state,
                dataclasses.replace(seed, component="tensor"),
            )
        elif isinstance(state, AttentionState):
            values.add(
                (*path, "keys"),
                state.keys,
                dataclasses.replace(seed, component="keys"),
            )
            values.add(
                (*path, "values"),
                state.values,
                dataclasses.replace(seed, component="values"),
            )

    def replace_state(
        path: Tuple[Hashable, ...], state: ReceiverState
    ) -> ReceiverState:
        if isinstance(state, Tensor):
            return values.get((*path, "tensor"), state)
        if isinstance(state, AttentionState):
            return AttentionState(
                state.positions,
                values.get((*path, "keys"), state.keys),
                values.get((*path, "values"), state.values),
            )
        return state

    for key, state in result.state.values.items():
        sequence_id, node_id = key
        collect_state(
            ("state", sequence_id, node_id),
            state,
            _ObservableSeed(
                "state", sequence_id=sequence_id, node_id=node_id
            ),
        )
    for region_id, stats in result.balance_stats.regions.items():
        values.add(
            ("balance", region_id),
            stats.soft_sum,
            _ObservableSeed("balance", region_id=region_id),
        )

    trace = result.trace
    event_lookup: Dict[Tuple[str, int], int] = {}
    batch, length = token_positions.shape
    for row, sequence_id in enumerate(sequence_ids):
        for token in range(length):
            if bool(execution_mask[row, token].item()):
                event_lookup[
                    (sequence_id, int(token_positions[row, token].item()))
                ] = row * length + token
    edge_source = {edge.edge_id: edge.source for edge in model.plan.edges}

    def event_of(event: Any) -> int:
        try:
            return event_lookup[(event.sequence_id, event.token_position)]
        except KeyError as exc:  # pragma: no cover - trace construction invariant
            raise AssertionError("trace event has no executed input position") from exc

    if trace is not None:
        for index, event in enumerate(trace.node_events):
            event_index = event_of(event)
            base = ("trace", "node", index)
            for field, kind in (
                ("input_hidden", "aggregate"),
                ("normalized_input", "normalized"),
                ("selector_read", "selector-read"),
                ("logit", "node-logit"),
                ("probability", "probability"),
                ("computed", "computed"),
                ("emitted", "emitted"),
            ):
                values.add(
                    (*base, field),
                    getattr(event, field),
                    _ObservableSeed(
                        kind,
                        event=event_index,
                        node_id=event.node_id,
                        region_id=event.region_id,
                    ),
                )
            for field, kind in (
                ("state_before", "state-before"),
                ("proposal", "state-proposal"),
                ("state_for_compute", "state-compute"),
            ):
                collect_state(
                    (*base, field),
                    getattr(event, field),
                    _ObservableSeed(
                        kind,
                        event=event_index,
                        node_id=event.node_id,
                        region_id=event.region_id,
                    ),
                )
            for parent_index, (edge_id, _, payload) in enumerate(
                event.parent_messages
            ):
                if payload is None:
                    continue
                if edge_id.startswith("boundary:"):
                    seed = _ObservableSeed("hidden", event=event_index)
                else:
                    seed = _ObservableSeed(
                        "emitted",
                        event=event_index,
                        node_id=edge_source[edge_id],
                        edge_id=edge_id,
                    )
                values.add((*base, "parent", parent_index), payload, seed)

        for index, event in enumerate(trace.edge_events):
            if event.payload is None:
                continue
            event_index = event_of(event)
            values.add(
                ("trace", "edge", index, "payload"),
                event.payload,
                _ObservableSeed(
                    "emitted",
                    event=event_index,
                    node_id=edge_source[event.edge_id],
                    edge_id=event.edge_id,
                ),
            )
        for index, event in enumerate(trace.boundary_events):
            values.add(
                ("trace", "boundary", index, "payload"),
                event.payload,
                _ObservableSeed("hidden", event=event_of(event)),
            )
        for index, event in enumerate(trace.region_events):
            event_index = event_of(event)
            values.add(
                ("trace", "region", index, "logits"),
                event.logits,
                _ObservableSeed(
                    "region-logits",
                    event=event_index,
                    region_id=event.region_id,
                ),
            )
            values.add(
                ("trace", "region", index, "probabilities"),
                event.probabilities,
                _ObservableSeed(
                    "probability",
                    event=event_index,
                    region_id=event.region_id,
                ),
            )
        for index, event in enumerate(trace.state_writes):
            collect_state(
                ("trace", "write", index, "value"),
                event.value,
                _ObservableSeed(
                    "state-proposal",
                    event=event_of(event),
                    node_id=event.owner_id,
                ),
            )
        for index, event in enumerate(trace.output_events):
            event_index = event_of(event)
            for message_index, (node_id, payload) in enumerate(
                event.terminal_messages
            ):
                values.add(
                    ("trace", "output", index, "message", message_index),
                    payload,
                    _ObservableSeed(
                        "emitted", event=event_index, node_id=node_id
                    ),
                )
            values.add(
                ("trace", "output", index, "output"),
                event.output,
                _ObservableSeed("event-output", event=event_index),
            )

    values.apply(tracker)

    state_values = {
        key: replace_state(("state", key[0], key[1]), state)
        for key, state in result.state.values.items()
    }
    state = StateStore(
        state_values,
        dict(result.state.selector_history),
        dict(result.state.next_position),
    )
    balance = BalanceStats(zero=result.balance_stats.zero)
    for region_id, stats in result.balance_stats.regions.items():
        balance.regions[region_id] = dataclasses.replace(
            stats,
            soft_sum=values.get(("balance", region_id), stats.soft_sum),
        )

    rebuilt_trace: Optional[ExecutionTrace] = None
    if trace is not None:
        nodes = []
        for index, event in enumerate(trace.node_events):
            base = ("trace", "node", index)
            fields = {
                field: (
                    values.get((*base, field), value)
                    if isinstance(value, Tensor)
                    else value
                )
                for field, value in (
                    ("input_hidden", event.input_hidden),
                    ("normalized_input", event.normalized_input),
                    ("selector_read", event.selector_read),
                    ("logit", event.logit),
                    ("probability", event.probability),
                    ("computed", event.computed),
                    ("emitted", event.emitted),
                )
            }
            fields.update(
                {
                    field: replace_state((*base, field), getattr(event, field))
                    for field in (
                        "state_before",
                        "proposal",
                        "state_for_compute",
                    )
                }
            )
            fields["parent_messages"] = tuple(
                (
                    edge_id,
                    status,
                    (
                        values.get((*base, "parent", parent_index), payload)
                        if payload is not None
                        else None
                    ),
                )
                for parent_index, (edge_id, status, payload) in enumerate(
                    event.parent_messages
                )
            )
            nodes.append(dataclasses.replace(event, **fields))

        edges = tuple(
            dataclasses.replace(
                event,
                payload=(
                    values.get(("trace", "edge", index, "payload"), event.payload)
                    if event.payload is not None
                    else None
                ),
            )
            for index, event in enumerate(trace.edge_events)
        )
        boundaries = tuple(
            dataclasses.replace(
                event,
                payload=values.get(
                    ("trace", "boundary", index, "payload"), event.payload
                ),
            )
            for index, event in enumerate(trace.boundary_events)
        )
        regions = tuple(
            dataclasses.replace(
                event,
                logits=(
                    values.get(("trace", "region", index, "logits"), event.logits)
                    if event.logits is not None
                    else None
                ),
                probabilities=(
                    values.get(
                        ("trace", "region", index, "probabilities"),
                        event.probabilities,
                    )
                    if event.probabilities is not None
                    else None
                ),
            )
            for index, event in enumerate(trace.region_events)
        )
        writes = tuple(
            dataclasses.replace(
                event,
                value=replace_state(
                    ("trace", "write", index, "value"), event.value
                ),
            )
            for index, event in enumerate(trace.state_writes)
        )
        outputs = tuple(
            dataclasses.replace(
                event,
                terminal_messages=tuple(
                    (
                        node_id,
                        values.get(
                            ("trace", "output", index, "message", message_index),
                            payload,
                        ),
                    )
                    for message_index, (node_id, payload) in enumerate(
                        event.terminal_messages
                    )
                ),
                output=values.get(
                    ("trace", "output", index, "output"), event.output
                ),
            )
            for index, event in enumerate(trace.output_events)
        )
        rebuilt_trace = ExecutionTrace(
            tuple(nodes), edges, boundaries, regions, writes, outputs
        )

    return ExecutionResult(
        values.get(("output",), result.output),
        state,
        balance,
        rebuilt_trace,
    )


class PackedSettleGraph:
    """Plan-bound generic packed prefill executor sharing an eager model's parameters."""

    def __init__(self, model: SettleGraph) -> None:
        if not isinstance(model, SettleGraph):
            raise TypeError("PackedSettleGraph requires a SettleGraph parameter owner")
        self.model = model
        self._support = inspect_packed_support(model)
        self._support.require_supported()
        self._schedules, self.schedule_identity = _compile_schedule(model)
        self.last_profile: Optional[PackedExecutionProfile] = None

    def support_report(self) -> PackedSupportReport:
        return self._support

    def named_parameters(self, *args: Any, **kwargs: Any) -> Any:
        """Expose the single shared parameter owner; no packed copies exist."""

        return self.model.named_parameters(*args, **kwargs)

    def state_dict(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.load_state_dict(*args, **kwargs)

    def decode(
        self,
        hidden: Tensor,
        execution_mask: Tensor,
        sequence_ids: Sequence[Any],
        token_positions: Tensor,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Run the packed path for one decode position without eager fallback."""

        # Match the eager entry point's contract ordering: a malformed
        # configuration is rejected before any Tensor shape is interpreted.
        _validate_detach_at_end(kwargs.get("detach_at_end", True))
        if hidden.ndim != 2:
            raise ExecutionContractError("packed decode hidden must have shape [B,d_model]")
        if execution_mask.ndim != 1 or token_positions.ndim != 1:
            raise ExecutionContractError("packed decode masks and positions must have shape [B]")
        # A public decode call is a chunk boundary.  Preserve a cross-call
        # graph only when the caller opts in with ``detach_at_end=False``.
        kwargs.setdefault("detach_at_end", True)
        lm_target_mask = kwargs.pop("lm_target_mask", None)
        routing_stats_mask = kwargs.pop("routing_stats_mask", None)
        result = self.prefill(
            hidden.unsqueeze(1),
            execution_mask.unsqueeze(1),
            sequence_ids,
            token_positions.unsqueeze(1),
            lm_target_mask=(
                lm_target_mask.unsqueeze(1) if lm_target_mask is not None else None
            ),
            routing_stats_mask=(
                routing_stats_mask.unsqueeze(1)
                if routing_stats_mask is not None
                else None
            ),
            **kwargs,
        )
        return ExecutionResult(
            result.output[:, 0], result.state, result.balance_stats, result.trace
        )

    @_track_connectivity
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
        """Execute a complete chunk without calling the eager event scheduler."""

        _validate_detach_at_end(detach_at_end)
        if hidden.ndim != 3:
            raise ExecutionContractError("packed prefill hidden must have shape [B,T,d_model]")
        batch, length, width = hidden.shape
        if width != self.model.plan.d_model:
            raise ExecutionContractError("packed prefill hidden width does not match Plan d_model")
        if execution_mask.shape != (batch, length):
            raise ExecutionContractError("execution_mask must have shape [B,T]")
        if token_positions.shape != (batch, length):
            raise ExecutionContractError("token_positions must have shape [B,T]")
        if len(sequence_ids) != batch:
            raise ExecutionContractError("sequence_ids length must equal batch size")
        if execution_mask.dtype != torch.bool:
            raise ExecutionContractError("execution_mask must have bool dtype")
        if token_positions.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise ExecutionContractError("token_positions must have an integer dtype")
        if not torch.is_floating_point(hidden):
            raise ExecutionContractError("hidden must have a floating dtype")
        _validate_optional_mask(
            "lm_target_mask", lm_target_mask, execution_mask, (batch, length)
        )
        _validate_optional_mask(
            "routing_stats_mask", routing_stats_mask, execution_mask, (batch, length)
        )
        sequence_keys = _validate_stable_id_sequence("sequence", sequence_ids)
        if len(set(sequence_keys)) != len(sequence_keys):
            raise ExecutionContractError("sequence_id values must be unique within one call")
        reset_ids = _validate_stable_id_sequence("reset sequence", reset_sequence_ids)
        if len(set(reset_ids)) != len(reset_ids):
            raise ExecutionContractError("reset sequence IDs must be unique")
        if requested_k is not None:
            raise ExecutionContractError(
                "requested_k is an input-K extension and is unavailable for core-v1 packed execution"
            )

        public_state = state if state is not None else StateStore()
        _validate_model_and_state(self.model, hidden, public_state)
        initial = public_state.reset(reset_ids)
        _validate_position_matrix(initial, execution_mask, sequence_keys, token_positions)
        if length == 0:
            result_state = initial.detached() if detach_at_end else initial
            self.last_profile = PackedExecutionProfile(
                PACKED_EXECUTOR_ID,
                self.schedule_identity,
                0,
                0,
                0,
                0,
                0,
                0,
                trace_materialized=record_trace,
            )
            empty_trace = _build_trace(self.model.plan, (), (), (), (), (), ()) if record_trace else None
            return ExecutionResult(hidden, result_state, BalanceStats(zero=hidden.new_zeros(())), empty_trace)

        # One logical input Tensor is shared by all packed event lanes.  The
        # identity lets backward preserve eager's None-vs-zero distinction
        # when a state- or trace-only objective does not read any entry data.
        hidden = _selective_input(hidden, _HIDDEN_SOURCE_TOKEN)
        route_mask = (
            routing_stats_mask if routing_stats_mask is not None else execution_mask
        ).reshape(-1)
        hidden_flat = hidden.reshape(batch * length, width)
        execution_flat = execution_mask.reshape(-1)
        event_count = batch * length

        pending: List[List[_MessageBatch]] = [[] for _ in self._schedules]
        terminal_batches: List[_MessageBatch] = []
        valid_events = execution_flat.nonzero(as_tuple=False).squeeze(-1)
        for schedule in self._schedules:
            if not schedule.entry_locals:
                continue
            entry_locals = _device_index(schedule.entry_locals, hidden)
            event = valid_events.repeat_interleave(len(schedule.entry_locals))
            target = entry_locals.repeat(valid_events.shape[0])
            pending[schedule.rank].append(
                _MessageBatch(
                    hidden_flat[event],
                    event,
                    target,
                    torch.zeros_like(event),
                    torch.full_like(event, -1),
                )
            )

        balance = BalanceStats(zero=hidden.new_zeros(()))
        runtimes: List[_RegionRuntime] = []
        formula_batches = 0
        state_scan_batches = 0
        fanout_batches = 0
        data_message_rows = 0
        active_compute_rows = 0

        for schedule in self._schedules:
            region = self.model.plan.region_by_id(schedule.region_id)
            records = _merge_message_batches(pending[schedule.rank], hidden_flat)
            data_message_rows += int(records.event.shape[0])
            message_hidden, message_mask = _region_message_view(
                records,
                event_count=event_count,
                node_count=len(schedule.node_ids),
                max_fanin=schedule.max_fanin,
                reference=hidden_flat,
            )
            reached = message_mask.any(dim=-1)
            aggregate = _aggregate_region(
                self.model,
                schedule,
                message_hidden,
                message_mask,
                batch=batch,
                length=length,
            )
            normalized = _normalize_region(self.model, schedule, aggregate, reached)
            formula_batches += len(schedule.aggregate_groups) + 1

            state_results: Tuple[Optional[_StateResult], ...] = tuple(
                None for _ in schedule.state_groups
            )
            if schedule.forced_singleton:
                logits, probabilities, active = _score_and_route(
                    self.model, schedule, None, reached
                )
                if region.profile == "N":
                    observed = torch.zeros_like(reached)
                elif region.profile == "SD":
                    observed = active
                else:
                    observed = reached
                if schedule.state_groups:
                    state_results = _run_state_groups(
                        self.model,
                        schedule,
                        normalized,
                        observed,
                        batch=batch,
                        length=length,
                        token_positions=token_positions,
                        initial=initial,
                        sequence_ids=sequence_keys,
                    )
                    state_scan_batches += len(schedule.state_groups)
            elif region.profile == "SD" and region.selector_timing == "pre":
                # Top-K is discrete, so first discover its causal hard route in
                # one no-grad scripted recurrence.  With that route fixed, the
                # state path and all selector values are recomputed with
                # autograd.  Tensor-state recurrence stays sequential here so
                # the logits that are recorded and differentiated use exactly
                # the same state timing/order as route discovery.
                with torch.no_grad():
                    discovered_active = _discover_sd_pre_active(
                        self.model,
                        schedule,
                        normalized,
                        reached,
                        batch=batch,
                        length=length,
                        initial=initial,
                        sequence_ids=sequence_keys,
                    )
                observed = discovered_active
                state_results = _run_state_groups(
                    self.model,
                    schedule,
                    normalized,
                    observed,
                    batch=batch,
                    length=length,
                    token_positions=token_positions,
                    initial=initial,
                    sequence_ids=sequence_keys,
                    sequential_tensor_scan=True,
                )
                state_scan_batches += len(schedule.state_groups)
                readouts = _selector_readouts(
                    self.model,
                    schedule,
                    normalized,
                    reached,
                    state_results,
                    "pre",
                )
                logits, probabilities, active = _score_and_route(
                    self.model, schedule, readouts, reached
                )
                formula_batches += len(schedule.selector_read_groups) + 1
                if not torch.equal(active, discovered_active):
                    raise AssertionError(
                        f"SD/pre packed route discovery disagreed with its "
                        f"differentiable replay in region {schedule.region_id!r}"
                    )
            elif region.selector_timing in {"pre", "post"}:
                # BO observes every reached node, so state evolution is known
                # before selection.
                observed = reached
                state_results = _run_state_groups(
                    self.model,
                    schedule,
                    normalized,
                    observed,
                    batch=batch,
                    length=length,
                    token_positions=token_positions,
                    initial=initial,
                    sequence_ids=sequence_keys,
                )
                state_scan_batches += len(schedule.state_groups)
                readouts = _selector_readouts(
                    self.model,
                    schedule,
                    normalized,
                    reached,
                    state_results,
                    region.selector_timing,
                )
                logits, probabilities, active = _score_and_route(
                    self.model, schedule, readouts, reached
                )
                formula_batches += len(schedule.selector_read_groups) + 1
            else:
                readouts = _selector_readouts(
                    self.model,
                    schedule,
                    normalized,
                    reached,
                    state_results,
                    "content",
                )
                logits, probabilities, active = _score_and_route(
                    self.model, schedule, readouts, reached
                )
                formula_batches += len(schedule.selector_read_groups) + 1
                if region.profile == "N":
                    observed = torch.zeros_like(reached)
                elif region.profile == "SD":
                    observed = active
                else:
                    observed = reached
                if schedule.state_groups:
                    state_results = _run_state_groups(
                        self.model,
                        schedule,
                        normalized,
                        observed,
                        batch=batch,
                        length=length,
                        token_positions=token_positions,
                        initial=initial,
                        sequence_ids=sequence_keys,
                    )
                    state_scan_batches += len(schedule.state_groups)

            assert probabilities is not None
            computed, emitted, active_rows, compute_batches = _compute_active(
                self.model,
                schedule,
                aggregate,
                normalized,
                active,
                probabilities,
                state_results,
                record_occurrences=record_trace,
            )
            active_compute_rows += active_rows
            formula_batches += len(schedule.compute_groups)

            stats = _balance_for_region(
                schedule, reached, active, probabilities, route_mask
            )
            if stats is not None:
                balance.regions[schedule.region_id] = stats

            for fanout in schedule.fanout_groups:
                source_local = _device_index(fanout.source_locals, hidden)
                active_edges = active[:, source_local]
                pairs = active_edges.nonzero(as_tuple=False)
                if pairs.numel():
                    event, edge_column = pairs[:, 0], pairs[:, 1]
                    payload = emitted[event, source_local[edge_column]]
                    target_local = _device_index(fanout.target_locals, hidden)[edge_column]
                    parent_slot = _device_index(fanout.target_parent_slots, hidden)[edge_column]
                    edge_index = _device_index(fanout.edge_indices, hidden)[edge_column]
                    batch_records = _MessageBatch(
                        payload, event, target_local, parent_slot, edge_index
                    )
                    pending[fanout.target_region_rank].append(batch_records)
                fanout_batches += 1

            if schedule.terminal_locals:
                terminal_local = _device_index(schedule.terminal_locals, hidden)
                pairs = active[:, terminal_local].nonzero(as_tuple=False)
                if pairs.numel():
                    event, terminal_column = pairs[:, 0], pairs[:, 1]
                    terminal_batches.append(
                        _MessageBatch(
                            emitted[event, terminal_local[terminal_column]],
                            event,
                            _device_index(schedule.terminal_ordinals, hidden)[terminal_column],
                            torch.zeros_like(event),
                            torch.full_like(event, -1),
                        )
                    )

            runtimes.append(
                _RegionRuntime(
                    schedule,
                    message_hidden,
                    message_mask,
                    reached,
                    aggregate,
                    normalized,
                    observed,
                    active,
                    None if schedule.forced_singleton else readouts,
                    logits,
                    probabilities,
                    computed,
                    emitted,
                    state_results,
                    compute_batches,
                )
            )

        output_flat, _, terminal_mask = _aggregate_terminals(
            self.model, terminal_batches, hidden_flat, execution_flat
        )
        output = output_flat.reshape(batch, length, width)
        next_positions = dict(initial.next_position)
        for batch_index, sequence_id in enumerate(sequence_keys):
            valid = execution_mask[batch_index]
            if bool(valid.any().item()):
                next_positions[sequence_id] = int(token_positions[batch_index][valid][-1].item()) + 1
        result_state = StateStore(
            _publish_receiver_states(initial, runtimes, sequence_keys),
            dict(initial.selector_history),
            next_positions,
        )
        if detach_at_end:
            result_state = result_state.detached()

        trace = (
            self._materialize_trace(
                hidden,
                execution_mask,
                token_positions,
                sequence_keys,
                initial,
                runtimes,
                terminal_mask,
            )
            if record_trace
            else None
        )
        self.last_profile = PackedExecutionProfile(
            PACKED_EXECUTOR_ID,
            self.schedule_identity,
            len(self._schedules),
            formula_batches,
            state_scan_batches,
            fanout_batches,
            data_message_rows,
            active_compute_rows,
            trace_materialized=record_trace,
        )
        tracker = _ACTIVE_CONNECTIVITY.get()
        result = ExecutionResult(output, result_state, balance, trace)
        if tracker is not None:
            tracker.finalize(
                lambda selected_roots: _semantic_connected_source_tokens(
                    self.model,
                    runtimes,
                    initial,
                    route_mask,
                    execution_mask,
                    sequence_keys,
                    selected_roots,
                    batch=batch,
                    length=length,
                    detach_at_end=detach_at_end,
                )
            )
            result = _apply_result_connectivity_boundary(
                self.model,
                result,
                tracker,
                initial,
                hidden,
                sequence_keys,
                token_positions,
                execution_mask,
            )
        return result

    def _materialize_trace(
        self,
        hidden: Tensor,
        execution_mask: Tensor,
        token_positions: Tensor,
        sequence_ids: Sequence[str],
        initial: StateStore,
        runtimes: Sequence[_RegionRuntime],
        terminal_mask: Tensor,
    ) -> Any:
        """Restore the canonical small-fixture trace after tensor execution.

        This diagnostic path intentionally walks semantic events in Python.
        It is never entered by the training/performance path and is recorded
        as ``trace_materialized=True`` in :class:`PackedExecutionProfile`.
        """

        batch, length, _ = hidden.shape
        valid_events = execution_mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        node_events: List[NodeEventTrace] = []
        edge_events: List[EdgeEventTrace] = []
        boundary_events: List[BoundaryEventTrace] = []
        region_events: List[RegionEventTrace] = []
        state_writes: List[StateWriteTrace] = []
        output_events: List[OutputEventTrace] = []
        raw_computed_occurrences: Dict[Tuple[int, str], Tensor] = {}
        raw_emitted_occurrences: Dict[Tuple[int, str], Tensor] = {}
        for runtime in runtimes:
            schedule = runtime.schedule
            for records in runtime.compute_batches:
                event_values = records.event.detach().to(device="cpu").tolist()
                local_values = records.local.detach().to(device="cpu").tolist()
                for record_index, (event, local) in enumerate(
                    zip(event_values, local_values)
                ):
                    key = (int(event), schedule.node_ids[int(local)])
                    if key in raw_computed_occurrences:  # pragma: no cover
                        raise AssertionError(
                            "packed trace found duplicate NodeCompute occurrence"
                        )
                    raw_computed_occurrences[key] = records.computed[record_index]
                    raw_emitted_occurrences[key] = records.emitted[record_index]

        edge_source = {
            edge.edge_id: edge.source for edge in self.model.plan.edges
        }
        entry_ids = set(self.model.plan.entry_node_ids)
        aggregate_occurrences: Dict[Tuple[int, str], Tensor] = {}
        computed_occurrences: Dict[Tuple[int, str], Tensor] = {}
        emitted_occurrences: Dict[Tuple[int, str], Tensor] = {}
        # Rebuild semantic Aggregate occurrences in region order.  The source
        # Emit occurrences for every non-entry receiver are already available
        # because fixed graph edges cross topologically ordered regions.
        for runtime in runtimes:
            schedule = runtime.schedule
            for event_tensor in valid_events:
                event = int(event_tensor.item())
                row, token = divmod(event, length)
                for local, node_id in enumerate(schedule.node_ids):
                    if not bool(runtime.reached[event, local].item()):
                        continue
                    if node_id in entry_ids:
                        messages = (hidden[row, token],)
                        edge_ids = (f"boundary:{node_id}",)
                    else:
                        data = tuple(
                            (
                                edge_id,
                                emitted_occurrences[(event, edge_source[edge_id])],
                            )
                            for slot, edge_id in enumerate(
                                schedule.parent_edge_ids[local]
                            )
                            if bool(
                                runtime.message_mask[event, local, slot].item()
                            )
                        )
                        edge_ids = tuple(edge_id for edge_id, _ in data)
                        messages = tuple(value for _, value in data)
                    key = (event, node_id)
                    aggregate = _aggregate_trace_occurrence(
                        self.model, node_id, messages, edge_ids
                    )
                    aggregate_occurrences[key] = aggregate
                    if not bool(runtime.active[event, local].item()):
                        continue
                    module = self.model.receivers[safe_module_key(node_id)]
                    computed = raw_computed_occurrences[key]
                    if module.compute_type == "identity":
                        computed = aggregate
                    computed_occurrences[key] = computed
                    emit_type = _kind(module.emit_config, "hard")
                    if emit_type == "hard":
                        emitted = computed
                    else:
                        # Non-hard Emit remains differentiable through its
                        # probability even when NodeCompute is identity.  Its
                        # grouped value has the correct formula graph; only a
                        # hard identity needs exact object provenance here.
                        emitted = raw_emitted_occurrences[key]
                    emitted_occurrences[key] = emitted

        def emitted_value(event: int, node_id: str) -> Tensor:
            try:
                return emitted_occurrences[(event, node_id)]
            except KeyError as exc:  # pragma: no cover - execution invariant
                raise AssertionError(
                    "packed trace has DATA from a node without an Emit occurrence"
                ) from exc

        def aggregate_output(
            messages: Sequence[Tuple[str, Tensor]],
        ) -> Tensor:
            if not messages:  # pragma: no cover - terminal reachability gate
                raise AssertionError("packed trace cannot aggregate no messages")
            stacked = torch.stack(tuple(value for _, value in messages), dim=0)
            if self.model.output_aggregate_type == "mean":
                return stacked.mean(dim=0)
            assert self.model.output_scores is not None
            scores = torch.stack(
                tuple(
                    self.model.output_scores[safe_module_key(node_id)]
                    for node_id, _ in messages
                )
            ).to(device=stacked.device, dtype=stacked.dtype)
            weights = torch.softmax(scores, dim=0)
            return (weights.unsqueeze(-1) * stacked).sum(dim=0)

        for schedule in self._schedules:
            for local in schedule.entry_locals:
                node_id = schedule.node_ids[local]
                for event_tensor in valid_events:
                    event = int(event_tensor.item())
                    row, token = divmod(event, length)
                    boundary_events.append(
                        BoundaryEventTrace(
                            sequence_ids[row],
                            int(token_positions[row, token].item()),
                            node_id,
                            hidden[row, token],
                        )
                    )

        for runtime in runtimes:
            schedule = runtime.schedule
            region = self.model.plan.region_by_id(schedule.region_id)
            state_lookup = _state_group_lookup(schedule)
            entry_states: Dict[Tuple[int, int], ReceiverState] = {}

            def entry_state(event: int, local: int) -> ReceiverState:
                key = (event, local)
                if key not in entry_states:
                    row, _ = divmod(event, length)
                    node_id = schedule.node_ids[local]
                    default = self.model.receiver(node_id).initial_state(
                        runtime.normalized[event, local]
                    )
                    entry_states[key] = initial.values.get(
                        (sequence_ids[row], node_id), default
                    )
                return entry_states[key]

            def state_value(event: int, local: int, which: str) -> ReceiverState:
                location = state_lookup.get(local)
                if location is None:
                    return None
                group_index, column = location
                result = runtime.state_results[group_index]
                assert result is not None
                row, token = divmod(event, length)
                observed = bool(result.observed[row, token, column].item())
                has_prior_observe = bool(
                    result.observed[row, :token, column].any().item()
                )
                if isinstance(result, _TensorStateResult):
                    if which == "before":
                        if not has_prior_observe:
                            return entry_state(event, local)
                        return result.before[row, token, column]
                    if which == "proposal":
                        return result.proposal[row, token, column]
                    if which == "compute":
                        if observed:
                            return result.proposal[row, token, column]
                        if not has_prior_observe:
                            return entry_state(event, local)
                        return result.before[row, token, column]
                    raise AssertionError(which)
                use_before = which == "before" or (
                    which == "compute" and not observed
                )
                if use_before and not has_prior_observe:
                    return entry_state(event, local)
                if use_before:
                    positions = result.before_positions[row, token, column]
                    keys = result.before_keys[row, token, column]
                    values = result.before_values[row, token, column]
                    mask = result.before_mask[row, token, column]
                else:
                    positions = result.after_positions[row, token, column]
                    keys = result.after_keys[row, token, column]
                    values = result.after_values[row, token, column]
                    mask = result.after_mask[row, token, column]
                return AttentionState(positions[mask], keys[mask], values[mask])

            for event_tensor in valid_events:
                event = int(event_tensor.item())
                row, token = divmod(event, length)
                sequence_id = sequence_ids[row]
                token_position = int(token_positions[row, token].item())
                reached_locals = runtime.reached[event].nonzero(as_tuple=False).squeeze(-1)
                active_locals = runtime.active[event].nonzero(as_tuple=False).squeeze(-1)
                reached_ids = tuple(
                    schedule.node_ids[int(local.item())] for local in reached_locals
                )
                active_ids = tuple(
                    schedule.node_ids[int(local.item())] for local in active_locals
                )
                forced = bool(reached_ids) and schedule.forced_singleton
                if reached_ids and not forced:
                    reached_mask = runtime.reached[event]
                    logits = runtime.logits[event, reached_mask] if runtime.logits is not None else None
                    probabilities = runtime.probabilities[event, reached_mask]
                    requested_k: Optional[int] = int(region.k_requested["value"])
                    effective_k: Optional[int] = min(requested_k, len(reached_ids))
                    top_k_ids: Optional[Tuple[str, ...]] = active_ids
                elif reached_ids:
                    logits = None
                    probabilities = runtime.probabilities[event, runtime.reached[event]]
                    requested_k = None
                    effective_k = None
                    top_k_ids = None
                else:
                    logits = None
                    probabilities = None
                    requested_k = None
                    effective_k = None
                    top_k_ids = None
                region_events.append(
                    RegionEventTrace(
                        sequence_id=sequence_id,
                        token_position=token_position,
                        region_id=schedule.region_id,
                        candidate_node_ids=reached_ids,
                        logits=logits,
                        probabilities=probabilities,
                        requested_k=requested_k,
                        effective_k=effective_k,
                        active_node_ids=active_ids,
                        forced_active=forced,
                        top_k_node_ids=top_k_ids,
                    )
                )

                for local, node_id in enumerate(schedule.node_ids):
                    reached = bool(runtime.reached[event, local].item())
                    observed = bool(runtime.observed[event, local].item())
                    active = bool(runtime.active[event, local].item())
                    if node_id in self.model.plan.entry_node_ids:
                        parents = ((f"boundary:{node_id}", "DATA", hidden[row, token]),)
                    else:
                        parent_records = []
                        for slot, edge_id in enumerate(schedule.parent_edge_ids[local]):
                            has_data = bool(runtime.message_mask[event, local, slot].item())
                            parent_records.append(
                                (
                                    edge_id,
                                    "DATA" if has_data else "CLOSED",
                                    (
                                        emitted_value(event, edge_source[edge_id])
                                        if has_data
                                        else None
                                    ),
                                )
                            )
                        parents = tuple(parent_records)
                    proposal = (
                        state_value(event, local, "proposal")
                        if observed and local in state_lookup
                        else None
                    )
                    node_events.append(
                        NodeEventTrace(
                            sequence_id,
                            token_position,
                            schedule.region_id,
                            node_id,
                            reached,
                            observed,
                            active,
                            (
                                aggregate_occurrences[(event, node_id)]
                                if reached
                                else None
                            ),
                            runtime.normalized[event, local] if reached else None,
                            state_value(event, local, "before") if reached else None,
                            proposal,
                            state_value(event, local, "compute") if reached else None,
                            (
                                runtime.selector_read[event, local]
                                if reached and runtime.selector_read is not None
                                else None
                            ),
                            (
                                runtime.logits[event, local]
                                if reached and runtime.logits is not None
                                else None
                            ),
                            runtime.probabilities[event, local] if reached else None,
                            (
                                computed_occurrences[(event, node_id)]
                                if active
                                else None
                            ),
                            emitted_value(event, node_id) if active else None,
                            parents,
                        )
                    )
                    if observed and proposal is not None:
                        state_writes.append(
                            StateWriteTrace(
                                sequence_id,
                                token_position,
                                "receiver",
                                node_id,
                                proposal,
                            )
                        )

        for event_tensor in valid_events:
            event = int(event_tensor.item())
            row, token = divmod(event, length)
            sequence_id = sequence_ids[row]
            token_position = int(token_positions[row, token].item())
            for edge in self.model.plan.edges:
                has_data = (event, edge.source) in emitted_occurrences
                edge_events.append(
                    EdgeEventTrace(
                        sequence_id,
                        token_position,
                        edge.edge_id,
                        "DATA" if has_data else "CLOSED",
                        emitted_value(event, edge.source) if has_data else None,
                    )
                )
            messages = tuple(
                (
                    node_id,
                    emitted_value(event, node_id),
                )
                for terminal_index, node_id in enumerate(self.model.plan.terminal_node_ids)
                if bool(terminal_mask[event, terminal_index].item())
            )
            output_events.append(
                OutputEventTrace(
                    sequence_id,
                    token_position,
                    messages,
                    aggregate_output(messages),
                )
            )

        return _build_trace(
            self.model.plan,
            node_events,
            edge_events,
            boundary_events,
            region_events,
            state_writes,
            output_events,
        )


# A descriptive alias makes the role clear at call sites that host multiple
# executor implementations.
GenericPackedExecutor = PackedSettleGraph


__all__ = [
    "GenericPackedExecutor",
    "PACKED_EXECUTOR_ID",
    "PackedExecutionProfile",
    "PackedSettleGraph",
    "PackedSupportIssue",
    "PackedSupportReport",
    "inspect_packed_support",
]

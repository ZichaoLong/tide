"""Reusable comparators and trace invariants for SettleGraph qualification.

The execution engines intentionally do not decide whether two runs are
equivalent.  This module implements that independent test boundary: discrete
structure is exact, floating values are finite and elementwise tolerant, and
an exact trace must satisfy the one-settlement invariants even when two
executors happen to agree with each other.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple

import torch
from torch import Tensor

from .engine import ExecutionTrace
from .ops import AttentionState
from .plan import Plan, validate_stable_id


@dataclass(frozen=True)
class Tolerance:
    """Elementwise absolute and relative floating-point tolerances."""

    atol: float
    rtol: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.atol, bool)
            or isinstance(self.rtol, bool)
            or not isinstance(self.atol, (int, float))
            or not isinstance(self.rtol, (int, float))
            or not math.isfinite(float(self.atol))
            or not math.isfinite(float(self.rtol))
            or self.atol < 0
            or self.rtol < 0
        ):
            raise ValueError("atol and rtol must be finite nonnegative numbers")


CPU_FLOAT64_TOLERANCE = Tolerance(atol=1e-10, rtol=1e-8)
SAME_BACKEND_FLOAT32_TOLERANCE = Tolerance(atol=1e-6, rtol=1e-5)
CPU_NPU_FLOAT32_TOLERANCE = Tolerance(atol=1e-4, rtol=1e-4)


@dataclass(frozen=True)
class TensorComparison:
    """Diagnostics for one successfully shape-compatible Tensor comparison."""

    path: str
    passed: bool
    max_absolute_error: float
    max_relative_error: float
    worst_path: str
    reference_value: float
    candidate_value: float
    tolerance_at_worst: float


@dataclass(frozen=True)
class EquivalenceReport:
    """Complete result of a nested equivalence comparison."""

    errors: Tuple[str, ...]
    tensors: Tuple[TensorComparison, ...]

    @property
    def passed(self) -> bool:
        return not self.errors and all(item.passed for item in self.tensors)

    def require_pass(self) -> None:
        if self.passed:
            return
        details = list(self.errors)
        details.extend(
            f"{item.worst_path}: |candidate-reference| exceeds "
            f"{item.tolerance_at_worst:.17g} "
            f"({item.candidate_value:.17g} vs {item.reference_value:.17g})"
            for item in self.tensors
            if not item.passed
        )
        raise EquivalenceError(details)


class EquivalenceError(AssertionError):
    """Two artifacts violate the declared equivalence comparator."""

    def __init__(self, errors: Iterable[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("equivalence comparison failed:\n- " + "\n- ".join(self.errors))


class TraceInvariantError(AssertionError):
    """An exact trace violates SettleGraph's one-settlement invariants."""

    def __init__(self, errors: Iterable[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("trace invariant failure:\n- " + "\n- ".join(self.errors))


@dataclass(frozen=True)
class RouteBoundary:
    """Tolerance-aware classification of one Top-K routing boundary."""

    classification: str
    delta: Optional[float]
    guard_band: Optional[float]
    kth_node_id: Optional[str]
    next_node_id: Optional[str]


def compare_nested(
    reference: Any,
    candidate: Any,
    *,
    tolerance: Tolerance,
    require_same_dtype: bool = True,
    path: str = "$",
) -> EquivalenceReport:
    """Compare nested dataclasses, mappings, sequences, scalars, and Tensors.

    Floating Tensors are checked for finiteness on their producing device,
    copied to CPU float64, and compared element by element.  Integer and bool
    Tensors, key sets, shapes, container kinds, and absent values are exact.
    """

    errors: List[str] = []
    tensors: List[TensorComparison] = []

    def visit(left: Any, right: Any, current: str) -> None:
        if isinstance(left, Tensor):
            _compare_tensor(
                left,
                right,
                tolerance=tolerance,
                require_same_dtype=require_same_dtype,
                path=current,
                errors=errors,
                comparisons=tensors,
            )
            return
        if isinstance(right, Tensor):
            errors.append(
                f"{current}: expected {type(left).__name__}, got Tensor"
            )
            return
        if dataclasses.is_dataclass(left) and not isinstance(left, type):
            if type(right) is not type(left):
                errors.append(
                    f"{current}: dataclass type mismatch "
                    f"{type(left).__name__} != {type(right).__name__}"
                )
                return
            for field in dataclasses.fields(left):
                visit(
                    getattr(left, field.name),
                    getattr(right, field.name),
                    f"{current}.{field.name}",
                )
            return
        if isinstance(left, Mapping):
            if not isinstance(right, Mapping):
                errors.append(
                    f"{current}: expected mapping, got {type(right).__name__}"
                )
                return
            left_keys = set(left)
            right_keys = set(right)
            if left_keys != right_keys:
                errors.append(
                    f"{current}: mapping keys differ; "
                    f"missing={sorted(left_keys - right_keys, key=repr)!r}, "
                    f"extra={sorted(right_keys - left_keys, key=repr)!r}"
                )
            for key in sorted(left_keys & right_keys, key=repr):
                visit(left[key], right[key], f"{current}[{key!r}]")
            return
        if isinstance(left, (tuple, list)):
            if type(right) is not type(left):
                errors.append(
                    f"{current}: sequence type mismatch "
                    f"{type(left).__name__} != {type(right).__name__}"
                )
                return
            if len(left) != len(right):
                errors.append(
                    f"{current}: sequence length mismatch {len(left)} != {len(right)}"
                )
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                visit(left_item, right_item, f"{current}[{index}]")
            return
        if isinstance(left, float):
            if not isinstance(right, float):
                errors.append(
                    f"{current}: scalar type mismatch float != {type(right).__name__}"
                )
                return
            if not math.isfinite(left) or not math.isfinite(right):
                errors.append(f"{current}: non-finite scalar is not comparable")
                return
            allowed = tolerance.atol + tolerance.rtol * abs(left)
            if abs(right - left) > allowed:
                errors.append(
                    f"{current}: {right:.17g} differs from {left:.17g}; "
                    f"allowed {allowed:.17g}"
                )
            return
        if type(left) is not type(right) or left != right:
            errors.append(f"{current}: exact value mismatch {left!r} != {right!r}")

    visit(reference, candidate, path)
    return EquivalenceReport(tuple(errors), tuple(tensors))


def _compare_tensor(
    reference: Tensor,
    candidate: Any,
    *,
    tolerance: Tolerance,
    require_same_dtype: bool,
    path: str,
    errors: List[str],
    comparisons: List[TensorComparison],
) -> None:
    if not isinstance(candidate, Tensor):
        errors.append(
            f"{path}: expected Tensor, got {type(candidate).__name__}"
        )
        return
    if reference.shape != candidate.shape:
        errors.append(
            f"{path}: Tensor shape mismatch "
            f"{tuple(reference.shape)} != {tuple(candidate.shape)}"
        )
        return
    if require_same_dtype and reference.dtype != candidate.dtype:
        errors.append(
            f"{path}: Tensor dtype mismatch {reference.dtype} != {candidate.dtype}"
        )
        return
    if reference.is_floating_point() != candidate.is_floating_point():
        errors.append(f"{path}: floating/integer Tensor kind mismatch")
        return
    if not reference.is_floating_point():
        if reference.dtype != candidate.dtype:
            errors.append(
                f"{path}: exact Tensor dtype mismatch "
                f"{reference.dtype} != {candidate.dtype}"
            )
            return
        if not torch.equal(reference.detach().cpu(), candidate.detach().cpu()):
            errors.append(f"{path}: exact Tensor values differ")
        return

    if not bool(torch.isfinite(reference).all().item()):
        errors.append(f"{path}: reference Tensor contains NaN or infinity")
        return
    if not bool(torch.isfinite(candidate).all().item()):
        errors.append(f"{path}: candidate Tensor contains NaN or infinity")
        return

    left = reference.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    right = candidate.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if left.numel() == 0:
        comparisons.append(
            TensorComparison(path, True, 0.0, 0.0, path, 0.0, 0.0, tolerance.atol)
        )
        return

    absolute = (right - left).abs()
    allowed = tolerance.atol + tolerance.rtol * left.abs()
    passed = bool((absolute <= allowed).all().item())
    relative = torch.where(
        left != 0,
        absolute / left.abs(),
        torch.where(absolute == 0, torch.zeros_like(absolute), torch.full_like(absolute, math.inf)),
    )
    scale = torch.where(
        allowed > 0,
        absolute / allowed,
        torch.where(absolute == 0, torch.zeros_like(absolute), torch.full_like(absolute, math.inf)),
    )
    worst_index = int(torch.argmax(scale).item())
    worst_path = f"{path}.flat[{worst_index}]"
    comparisons.append(
        TensorComparison(
            path=path,
            passed=passed,
            max_absolute_error=float(absolute.max().item()),
            max_relative_error=float(relative.max().item()),
            worst_path=worst_path,
            reference_value=float(left[worst_index].item()),
            candidate_value=float(right[worst_index].item()),
            tolerance_at_worst=float(allowed[worst_index].item()),
        )
    )


def classify_route_boundary(
    logits: Tensor,
    candidate_node_ids: Sequence[str],
    effective_k: int,
    *,
    tolerance: Tolerance,
) -> RouteBoundary:
    """Classify one routing event using the contract's Top-K guard band."""

    if logits.ndim != 1 or logits.shape[0] != len(candidate_node_ids):
        raise ValueError("logits and candidate_node_ids must describe one event")
    if len(set(candidate_node_ids)) != len(candidate_node_ids):
        raise ValueError("candidate_node_ids must be unique")
    if isinstance(effective_k, bool) or not isinstance(effective_k, int):
        raise ValueError("effective_k must be an integer")
    if effective_k < 0 or effective_k > len(candidate_node_ids):
        raise ValueError("effective_k is outside the candidate set")
    if not logits.is_floating_point() or not bool(torch.isfinite(logits).all().item()):
        raise ValueError("logits must be a finite floating Tensor")
    if effective_k >= len(candidate_node_ids):
        return RouteBoundary("all-active", None, None, None, None)
    if effective_k == 0:
        raise ValueError("a nonempty standard routing event cannot have K=0")

    values = logits.detach().to(device="cpu", dtype=torch.float64).tolist()
    order = sorted(
        range(len(values)),
        key=lambda index: (-values[index], candidate_node_ids[index]),
    )
    kth = order[effective_k - 1]
    following = order[effective_k]
    delta = float(values[kth] - values[following])
    guard = 4.0 * (
        tolerance.atol
        + tolerance.rtol * max(abs(values[kth]), abs(values[following]))
    )
    if delta == 0.0:
        classification = "exact-tie"
    elif delta <= guard:
        classification = "near-boundary"
    else:
        classification = "margin-safe"
    return RouteBoundary(
        classification,
        delta,
        guard,
        candidate_node_ids[kth],
        candidate_node_ids[following],
    )


def validate_trace_invariants(
    plan: Plan,
    trace: ExecutionTrace,
    executed_tokens: Iterable[Tuple[str, int]],
    *,
    tolerance: Tolerance,
) -> None:
    """Require an exact trace to satisfy independent SettleGraph invariants.

    ``executed_tokens`` contains only positions whose execution mask is true.
    Consequently, any event for a padding or otherwise bypassed position is an
    error, and every listed token must have the full fixed-graph settlement.
    """

    plan.validate()
    errors: List[str] = []
    expected_items = tuple(executed_tokens)
    expected_list: List[Tuple[str, int]] = []
    for index, item in enumerate(expected_items):
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            errors.append(
                f"executed_tokens[{index}] must be a (sequence_id, position) pair"
            )
            continue
        sequence_id, position = item
        try:
            validated_id = validate_stable_id(
                sequence_id, kind="executed-token sequence"
            )
        except ValueError as exc:
            errors.append(f"executed_tokens[{index}]: {exc}")
            continue
        if type(position) is not int or position < 0:
            errors.append(
                f"executed_tokens[{index}] position must be a nonnegative integer"
            )
            continue
        expected_list.append((validated_id, position))

    for group_name, events in (
        ("boundary_events", trace.boundary_events),
        ("region_events", trace.region_events),
        ("node_events", trace.node_events),
        ("edge_events", trace.edge_events),
        ("state_writes", trace.state_writes),
        ("output_events", trace.output_events),
    ):
        for index, event in enumerate(events):
            try:
                validate_stable_id(
                    event.sequence_id, kind="trace sequence"
                )
            except ValueError as exc:
                errors.append(f"{group_name}[{index}]: {exc}")
            if type(event.token_position) is not int or event.token_position < 0:
                errors.append(
                    f"{group_name}[{index}] token_position must be a "
                    "nonnegative integer"
                )

    # Invalid token keys must fail at the comparator boundary.  In
    # particular, neither bool/int coercion nor Unicode normalization is part
    # of qualification semantics, and attempting to sort malformed keys can
    # otherwise hide the schema failure behind a Python TypeError.
    if errors:
        raise TraceInvariantError(errors)

    expected = tuple(expected_list)
    expected_set = set(expected)
    if len(expected_set) != len(expected):
        errors.append("executed_tokens contains duplicate sequence/position keys")

    region_rank = {
        region.region_id: index
        for index, region in enumerate(plan.topological_regions)
    }
    canonical_groups = (
        (
            "boundary_events",
            trace.boundary_events,
            lambda event: (event.sequence_id, event.token_position, event.node_id),
        ),
        (
            "region_events",
            trace.region_events,
            lambda event: (
                event.sequence_id,
                event.token_position,
                region_rank.get(event.region_id, len(region_rank)),
                event.region_id,
            ),
        ),
        (
            "node_events",
            trace.node_events,
            lambda event: (
                event.sequence_id,
                event.token_position,
                region_rank.get(event.region_id, len(region_rank)),
                event.region_id,
                event.node_id,
            ),
        ),
        (
            "edge_events",
            trace.edge_events,
            lambda event: (event.sequence_id, event.token_position, event.edge_id),
        ),
        (
            "state_writes",
            trace.state_writes,
            lambda event: (
                event.sequence_id,
                event.token_position,
                event.owner_kind,
                event.owner_id,
            ),
        ),
        (
            "output_events",
            trace.output_events,
            lambda event: (event.sequence_id, event.token_position),
        ),
    )
    for name, events, key in canonical_groups:
        event_keys = tuple(key(event) for event in events)
        if event_keys != tuple(sorted(event_keys)):
            errors.append(f"{name} is not in canonical order")
        for event in events:
            token_key = (event.sequence_id, event.token_position)
            if token_key not in expected_set:
                errors.append(f"{name} contains event for non-executed token {token_key!r}")

    nodes_by_id = {node.node_id: node for node in plan.nodes}
    regions_by_id = {region.region_id: region for region in plan.regions}
    edges_by_id = {edge.edge_id: edge for edge in plan.edges}
    outgoing = {node.node_id: [] for node in plan.nodes}
    incoming = {node.node_id: [] for node in plan.nodes}
    for edge in plan.edges:
        outgoing[edge.source].append(edge)
        incoming[edge.target].append(edge)
    for edge_list in tuple(outgoing.values()) + tuple(incoming.values()):
        edge_list.sort(key=lambda edge: edge.edge_id)

    for token_key in sorted(expected_set):
        sequence_id, token_position = token_key
        prefix = f"token[{sequence_id!r},{token_position}]"
        boundary_events = [
            event
            for event in trace.boundary_events
            if (event.sequence_id, event.token_position) == token_key
        ]
        node_events = [
            event
            for event in trace.node_events
            if (event.sequence_id, event.token_position) == token_key
        ]
        region_events = [
            event
            for event in trace.region_events
            if (event.sequence_id, event.token_position) == token_key
        ]
        edge_events = [
            event
            for event in trace.edge_events
            if (event.sequence_id, event.token_position) == token_key
        ]
        output_events = [
            event
            for event in trace.output_events
            if (event.sequence_id, event.token_position) == token_key
        ]
        state_writes = [
            event
            for event in trace.state_writes
            if (event.sequence_id, event.token_position) == token_key
        ]

        _require_unique_ids(
            errors,
            prefix,
            "boundary node",
            [event.node_id for event in boundary_events],
            set(plan.entry_node_ids),
        )
        _require_unique_ids(
            errors,
            prefix,
            "node",
            [event.node_id for event in node_events],
            set(nodes_by_id),
        )
        _require_unique_ids(
            errors,
            prefix,
            "region",
            [event.region_id for event in region_events],
            set(regions_by_id),
        )
        _require_unique_ids(
            errors,
            prefix,
            "edge",
            [event.edge_id for event in edge_events],
            set(edges_by_id),
        )
        if len(output_events) != 1:
            errors.append(f"{prefix}: expected exactly one output event")

        node_event_by_id = {event.node_id: event for event in node_events}
        edge_event_by_id = {event.edge_id: event for event in edge_events}
        region_event_by_id = {event.region_id: event for event in region_events}
        boundary_event_by_id = {
            event.node_id: event for event in boundary_events
        }
        receiver_writes = [
            event for event in state_writes if event.owner_kind == "receiver"
        ]
        selector_history_writes = [
            event
            for event in state_writes
            if event.owner_kind == "selector_history"
        ]
        unknown_write_kinds = sorted(
            {
                event.owner_kind
                for event in state_writes
                if event.owner_kind not in {"receiver", "selector_history"}
            }
        )
        if unknown_write_kinds:
            errors.append(
                f"{prefix}: unknown state-write owner kinds {unknown_write_kinds!r}"
            )
        expected_receiver_writes = {
            node_id
            for node_id, node in nodes_by_id.items()
            if node.update.get("type") != "none"
            and node_event_by_id.get(node_id) is not None
            and node_event_by_id[node_id].observed
        }
        _require_unique_ids(
            errors,
            prefix,
            "receiver state-write owner",
            [event.owner_id for event in receiver_writes],
            expected_receiver_writes,
        )
        receiver_write_by_id = {
            event.owner_id: event for event in receiver_writes
        }
        if all(
            region.selector_history.get("type") == "none"
            for region in plan.regions
        ):
            _require_unique_ids(
                errors,
                prefix,
                "selector-history state-write owner",
                [event.owner_id for event in selector_history_writes],
                set(),
            )

        for event in boundary_events:
            _check_hidden_tensor(errors, f"{prefix}.boundary[{event.node_id!r}]", event.payload, plan.d_model)

        for region_id, region in regions_by_id.items():
            event = region_event_by_id.get(region_id)
            if event is None:
                continue
            reached = tuple(
                node_id
                for node_id in region.node_ids
                if node_event_by_id.get(node_id) is not None
                and node_event_by_id[node_id].reached
            )
            if event.candidate_node_ids != reached:
                errors.append(
                    f"{prefix}.region[{region_id!r}]: candidates are not the reached members"
                )
            active_set = set(event.active_node_ids)
            if len(active_set) != len(event.active_node_ids) or not active_set <= set(reached):
                errors.append(
                    f"{prefix}.region[{region_id!r}]: active IDs are not a unique candidate subset"
                )
            if event.effective_k != len(event.active_node_ids):
                errors.append(
                    f"{prefix}.region[{region_id!r}]: effective K does not equal active count"
                )
            if type(event.effective_k) is not int or event.effective_k < 0:
                errors.append(
                    f"{prefix}.region[{region_id!r}]: effective K must be a "
                    "nonnegative integer"
                )
            expected_forced = bool(reached) and (
                len(region.node_ids) == 1
                and nodes_by_id[region.node_ids[0]].forced_active
            )
            if type(event.forced_active) is not bool or event.forced_active is not expected_forced:
                errors.append(
                    f"{prefix}.region[{region_id!r}]: forced-active flag "
                    "does not match the Plan and reached set"
                )
            if not reached:
                if (
                    event.logits is not None
                    or event.probabilities is not None
                    or event.requested_k is not None
                    or event.effective_k != 0
                    or event.active_node_ids
                ):
                    errors.append(
                        f"{prefix}.region[{region_id!r}]: empty candidates have routing values"
                    )
                continue
            if event.logits is None or event.probabilities is None:
                errors.append(
                    f"{prefix}.region[{region_id!r}]: nonempty candidates lack logits/probabilities"
                )
                continue
            _check_vector(errors, f"{prefix}.region[{region_id!r}].logits", event.logits, len(reached))
            _check_vector(errors, f"{prefix}.region[{region_id!r}].probabilities", event.probabilities, len(reached))
            if (
                event.logits.numel() == len(reached)
                and event.probabilities.numel() == len(reached)
            ):
                for index, node_id in enumerate(reached):
                    node_event = node_event_by_id.get(node_id)
                    if node_event is None:
                        continue
                    if node_event.logit is not None:
                        _append_report_errors(
                            errors,
                            compare_nested(
                                event.logits[index],
                                node_event.logit,
                                tolerance=tolerance,
                                path=(
                                    f"{prefix}.region[{region_id!r}]"
                                    f".node[{node_id!r}].logit"
                                ),
                            ),
                        )
                    if node_event.probability is not None:
                        _append_report_errors(
                            errors,
                            compare_nested(
                                event.probabilities[index],
                                node_event.probability,
                                tolerance=tolerance,
                                path=(
                                    f"{prefix}.region[{region_id!r}]"
                                    f".node[{node_id!r}].probability"
                                ),
                            ),
                        )
            if (
                type(event.requested_k) is not int
                or not 1 <= event.requested_k <= region.k_max
            ):
                errors.append(
                    f"{prefix}.region[{region_id!r}]: requested K is outside the Plan bound"
                )
            else:
                if region.k_requested.get("type") == "fixed":
                    fixed_k = region.k_requested.get("value")
                    if event.requested_k != fixed_k:
                        errors.append(
                            f"{prefix}.region[{region_id!r}]: requested K "
                            "does not match the Plan fixed value"
                        )
                expected_k = min(event.requested_k, len(reached))
                if event.effective_k != expected_k:
                    errors.append(
                        f"{prefix}.region[{region_id!r}]: effective K is not min(requested,candidates)"
                    )
            if event.probabilities.numel() == len(reached) and bool(torch.isfinite(event.probabilities).all().item()):
                if bool((event.probabilities < -tolerance.atol).any().item()):
                    errors.append(
                        f"{prefix}.region[{region_id!r}]: selector probability is negative"
                    )
                probability_sum = float(event.probabilities.detach().to(device="cpu", dtype=torch.float64).sum().item())
                allowed = tolerance.atol + tolerance.rtol
                if abs(probability_sum - 1.0) > allowed:
                    errors.append(
                        f"{prefix}.region[{region_id!r}]: probabilities do not sum to one"
                    )
            if event.forced_active:
                if (
                    len(region.node_ids) != 1
                    or len(reached) != 1
                    or event.requested_k != 1
                    or event.effective_k != 1
                    or event.probabilities.numel() != 1
                    or float(event.probabilities.detach().cpu()[0].item()) != 1.0
                ):
                    errors.append(
                        f"{prefix}.region[{region_id!r}]: forced-active singleton is malformed"
                    )
            elif event.logits.numel() == len(reached) and event.effective_k > 0:
                values = event.logits.detach().to(device="cpu", dtype=torch.float64).tolist()
                ranking = sorted(
                    range(len(reached)),
                    key=lambda index: (-values[index], reached[index]),
                )
                selected = set(ranking[: event.effective_k])
                expected_active = tuple(
                    node_id for index, node_id in enumerate(reached) if index in selected
                )
                if event.active_node_ids != expected_active:
                    errors.append(
                        f"{prefix}.region[{region_id!r}]: active IDs do not match stable Top-K"
                    )

            expected_observed = (
                set()
                if region.profile == "N"
                else active_set
                if region.profile == "SD"
                else set(reached)
            )
            actual_observed = {
                node_id
                for node_id in region.node_ids
                if node_event_by_id.get(node_id) is not None
                and node_event_by_id[node_id].observed
            }
            if actual_observed != expected_observed:
                errors.append(
                    f"{prefix}.region[{region_id!r}]: Observe set violates profile {region.profile}"
                )

        for node_id, node in nodes_by_id.items():
            event = node_event_by_id.get(node_id)
            if event is None:
                continue
            node_prefix = f"{prefix}.node[{node_id!r}]"
            if event.region_id != node.region_id:
                errors.append(f"{node_prefix}: region ID does not match Plan")
            if node_id in plan.entry_node_ids:
                boundary = boundary_event_by_id.get(node_id)
                expected_parents = (
                    (
                        f"boundary:{node_id}",
                        "DATA",
                        boundary.payload if boundary is not None else None,
                    ),
                )
                expected_reached = True
            else:
                expected_parents = tuple(
                    (
                        edge.edge_id,
                        edge_event_by_id[edge.edge_id].status,
                        edge_event_by_id[edge.edge_id].payload,
                    )
                    for edge in incoming[node_id]
                    if edge.edge_id in edge_event_by_id
                )
                expected_reached = any(
                    parent[1] == "DATA" for parent in expected_parents
                )
            if type(event.reached) is not bool or event.reached is not expected_reached:
                errors.append(
                    f"{node_prefix}: reached flag does not match parent DATA settlements"
                )
            parent_report = compare_nested(
                expected_parents,
                event.parent_messages,
                tolerance=tolerance,
                path=f"{node_prefix}.parent_messages",
            )
            _append_report_errors(errors, parent_report)
            if event.active and not event.reached:
                errors.append(f"{node_prefix}: active receiver was not reached")
            if event.observed and not event.reached:
                errors.append(f"{node_prefix}: observed receiver was not reached")
            stateful = node.update.get("type") != "none"
            if not stateful and event.proposal is not None:
                errors.append(f"{node_prefix}: stateless receiver has a proposal")
            if stateful and event.observed and event.proposal is None:
                errors.append(f"{node_prefix}: observed stateful receiver lacks a proposal")
            if not event.reached:
                for field_name in (
                    "state_before",
                    "proposal",
                    "state_for_compute",
                ):
                    if getattr(event, field_name) is not None:
                        errors.append(
                            f"{node_prefix}: unreached receiver has {field_name}"
                        )
            elif stateful:
                _check_receiver_state(
                    errors,
                    f"{node_prefix}.state_before",
                    event.state_before,
                    node,
                    event.input_hidden,
                    token_position=token_position,
                    proposal=False,
                )
                expected_proposal = (
                    regions_by_id[node.region_id].selector_timing == "post"
                    or event.observed
                )
                if expected_proposal and event.proposal is None:
                    errors.append(
                        f"{node_prefix}: reached receiver lacks required proposal"
                    )
                if event.proposal is not None:
                    _check_receiver_state(
                        errors,
                        f"{node_prefix}.proposal",
                        event.proposal,
                        node,
                        event.input_hidden,
                        token_position=token_position,
                        proposal=True,
                    )
                expected_compute_state = (
                    event.proposal if event.observed else event.state_before
                )
                _append_report_errors(
                    errors,
                    compare_nested(
                        expected_compute_state,
                        event.state_for_compute,
                        tolerance=tolerance,
                        path=f"{node_prefix}.state_for_compute",
                    ),
                )
            else:
                for field_name in (
                    "state_before",
                    "proposal",
                    "state_for_compute",
                ):
                    if getattr(event, field_name) is not None:
                        errors.append(
                            f"{node_prefix}: stateless receiver has {field_name}"
                        )
            state_write = receiver_write_by_id.get(node_id)
            if state_write is not None and event.proposal is not None:
                report = compare_nested(
                    event.proposal,
                    state_write.value,
                    tolerance=tolerance,
                    path=f"{node_prefix}.state_write",
                )
                _append_report_errors(errors, report)
            for field_name in ("input_hidden", "normalized_input", "selector_read", "logit", "probability"):
                value = getattr(event, field_name)
                if event.reached and value is None:
                    errors.append(f"{node_prefix}: reached receiver lacks {field_name}")
                if not event.reached and value is not None:
                    errors.append(f"{node_prefix}: unreached receiver has {field_name}")
            for field_name in ("computed", "emitted"):
                value = getattr(event, field_name)
                if event.active and value is None:
                    errors.append(f"{node_prefix}: active receiver lacks {field_name}")
                if not event.active and value is not None:
                    errors.append(f"{node_prefix}: inactive receiver has {field_name}")
            for field_name in ("input_hidden", "normalized_input", "computed", "emitted"):
                value = getattr(event, field_name)
                if value is not None:
                    _check_role_tensor(
                        errors,
                        f"{node_prefix}.{field_name}",
                        value,
                        (plan.d_model,),
                        event.input_hidden if field_name != "input_hidden" else None,
                    )
            if event.selector_read is not None:
                _check_role_tensor(
                    errors,
                    f"{node_prefix}.selector_read",
                    event.selector_read,
                    tuple(node.selector_read_shape),
                    event.input_hidden,
                )
            for field_name in ("logit", "probability"):
                value = getattr(event, field_name)
                if value is not None:
                    _check_role_tensor(
                        errors,
                        f"{node_prefix}.{field_name}",
                        value,
                        (),
                        event.input_hidden,
                    )

        for edge_id, edge in edges_by_id.items():
            event = edge_event_by_id.get(edge_id)
            source = node_event_by_id.get(edge.source)
            if event is None or source is None:
                continue
            expected_status = "DATA" if source.active else "CLOSED"
            if event.status != expected_status:
                errors.append(
                    f"{prefix}.edge[{edge_id!r}]: status does not match source activity"
                )
            if (event.status == "DATA") != (event.payload is not None):
                errors.append(
                    f"{prefix}.edge[{edge_id!r}]: status/payload presence mismatch"
                )
            if event.payload is not None:
                _check_hidden_tensor(errors, f"{prefix}.edge[{edge_id!r}].payload", event.payload, plan.d_model)
                if source.emitted is not None:
                    report = compare_nested(
                        source.emitted,
                        event.payload,
                        tolerance=tolerance,
                        path=f"{prefix}.edge[{edge_id!r}].payload",
                    )
                    _append_report_errors(errors, report)

        if output_events:
            output = output_events[0]
            terminal_ids = tuple(node_id for node_id, _ in output.terminal_messages)
            expected_terminal_ids = tuple(
                node_id
                for node_id in sorted(plan.terminal_node_ids)
                if node_event_by_id.get(node_id) is not None
                and node_event_by_id[node_id].active
            )
            if not terminal_ids:
                errors.append(f"{prefix}: successful output has no terminal message")
            if terminal_ids != expected_terminal_ids:
                errors.append(f"{prefix}: terminal messages do not match active terminals")
            for node_id, payload in output.terminal_messages:
                _check_hidden_tensor(errors, f"{prefix}.terminal[{node_id!r}]", payload, plan.d_model)
                node_event = node_event_by_id.get(node_id)
                if node_event is not None and node_event.emitted is not None:
                    report = compare_nested(
                        node_event.emitted,
                        payload,
                        tolerance=tolerance,
                        path=f"{prefix}.terminal[{node_id!r}]",
                    )
                    _append_report_errors(errors, report)
            _check_hidden_tensor(errors, f"{prefix}.output", output.output, plan.d_model)

    if errors:
        raise TraceInvariantError(errors)


def _require_unique_ids(
    errors: List[str],
    prefix: str,
    kind: str,
    actual: Sequence[str],
    expected: set[str],
) -> None:
    actual_set = set(actual)
    if len(actual_set) != len(actual):
        errors.append(f"{prefix}: {kind} events contain duplicate IDs")
    if actual_set != expected:
        errors.append(
            f"{prefix}: {kind} IDs differ; "
            f"missing={sorted(expected - actual_set)!r}, "
            f"extra={sorted(actual_set - expected)!r}"
        )


def _check_vector(errors: List[str], path: str, value: Any, length: int) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point() or value.shape != (length,):
        errors.append(f"{path}: expected floating Tensor shape [{length}]")
        return
    if not bool(torch.isfinite(value).all().item()):
        errors.append(f"{path}: contains NaN or infinity")


def _check_role_tensor(
    errors: List[str],
    path: str,
    value: Any,
    shape: Tuple[int, ...],
    reference: Any,
) -> None:
    """Check a floating trace Tensor against its declared role and event binding."""

    if (
        not isinstance(value, Tensor)
        or not value.is_floating_point()
        or tuple(value.shape) != shape
    ):
        errors.append(
            f"{path}: expected floating Tensor shape {list(shape)}"
        )
        return
    if not bool(torch.isfinite(value).all().item()):
        errors.append(f"{path}: contains NaN or infinity")
    if isinstance(reference, Tensor) and (
        value.dtype != reference.dtype or value.device != reference.device
    ):
        errors.append(
            f"{path}: dtype/device does not match the event hidden binding"
        )


def _check_receiver_state(
    errors: List[str],
    path: str,
    value: Any,
    node: Any,
    reference: Any,
    *,
    token_position: int,
    proposal: bool,
) -> None:
    """Validate the canonical state representation recorded in an exact trace."""

    update_type = node.update.get("type")
    if update_type in {"ema", "gdn"}:
        _check_role_tensor(
            errors,
            path,
            value,
            tuple(node.state_shape),
            reference,
        )
        return
    if update_type != "attention_window":
        errors.append(f"{path}: unsupported receiver-state representation")
        return
    if not isinstance(value, AttentionState):
        errors.append(f"{path}: expected canonical AttentionState")
        return

    positions = value.positions
    keys = value.keys
    values = value.values
    if not isinstance(positions, Tensor) or positions.dtype != torch.int64:
        errors.append(f"{path}.positions: expected int64 Tensor")
        return
    if positions.ndim != 1:
        errors.append(f"{path}.positions: expected one-dimensional Tensor")
        return
    length = positions.shape[0]
    window = node.update.get("window")
    key_dim = node.update.get("key_dim")
    value_dim = node.update.get("value_dim")
    if type(window) is not int or length > window:
        errors.append(f"{path}: Attention valid length exceeds the Plan window")
    _check_role_tensor(
        errors,
        f"{path}.keys",
        keys,
        (length, key_dim),
        reference,
    )
    _check_role_tensor(
        errors,
        f"{path}.values",
        values,
        (length, value_dim),
        reference,
    )
    if isinstance(reference, Tensor) and positions.device != reference.device:
        errors.append(
            f"{path}.positions: device does not match the event hidden binding"
        )
    if length:
        position_values = positions.detach().to(device="cpu").tolist()
        if position_values[0] < 0 or any(
            later <= earlier
            for earlier, later in zip(position_values, position_values[1:])
        ):
            errors.append(
                f"{path}.positions: positions must be nonnegative and strictly increasing"
            )
        if proposal:
            if position_values[-1] != token_position:
                errors.append(
                    f"{path}.positions: proposal must end at the current Token position"
                )
        elif position_values[-1] >= token_position:
            errors.append(
                f"{path}.positions: prior state must precede the current Token position"
            )
    elif proposal:
        errors.append(f"{path}: Attention proposal must contain the current Token")


def _check_hidden_tensor(errors: List[str], path: str, value: Any, width: int) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point() or value.shape != (width,):
        errors.append(f"{path}: expected floating Tensor shape [{width}]")
        return
    if not bool(torch.isfinite(value).all().item()):
        errors.append(f"{path}: contains NaN or infinity")


def _append_report_errors(
    errors: List[str], report: EquivalenceReport
) -> None:
    errors.extend(report.errors)
    errors.extend(
        f"{item.worst_path}: linked trace payload differs from its source"
        for item in report.tensors
        if not item.passed
    )


__all__ = [
    "CPU_FLOAT64_TOLERANCE",
    "CPU_NPU_FLOAT32_TOLERANCE",
    "SAME_BACKEND_FLOAT32_TOLERANCE",
    "EquivalenceError",
    "EquivalenceReport",
    "RouteBoundary",
    "TensorComparison",
    "Tolerance",
    "TraceInvariantError",
    "classify_route_boundary",
    "compare_nested",
    "validate_trace_invariants",
]

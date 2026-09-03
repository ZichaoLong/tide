"""Versioned logical parameter schemas and eager ``state_dict`` bindings.

The logical schema names parameters by their role in the declared formula,
never by a Python module attribute.  Eager ``state_dict`` paths are kept in a
separate binding record so another executor can preserve the logical schema
and use different load locators.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn

from .engine import SettleGraph
from .ops import safe_module_key
from .plan import Plan, validate_stable_id


PARAMETER_SCHEMA_VERSION = "tide.parameter-schema.v1"
PARAMETER_SCHEMA_CANONICALIZER_ID = "tide-parameter-schema-json-v1"
EAGER_PARAMETER_BINDING_VERSION = "tide.eager-parameter-binding.v1"
EAGER_EXECUTOR_ID = "tide.settlegraph.eager-reference.v1"


class ParameterManifestError(ValueError):
    """A parameter schema or executor binding is incomplete or inconsistent."""


_NODE_FIELDS = frozenset(
    {
        "input_norm",
        "ffn_norm",
        "update",
        "selector_read",
        "ffn_read",
        "node_compute",
    }
)
_ALL_PARAMETER_FIELDS = _NODE_FIELDS | {
    "aggregate",
    "score",
    "output_aggregate",
}


def _checked_id(value: object, *, kind: str) -> str:
    try:
        return validate_stable_id(value, kind=kind)
    except ValueError as exc:
        raise ParameterManifestError(str(exc)) from exc


@dataclass(frozen=True)
class LogicalParameterKey:
    """Implementation-independent identity of one formula parameter.

    Receiver-local fields carry both their stable region and node IDs.  Edge
    Aggregate parameters additionally carry the fixed edge ID, Score
    parameters identify their region and candidate node, and output Aggregate
    parameters identify the stable terminal node.
    """

    field: str
    parameter_role: str
    node_id: Optional[str] = None
    region_id: Optional[str] = None
    edge_id: Optional[str] = None
    terminal_node_id: Optional[str] = None

    def __post_init__(self) -> None:
        _checked_id(self.field, kind="parameter field")
        _checked_id(self.parameter_role, kind="parameter role")
        for name in ("node_id", "region_id", "edge_id", "terminal_node_id"):
            value = getattr(self, name)
            if value is not None:
                _checked_id(value, kind=name.removesuffix("_id"))

        if self.field not in _ALL_PARAMETER_FIELDS:
            raise ParameterManifestError(
                f"parameter field {self.field!r} is not in schema v1"
            )
        if self.field in _NODE_FIELDS:
            valid = (
                self.node_id is not None
                and self.region_id is not None
                and self.edge_id is None
                and self.terminal_node_id is None
            )
        elif self.field == "aggregate":
            valid = (
                self.node_id is not None
                and self.region_id is not None
                and self.edge_id is not None
                and self.terminal_node_id is None
            )
        elif self.field == "score":
            valid = (
                self.node_id is not None
                and self.region_id is not None
                and self.edge_id is None
                and self.terminal_node_id is None
            )
        else:
            valid = (
                self.node_id is None
                and self.region_id is None
                and self.edge_id is None
                and self.terminal_node_id is not None
            )
        if not valid:
            raise ParameterManifestError(
                f"logical parameter key has invalid owner fields for {self.field!r}"
            )

    def sort_key(self) -> Tuple[str, str, str, str, str, str]:
        """Return the schema-v1 scalar-value ordering tuple."""

        return (
            self.field,
            self.region_id or "",
            self.node_id or "",
            self.edge_id or "",
            self.terminal_node_id or "",
            self.parameter_role,
        )

    def canonical_dict(self) -> Dict[str, Optional[str]]:
        return {
            "field": self.field,
            "region_id": self.region_id,
            "node_id": self.node_id,
            "edge_id": self.edge_id,
            "terminal_node_id": self.terminal_node_id,
            "parameter_role": self.parameter_role,
        }


def logical_parameter_tensor_key(logical_key: Mapping[str, Any]) -> str:
    """Encode one logical parameter key for portable Tensor mappings."""

    if not isinstance(logical_key, Mapping):
        raise TypeError("logical_key must be a mapping")
    try:
        normalized = LogicalParameterKey(**dict(logical_key)).canonical_dict()
    except (TypeError, ParameterManifestError) as exc:
        raise ParameterManifestError(f"invalid logical parameter key: {exc}") from exc
    return "tide.logical-parameter.v1:" + json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class ParameterManifestEntry:
    """One logical parameter plus its eager-reference load locator."""

    logical_key: LogicalParameterKey
    formula_id: str
    shape: Tuple[int, ...]
    dtype_role: str
    parameter_group: Optional[str]
    state_dict_locator: str
    state_dict_shape: Optional[Tuple[int, ...]] = None
    logical_to_state_dict: str = "identity"

    def __post_init__(self) -> None:
        if not isinstance(self.logical_key, LogicalParameterKey):
            raise ParameterManifestError(
                "logical_key must be a LogicalParameterKey"
            )
        _checked_id(self.formula_id, kind="formula")
        _checked_id(self.dtype_role, kind="dtype role")
        _checked_id(self.state_dict_locator, kind="state_dict locator")
        if self.dtype_role != "parameter":
            raise ParameterManifestError(
                "parameter schema v1 requires dtype_role='parameter'"
            )
        if self.parameter_group is not None:
            _checked_id(self.parameter_group, kind="parameter group")
            raise ParameterManifestError(
                "parameter groups are not closed in parameter schema v1"
            )
        logical_shape = _normalize_parameter_shape(self.shape, context="parameter")
        state_dict_shape = (
            logical_shape
            if self.state_dict_shape is None
            else _normalize_parameter_shape(
                self.state_dict_shape, context="state_dict parameter"
            )
        )
        if (
            not isinstance(self.logical_to_state_dict, str)
            or self.logical_to_state_dict
            not in {"identity", "reshape-row-major"}
        ):
            raise ParameterManifestError(
                "logical_to_state_dict must be 'identity' or 'reshape-row-major'"
            )
        if self.logical_to_state_dict == "identity" and logical_shape != state_dict_shape:
            raise ParameterManifestError(
                "identity parameter binding requires equal logical and state_dict shapes"
            )
        if math.prod(logical_shape) != math.prod(state_dict_shape):
            raise ParameterManifestError(
                "logical and state_dict parameter shapes must have equal element counts"
            )
        object.__setattr__(self, "shape", logical_shape)
        object.__setattr__(self, "state_dict_shape", state_dict_shape)

    def logical_dict(self) -> Dict[str, Any]:
        """Return the executor-independent portion of this entry."""

        return {
            "logical_key": self.logical_key.canonical_dict(),
            "formula_id": self.formula_id,
            "shape": list(self.shape),
            "dtype_role": self.dtype_role,
            "parameter_group": self.parameter_group,
        }

    def locator_dict(self) -> Dict[str, Any]:
        return {
            "logical_key": self.logical_key.canonical_dict(),
            "state_dict_locator": self.state_dict_locator,
            "state_dict_shape": list(self.state_dict_shape or ()),
            "logical_to_state_dict": self.logical_to_state_dict,
        }


def _normalize_parameter_shape(
    shape: object, *, context: str
) -> Tuple[int, ...]:
    if not isinstance(shape, (tuple, list)):
        raise ParameterManifestError(f"{context} shape must be an array")
    normalized: List[int] = []
    for dimension in shape:
        if type(dimension) is not int or dimension <= 0:
            raise ParameterManifestError(
                f"{context} shapes must contain only positive integers"
            )
        normalized.append(dimension)
    return tuple(normalized)


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return text.encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ParameterManifestError(
            "parameter manifest is not canonical UTF-8 JSON"
        ) from exc


@dataclass(frozen=True)
class ParameterSchemaManifest:
    """Logical parameter schema paired with an eager-reference locator map."""

    logical_plan_hash: str
    entries: Tuple[ParameterManifestEntry, ...]
    schema_version: str = PARAMETER_SCHEMA_VERSION
    canonicalizer_id: str = PARAMETER_SCHEMA_CANONICALIZER_ID
    executor_id: str = EAGER_EXECUTOR_ID
    binding_schema_version: str = EAGER_PARAMETER_BINDING_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PARAMETER_SCHEMA_VERSION:
            raise ParameterManifestError(
                f"unsupported parameter schema version {self.schema_version!r}"
            )
        if self.canonicalizer_id != PARAMETER_SCHEMA_CANONICALIZER_ID:
            raise ParameterManifestError(
                f"unsupported parameter canonicalizer {self.canonicalizer_id!r}"
            )
        if self.binding_schema_version != EAGER_PARAMETER_BINDING_VERSION:
            raise ParameterManifestError(
                "unsupported eager parameter binding schema "
                f"{self.binding_schema_version!r}"
            )
        _checked_id(self.executor_id, kind="executor")
        if self.executor_id != EAGER_EXECUTOR_ID:
            raise ParameterManifestError(
                f"unsupported eager executor {self.executor_id!r}"
            )
        if (
            not isinstance(self.logical_plan_hash, str)
            or len(self.logical_plan_hash) != 64
            or any(character not in "0123456789abcdef" for character in self.logical_plan_hash)
        ):
            raise ParameterManifestError(
                "logical_plan_hash must be a lowercase SHA-256 hex digest"
            )

        if not isinstance(self.entries, (tuple, list)) or any(
            not isinstance(entry, ParameterManifestEntry) for entry in self.entries
        ):
            raise ParameterManifestError(
                "manifest entries must be ParameterManifestEntry values"
            )
        ordered = tuple(
            sorted(self.entries, key=lambda item: item.logical_key.sort_key())
        )
        logical_keys = [entry.logical_key for entry in ordered]
        if len(logical_keys) != len(set(logical_keys)):
            raise ParameterManifestError("logical parameter keys must be unique")
        locators = [entry.state_dict_locator for entry in ordered]
        if len(locators) != len(set(locators)):
            raise ParameterManifestError("state_dict locators must be unique")
        object.__setattr__(self, "entries", ordered)

    def canonical_dict(self) -> Dict[str, Any]:
        """Return the implementation-independent canonical schema record."""

        return {
            "schema_version": self.schema_version,
            "canonicalizer_id": self.canonicalizer_id,
            "logical_plan_hash": self.logical_plan_hash,
            "parameters": [entry.logical_dict() for entry in self.entries],
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(self.canonical_dict())

    def canonical_hash(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def eager_binding_dict(self) -> Dict[str, Any]:
        """Return the eager-only logical-key to ``state_dict`` locator map."""

        return {
            "schema_version": self.binding_schema_version,
            "executor_id": self.executor_id,
            "parameter_schema_hash": self.canonical_hash(),
            "locators": [entry.locator_dict() for entry in self.entries],
        }

    def eager_binding_bytes(self) -> bytes:
        return _canonical_json_bytes(self.eager_binding_dict())

    def eager_binding_hash(self) -> str:
        return hashlib.sha256(self.eager_binding_bytes()).hexdigest()

    def to_record_dict(self) -> Dict[str, Any]:
        """Return the complete portable schema plus this executor's binding."""

        return {
            "parameter_schema": self.canonical_dict(),
            "eager_binding": self.eager_binding_dict(),
        }

    def validate_model(self, model: nn.Module) -> "ParameterSchemaManifest":
        """Prove that every eager locator names exactly one matching parameter."""

        if not isinstance(model, SettleGraph):
            raise ParameterManifestError(
                "eager parameter binding requires a SettleGraph model"
            )
        if model.plan.canonical_hash() != self.logical_plan_hash:
            raise ParameterManifestError(
                "manifest logical Plan hash does not match the model"
            )

        derived_entries = tuple(
            _receiver_entries(model.plan)
            + _selector_entries(model.plan)
            + _output_entries(model.plan)
        )
        derived_schema = {
            entry.logical_key: entry.logical_dict() for entry in derived_entries
        }
        declared_schema = {
            entry.logical_key: entry.logical_dict() for entry in self.entries
        }
        if declared_schema != derived_schema:
            raise ParameterManifestError(
                "manifest logical entries do not match the Plan formula schema"
            )

        derived_binding = {
            entry.logical_key: entry.locator_dict() for entry in derived_entries
        }
        declared_binding = {
            entry.logical_key: entry.locator_dict() for entry in self.entries
        }
        if declared_binding != derived_binding:
            mismatched = [
                key.canonical_dict()
                for key in sorted(declared_binding, key=lambda item: item.sort_key())
                if declared_binding[key] != derived_binding[key]
            ]
            raise ParameterManifestError(
                "manifest eager locator/shape/transform bindings do not match "
                f"the Plan-derived eager binding for logical keys {mismatched!r}"
            )

        named_with_duplicates = list(model.named_parameters(remove_duplicate=False))
        parameter_ids: Dict[int, str] = {}
        aliases: List[Tuple[str, str]] = []
        for locator, parameter in named_with_duplicates:
            previous = parameter_ids.get(id(parameter))
            if previous is not None:
                aliases.append((previous, locator))
            else:
                parameter_ids[id(parameter)] = locator
        if aliases:
            details = ", ".join(f"{left!r}/{right!r}" for left, right in aliases)
            raise ParameterManifestError(
                "parameter aliases require a closed parameter_group schema: " + details
            )

        storage_owners: Dict[Tuple[str, Optional[int], str, int], str] = {}
        storage_ranges: List[Tuple[str, Optional[int], int, int, str]] = []
        for locator, parameter in named_with_duplicates:
            storage_id, storage_range = _parameter_storage_descriptor(
                parameter, locator=locator
            )
            previous = storage_owners.get(storage_id)
            if previous is not None and previous != locator:
                raise ParameterManifestError(
                    "parameter backing-storage aliases require a closed "
                    f"parameter_group schema: {previous!r}/{locator!r}"
                )
            if storage_range is not None:
                device_type, device_index, start, end = storage_range
                for (
                    other_type,
                    other_index,
                    other_start,
                    other_end,
                    other_locator,
                ) in storage_ranges:
                    if (
                        other_locator != locator
                        and other_type == device_type
                        and other_index == device_index
                        and start < other_end
                        and other_start < end
                    ):
                        raise ParameterManifestError(
                            "parameter backing-storage aliases require a closed "
                            "parameter_group schema: "
                            f"{other_locator!r}/{locator!r}"
                        )
                storage_ranges.append(
                    (device_type, device_index, start, end, locator)
                )
            storage_owners[storage_id] = locator

        named = dict(named_with_duplicates)
        expected_locators = {entry.state_dict_locator for entry in self.entries}
        actual_locators = set(named)
        if expected_locators != actual_locators:
            missing = sorted(expected_locators - actual_locators)
            unexpected = sorted(actual_locators - expected_locators)
            raise ParameterManifestError(
                "eager parameter locator set differs from model.named_parameters(); "
                f"missing={missing!r}, unexpected={unexpected!r}"
            )

        state_dict = model.state_dict()
        expected_dtype = _expected_parameter_dtype(model)
        for entry in self.entries:
            parameter = named[entry.state_dict_locator]
            if tuple(parameter.shape) != entry.state_dict_shape:
                raise ParameterManifestError(
                    f"parameter {entry.logical_key.canonical_dict()!r} expects "
                    f"state_dict shape {entry.state_dict_shape!r}, locator "
                    f"{entry.state_dict_locator!r} has "
                    f"{tuple(parameter.shape)!r}"
                )
            if not parameter.is_floating_point():
                raise ParameterManifestError(
                    f"parameter locator {entry.state_dict_locator!r} must be floating point"
                )
            if expected_dtype is not None and parameter.dtype != expected_dtype:
                raise ParameterManifestError(
                    f"parameter locator {entry.state_dict_locator!r} has dtype "
                    f"{parameter.dtype}, expected {expected_dtype} for role 'parameter'"
                )
            if entry.state_dict_locator not in state_dict:
                raise ParameterManifestError(
                    f"parameter locator {entry.state_dict_locator!r} is absent from state_dict"
                )
            stored = state_dict[entry.state_dict_locator]
            if (
                tuple(stored.shape) != entry.state_dict_shape
                or stored.dtype != parameter.dtype
            ):
                raise ParameterManifestError(
                    f"state_dict tensor at {entry.state_dict_locator!r} disagrees with "
                    "model.named_parameters()"
                )
            parameter_storage, _ = _parameter_storage_descriptor(
                parameter, locator=entry.state_dict_locator
            )
            stored_storage, _ = _parameter_storage_descriptor(
                stored, locator=entry.state_dict_locator
            )
            if (
                stored_storage != parameter_storage
                or stored.storage_offset() != parameter.storage_offset()
                or tuple(stored.stride()) != tuple(parameter.stride())
            ):
                raise ParameterManifestError(
                    f"state_dict tensor at {entry.state_dict_locator!r} is not "
                    "bound to the named parameter storage"
                )
        return self


def _parameter_storage_descriptor(
    parameter: torch.Tensor, *, locator: str
) -> Tuple[
    Tuple[str, Optional[int], str, int],
    Optional[Tuple[str, Optional[int], int, int]],
]:
    """Return version-tolerant storage identity and address-range metadata."""

    if parameter.layout != torch.strided:
        raise ParameterManifestError(
            f"parameter locator {locator!r} uses unsupported layout "
            f"{parameter.layout}"
        )
    span = 0 if parameter.numel() == 0 else 1
    if span:
        for stride, size in sorted(
            (stride, size)
            for size, stride in zip(parameter.shape, parameter.stride())
            if size > 1
        ):
            if stride <= 0 or stride < span:
                raise ParameterManifestError(
                    f"parameter locator {locator!r} has an internally "
                    "overlapping or uncertifiable strided layout"
                )
            span += (size - 1) * stride
    if parameter.storage_offset() < 0:
        raise ParameterManifestError(
            f"parameter locator {locator!r} has a negative storage offset"
        )
    try:
        storage = parameter.untyped_storage()
    except AttributeError:  # pragma: no cover - legacy Torch compatibility
        try:
            storage = parameter.storage()
        except (
            AttributeError,
            NotImplementedError,
            RuntimeError,
            TypeError,
        ) as exc:
            raise ParameterManifestError(
                f"cannot establish backing-storage ownership for parameter "
                f"locator {locator!r}"
            ) from exc
    except (NotImplementedError, RuntimeError, TypeError) as exc:
        raise ParameterManifestError(
            f"cannot establish backing-storage ownership for parameter "
            f"locator {locator!r}"
        ) from exc

    device_type = parameter.device.type
    device_index = parameter.device.index
    storage_cdata = getattr(storage, "_cdata", None)
    try:
        start = int(storage.data_ptr())
    except (AttributeError, NotImplementedError, RuntimeError, TypeError):
        start = 0

    if storage_cdata is not None:
        storage_id = (
            device_type,
            device_index,
            "storage-cdata",
            int(storage_cdata),
        )
    elif start:
        storage_id = (device_type, device_index, "storage-data-ptr", start)
    else:  # pragma: no cover - no supported Torch release takes this branch
        raise ParameterManifestError(
            f"cannot establish unique backing storage for parameter locator "
            f"{locator!r}"
        )

    try:
        storage_nbytes = int(storage.nbytes())
    except (AttributeError, NotImplementedError, RuntimeError, TypeError):
        try:  # pragma: no cover - legacy typed-storage compatibility
            storage_nbytes = int(len(storage)) * parameter.element_size()
        except (
            NotImplementedError,
            RuntimeError,
            TypeError,
        ) as exc:  # pragma: no cover
            raise ParameterManifestError(
                f"cannot determine backing-storage extent for parameter locator "
                f"{locator!r}"
            ) from exc
    storage_range = (
        (device_type, device_index, start, start + storage_nbytes)
        if start and storage_nbytes > 0
        else None
    )
    return storage_id, storage_range


def _expected_parameter_dtype(model: SettleGraph) -> Optional[torch.dtype]:
    if model.typed_plan is None:
        return None
    dtype_name = model.typed_plan.binding.dtype_roles["parameter"]
    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise ParameterManifestError(
            f"typed Plan has unknown parameter dtype {dtype_name!r}"
        )
    return dtype


def _formula_id(config: Mapping[str, Any], *, field: str) -> str:
    formula_id = config.get("formula_id")
    if not isinstance(formula_id, str) or not formula_id:
        raise ParameterManifestError(
            f"{field} does not declare a nonempty formula_id"
        )
    return formula_id


def _entry(
    entries: List[ParameterManifestEntry],
    *,
    field: str,
    role: str,
    formula_id: str,
    shape: Sequence[int],
    locator: str,
    node_id: Optional[str] = None,
    region_id: Optional[str] = None,
    edge_id: Optional[str] = None,
    terminal_node_id: Optional[str] = None,
    state_dict_shape: Optional[Sequence[int]] = None,
) -> None:
    logical_shape = tuple(shape)
    eager_shape = logical_shape if state_dict_shape is None else tuple(state_dict_shape)
    entries.append(
        ParameterManifestEntry(
            logical_key=LogicalParameterKey(
                field=field,
                parameter_role=role,
                node_id=node_id,
                region_id=region_id,
                edge_id=edge_id,
                terminal_node_id=terminal_node_id,
            ),
            formula_id=formula_id,
            shape=logical_shape,
            dtype_role="parameter",
            parameter_group=None,
            state_dict_locator=locator,
            state_dict_shape=eager_shape,
            logical_to_state_dict=(
                "identity" if logical_shape == eager_shape else "reshape-row-major"
            ),
        )
    )


def _receiver_entries(plan: Plan) -> List[ParameterManifestEntry]:
    entries: List[ParameterManifestEntry] = []
    d_model = plan.d_model
    incoming = {
        node.node_id: tuple(
            edge for edge in plan.edges if edge.target == node.node_id
        )
        for node in plan.nodes
    }
    for node in plan.nodes:
        if node.parameter_group is not None:
            raise ParameterManifestError(
                f"node {node.node_id!r} requests parameter_group "
                f"{node.parameter_group!r}; shared parameter schema is not closed"
            )
        node_key = safe_module_key(node.node_id)
        prefix = f"receivers.{node_key}"
        owner = {"node_id": node.node_id, "region_id": node.region_id}

        for field, suffix in (
            ("input_norm", "input_norm.weight"),
            ("ffn_norm", "ffn_norm.weight"),
        ):
            _entry(
                entries,
                field=field,
                role="w",
                formula_id=_formula_id(getattr(node, field), field=field),
                shape=(d_model,),
                locator=f"{prefix}.{suffix}",
                **owner,
            )

        aggregate_type = node.aggregate["type"]
        aggregate_formula = _formula_id(node.aggregate, field="aggregate")
        for edge in incoming[node.node_id]:
            edge_key = safe_module_key(edge.edge_id)
            if aggregate_type == "edge_softmax":
                _entry(
                    entries,
                    field="aggregate",
                    role="eta",
                    formula_id=aggregate_formula,
                    shape=(),
                    locator=f"{prefix}.edge_scores.{edge_key}",
                    edge_id=edge.edge_id,
                    **owner,
                )
            elif aggregate_type == "edge_linear_mean":
                for role, suffix, shape in (
                    ("W", "weight", (d_model, d_model)),
                    ("b", "bias", (d_model,)),
                ):
                    _entry(
                        entries,
                        field="aggregate",
                        role=role,
                        formula_id=aggregate_formula,
                        shape=shape,
                        locator=f"{prefix}.edge_transforms.{edge_key}.{suffix}",
                        edge_id=edge.edge_id,
                        **owner,
                    )

        update_type = node.update["type"]
        update_formula = _formula_id(node.update, field="update")
        if update_type == "ema":
            state_dim = int(node.update["state_dim"])
            for role, suffix, shape in (
                ("W_obs", "ema_observe.weight", (state_dim, d_model)),
                ("b_obs", "ema_observe.bias", (state_dim,)),
            ):
                _entry(
                    entries,
                    field="update",
                    role=role,
                    formula_id=update_formula,
                    shape=shape,
                    locator=f"{prefix}.{suffix}",
                    **owner,
                )
        elif update_type == "gdn":
            key_dim = int(node.update["key_dim"])
            value_dim = int(node.update["value_dim"])
            for role, suffix, shape in (
                ("W_k", "gdn_key.weight", (key_dim, d_model)),
                ("W_nu", "gdn_value.weight", (value_dim, d_model)),
                ("w_eta", "gdn_eta.weight", (d_model,)),
                ("b_eta", "gdn_eta.bias", ()),
                ("w_gamma", "gdn_gamma.weight", (d_model,)),
                ("b_gamma", "gdn_gamma.bias", ()),
                ("beta", "gdn_beta", ()),
            ):
                _entry(
                    entries,
                    field="update",
                    role=role,
                    formula_id=update_formula,
                    shape=shape,
                    locator=f"{prefix}.{suffix}",
                    state_dict_shape=(
                        (1, d_model)
                        if role in {"w_eta", "w_gamma"}
                        else (1,)
                        if role in {"b_eta", "b_gamma"}
                        else shape
                    ),
                    **owner,
                )
        elif update_type == "attention_window":
            key_dim = int(node.update["key_dim"])
            value_dim = int(node.update["value_dim"])
            for role, suffix, shape in (
                ("W_k", "attn_key.weight", (key_dim, d_model)),
                ("W_nu", "attn_value.weight", (value_dim, d_model)),
            ):
                _entry(
                    entries,
                    field="update",
                    role=role,
                    formula_id=update_formula,
                    shape=shape,
                    locator=f"{prefix}.{suffix}",
                    **owner,
                )

        read_type = node.selector_read["type"]
        if read_type in {
            "content_linear",
            "content_state_linear",
            "content_state_summary_linear",
        }:
            read_dim = int(node.selector_read["out_dim"])
            if read_type == "content_linear":
                input_dim = d_model
            elif read_type == "content_state_linear":
                input_dim = d_model + math.prod(node.state_shape)
            else:
                input_dim = d_model + 1
            formula = _formula_id(node.selector_read, field="selector_read")
            for role, suffix, shape in (
                ("W_sel", "selector_read_linear.weight", (read_dim, input_dim)),
                ("b_sel", "selector_read_linear.bias", (read_dim,)),
            ):
                _entry(
                    entries,
                    field="selector_read",
                    role=role,
                    formula_id=formula,
                    shape=shape,
                    locator=f"{prefix}.{suffix}",
                    **owner,
                )

        ffn_type = node.ffn_read["type"]
        if ffn_type == "state_default":
            formula = _formula_id(node.ffn_read, field="ffn_read")
            if update_type == "ema":
                _entry(
                    entries,
                    field="ffn_read",
                    role="W_out",
                    formula_id=formula,
                    shape=(d_model, int(node.update["state_dim"])),
                    locator=f"{prefix}.state_out.weight",
                    **owner,
                )
            elif update_type in {"gdn", "attention_window"}:
                stem = "gdn" if update_type == "gdn" else "attn"
                key_dim = int(node.update["key_dim"])
                value_dim = int(node.update["value_dim"])
                for role, suffix, shape in (
                    ("W_q", f"{stem}_query.weight", (key_dim, d_model)),
                    ("W_out", f"{stem}_out.weight", (d_model, value_dim)),
                ):
                    _entry(
                        entries,
                        field="ffn_read",
                        role=role,
                        formula_id=formula,
                        shape=shape,
                        locator=f"{prefix}.{suffix}",
                        **owner,
                    )

        compute_type = node.node_compute["type"]
        compute_formula = _formula_id(node.node_compute, field="node_compute")
        if compute_type == "affine_residual":
            for role, suffix, shape in (
                ("W_node", "down_proj.weight", (d_model, d_model)),
                ("b_node", "down_proj.bias", (d_model,)),
            ):
                _entry(
                    entries,
                    field="node_compute",
                    role=role,
                    formula_id=compute_formula,
                    shape=shape,
                    locator=f"{prefix}.{suffix}",
                    **owner,
                )
        elif compute_type == "double_residual_swiglu":
            hidden_dim = int(node.node_compute["hidden_dim"])
            for role, suffix, shape in (
                ("W_g", "gate_proj.weight", (hidden_dim, d_model)),
                ("b_g", "gate_proj.bias", (hidden_dim,)),
                ("W_u", "up_proj.weight", (hidden_dim, d_model)),
                ("b_u", "up_proj.bias", (hidden_dim,)),
                ("W_o", "down_proj.weight", (d_model, hidden_dim)),
                ("b_o", "down_proj.bias", (d_model,)),
            ):
                _entry(
                    entries,
                    field="node_compute",
                    role=role,
                    formula_id=compute_formula,
                    shape=shape,
                    locator=f"{prefix}.{suffix}",
                    **owner,
                )
    return entries


def _selector_entries(plan: Plan) -> List[ParameterManifestEntry]:
    entries: List[ParameterManifestEntry] = []
    nodes = {node.node_id: node for node in plan.nodes}
    for region in plan.regions:
        score_type = region.score["type"]
        if score_type not in {"linear", "mlp"}:
            continue
        formula = _formula_id(region.score, field="score")
        input_dim = int(nodes[region.node_ids[0]].selector_read_shape[0])
        prefix = f"selectors.{safe_module_key(region.region_id)}"
        for node_id in region.node_ids:
            node_key = safe_module_key(node_id)
            owner = {"node_id": node_id, "region_id": region.region_id}
            if score_type == "linear":
                for role, suffix, shape in (
                    ("w_score", f"linears.{node_key}.weight", (input_dim,)),
                    ("b_score", f"linears.{node_key}.bias", ()),
                ):
                    _entry(
                        entries,
                        field="score",
                        role=role,
                        formula_id=formula,
                        shape=shape,
                        locator=f"{prefix}.{suffix}",
                        state_dict_shape=(
                            (1, input_dim) if role == "w_score" else (1,)
                        ),
                        **owner,
                    )
            else:
                hidden_dim = int(region.score["hidden_dim"])
                for role, suffix, shape in (
                    ("W_1", f"hidden_layers.{node_key}.weight", (hidden_dim, input_dim)),
                    ("b_1", f"hidden_layers.{node_key}.bias", (hidden_dim,)),
                    ("w_2", f"output_layers.{node_key}.weight", (hidden_dim,)),
                    ("b_2", f"output_layers.{node_key}.bias", ()),
                ):
                    _entry(
                        entries,
                        field="score",
                        role=role,
                        formula_id=formula,
                        shape=shape,
                        locator=f"{prefix}.{suffix}",
                        state_dict_shape=(
                            (1, hidden_dim)
                            if role == "w_2"
                            else (1,)
                            if role == "b_2"
                            else shape
                        ),
                        **owner,
                    )
    return entries


def _output_entries(plan: Plan) -> List[ParameterManifestEntry]:
    if plan.output_aggregate["type"] != "node_softmax":
        return []
    formula = _formula_id(plan.output_aggregate, field="output_aggregate")
    entries: List[ParameterManifestEntry] = []
    for node_id in plan.terminal_node_ids:
        _entry(
            entries,
            field="output_aggregate",
            role="eta_out",
            formula_id=formula,
            shape=(),
            locator=f"output_scores.{safe_module_key(node_id)}",
            terminal_node_id=node_id,
        )
    return entries


def build_parameter_schema_manifest(plan: Plan) -> ParameterSchemaManifest:
    """Derive the complete logical parameter schema from a validated Plan.

    The returned object's canonical record contains no executor locators.  Its
    entries also carry the eager binding locators so
    :func:`build_eager_parameter_manifest` can validate that implementation
    without deriving a second logical schema.
    """

    plan = plan.validate()
    entries = tuple(
        _receiver_entries(plan) + _selector_entries(plan) + _output_entries(plan)
    )
    return ParameterSchemaManifest(
        logical_plan_hash=plan.canonical_hash(), entries=entries
    )


def build_eager_parameter_manifest(model: nn.Module) -> ParameterSchemaManifest:
    """Derive and validate schema v1 for an eager-reference SettleGraph."""

    if not isinstance(model, SettleGraph):
        raise ParameterManifestError(
            "eager parameter manifests require a SettleGraph model"
        )
    manifest = build_parameter_schema_manifest(model.plan)
    return manifest.validate_model(model)


def export_eager_parameter_tensors(
    model: nn.Module, manifest: ParameterSchemaManifest
) -> Mapping[str, torch.Tensor]:
    """Expose eager parameters under their implementation-independent keys."""

    manifest.validate_model(model)
    named = dict(model.named_parameters())
    return {
        logical_parameter_tensor_key(entry.logical_key.canonical_dict()): named[
            entry.state_dict_locator
        ].reshape(entry.shape)
        for entry in manifest.entries
    }


def load_eager_parameter_tensors(
    model: nn.Module,
    manifest: ParameterSchemaManifest,
    tensors: Mapping[str, torch.Tensor],
) -> nn.Module:
    """Validate and stage a portable logical Tensor mapping before copying.

    The copy is intended for an executor-loading safe point with no active
    autograd graph or concurrent parameter readers.  If a copy raises, all
    parameter *values* are restored, but Torch's in-place version counters
    cannot be rolled back and the sequence of in-place copies is not an
    atomic visibility boundary for concurrent readers.
    """

    manifest.validate_model(model)
    if not isinstance(tensors, Mapping):
        raise ParameterManifestError("parameter tensors must be a mapping")
    supplied = dict(tensors)
    if any(not isinstance(key, str) for key in supplied):
        raise ParameterManifestError(
            "logical parameter Tensor keys must all be strings"
        )
    named = dict(model.named_parameters())
    entries = {
        logical_parameter_tensor_key(entry.logical_key.canonical_dict()): entry
        for entry in manifest.entries
    }
    if set(supplied) != set(entries):
        missing = sorted(set(entries) - set(supplied))
        unexpected = sorted(set(supplied) - set(entries))
        raise ParameterManifestError(
            "logical parameter Tensor set differs from the manifest; "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )
    source_storage_owners: Dict[
        Tuple[str, Optional[int], str, int], str
    ] = {}
    source_storage_ranges: List[
        Tuple[str, Optional[int], int, int, str]
    ] = []
    for key, source in supplied.items():
        entry = entries[key]
        target = named[entry.state_dict_locator]
        if (
            not isinstance(source, torch.Tensor)
            or source.device.type != "cpu"
            or source.layout != torch.strided
            or not source.is_floating_point()
            or tuple(source.shape) != entry.shape
            or source.dtype != target.dtype
        ):
            raise ParameterManifestError(
                f"logical parameter Tensor {key!r} must be a CPU floating "
                "strided Tensor with shape "
                f"{entry.shape!r} and dtype {target.dtype}"
            )
        storage_id, storage_range = _parameter_storage_descriptor(
            source, locator=key
        )
        previous = source_storage_owners.get(storage_id)
        if previous is not None and previous != key:
            raise ParameterManifestError(
                "logical parameter Tensor backing storage must be independent "
                f"without a parameter_group: {previous!r}/{key!r}"
            )
        if storage_range is not None:
            device_type, device_index, start, end = storage_range
            for (
                other_type,
                other_index,
                other_start,
                other_end,
                other_key,
            ) in source_storage_ranges:
                if (
                    other_key != key
                    and other_type == device_type
                    and other_index == device_index
                    and start < other_end
                    and other_start < end
                ):
                    raise ParameterManifestError(
                        "logical parameter Tensor backing storage must be "
                        "independent without a parameter_group: "
                        f"{other_key!r}/{key!r}"
                    )
            source_storage_ranges.append(
                (device_type, device_index, start, end, key)
            )
        source_storage_owners[storage_id] = key

    staged_cpu = {
        key: supplied[key].detach().clone(memory_format=torch.preserve_format)
        for key in entries
    }
    staged = {}
    for key, source in staged_cpu.items():
        entry = entries[key]
        target = named[entry.state_dict_locator]
        try:
            staged[key] = source.reshape(entry.state_dict_shape).to(
                device=target.device
            )
        except (NotImplementedError, RuntimeError, TypeError, ValueError) as exc:
            raise ParameterManifestError(
                f"cannot stage logical parameter Tensor {key!r} on the "
                f"target device {target.device}"
            ) from exc
    snapshot = {
        name: parameter.detach().clone(memory_format=torch.preserve_format)
        for name, parameter in named.items()
    }
    try:
        with torch.no_grad():
            for key, source in staged.items():
                entry = entries[key]
                target = named[entry.state_dict_locator]
                target.copy_(source)
    except BaseException as copy_error:
        rollback_errors = []
        with torch.no_grad():
            for name, target in named.items():
                try:
                    target.copy_(snapshot[name])
                except BaseException as rollback_error:
                    rollback_errors.append((name, rollback_error))
        if rollback_errors:
            # Diagnostics must never replace the primary copy failure.  A
            # user-defined exception may reject both attributes and notes.
            try:
                setattr(
                    copy_error,
                    "tide_parameter_rollback_failures",
                    tuple(rollback_errors),
                )
            except BaseException:
                pass
            try:
                details = ", ".join(
                    f"{name}: {type(error).__name__}"
                    for name, error in rollback_errors
                )
                add_note = getattr(copy_error, "add_note", None)
                if callable(add_note):
                    add_note(
                        "parameter value rollback also failed for " + details
                    )
            except BaseException:
                pass
        raise
    return model


__all__ = [
    "EAGER_EXECUTOR_ID",
    "EAGER_PARAMETER_BINDING_VERSION",
    "LogicalParameterKey",
    "PARAMETER_SCHEMA_CANONICALIZER_ID",
    "PARAMETER_SCHEMA_VERSION",
    "ParameterManifestEntry",
    "ParameterManifestError",
    "ParameterSchemaManifest",
    "build_eager_parameter_manifest",
    "build_parameter_schema_manifest",
    "export_eager_parameter_tensors",
    "load_eager_parameter_tensors",
    "logical_parameter_tensor_key",
]

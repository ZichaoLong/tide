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
from .plan import (
    OperationParameterSlot,
    Plan,
    operation_parameter_slots,
    validate_stable_id,
)


PARAMETER_SCHEMA_VERSION = "tide.parameter-schema.v1"
PARAMETER_SCHEMA_CANONICALIZER_ID = "tide-parameter-schema-json-v1"
EAGER_PARAMETER_BINDING_VERSION = "tide.eager-parameter-binding.v1"
PARAMETER_SCHEMA_VERSION_V2 = "tide.parameter-schema.v2"
PARAMETER_SCHEMA_CANONICALIZER_ID_V2 = "tide-parameter-schema-json-v2"
EAGER_PARAMETER_BINDING_VERSION_V2 = "tide.eager-parameter-binding.v2"
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
    """Implementation-independent key for one formula-parameter use-site.

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


def _operation_use_key(
    logical_key: LogicalParameterKey,
) -> Tuple[str, str, str]:
    if logical_key.field == "score":
        assert logical_key.region_id is not None
        return ("region", logical_key.region_id, "score")
    if logical_key.field == "output_aggregate":
        return ("graph", "graph", "output_aggregate")
    assert logical_key.node_id is not None
    return ("node", logical_key.node_id, logical_key.field)


def _entry_tensor_key(
    entry: "ParameterManifestEntry", schema_version: str
) -> str:
    if schema_version == PARAMETER_SCHEMA_VERSION:
        return logical_parameter_tensor_key(entry.logical_key.canonical_dict())
    return entry.parameter_identity_key()


@dataclass(frozen=True)
class ParameterManifestEntry:
    """One operation-local parameter use plus its eager-reference locator."""

    logical_key: LogicalParameterKey
    formula_id: str
    shape: Tuple[int, ...]
    dtype_role: str
    parameter_group: Optional[str]
    state_dict_locator: str
    state_dict_shape: Optional[Tuple[int, ...]] = None
    logical_to_state_dict: str = "identity"
    parameter_slot: Optional[str] = None

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
        if self.parameter_slot is not None:
            _checked_id(self.parameter_slot, kind="parameter slot")
        if (self.parameter_group is None) != (self.parameter_slot is None):
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

    def parameter_identity_dict(self) -> Dict[str, Any]:
        """Return the schema-v2 identity of the Tensor used at this site."""

        if self.parameter_group is None:
            return {
                "kind": "use",
                "logical_key": self.logical_key.canonical_dict(),
            }
        return {
            "kind": "set",
            "parameter_set_id": self.parameter_group,
            "parameter_slot": self.parameter_slot,
        }

    def parameter_identity_key(self) -> str:
        return "tide.logical-parameter.v2:" + json.dumps(
            self.parameter_identity_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def use_schema_dict_v2(self) -> Dict[str, Any]:
        return {
            "logical_key": self.logical_key.canonical_dict(),
            "formula_id": self.formula_id,
            "shape": list(self.shape),
            "dtype_role": self.dtype_role,
            "parameter_id": self.parameter_identity_key(),
            "parameter_set_id": self.parameter_group,
            "parameter_slot": self.parameter_slot,
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
        supported_versions = {
            PARAMETER_SCHEMA_VERSION: (
                PARAMETER_SCHEMA_CANONICALIZER_ID,
                EAGER_PARAMETER_BINDING_VERSION,
            ),
            PARAMETER_SCHEMA_VERSION_V2: (
                PARAMETER_SCHEMA_CANONICALIZER_ID_V2,
                EAGER_PARAMETER_BINDING_VERSION_V2,
            ),
        }
        if self.schema_version not in supported_versions:
            raise ParameterManifestError(
                f"unsupported parameter schema version {self.schema_version!r}"
            )
        expected_canonicalizer, expected_binding = supported_versions[
            self.schema_version
        ]
        if self.canonicalizer_id != expected_canonicalizer:
            raise ParameterManifestError(
                f"unsupported parameter canonicalizer {self.canonicalizer_id!r}"
            )
        if self.binding_schema_version != expected_binding:
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
        if self.schema_version == PARAMETER_SCHEMA_VERSION:
            if any(
                entry.parameter_group is not None
                or entry.parameter_slot is not None
                for entry in ordered
            ):
                raise ParameterManifestError(
                    "parameter groups are not closed in parameter schema v1"
                )
        else:
            self._validate_v2_parameter_identities(ordered)
        object.__setattr__(self, "entries", ordered)

    @staticmethod
    def _validate_v2_parameter_identities(
        entries: Sequence[ParameterManifestEntry],
    ) -> None:
        contracts: Dict[str, Tuple[str, Tuple[int, ...], str]] = {}
        uses_by_set: Dict[
            str,
            Dict[Tuple[str, str, str], Dict[str, Tuple[str, Tuple[int, ...], str]]],
        ] = {}
        for entry in entries:
            identity = entry.parameter_identity_key()
            contract = (entry.formula_id, entry.shape, entry.dtype_role)
            previous = contracts.get(identity)
            if previous is not None and previous != contract:
                raise ParameterManifestError(
                    f"logical parameter {identity!r} has incompatible contracts"
                )
            contracts[identity] = contract
            if entry.parameter_group is None:
                continue
            assert entry.parameter_slot is not None
            use = _operation_use_key(entry.logical_key)
            slots = uses_by_set.setdefault(entry.parameter_group, {}).setdefault(
                use, {}
            )
            if entry.parameter_slot in slots:
                raise ParameterManifestError(
                    f"parameter set {entry.parameter_group!r} repeats slot "
                    f"{entry.parameter_slot!r} at operation use {use!r}"
                )
            slots[entry.parameter_slot] = contract

        for parameter_set_id, uses in uses_by_set.items():
            expected: Optional[Dict[str, Tuple[str, Tuple[int, ...], str]]] = None
            expected_use: Optional[Tuple[str, str, str]] = None
            for use, slots in sorted(uses.items()):
                if expected is None:
                    expected = slots
                    expected_use = use
                elif slots != expected:
                    raise ParameterManifestError(
                        f"parameter set {parameter_set_id!r} has incompatible "
                        f"operation slots at {expected_use!r} and {use!r}"
                    )

    def canonical_dict(self) -> Dict[str, Any]:
        """Return the implementation-independent canonical schema record."""

        if self.schema_version == PARAMETER_SCHEMA_VERSION_V2:
            parameters: Dict[str, Dict[str, Any]] = {}
            uses = []
            for entry in self.entries:
                parameter_id = entry.parameter_identity_key()
                uses.append(
                    {
                        "logical_key": entry.logical_key.canonical_dict(),
                        "parameter_id": parameter_id,
                        "parameter_set_id": entry.parameter_group,
                        "parameter_slot": entry.parameter_slot,
                    }
                )
                parameters.setdefault(
                    parameter_id,
                    {
                        "parameter_id": parameter_id,
                        "formula_id": entry.formula_id,
                        "shape": list(entry.shape),
                        "dtype_role": entry.dtype_role,
                    },
                )
            return {
                "schema_version": self.schema_version,
                "canonicalizer_id": self.canonicalizer_id,
                "logical_plan_hash": self.logical_plan_hash,
                "parameter_uses": uses,
                "parameters": [parameters[key] for key in sorted(parameters)],
            }
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

        derived_manifest = build_parameter_schema_manifest(model.plan)
        if (
            self.schema_version != derived_manifest.schema_version
            or self.canonicalizer_id != derived_manifest.canonicalizer_id
            or self.binding_schema_version
            != derived_manifest.binding_schema_version
        ):
            raise ParameterManifestError(
                "manifest version does not match the Plan schema version"
            )
        derived_entries = derived_manifest.entries
        schema_record = (
            (lambda entry: entry.use_schema_dict_v2())
            if self.schema_version == PARAMETER_SCHEMA_VERSION_V2
            else (lambda entry: entry.logical_dict())
        )
        derived_schema = {
            entry.logical_key: schema_record(entry) for entry in derived_entries
        }
        declared_schema = {
            entry.logical_key: schema_record(entry) for entry in self.entries
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
        entry_by_locator = {
            entry.state_dict_locator: entry for entry in self.entries
        }
        identity_by_object: Dict[int, Tuple[str, str]] = {}
        object_by_identity: Dict[str, Tuple[int, str]] = {}
        for locator, parameter in named_with_duplicates:
            entry = entry_by_locator.get(locator)
            if entry is None:
                continue
            logical_identity = _entry_tensor_key(entry, self.schema_version)
            previous_identity = identity_by_object.get(id(parameter))
            if (
                previous_identity is not None
                and previous_identity[0] != logical_identity
            ):
                raise ParameterManifestError(
                    "parameter aliases are not declared by one parameter set: "
                    f"{previous_identity[1]!r}/{locator!r}"
                )
            identity_by_object[id(parameter)] = (logical_identity, locator)
            previous_object = object_by_identity.get(logical_identity)
            if previous_object is not None and previous_object[0] != id(parameter):
                raise ParameterManifestError(
                    f"logical parameter {logical_identity!r} is bound to distinct "
                    f"Parameters at {previous_object[1]!r}/{locator!r}"
                )
            object_by_identity[logical_identity] = (id(parameter), locator)

        storage_owners: Dict[
            Tuple[str, Optional[int], str, int], Tuple[str, int, str]
        ] = {}
        storage_ranges: List[
            Tuple[str, Optional[int], int, int, str, int, str]
        ] = []
        for locator, parameter in named_with_duplicates:
            entry = entry_by_locator.get(locator)
            if entry is None:
                continue
            logical_identity = _entry_tensor_key(entry, self.schema_version)
            storage_id, storage_range = _parameter_storage_descriptor(
                parameter, locator=locator
            )
            previous = storage_owners.get(storage_id)
            if previous is not None and (
                previous[0] != logical_identity or previous[1] != id(parameter)
            ):
                raise ParameterManifestError(
                    "parameter backing-storage aliases are not one declared "
                    f"Parameter: {previous[2]!r}/{locator!r}"
                )
            if storage_range is not None:
                device_type, device_index, start, end = storage_range
                for (
                    other_type,
                    other_index,
                    other_start,
                    other_end,
                    other_identity,
                    other_object,
                    other_locator,
                ) in storage_ranges:
                    if (
                        other_type == device_type
                        and other_index == device_index
                        and start < other_end
                        and other_start < end
                        and (
                            other_identity != logical_identity
                            or other_object != id(parameter)
                        )
                    ):
                        raise ParameterManifestError(
                            "parameter backing-storage aliases are not one "
                            "declared Parameter: "
                            f"{other_locator!r}/{locator!r}"
                        )
                storage_ranges.append(
                    (
                        device_type,
                        device_index,
                        start,
                        end,
                        logical_identity,
                        id(parameter),
                        locator,
                    )
                )
            storage_owners[storage_id] = (
                logical_identity,
                id(parameter),
                locator,
            )

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


def _entry(
    entries: List[ParameterManifestEntry],
    *,
    field: str,
    contract: OperationParameterSlot,
    locator: str,
    node_id: Optional[str] = None,
    region_id: Optional[str] = None,
    edge_id: Optional[str] = None,
    terminal_node_id: Optional[str] = None,
    state_dict_shape: Optional[Sequence[int]] = None,
    parameter_set_id: Optional[str] = None,
    parameter_slot: Optional[str] = None,
) -> None:
    logical_shape = contract.shape
    eager_shape = logical_shape if state_dict_shape is None else tuple(state_dict_shape)
    entries.append(
        ParameterManifestEntry(
            logical_key=LogicalParameterKey(
                field=field,
                parameter_role=contract.parameter_role,
                node_id=node_id,
                region_id=region_id,
                edge_id=edge_id,
                terminal_node_id=terminal_node_id,
            ),
            formula_id=contract.formula_id,
            shape=logical_shape,
            dtype_role=contract.dtype_role,
            parameter_group=parameter_set_id,
            state_dict_locator=locator,
            state_dict_shape=eager_shape,
            logical_to_state_dict=(
                "identity" if logical_shape == eager_shape else "reshape-row-major"
            ),
            parameter_slot=parameter_slot,
        )
    )


def _operation_slot_map(
    plan: Plan,
    use: Tuple[str, str, str],
) -> Mapping[str, OperationParameterSlot]:
    slots = operation_parameter_slots(plan, *use)
    mapped = {slot.parameter_slot: slot for slot in slots}
    if len(mapped) != len(slots):  # pragma: no cover - module invariant
        raise ParameterManifestError(
            f"operation use {use!r} repeats a canonical parameter slot"
        )
    return mapped


def _required_operation_slot(
    contracts: Mapping[str, OperationParameterSlot],
    use: Tuple[str, str, str],
    slot: str,
) -> OperationParameterSlot:
    try:
        return contracts[slot]
    except KeyError as exc:  # pragma: no cover - executor/schema invariant
        raise ParameterManifestError(
            f"eager binding for operation use {use!r} has no canonical "
            f"parameter slot {slot!r}"
        ) from exc


def _parameter_binding_map(
    plan: Plan,
) -> Dict[Tuple[str, str, str], str]:
    if plan.schema_version != "2":
        return {}
    return {
        (binding.owner_kind, binding.owner_id, binding.operation):
        binding.parameter_set_id
        for binding in plan.parameter_bindings
    }


def _bound_parameter_kwargs(
    bindings: Mapping[Tuple[str, str, str], str],
    use: Tuple[str, str, str],
    slot: str,
) -> Dict[str, Optional[str]]:
    parameter_set_id = bindings.get(use)
    return {
        "parameter_set_id": parameter_set_id,
        "parameter_slot": slot if parameter_set_id is not None else None,
    }


def _receiver_entries(plan: Plan) -> List[ParameterManifestEntry]:
    entries: List[ParameterManifestEntry] = []
    incoming = {
        node.node_id: tuple(
            edge for edge in plan.edges if edge.target == node.node_id
        )
        for node in plan.nodes
    }
    bindings = _parameter_binding_map(plan)
    for node in plan.nodes:
        node_key = safe_module_key(node.node_id)
        prefix = f"receivers.{node_key}"
        owner = {"node_id": node.node_id, "region_id": node.region_id}
        contracts_by_operation: Dict[
            str, Mapping[str, OperationParameterSlot]
        ] = {}

        def add_node_parameter(
            field: str,
            slot: str,
            locator: str,
            *,
            edge_id: Optional[str] = None,
            state_dict_shape: Optional[Sequence[int]] = None,
        ) -> None:
            use = ("node", node.node_id, field)
            contracts = contracts_by_operation.get(field)
            if contracts is None:
                contracts = _operation_slot_map(plan, use)
                contracts_by_operation[field] = contracts
            _entry(
                entries,
                field=field,
                contract=_required_operation_slot(contracts, use, slot),
                locator=locator,
                edge_id=edge_id,
                state_dict_shape=state_dict_shape,
                **_bound_parameter_kwargs(bindings, use, slot),
                **owner,
            )

        for field, suffix in (
            ("input_norm", "input_norm.weight"),
            ("ffn_norm", "ffn_norm.weight"),
        ):
            add_node_parameter(field, "w", f"{prefix}.{suffix}")

        aggregate_type = node.aggregate["type"]
        for edge_ordinal, edge in enumerate(incoming[node.node_id]):
            edge_key = safe_module_key(edge.edge_id)
            if aggregate_type == "edge_softmax":
                add_node_parameter(
                    "aggregate",
                    f"edge.{edge_ordinal}.eta",
                    f"{prefix}.edge_scores.{edge_key}",
                    edge_id=edge.edge_id,
                )
            elif aggregate_type == "edge_linear_mean":
                for role, suffix in (
                    ("W", "weight"),
                    ("b", "bias"),
                ):
                    add_node_parameter(
                        "aggregate",
                        f"edge.{edge_ordinal}.{role}",
                        f"{prefix}.edge_transforms.{edge_key}.{suffix}",
                        edge_id=edge.edge_id,
                    )

        update_type = node.update["type"]
        if update_type == "ema":
            for role, suffix in (
                ("W_obs", "ema_observe.weight"),
                ("b_obs", "ema_observe.bias"),
            ):
                add_node_parameter("update", role, f"{prefix}.{suffix}")
        elif update_type == "gdn":
            for role, suffix in (
                ("W_k", "gdn_key.weight"),
                ("W_nu", "gdn_value.weight"),
                ("w_eta", "gdn_eta.weight"),
                ("b_eta", "gdn_eta.bias"),
                ("w_gamma", "gdn_gamma.weight"),
                ("b_gamma", "gdn_gamma.bias"),
                ("beta", "gdn_beta"),
            ):
                use = ("node", node.node_id, "update")
                contracts = contracts_by_operation.get("update")
                if contracts is None:
                    contracts = _operation_slot_map(plan, use)
                    contracts_by_operation["update"] = contracts
                contract = _required_operation_slot(contracts, use, role)
                state_dict_shape = (
                    (1, contract.shape[0])
                    if role in {"w_eta", "w_gamma"}
                    else (1,)
                    if role in {"b_eta", "b_gamma"}
                    else contract.shape
                )
                add_node_parameter(
                    "update",
                    role,
                    f"{prefix}.{suffix}",
                    state_dict_shape=state_dict_shape,
                )
        elif update_type == "attention_window":
            for role, suffix in (
                ("W_k", "attn_key.weight"),
                ("W_nu", "attn_value.weight"),
            ):
                add_node_parameter("update", role, f"{prefix}.{suffix}")

        read_type = node.selector_read["type"]
        if read_type in {
            "content_linear",
            "content_state_linear",
            "content_state_summary_linear",
        }:
            for role, suffix in (
                ("W_sel", "selector_read_linear.weight"),
                ("b_sel", "selector_read_linear.bias"),
            ):
                add_node_parameter(
                    "selector_read", role, f"{prefix}.{suffix}"
                )

        ffn_type = node.ffn_read["type"]
        if ffn_type == "state_default":
            if update_type == "ema":
                add_node_parameter(
                    "ffn_read", "W_out", f"{prefix}.state_out.weight"
                )
            elif update_type in {"gdn", "attention_window"}:
                stem = "gdn" if update_type == "gdn" else "attn"
                for role, suffix in (
                    ("W_q", f"{stem}_query.weight"),
                    ("W_out", f"{stem}_out.weight"),
                ):
                    add_node_parameter("ffn_read", role, f"{prefix}.{suffix}")

        compute_type = node.node_compute["type"]
        if compute_type == "affine_residual":
            for role, suffix in (
                ("W_node", "down_proj.weight"),
                ("b_node", "down_proj.bias"),
            ):
                add_node_parameter("node_compute", role, f"{prefix}.{suffix}")
        elif compute_type == "double_residual_swiglu":
            for role, suffix in (
                ("W_g", "gate_proj.weight"),
                ("b_g", "gate_proj.bias"),
                ("W_u", "up_proj.weight"),
                ("b_u", "up_proj.bias"),
                ("W_o", "down_proj.weight"),
                ("b_o", "down_proj.bias"),
            ):
                add_node_parameter("node_compute", role, f"{prefix}.{suffix}")
    return entries


def _selector_entries(plan: Plan) -> List[ParameterManifestEntry]:
    entries: List[ParameterManifestEntry] = []
    bindings = _parameter_binding_map(plan)
    for region in plan.regions:
        score_type = region.score["type"]
        if score_type not in {"linear", "mlp"}:
            continue
        prefix = f"selectors.{safe_module_key(region.region_id)}"
        use = ("region", region.region_id, "score")
        contracts = _operation_slot_map(plan, use)
        for candidate_ordinal, node_id in enumerate(region.node_ids):
            node_key = safe_module_key(node_id)
            owner = {"node_id": node_id, "region_id": region.region_id}

            def add_score_parameter(
                role: str,
                suffix: str,
                *,
                state_dict_shape: Optional[Sequence[int]] = None,
            ) -> None:
                slot = f"candidate.{candidate_ordinal}.{role}"
                _entry(
                    entries,
                    field="score",
                    contract=_required_operation_slot(contracts, use, slot),
                    locator=f"{prefix}.{suffix}",
                    state_dict_shape=state_dict_shape,
                    **_bound_parameter_kwargs(bindings, use, slot),
                    **owner,
                )

            if score_type == "linear":
                for role, suffix in (
                    ("w_score", f"linears.{node_key}.weight"),
                    ("b_score", f"linears.{node_key}.bias"),
                ):
                    contract = _required_operation_slot(
                        contracts,
                        use,
                        f"candidate.{candidate_ordinal}.{role}",
                    )
                    add_score_parameter(
                        role,
                        suffix,
                        state_dict_shape=(
                            (1, contract.shape[0])
                            if role == "w_score"
                            else (1,)
                        ),
                    )
            else:
                for role, suffix in (
                    ("W_1", f"hidden_layers.{node_key}.weight"),
                    ("b_1", f"hidden_layers.{node_key}.bias"),
                    ("w_2", f"output_layers.{node_key}.weight"),
                    ("b_2", f"output_layers.{node_key}.bias"),
                ):
                    contract = _required_operation_slot(
                        contracts,
                        use,
                        f"candidate.{candidate_ordinal}.{role}",
                    )
                    add_score_parameter(
                        role,
                        suffix,
                        state_dict_shape=(
                            (1, contract.shape[0])
                            if role == "w_2"
                            else (1,)
                            if role == "b_2"
                            else contract.shape
                        ),
                    )
    return entries


def _output_entries(plan: Plan) -> List[ParameterManifestEntry]:
    if plan.output_aggregate["type"] != "node_softmax":
        return []
    entries: List[ParameterManifestEntry] = []
    bindings = _parameter_binding_map(plan)
    use = ("graph", "graph", "output_aggregate")
    contracts = _operation_slot_map(plan, use)
    for terminal_ordinal, node_id in enumerate(plan.terminal_node_ids):
        slot = f"terminal.{terminal_ordinal}.eta_out"
        _entry(
            entries,
            field="output_aggregate",
            contract=_required_operation_slot(contracts, use, slot),
            locator=f"output_scores.{safe_module_key(node_id)}",
            terminal_node_id=node_id,
            **_bound_parameter_kwargs(bindings, use, slot),
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
    if plan.schema_version == "2":
        return ParameterSchemaManifest(
            logical_plan_hash=plan.canonical_hash(),
            entries=entries,
            schema_version=PARAMETER_SCHEMA_VERSION_V2,
            canonicalizer_id=PARAMETER_SCHEMA_CANONICALIZER_ID_V2,
            binding_schema_version=EAGER_PARAMETER_BINDING_VERSION_V2,
        )
    return ParameterSchemaManifest(
        logical_plan_hash=plan.canonical_hash(), entries=entries
    )


def _parameter_at_locator(model: nn.Module, locator: str) -> nn.Parameter:
    current: Any = model
    parts = locator.split(".")
    for part in parts[:-1]:
        if isinstance(current, (nn.ModuleDict, nn.ParameterDict)):
            current = current[part]
        else:
            current = getattr(current, part)
    leaf = parts[-1]
    value = (
        current[leaf]
        if isinstance(current, nn.ParameterDict)
        else getattr(current, leaf)
    )
    if not isinstance(value, nn.Parameter):
        raise ParameterManifestError(
            f"eager locator {locator!r} does not name an nn.Parameter"
        )
    return value


def _set_parameter_at_locator(
    model: nn.Module, locator: str, parameter: nn.Parameter
) -> None:
    current: Any = model
    parts = locator.split(".")
    for part in parts[:-1]:
        if isinstance(current, (nn.ModuleDict, nn.ParameterDict)):
            current = current[part]
        else:
            current = getattr(current, part)
    leaf = parts[-1]
    if isinstance(current, nn.ParameterDict):
        current[leaf] = parameter
    else:
        setattr(current, leaf, parameter)


def tie_eager_parameters(
    model: nn.Module,
    manifest: Optional[ParameterSchemaManifest] = None,
) -> ParameterSchemaManifest:
    """Apply explicit Plan-v2 parameter-set bindings to an eager model.

    Every parameter-set slot keeps one ``nn.Parameter`` identity.  This helper
    changes only trainable parameter bindings; receiver and selector state are
    outside the parameter manifest and are never aliased here.  Full dtype and
    storage validation remains the responsibility of
    :func:`build_eager_parameter_manifest`, because construction may precede a
    caller's ``model.to(dtype=...)`` conversion.
    """

    if not isinstance(model, SettleGraph):
        raise ParameterManifestError(
            "eager parameter tying requires a SettleGraph model"
        )
    expected = build_parameter_schema_manifest(model.plan)
    if manifest is not None and manifest.canonical_dict() != expected.canonical_dict():
        raise ParameterManifestError(
            "parameter tying manifest does not match the model Plan"
        )
    manifest = expected
    if manifest.schema_version == PARAMETER_SCHEMA_VERSION:
        return manifest

    grouped: Dict[str, List[ParameterManifestEntry]] = {}
    originals: Dict[str, nn.Parameter] = {}
    for entry in manifest.entries:
        parameter = _parameter_at_locator(model, entry.state_dict_locator)
        originals[entry.state_dict_locator] = parameter
        grouped.setdefault(entry.parameter_identity_key(), []).append(entry)

    replacements: List[Tuple[str, nn.Parameter]] = []
    for entries in grouped.values():
        if len(entries) < 2 or entries[0].parameter_group is None:
            continue
        canonical = originals[entries[0].state_dict_locator]
        for entry in entries[1:]:
            candidate = originals[entry.state_dict_locator]
            if (
                candidate.shape != canonical.shape
                or candidate.dtype != canonical.dtype
                or candidate.device != canonical.device
                or candidate.layout != canonical.layout
                or candidate.requires_grad != canonical.requires_grad
            ):
                raise ParameterManifestError(
                    f"parameter set {entry.parameter_group!r} cannot tie "
                    "incompatible eager Parameters"
                )
            replacements.append((entry.state_dict_locator, canonical))

    try:
        for locator, parameter in replacements:
            _set_parameter_at_locator(model, locator, parameter)
        return manifest
    except BaseException:
        for locator, parameter in originals.items():
            _set_parameter_at_locator(model, locator, parameter)
        raise


def build_eager_parameter_manifest(model: nn.Module) -> ParameterSchemaManifest:
    """Derive and validate the Plan-matched eager parameter schema."""

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
    named = dict(model.named_parameters(remove_duplicate=False))
    exported: Dict[str, torch.Tensor] = {}
    for entry in manifest.entries:
        key = _entry_tensor_key(entry, manifest.schema_version)
        if key not in exported:
            exported[key] = named[entry.state_dict_locator].reshape(entry.shape)
    return exported


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
    named = dict(model.named_parameters(remove_duplicate=False))
    entries: Dict[str, List[ParameterManifestEntry]] = {}
    for entry in manifest.entries:
        entries.setdefault(
            _entry_tensor_key(entry, manifest.schema_version), []
        ).append(entry)
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
        entry = entries[key][0]
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
        entry = entries[key][0]
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
    unique_targets: Dict[int, Tuple[str, nn.Parameter]] = {}
    for key, entry_group in entries.items():
        target = named[entry_group[0].state_dict_locator]
        unique_targets[id(target)] = (key, target)
    snapshot = {
        identity: parameter.detach().clone(memory_format=torch.preserve_format)
        for identity, (_, parameter) in unique_targets.items()
    }
    try:
        with torch.no_grad():
            for key, source in staged.items():
                entry = entries[key][0]
                target = named[entry.state_dict_locator]
                target.copy_(source)
    except BaseException as copy_error:
        rollback_errors = []
        with torch.no_grad():
            for identity, (key, target) in unique_targets.items():
                try:
                    target.copy_(snapshot[identity])
                except BaseException as rollback_error:
                    rollback_errors.append((key, rollback_error))
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
    "EAGER_PARAMETER_BINDING_VERSION_V2",
    "LogicalParameterKey",
    "PARAMETER_SCHEMA_CANONICALIZER_ID",
    "PARAMETER_SCHEMA_CANONICALIZER_ID_V2",
    "PARAMETER_SCHEMA_VERSION",
    "PARAMETER_SCHEMA_VERSION_V2",
    "ParameterManifestEntry",
    "ParameterManifestError",
    "ParameterSchemaManifest",
    "build_eager_parameter_manifest",
    "build_parameter_schema_manifest",
    "export_eager_parameter_tensors",
    "load_eager_parameter_tensors",
    "logical_parameter_tensor_key",
    "tie_eager_parameters",
]

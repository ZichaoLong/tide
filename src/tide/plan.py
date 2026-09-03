"""Canonical, framework-neutral SettleGraph plans.

The classes in this module describe logical graph semantics only.  They do
not contain parameters, runtime state, reached/active events, or compiled
schedules.  Identifiers are NFC-normalized Unicode scalar-value sequences.
Python's string order is therefore their scalar-value lexicographic order;
declaration order has no effect on canonical JSON or the plan digest.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
)


JsonValue = Any

_CONCRETE_DTYPES = {
    "float64",
    "float32",
    "float16",
    "bfloat16",
}
_REQUIRED_DTYPE_ROLES = {"hidden", "parameter", "state", "readout"}
_PROFILES = {"N", "SD", "BO"}
_SELECTOR_TIMINGS = {"content", "pre", "post"}
_HB_EDGE_LABELS = {"tree", "local", "shortcut", "mirror"}
_MAX_JSON_SAFE_INTEGER = (1 << 53) - 1

# Fixture bundles use this identifier to select these exact reference bytes;
# it deliberately does not claim conformance to an unrelated JSON standard.
PLAN_CANONICALIZER_ID = "tide-plan-json-v1"


def _contains_surrogate(value: str) -> bool:
    """Return whether a Python string contains a non-scalar code point."""

    return any(0xD800 <= ord(character) <= 0xDFFF for character in value)


def _is_stable_id_boundary_whitespace(character: str) -> bool:
    """Implement the stable-ID boundary-space set without locale dependence."""

    code_point = ord(character)
    return (
        0x0009 <= code_point <= 0x000D
        or 0x001C <= code_point <= 0x0020
        or code_point
        in {
            0x0085,
            0x00A0,
            0x1680,
            0x2028,
            0x2029,
            0x202F,
            0x205F,
            0x3000,
        }
        or 0x2000 <= code_point <= 0x200A
    )


def _stable_id_errors(kind: str, value: object) -> List[str]:
    if not isinstance(value, str) or not value:
        return [f"{kind} ID must be a nonempty string"]

    errors: List[str] = []
    if _is_stable_id_boundary_whitespace(value[0]) or _is_stable_id_boundary_whitespace(
        value[-1]
    ):
        errors.append(
            f"{kind} ID {value!r} must not have surrounding Unicode whitespace"
        )
    if "\x00" in value:
        errors.append(f"{kind} ID {value!r} must not contain NUL")
    if _contains_surrogate(value):
        errors.append(
            f"{kind} ID {value!r} must contain only Unicode scalar values"
        )
    elif unicodedata.normalize("NFC", value) != value:
        errors.append(f"{kind} ID {value!r} must use Unicode NFC normalization")
    return errors


def validate_stable_id(value: object, *, kind: str = "stable") -> str:
    """Return a valid stable ID or raise ``ValueError``.

    Runtime inputs such as sequence and reset IDs can use this strict contract
    instead of coercing arbitrary Python objects with ``str()``.
    """

    errors = _stable_id_errors(kind, value)
    if errors:
        raise ValueError("; ".join(errors))
    assert isinstance(value, str)  # Narrowed by _stable_id_errors.
    return value


def _require_unicode_scalar_sequence(value: str, *, path: str) -> None:
    if _contains_surrogate(value):
        raise ValueError(
            f"{path} must contain only Unicode scalar values; "
            "surrogate code points are not valid UTF-8 text"
        )


class PlanValidationError(ValueError):
    """A Plan violates one or more static SettleGraph invariants."""

    def __init__(
        self,
        errors: Sequence[str],
        *,
        failure_codes: Sequence[str] = (),
    ) -> None:
        self.errors = tuple(errors)
        self.failure_codes = tuple(sorted(set(failure_codes)))
        super().__init__("invalid Plan:\n- " + "\n- ".join(self.errors))


class _ValidationErrorSink(Protocol):
    """The minimal append/extend surface shared by validation collectors."""

    def append(self, message: str) -> None: ...

    def extend(self, messages: Iterable[str]) -> None: ...


class _CategorizedValidationErrors:
    """List-like view that records a stable category beside each message."""

    def __init__(
        self, messages: List[str], codes: Set[str], failure_code: str
    ) -> None:
        self._messages = messages
        self._codes = codes
        self._failure_code = failure_code

    def append(self, message: str) -> None:
        self._messages.append(message)
        self._codes.add(self._failure_code)

    def extend(self, messages: Iterable[str]) -> None:
        for message in messages:
            self.append(message)


class FrozenConfig(Mapping):
    """A recursively immutable JSON object used for operation configs."""

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, JsonValue]) -> None:
        if not isinstance(values, Mapping):
            raise TypeError("an operation config must be a JSON object")
        frozen: Dict[str, JsonValue] = {}
        for key, value in values.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            _require_unicode_scalar_sequence(
                key, path=f"JSON object key {key!r}"
            )
            frozen[key] = _freeze_json(value, path=key)
        self._values = frozen

    def __getitem__(self, key: str) -> JsonValue:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return f"FrozenConfig({self._values!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mapping):
            return False
        return dict(self.items()) == dict(other.items())


def _freeze_json(value: JsonValue, *, path: str) -> JsonValue:
    if isinstance(value, str):
        _require_unicode_scalar_sequence(value, path=path)
        return value
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        return FrozenConfig(value)
    if isinstance(value, list):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(
        f"{path} contains {type(value).__name__}; operation configs must be "
        "JSON-safe objects"
    )


def _thaw_json(value: JsonValue) -> JsonValue:
    if isinstance(value, Mapping):
        return {
            key: _thaw_json(value[key])
            for key in sorted(value)
        }
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _config(value: Mapping[str, JsonValue]) -> FrozenConfig:
    if isinstance(value, FrozenConfig):
        return value
    return FrozenConfig(value)


def _freeze_config_if_possible(value: object) -> object:
    """Freeze a valid JSON object while deferring schema errors to validate()."""

    if not isinstance(value, Mapping):
        return value
    try:
        return _config(value)
    except (TypeError, ValueError):
        # Public dataclass constructors deliberately remain total for malformed
        # declarations.  The schema gate owns their stable failure category.
        return value


def _is_sequence_declaration(value: object) -> bool:
    """Return whether value is an accepted in-memory JSON-array declaration."""

    return isinstance(value, (list, tuple))


def _validate_sequence_schema(
    value: object, name: str, errors: _ValidationErrorSink
) -> bool:
    if not isinstance(value, tuple):
        errors.append(f"{name} must be a JSON array")
        return False
    return True


def _validate_shape_schema(
    value: object, name: str, errors: _ValidationErrorSink
) -> None:
    if not _validate_sequence_schema(value, name, errors):
        return
    if any(type(dimension) is not int for dimension in value):
        errors.append(f"{name} must contain only integer dimensions")


def _validate_config_schema(
    value: object,
    name: str,
    errors: _ValidationErrorSink,
    *,
    operation: bool = True,
) -> bool:
    if not isinstance(value, FrozenConfig):
        if isinstance(value, Mapping):
            errors.append(f"{name} must be a JSON-safe object with string keys")
        else:
            errors.append(f"{name} must be a JSON object")
        return False
    if not operation:
        return True
    for field_name in ("type", "formula_id"):
        field_value = value.get(field_name)
        if field_name in value and not isinstance(field_value, str):
            errors.append(f"{name}.{field_name} must be a string")
    if "formula" in value and not isinstance(value.get("formula"), str):
        errors.append(f"{name}.formula must be a string")
    return True


def _mean_config() -> Mapping[str, JsonValue]:
    return {"type": "mean", "formula_id": "agg.mean.v1"}


def _no_update_config() -> Mapping[str, JsonValue]:
    return {"type": "none", "formula_id": "update.none.v1"}


def _content_config() -> Mapping[str, JsonValue]:
    return {
        "type": "content",
        "formula_id": "read.selector.content.v1",
    }


def _zero_config() -> Mapping[str, JsonValue]:
    return {"type": "zero", "formula_id": "read.ffn.zero.v1"}


def _identity_config() -> Mapping[str, JsonValue]:
    return {"type": "identity", "formula_id": "node.identity.v1"}


def _rmsnorm_config() -> Mapping[str, JsonValue]:
    return {
        "type": "rmsnorm",
        "formula_id": "norm.rms.v1",
        "eps": 1e-6,
    }


def _hard_config() -> Mapping[str, JsonValue]:
    return {"type": "hard", "formula_id": "emit.hard.v1"}


def _fixed_one_config() -> Mapping[str, JsonValue]:
    return {"type": "fixed", "formula_id": "k.fixed.v1", "value": 1}


def _score_config() -> Mapping[str, JsonValue]:
    return {
        "type": "constant",
        "formula_id": "score.constant.v1",
        "value": 0.0,
    }


def _no_context_config() -> Mapping[str, JsonValue]:
    return {"type": "none", "formula_id": "context.none.v1"}


def _no_history_config() -> Mapping[str, JsonValue]:
    return {"type": "none", "formula_id": "history.none.v1"}


def _id_sort_key(value: object) -> Tuple[int, str, str]:
    """Order malformed declarations deterministically until validation."""

    if isinstance(value, str):
        return (0, value, "")
    return (1, type(value).__name__, repr(value))


@dataclass(frozen=True)
class _DefaultFromContext:
    name: str


@dataclass(frozen=True)
class OperationConfigSchema:
    """Exact serialized config shape for one executable formula dispatch."""

    canonical_type: str
    required_keys: frozenset[str] = frozenset()
    defaults: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )
    fixed_values: Mapping[str, JsonValue] = field(
        default_factory=lambda: MappingProxyType({})
    )

    @property
    def allowed_keys(self) -> frozenset[str]:
        return frozenset(
            {"type", "formula_id"}
            | set(self.required_keys)
            | set(self.defaults)
        )


class ReferenceOperationConfigError(ValueError):
    """A config cannot be dispatched as its declared eager formula."""


def _schema(
    canonical_type: str,
    *,
    required: Sequence[str] = (),
    defaults: Optional[Mapping[str, object]] = None,
    fixed: Optional[Mapping[str, JsonValue]] = None,
) -> OperationConfigSchema:
    return OperationConfigSchema(
        canonical_type=canonical_type,
        required_keys=frozenset(required),
        defaults=MappingProxyType(dict(defaults or {})),
        fixed_values=MappingProxyType(dict(fixed or {})),
    )


_OUTPUT_SHAPE = _DefaultFromContext("output_shape")
_STATE_SHAPE = _DefaultFromContext("state_shape")
_STATE_DIM = _DefaultFromContext("state_dim")
_SELECTOR_OUT_DIM = _DefaultFromContext("selector_out_dim")
_NODE_HIDDEN_DIM = _DefaultFromContext("node_hidden_dim")
_SCORE_HIDDEN_DIM = _DefaultFromContext("score_hidden_dim")


def _reference_operation_schemas() -> Mapping[
    Tuple[str, str, str], OperationConfigSchema
]:
    schemas: Dict[Tuple[str, str, str], OperationConfigSchema] = {}

    def add(
        operation_field: str,
        operation_types: Sequence[str],
        formula_ids: Sequence[str],
        schema: OperationConfigSchema,
    ) -> None:
        for operation_type in operation_types:
            for formula_id in formula_ids:
                key = (operation_field, operation_type, formula_id)
                if key in schemas:  # pragma: no cover - module invariant
                    raise AssertionError(f"duplicate operation schema {key!r}")
                schemas[key] = schema

    norm_schema = _schema("rmsnorm", defaults={"eps": 1e-6})
    for norm_field in ("input_norm", "ffn_norm"):
        add(
            norm_field,
            ("rmsnorm",),
            ("norm.rms.v1", "TEST-RMSNORM-V1"),
            norm_schema,
        )

    add(
        "aggregate",
        ("mean",),
        ("agg.mean.v1",),
        _schema("mean", defaults={"output_shape": _OUTPUT_SHAPE}),
    )
    add(
        "aggregate",
        ("edge_softmax",),
        ("TEST-AGG-EDGE-SOFTMAX-V1",),
        _schema("edge_softmax", defaults={"output_shape": _OUTPUT_SHAPE}),
    )
    add(
        "aggregate",
        ("edge_linear_mean",),
        ("TEST-AGG-EDGE-AFFINE-MEAN-V1",),
        _schema(
            "edge_linear_mean",
            defaults={"bias": True, "output_shape": _OUTPUT_SHAPE},
            fixed={"bias": True},
        ),
    )

    add(
        "update",
        ("none",),
        ("update.none.v1",),
        _schema("none", defaults={"state_shape": _STATE_SHAPE}),
    )
    add(
        "update",
        ("ema",),
        ("state.ema.v1",),
        _schema(
            "ema",
            defaults={
                "state_dim": _STATE_DIM,
                "decay": 0.9,
                "learnable_decay": False,
                "state_shape": _STATE_SHAPE,
            },
            fixed={"learnable_decay": False},
        ),
    )
    add(
        "update",
        ("gdn",),
        ("state.gdn.v1",),
        _schema(
            "gdn",
            required=("key_dim", "value_dim"),
            defaults={"norm_eps": 1e-12, "state_shape": _STATE_SHAPE},
        ),
    )
    add(
        "update",
        ("attention_window",),
        ("state.attention-window.v1",),
        _schema(
            "attention_window",
            required=("key_dim", "value_dim", "window"),
            defaults={"norm_eps": 1e-12, "state_shape": _STATE_SHAPE},
        ),
    )

    for read_type, formula_id in (
        ("content", "read.selector.content.v1"),
        ("content_norm", "read.selector.content-rms.v1"),
        ("content_linear", "TEST-READ-PROJ-V1"),
        ("content_state_linear", "TEST-READ-PROJ-V1"),
        (
            "content_state_summary_linear",
            "TEST-READ-STATE-RMS-SUMMARY-PROJ-V1",
        ),
    ):
        add(
            "selector_read",
            (read_type,),
            (formula_id,),
            _schema(
                read_type,
                defaults={
                    "out_dim": _SELECTOR_OUT_DIM,
                    "output_shape": _OUTPUT_SHAPE,
                },
            ),
        )

    add(
        "ffn_read",
        ("zero",),
        ("read.ffn.zero.v1",),
        _schema("zero", defaults={"output_shape": _OUTPUT_SHAPE}),
    )
    for state_formula in (
        "read.ffn.zero.v1",
        "read.ffn.ema.v1",
        "read.ffn.gdn.v1",
        "read.ffn.attention-window.v1",
    ):
        add(
            "ffn_read",
            ("state_default",),
            (state_formula,),
            _schema(
                "state_default", defaults={"output_shape": _OUTPUT_SHAPE}
            ),
        )

    add(
        "node_compute",
        ("identity",),
        ("node.identity.v1",),
        _schema("identity", defaults={"output_shape": _OUTPUT_SHAPE}),
    )
    add(
        "node_compute",
        ("affine_residual",),
        ("TEST-NODE-AFFINE-V1",),
        _schema(
            "affine_residual",
            defaults={"bias": True, "output_shape": _OUTPUT_SHAPE},
            fixed={"bias": True},
        ),
    )
    add(
        "node_compute",
        ("double_residual_swiglu",),
        ("TEST-NODE-SWIGLU-V1",),
        _schema(
            "double_residual_swiglu",
            defaults={
                "hidden_dim": _NODE_HIDDEN_DIM,
                "bias": True,
                "output_shape": _OUTPUT_SHAPE,
            },
            fixed={"bias": True},
        ),
    )

    for emit_type, formula_id, defaults in (
        ("hard", "emit.hard.v1", {}),
        ("hst", "emit.hst.v1", {"zeta": 1.0}),
        ("softp", "emit.softp.v1", {}),
    ):
        add(
            "emit",
            (emit_type,),
            (formula_id,),
            _schema(
                emit_type,
                defaults={**defaults, "output_shape": _OUTPUT_SHAPE},
            ),
        )

    add(
        "score",
        ("constant",),
        ("score.constant.v1",),
        _schema("constant", defaults={"value": 0.0}),
    )
    add(
        "score",
        ("fixed",),
        ("score.fixed-by-node.v1", "TEST-SCORE-CONST-V1"),
        _schema("fixed", required=("values_by_node",)),
    )
    add(
        "score",
        ("linear",),
        ("TEST-SCORE-LINEAR-V1",),
        _schema(
            "linear",
            defaults={
                "bias": True,
                "shared_parameters": False,
                "context_dim": 0,
            },
            fixed={
                "bias": True,
                "shared_parameters": False,
                "context_dim": 0,
            },
        ),
    )
    add(
        "score",
        ("mlp",),
        ("TEST-SCORE-MLP-V1",),
        _schema(
            "mlp",
            defaults={
                "hidden_dim": _SCORE_HIDDEN_DIM,
                "bias": True,
                "shared_parameters": False,
                "context_dim": 0,
            },
            fixed={
                "bias": True,
                "shared_parameters": False,
                "context_dim": 0,
            },
        ),
    )
    add(
        "score",
        ("read_sum",),
        ("score.read-sum.v1",),
        _schema(
            "read_sum",
            defaults={"context_dim": 0},
            fixed={"context_dim": 0},
        ),
    )

    add(
        "selector_context",
        ("none",),
        ("context.none.v1",),
        _schema("none"),
    )
    add(
        "selector_history",
        ("none",),
        ("history.none.v1",),
        _schema("none"),
    )
    add(
        "k_requested",
        ("fixed",),
        ("k.fixed.v1",),
        _schema("fixed", required=("value",)),
    )
    add(
        "k_requested",
        ("input",),
        ("k.input.v1",),
        _schema(
            "input", required=("field", "minimum", "maximum")
        ),
    )

    add(
        "output_aggregate",
        ("mean",),
        ("agg.mean.v1",),
        _schema("mean", defaults={"output_shape": _OUTPUT_SHAPE}),
    )
    add(
        "output_aggregate",
        ("node_softmax",),
        ("TEST-AGG-TERMINAL-SOFTMAX-V1",),
        _schema(
            "node_softmax", defaults={"output_shape": _OUTPUT_SHAPE}
        ),
    )
    return MappingProxyType(schemas)


REFERENCE_OPERATION_CONFIG_SCHEMAS = _reference_operation_schemas()


_STATE_DEFAULT_READ_FORMULA_BY_UPDATE: Mapping[str, str] = MappingProxyType(
    {
        "none": "read.ffn.zero.v1",
        "ema": "read.ffn.ema.v1",
        "gdn": "read.ffn.gdn.v1",
        "attention_window": "read.ffn.attention-window.v1",
    }
)


def _normalized_operation_type(value: object) -> object:
    if not isinstance(value, str):
        return value
    return value.lower().replace("-", "_")


def _operation_schema(
    operation_field: str, config: Mapping[str, JsonValue]
) -> Optional[OperationConfigSchema]:
    operation_type = _normalized_operation_type(config.get("type"))
    formula_id = config.get("formula_id")
    if not isinstance(operation_type, str) or not isinstance(formula_id, str):
        return None
    return REFERENCE_OPERATION_CONFIG_SCHEMAS.get(
        (operation_field, operation_type, formula_id)
    )


def _normalize_operation_config(
    operation_field: str,
    config: Mapping[str, JsonValue],
    semantic_context: Mapping[str, JsonValue],
) -> FrozenConfig:
    values = _thaw_json(config)
    values["type"] = _normalized_operation_type(values.get("type"))
    schema = _operation_schema(operation_field, values)
    if schema is None:
        return _config(values)
    values["type"] = schema.canonical_type
    for key, default in schema.defaults.items():
        if key in values:
            if type(default) is float:
                values[key] = _normalize_formula_number(values[key])
            continue
        if isinstance(default, _DefaultFromContext):
            values[key] = _thaw_json(semantic_context[default.name])
        else:
            values[key] = _thaw_json(default)
    if operation_field == "score" and schema.canonical_type == "fixed":
        fixed_values = values.get("values_by_node")
        if isinstance(fixed_values, Mapping):
            values["values_by_node"] = {
                key: _normalize_formula_number(value)
                for key, value in fixed_values.items()
            }
    return _config(values)


def _normalize_formula_number(value: JsonValue) -> JsonValue:
    normalized = _finite_formula_number(value)
    if normalized is None:
        return value
    return 0.0 if normalized == 0.0 else normalized


def _finite_formula_number(value: object) -> Optional[float]:
    """Return the lossless reference numeric value, or ``None`` if invalid."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, int) and abs(value) > _MAX_JSON_SAFE_INTEGER:
        return None
    try:
        normalized = float(value)
    except (OverflowError, ValueError):
        return None
    return normalized if math.isfinite(normalized) else None


def _same_json_scalar(actual: object, expected: object) -> bool:
    return type(actual) is type(expected) and actual == expected


def _require_positive_config_integer(
    config: Mapping[str, JsonValue], key: str, context: str
) -> None:
    value = config.get(key)
    if type(value) is not int or value <= 0:
        raise ReferenceOperationConfigError(
            f"{context}.{key} must be a positive integer"
        )


def _require_positive_config_number(
    config: Mapping[str, JsonValue], key: str, context: str
) -> None:
    value = config.get(key)
    normalized = _finite_formula_number(value)
    if normalized is None or normalized <= 0.0:
        raise ReferenceOperationConfigError(
            f"{context}.{key} must be a positive finite number with integer "
            "literals in the JSON-safe range"
        )


def validate_reference_operation_config(
    operation_field: str,
    config: Mapping[str, JsonValue],
    *,
    state_update_type: Optional[str] = None,
    semantic_context: Optional[Mapping[str, JsonValue]] = None,
) -> str:
    """Validate one canonical field/type/formula config for eager execution."""

    operation_type = _normalized_operation_type(config.get("type"))
    if not isinstance(operation_type, str):
        raise ReferenceOperationConfigError(
            f"{operation_field} operation type must be a string"
        )
    formula_id = config.get("formula_id")
    if not isinstance(formula_id, str) or not formula_id:
        raise ReferenceOperationConfigError(
            f"{operation_field} type {operation_type!r} must declare a "
            "nonempty string formula_id"
        )
    schema = REFERENCE_OPERATION_CONFIG_SCHEMAS.get(
        (operation_field, operation_type, formula_id)
    )
    if schema is None:
        field_schemas = {
            key: value
            for key, value in REFERENCE_OPERATION_CONFIG_SCHEMAS.items()
            if key[0] == operation_field
        }
        if not field_schemas:
            raise ValueError(
                f"unknown reference operation field: {operation_field!r}"
            )
        supported_types = {key[1] for key in field_schemas}
        if operation_type not in supported_types:
            raise ReferenceOperationConfigError(
                f"{operation_field} type {operation_type!r} is not supported "
                "by the eager reference executor"
            )
        supported_ids = sorted(
            key[2] for key in field_schemas if key[1] == operation_type
        )
        expected = ", ".join(repr(item) for item in supported_ids)
        raise ReferenceOperationConfigError(
            f"{operation_field} type {operation_type!r} does not support "
            f"formula_id {formula_id!r}; expected one of: {expected}"
        )

    unknown = sorted(set(config) - set(schema.allowed_keys))
    if unknown:
        raise ReferenceOperationConfigError(
            f"{operation_field} type {operation_type!r} formula_id "
            f"{formula_id!r} has unknown config keys: {unknown!r}"
        )
    missing = sorted(
        (set(schema.required_keys) | set(schema.defaults)) - set(config)
    )
    if missing:
        raise ReferenceOperationConfigError(
            f"{operation_field} type {operation_type!r} formula_id "
            f"{formula_id!r} is missing canonical config keys: {missing!r}"
        )
    for key, expected in schema.fixed_values.items():
        actual = config.get(key)
        if not _same_json_scalar(actual, expected):
            raise ReferenceOperationConfigError(
                f"{operation_field} formula_id {formula_id!r} requires "
                f"{key}={expected!r}, got {actual!r}"
            )

    context = f"{operation_field} formula_id {formula_id!r}"
    if operation_field in {"input_norm", "ffn_norm"}:
        _require_positive_config_number(config, "eps", context)
    elif operation_field == "update":
        if operation_type == "ema":
            _require_positive_config_integer(config, "state_dim", context)
            decay = config.get("decay")
            normalized_decay = _finite_formula_number(decay)
            if normalized_decay is None or not 0.0 <= normalized_decay < 1.0:
                raise ReferenceOperationConfigError(
                    f"{context}.decay must satisfy 0 <= decay < 1"
                )
        elif operation_type in {"gdn", "attention_window"}:
            _require_positive_config_integer(config, "key_dim", context)
            _require_positive_config_integer(config, "value_dim", context)
            if operation_type == "attention_window":
                _require_positive_config_integer(config, "window", context)
            _require_positive_config_number(config, "norm_eps", context)
    elif operation_field == "selector_read":
        _require_positive_config_integer(config, "out_dim", context)
    elif operation_field == "node_compute" and operation_type == "double_residual_swiglu":
        _require_positive_config_integer(config, "hidden_dim", context)
    elif operation_field == "emit" and operation_type == "hst":
        zeta = config.get("zeta")
        if _finite_formula_number(zeta) is None:
            raise ReferenceOperationConfigError(
                f"{context}.zeta must be a finite number with integer literals "
                "in the JSON-safe range"
            )
    elif operation_field == "score":
        if operation_type == "constant":
            value = config.get("value")
            if _finite_formula_number(value) is None:
                raise ReferenceOperationConfigError(
                    f"{context}.value must be finite numeric data with integer "
                    "literals in the JSON-safe range"
                )
        elif operation_type == "mlp":
            _require_positive_config_integer(config, "hidden_dim", context)

    normalized_update = (
        _normalized_operation_type(state_update_type)
        if state_update_type is not None
        else None
    )
    if operation_field == "ffn_read" and operation_type == "state_default":
        expected_formula = _STATE_DEFAULT_READ_FORMULA_BY_UPDATE.get(
            normalized_update
        )
        if expected_formula is None:
            raise ReferenceOperationConfigError(
                "ffn_read type 'state_default' has no formula for Update type "
                f"{normalized_update!r}"
            )
        if formula_id != expected_formula:
            raise ReferenceOperationConfigError(
                "ffn_read type 'state_default' with Update type "
                f"{normalized_update!r} requires formula_id "
                f"{expected_formula!r}, got {formula_id!r}"
            )

    semantic_context = semantic_context or {}
    declared_state_shape = tuple(semantic_context.get("state_shape", ()))
    if operation_field == "update":
        expected_state_shape: Optional[Tuple[int, ...]] = None
        if operation_type == "none":
            expected_state_shape = ()
        elif operation_type == "ema":
            expected_state_shape = (int(config["state_dim"]),)
        elif operation_type == "gdn":
            expected_state_shape = (
                int(config["key_dim"]),
                int(config["value_dim"]),
            )
        elif operation_type == "attention_window":
            expected_state_shape = (
                int(config["window"]),
                int(config["key_dim"]),
                int(config["value_dim"]),
            )
        if (
            "state_shape" in semantic_context
            and expected_state_shape != declared_state_shape
        ):
            raise ReferenceOperationConfigError(
                f"{context} dimensions require state_shape "
                f"{expected_state_shape!r}, got {declared_state_shape!r}"
            )
    if operation_field == "selector_read" and "selector_read_shape" in semantic_context:
        expected_read_shape = (int(config["out_dim"]),)
        declared_read_shape = tuple(semantic_context["selector_read_shape"])
        if declared_read_shape != expected_read_shape:
            raise ReferenceOperationConfigError(
                f"{context}.out_dim requires selector_read_shape "
                f"{expected_read_shape!r}, got {declared_read_shape!r}"
            )
        if operation_type == "content" and semantic_context.get("d_model") != config["out_dim"]:
            raise ReferenceOperationConfigError(
                f"{context}.out_dim must equal d_model"
            )
        if operation_type == "content_norm" and config["out_dim"] != 1:
            raise ReferenceOperationConfigError(
                f"{context}.out_dim must be 1"
            )
        if operation_type == "content_state_linear" and normalized_update == "attention_window":
            raise ReferenceOperationConfigError(
                "content_state_linear requires fixed-shape Tensor state; use "
                "content_state_summary_linear for window Attention"
            )
        if operation_type == "content_state_linear" and not declared_state_shape:
            raise ReferenceOperationConfigError(
                "content_state_linear requires a non-empty receiver state"
            )
    return schema.canonical_type


@dataclass(frozen=True)
class NodeSpec:
    """One receiver and its complete local logical-operation contract."""

    node_id: str
    region_id: str
    hidden_shape: Tuple[int, ...]
    selector_read_shape: Tuple[int, ...] = (1,)
    state_shape: Tuple[int, ...] = ()
    state_owner: Optional[str] = None
    forced_active: bool = False
    parameter_group: Optional[str] = None
    input_norm: Mapping[str, JsonValue] = field(
        default_factory=_rmsnorm_config
    )
    ffn_norm: Mapping[str, JsonValue] = field(
        default_factory=_rmsnorm_config
    )
    aggregate: Mapping[str, JsonValue] = field(default_factory=_mean_config)
    update: Mapping[str, JsonValue] = field(default_factory=_no_update_config)
    selector_read: Mapping[str, JsonValue] = field(default_factory=_content_config)
    ffn_read: Mapping[str, JsonValue] = field(default_factory=_zero_config)
    node_compute: Mapping[str, JsonValue] = field(default_factory=_identity_config)
    emit: Mapping[str, JsonValue] = field(default_factory=_hard_config)

    def __post_init__(self) -> None:
        for name in ("hidden_shape", "selector_read_shape", "state_shape"):
            value = getattr(self, name)
            if _is_sequence_declaration(value):
                object.__setattr__(self, name, tuple(value))
        for name in (
            "input_norm",
            "ffn_norm",
            "aggregate",
            "update",
            "selector_read",
            "ffn_read",
            "node_compute",
            "emit",
        ):
            object.__setattr__(
                self, name, _freeze_config_if_possible(getattr(self, name))
            )


@dataclass(frozen=True)
class EdgeSpec:
    """One fixed receiver-to-receiver edge."""

    edge_id: str
    source: str
    target: str
    label: str = "data"


@dataclass(frozen=True)
class RegionSpec:
    """A fixed candidate set that shares one selector invocation."""

    region_id: str
    node_ids: Tuple[str, ...]
    profile: str = "N"
    selector_timing: str = "content"
    k_max: int = 1
    k_requested: Mapping[str, JsonValue] = field(
        default_factory=_fixed_one_config
    )
    score: Mapping[str, JsonValue] = field(default_factory=_score_config)
    selector_context: Mapping[str, JsonValue] = field(
        default_factory=_no_context_config
    )
    selector_history: Mapping[str, JsonValue] = field(
        default_factory=_no_history_config
    )
    control_dependencies: Tuple[str, ...] = ()
    line: Optional[int] = None
    phase: Optional[str] = None

    def __post_init__(self) -> None:
        if _is_sequence_declaration(self.node_ids):
            object.__setattr__(
                self,
                "node_ids",
                tuple(sorted(self.node_ids, key=_id_sort_key)),
            )
        if _is_sequence_declaration(self.control_dependencies):
            object.__setattr__(
                self,
                "control_dependencies",
                tuple(sorted(self.control_dependencies, key=_id_sort_key)),
            )
        if isinstance(self.profile, str):
            object.__setattr__(self, "profile", self.profile.upper())
        if isinstance(self.selector_timing, str):
            object.__setattr__(
                self, "selector_timing", self.selector_timing.lower()
            )
        for name in (
            "k_requested",
            "score",
            "selector_context",
            "selector_history",
        ):
            object.__setattr__(
                self, name, _freeze_config_if_possible(getattr(self, name))
            )


_NODE_OPERATION_FIELDS = (
    "input_norm",
    "ffn_norm",
    "aggregate",
    "update",
    "selector_read",
    "ffn_read",
    "node_compute",
    "emit",
)

_REGION_OPERATION_FIELDS = (
    "k_requested",
    "score",
    "selector_context",
    "selector_history",
)


def _node_can_be_normalized(node: NodeSpec) -> bool:
    return (
        isinstance(node.hidden_shape, tuple)
        and all(type(dimension) is int for dimension in node.hidden_shape)
        and isinstance(node.selector_read_shape, tuple)
        and all(
            type(dimension) is int for dimension in node.selector_read_shape
        )
        and isinstance(node.state_shape, tuple)
        and all(type(dimension) is int for dimension in node.state_shape)
        and all(
            isinstance(getattr(node, name), FrozenConfig)
            for name in _NODE_OPERATION_FIELDS
        )
    )


def _region_can_be_normalized(region: RegionSpec) -> bool:
    return (
        isinstance(region.node_ids, tuple)
        and all(isinstance(node_id, str) for node_id in region.node_ids)
        and isinstance(region.control_dependencies, tuple)
        and all(
            isinstance(dependency, str)
            for dependency in region.control_dependencies
        )
        and isinstance(region.profile, str)
        and isinstance(region.selector_timing, str)
        and all(
            isinstance(getattr(region, name), FrozenConfig)
            for name in _REGION_OPERATION_FIELDS
        )
    )


def _first_positive_dimension(shape: object, fallback: int = 1) -> int:
    if (
        isinstance(shape, (list, tuple))
        and shape
        and type(shape[0]) is int
        and shape[0] > 0
    ):
        return shape[0]
    return fallback


def _normalize_node_operations(node: NodeSpec, d_model: object) -> NodeSpec:
    hidden_width = (
        d_model
        if type(d_model) is int and d_model > 0
        else _first_positive_dimension(node.hidden_shape)
    )
    state_dim = _first_positive_dimension(node.state_shape, 0)
    selector_out_dim = _first_positive_dimension(node.selector_read_shape)
    common = {
        "output_shape": list(node.hidden_shape),
        "state_shape": list(node.state_shape),
        "state_dim": state_dim,
        "selector_out_dim": selector_out_dim,
        "node_hidden_dim": 4 * hidden_width,
    }
    return replace(
        node,
        input_norm=_normalize_operation_config(
            "input_norm", node.input_norm, common
        ),
        ffn_norm=_normalize_operation_config("ffn_norm", node.ffn_norm, common),
        aggregate=_normalize_operation_config(
            "aggregate", node.aggregate, common
        ),
        update=_normalize_operation_config("update", node.update, common),
        selector_read=_normalize_operation_config(
            "selector_read",
            node.selector_read,
            {**common, "output_shape": list(node.selector_read_shape)},
        ),
        ffn_read=_normalize_operation_config(
            "ffn_read", node.ffn_read, common
        ),
        node_compute=_normalize_operation_config(
            "node_compute", node.node_compute, common
        ),
        emit=_normalize_operation_config("emit", node.emit, common),
    )


def _normalize_region_operations(
    region: RegionSpec, node_lookup: Mapping[str, NodeSpec]
) -> RegionSpec:
    member = next(
        (node_lookup[node_id] for node_id in region.node_ids if node_id in node_lookup),
        None,
    )
    read_dim = (
        _first_positive_dimension(member.selector_read_shape)
        if member is not None
        else 1
    )
    context = {"score_hidden_dim": max(4, read_dim)}
    return replace(
        region,
        k_requested=_normalize_operation_config(
            "k_requested", region.k_requested, context
        ),
        score=_normalize_operation_config("score", region.score, context),
        selector_context=_normalize_operation_config(
            "selector_context", region.selector_context, context
        ),
        selector_history=_normalize_operation_config(
            "selector_history", region.selector_history, context
        ),
    )


@dataclass(frozen=True)
class Plan:
    """A normalized logical SettleGraph Plan.

    ``plan_id`` and ``builder`` are provenance.  They are emitted by
    :meth:`to_record_dict`, but deliberately excluded from the canonical
    semantic digest: renaming a Plan or rebuilding the same expanded graph
    does not change what the graph computes.
    """

    plan_id: str
    d_model: int
    dtype_roles: Mapping[str, str]
    nodes: Tuple[NodeSpec, ...]
    edges: Tuple[EdgeSpec, ...]
    regions: Tuple[RegionSpec, ...]
    entry_node_ids: Tuple[str, ...] = ()
    terminal_node_ids: Tuple[str, ...] = ()
    output_aggregate: Mapping[str, JsonValue] = field(
        default_factory=_mean_config
    )
    topology_kind: str = "general"
    schema_version: str = "1"
    builder: Mapping[str, JsonValue] = field(default_factory=dict)
    _node_index: Mapping[str, NodeSpec] = field(
        init=False, repr=False, compare=False
    )
    _edge_index: Mapping[str, EdgeSpec] = field(
        init=False, repr=False, compare=False
    )
    _region_index: Mapping[str, RegionSpec] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        nodes_value: object = self.nodes
        if _is_sequence_declaration(nodes_value):
            nodes_value = tuple(nodes_value)
        if isinstance(nodes_value, tuple) and all(
            isinstance(node, NodeSpec) for node in nodes_value
        ):
            nodes = tuple(
                _normalize_node_operations(node, self.d_model)
                if _node_can_be_normalized(node)
                else node
                for node in sorted(
                    nodes_value, key=lambda item: _id_sort_key(item.node_id)
                )
            )
        else:
            nodes = nodes_value

        edges_value: object = self.edges
        if _is_sequence_declaration(edges_value):
            edges_value = tuple(edges_value)
        if isinstance(edges_value, tuple) and all(
            isinstance(edge, EdgeSpec) for edge in edges_value
        ):
            edges = tuple(
                sorted(edges_value, key=lambda item: _id_sort_key(item.edge_id))
            )
        else:
            edges = edges_value

        node_lookup = (
            {
                node.node_id: node
                for node in nodes
                if isinstance(node, NodeSpec) and isinstance(node.node_id, str)
            }
            if isinstance(nodes, tuple)
            else {}
        )

        regions_value: object = self.regions
        if _is_sequence_declaration(regions_value):
            regions_value = tuple(regions_value)
        if isinstance(regions_value, tuple) and all(
            isinstance(region, RegionSpec) for region in regions_value
        ):
            regions = tuple(
                _normalize_region_operations(region, node_lookup)
                if _region_can_be_normalized(region)
                else region
                for region in sorted(
                    regions_value,
                    key=lambda item: _id_sort_key(item.region_id),
                )
            )
        else:
            regions = regions_value
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "regions", regions)

        for name in ("entry_node_ids", "terminal_node_ids"):
            value = getattr(self, name)
            if _is_sequence_declaration(value):
                object.__setattr__(
                    self, name, tuple(sorted(value, key=_id_sort_key))
                )

        output_aggregate = _freeze_config_if_possible(self.output_aggregate)
        if (
            isinstance(output_aggregate, FrozenConfig)
            and type(self.d_model) is int
            and self.d_model > 0
        ):
            output_aggregate = _normalize_operation_config(
                "output_aggregate",
                output_aggregate,
                {"output_shape": [self.d_model]},
            )
        object.__setattr__(self, "output_aggregate", output_aggregate)
        object.__setattr__(
            self, "builder", _freeze_config_if_possible(self.builder)
        )
        object.__setattr__(
            self, "dtype_roles", _freeze_config_if_possible(self.dtype_roles)
        )
        if isinstance(self.topology_kind, str):
            object.__setattr__(
                self, "topology_kind", self.topology_kind.lower()
            )
        object.__setattr__(
            self,
            "_node_index",
            MappingProxyType(
                {
                    node.node_id: node
                    for node in nodes
                    if isinstance(node, NodeSpec)
                    and isinstance(node.node_id, str)
                }
                if isinstance(nodes, tuple)
                else {}
            ),
        )
        object.__setattr__(
            self,
            "_edge_index",
            MappingProxyType(
                {
                    edge.edge_id: edge
                    for edge in edges
                    if isinstance(edge, EdgeSpec)
                    and isinstance(edge.edge_id, str)
                }
                if isinstance(edges, tuple)
                else {}
            ),
        )
        object.__setattr__(
            self,
            "_region_index",
            MappingProxyType(
                {
                    region.region_id: region
                    for region in regions
                    if isinstance(region, RegionSpec)
                    and isinstance(region.region_id, str)
                }
                if isinstance(regions, tuple)
                else {}
            ),
        )

        # Boundary identities are semantic consequences of fixed edges.  An
        # omitted declaration is filled during normalization; an explicit but
        # incorrect declaration remains intact and is rejected by validate().
        safe_graph = (
            isinstance(nodes, tuple)
            and all(
                isinstance(node, NodeSpec) and isinstance(node.node_id, str)
                for node in nodes
            )
            and isinstance(edges, tuple)
            and all(
                isinstance(edge, EdgeSpec)
                and isinstance(edge.source, str)
                and isinstance(edge.target, str)
                for edge in edges
            )
        )
        node_ids = {node.node_id for node in nodes} if safe_graph else set()
        sources_with_incoming = (
            {edge.target for edge in edges} if safe_graph else set()
        )
        sources_with_outgoing = (
            {edge.source for edge in edges} if safe_graph else set()
        )
        if (
            node_ids
            and isinstance(self.entry_node_ids, tuple)
            and not self.entry_node_ids
        ):
            object.__setattr__(
                self,
                "entry_node_ids",
                tuple(sorted(node_ids - sources_with_incoming)),
            )
        if (
            node_ids
            and isinstance(self.terminal_node_ids, tuple)
            and not self.terminal_node_ids
        ):
            object.__setattr__(
                self,
                "terminal_node_ids",
                tuple(sorted(node_ids - sources_with_outgoing)),
            )

    def node_by_id(self, node_id: str) -> NodeSpec:
        """Return a node or raise ``KeyError`` for an unknown stable ID."""

        return self._node_index[node_id]

    def region_by_id(self, region_id: str) -> RegionSpec:
        """Return a region or raise ``KeyError`` for an unknown stable ID."""

        return self._region_index[region_id]

    def edge_by_id(self, edge_id: str) -> EdgeSpec:
        """Return an edge or raise ``KeyError`` for an unknown stable ID."""

        return self._edge_index[edge_id]

    @property
    def incoming_edges(self) -> Mapping[str, Tuple[EdgeSpec, ...]]:
        grouped: Dict[str, List[EdgeSpec]] = {
            node.node_id: [] for node in self.nodes
        }
        for edge in self.edges:
            if edge.target in grouped:
                grouped[edge.target].append(edge)
        return {
            node_id: tuple(sorted(items, key=lambda item: item.edge_id))
            for node_id, items in grouped.items()
        }

    @property
    def outgoing_edges(self) -> Mapping[str, Tuple[EdgeSpec, ...]]:
        grouped: Dict[str, List[EdgeSpec]] = {
            node.node_id: [] for node in self.nodes
        }
        for edge in self.edges:
            if edge.source in grouped:
                grouped[edge.source].append(edge)
        return {
            node_id: tuple(sorted(items, key=lambda item: item.edge_id))
            for node_id, items in grouped.items()
        }

    @property
    def topological_node_ids(self) -> Tuple[str, ...]:
        """Return the canonical receiver topological order."""

        self.validate()
        pairs = [(edge.source, edge.target) for edge in self.edges]
        order, _ = _topological_order(
            [node.node_id for node in self.nodes], pairs
        )
        return order

    @property
    def topological_region_ids(self) -> Tuple[str, ...]:
        """Return the canonical dependency order, including HB barriers."""

        self.validate()
        order, _ = _topological_order(
            [region.region_id for region in self.regions],
            self._region_dependency_pairs(),
        )
        return order

    @property
    def topological_regions(self) -> Tuple[RegionSpec, ...]:
        """Return region specs in the canonical legal execution order."""

        return tuple(
            self.region_by_id(region_id)
            for region_id in self.topological_region_ids
        )

    def validate(self) -> "Plan":
        """Validate all static invariants and return ``self``."""

        schema_messages: List[str] = []
        schema_codes: Set[str] = set()
        schema_errors = _CategorizedValidationErrors(
            schema_messages, schema_codes, "plan.schema"
        )
        self._validate_schema_contracts(schema_errors)
        if schema_messages:
            raise PlanValidationError(
                schema_messages, failure_codes=("plan.schema",)
            )

        errors: List[str] = []
        failure_codes: Set[str] = set()
        topology_errors = _CategorizedValidationErrors(
            errors, failure_codes, "plan.topology"
        )
        formula_errors = _CategorizedValidationErrors(
            errors, failure_codes, "plan.formula"
        )
        self._validate_formula_contracts(formula_errors)
        node_ids = [node.node_id for node in self.nodes]
        edge_ids = [edge.edge_id for edge in self.edges]
        region_ids = [region.region_id for region in self.regions]

        _validate_unique_ids("node", node_ids, topology_errors)
        _validate_unique_ids("edge", edge_ids, topology_errors)
        _validate_unique_ids("region", region_ids, topology_errors)
        if not node_ids:
            topology_errors.append("a Plan must contain at least one node")
        if not region_ids:
            topology_errors.append("a Plan must contain at least one region")

        node_set = set(node_ids)
        region_set = set(region_ids)
        self._validate_nodes(
            node_set,
            region_set,
            topology_errors,
            formula_errors,
        )
        self._validate_regions(
            node_set,
            region_set,
            topology_errors,
            formula_errors,
        )
        self._validate_edges(node_set, topology_errors)
        self._validate_boundaries_and_paths(node_set, topology_errors)
        self._validate_region_graph(topology_errors)
        self._validate_hb(topology_errors)

        if errors:
            raise PlanValidationError(
                errors, failure_codes=tuple(sorted(failure_codes))
            )
        return self

    def canonical_dict(self) -> Dict[str, JsonValue]:
        """Return the normalized semantic record used for SHA-256."""

        self.validate()
        return {
            "schema_version": self.schema_version,
            "topology_kind": self.topology_kind,
            "d_model": self.d_model,
            "dtype_roles": _thaw_json(self.dtype_roles),
            "entry_node_ids": list(self.entry_node_ids),
            "terminal_node_ids": list(self.terminal_node_ids),
            "output_aggregate": _thaw_json(self.output_aggregate),
            "nodes": [self._node_dict(node) for node in self.nodes],
            "edges": [self._edge_dict(edge) for edge in self.edges],
            "regions": [self._region_dict(region) for region in self.regions],
        }

    def canonical_json(self) -> str:
        """Serialize the logical Plan with a stable JSON encoding."""

        return _canonical_json_text(self.canonical_dict())

    def canonical_bytes(self) -> bytes:
        """Return the ``tide-plan-json-v1`` UTF-8 byte sequence."""

        return self.canonical_json().encode("utf-8")

    def canonical_hash(self) -> str:
        """Return the lowercase SHA-256 hex digest of canonical JSON."""

        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def logical_hash(self) -> str:
        """Explicit alias for :meth:`canonical_hash` at the logical layer."""

        return self.canonical_hash()

    def to_record_dict(self) -> Dict[str, JsonValue]:
        """Return semantic content plus non-semantic human provenance."""

        record = self.canonical_dict()
        record["plan_id"] = self.plan_id
        record["builder"] = _thaw_json(self.builder)
        record["canonical_hash"] = self.canonical_hash()
        return record

    @staticmethod
    def _node_dict(node: NodeSpec) -> Dict[str, JsonValue]:
        return {
            "node_id": node.node_id,
            "region_id": node.region_id,
            "hidden_shape": list(node.hidden_shape),
            "selector_read_shape": list(node.selector_read_shape),
            "state_shape": list(node.state_shape),
            "state_owner": node.state_owner,
            "forced_active": node.forced_active,
            "parameter_group": node.parameter_group,
            "input_norm": _thaw_json(node.input_norm),
            "ffn_norm": _thaw_json(node.ffn_norm),
            "aggregate": _thaw_json(node.aggregate),
            "update": _thaw_json(node.update),
            "selector_read": _thaw_json(node.selector_read),
            "ffn_read": _thaw_json(node.ffn_read),
            "node_compute": _thaw_json(node.node_compute),
            "emit": _thaw_json(node.emit),
        }

    @staticmethod
    def _edge_dict(edge: EdgeSpec) -> Dict[str, JsonValue]:
        return {
            "edge_id": edge.edge_id,
            "source": edge.source,
            "target": edge.target,
            "label": edge.label,
        }

    @staticmethod
    def _region_dict(region: RegionSpec) -> Dict[str, JsonValue]:
        return {
            "region_id": region.region_id,
            "node_ids": list(region.node_ids),
            "profile": region.profile,
            "selector_timing": region.selector_timing,
            "k_max": region.k_max,
            "k_requested": _thaw_json(region.k_requested),
            "score": _thaw_json(region.score),
            "selector_context": _thaw_json(region.selector_context),
            "selector_history": _thaw_json(region.selector_history),
            "control_dependencies": list(region.control_dependencies),
            "line": region.line,
            "phase": region.phase,
        }

    def _validate_schema_contracts(
        self,
        schema_errors: _CategorizedValidationErrors,
    ) -> None:
        _validate_identifier("plan", self.plan_id, schema_errors)
        _validate_identifier("schema version", self.schema_version, schema_errors)
        if self.schema_version != "1":
            schema_errors.append("schema_version must be exactly '1'")
        if type(self.d_model) is not int or self.d_model <= 0:
            schema_errors.append("d_model must be a positive integer")
        if not isinstance(self.topology_kind, str):
            schema_errors.append("topology_kind must be a string")
        elif self.topology_kind not in {"general", "hb"}:
            schema_errors.append("topology_kind must be 'general' or 'hb'")

        if _validate_config_schema(
            self.dtype_roles,
            "dtype_roles",
            schema_errors,
            operation=False,
        ):
            missing = sorted(_REQUIRED_DTYPE_ROLES - set(self.dtype_roles))
            if missing:
                schema_errors.append(
                    "dtype_roles is missing required roles: " + ", ".join(missing)
                )
            for role, dtype in self.dtype_roles.items():
                _validate_identifier("dtype role", role, schema_errors)
                if not isinstance(dtype, str):
                    schema_errors.append(
                        f"logical dtype role {role!r} must be a string"
                    )
                elif dtype != "runtime":
                    schema_errors.append(
                        f"logical dtype role {role!r} must remain symbolic as "
                        f"'runtime', got concrete declaration {dtype!r}"
                    )
        _validate_config_schema(
            self.builder, "builder", schema_errors, operation=False
        )
        _validate_config_schema(
            self.output_aggregate, "output_aggregate", schema_errors
        )

        for field_name, values, item_kind in (
            ("entry_node_ids", self.entry_node_ids, "entry node"),
            ("terminal_node_ids", self.terminal_node_ids, "terminal node"),
        ):
            if _validate_sequence_schema(values, field_name, schema_errors):
                for node_id in values:
                    _validate_identifier(item_kind, node_id, schema_errors)

        if _validate_sequence_schema(self.nodes, "nodes", schema_errors):
            for index, node in enumerate(self.nodes):
                if not isinstance(node, NodeSpec):
                    schema_errors.append(
                        f"nodes[{index}] must be a NodeSpec declaration"
                    )
                    continue
                _validate_identifier("node", node.node_id, schema_errors)
                _validate_identifier(
                    f"node {node.node_id!r} region",
                    node.region_id,
                    schema_errors,
                )
                for shape_name in (
                    "hidden_shape",
                    "selector_read_shape",
                    "state_shape",
                ):
                    _validate_shape_schema(
                        getattr(node, shape_name),
                        f"node {node.node_id!r} {shape_name}",
                        schema_errors,
                    )
                if node.state_owner is not None:
                    _validate_identifier(
                        f"node {node.node_id!r} state owner",
                        node.state_owner,
                        schema_errors,
                    )
                if node.parameter_group is not None:
                    schema_errors.append(
                        f"node {node.node_id!r} parameter_group must be null "
                        "in Plan schema version '1'"
                    )
                if type(node.forced_active) is not bool:
                    schema_errors.append(
                        f"node {node.node_id!r} forced_active must be a boolean"
                    )
                for name in _NODE_OPERATION_FIELDS:
                    _validate_config_schema(
                        getattr(node, name),
                        f"node {node.node_id!r} {name}",
                        schema_errors,
                    )

        if _validate_sequence_schema(self.regions, "regions", schema_errors):
            for index, region in enumerate(self.regions):
                if not isinstance(region, RegionSpec):
                    schema_errors.append(
                        f"regions[{index}] must be a RegionSpec declaration"
                    )
                    continue
                _validate_identifier("region", region.region_id, schema_errors)
                if not isinstance(region.profile, str):
                    schema_errors.append(
                        f"region {region.region_id!r} profile must be a string"
                    )
                elif _contains_surrogate(region.profile):
                    schema_errors.append(
                        f"region {region.region_id!r} profile must contain only "
                        "Unicode scalar values"
                    )
                if not isinstance(region.selector_timing, str):
                    schema_errors.append(
                        f"region {region.region_id!r} selector_timing must be a string"
                    )
                elif _contains_surrogate(region.selector_timing):
                    schema_errors.append(
                        f"region {region.region_id!r} selector_timing must contain "
                        "only Unicode scalar values"
                    )
                if type(region.k_max) is not int:
                    schema_errors.append(
                        f"region {region.region_id!r} k_max must be an integer"
                    )
                if region.line is not None and type(region.line) is not int:
                    schema_errors.append(
                        f"region {region.region_id!r} line must be null or an integer"
                    )
                if region.phase is not None and not isinstance(region.phase, str):
                    schema_errors.append(
                        f"region {region.region_id!r} phase must be null or a string"
                    )
                elif isinstance(region.phase, str) and _contains_surrogate(
                    region.phase
                ):
                    schema_errors.append(
                        f"region {region.region_id!r} phase must contain only "
                        "Unicode scalar values"
                    )
                for field_name, values, item_kind in (
                    ("node_ids", region.node_ids, "member node"),
                    (
                        "control_dependencies",
                        region.control_dependencies,
                        "control dependency",
                    ),
                ):
                    context = f"region {region.region_id!r} {field_name}"
                    if _validate_sequence_schema(values, context, schema_errors):
                        for value in values:
                            _validate_identifier(
                                f"region {region.region_id!r} {item_kind}",
                                value,
                                schema_errors,
                            )
                for name in _REGION_OPERATION_FIELDS:
                    _validate_config_schema(
                        getattr(region, name),
                        f"region {region.region_id!r} {name}",
                        schema_errors,
                    )

        if _validate_sequence_schema(self.edges, "edges", schema_errors):
            for index, edge in enumerate(self.edges):
                if not isinstance(edge, EdgeSpec):
                    schema_errors.append(
                        f"edges[{index}] must be an EdgeSpec declaration"
                    )
                    continue
                _validate_identifier("edge", edge.edge_id, schema_errors)
                _validate_identifier(
                    f"edge {edge.edge_id!r} source", edge.source, schema_errors
                )
                _validate_identifier(
                    f"edge {edge.edge_id!r} target", edge.target, schema_errors
                )
                _validate_identifier(
                    f"edge {edge.edge_id!r} label", edge.label, schema_errors
                )

    def _validate_formula_contracts(
        self, formula_errors: _CategorizedValidationErrors
    ) -> None:
        _validate_operation_config(
            self.output_aggregate, "output_aggregate", formula_errors
        )
        _validate_known_reference_config(
            "output_aggregate",
            self.output_aggregate,
            "Plan output_aggregate",
            formula_errors,
        )
        _validate_optional_shape_field(
            self.output_aggregate,
            "output_shape",
            (self.d_model,),
            "output_aggregate",
            formula_errors,
        )

    def _validate_nodes(
        self,
        node_set: Set[str],
        region_set: Set[str],
        topology_errors: _CategorizedValidationErrors,
        formula_errors: _CategorizedValidationErrors,
    ) -> None:
        for node in self.nodes:
            if node.region_id not in region_set:
                topology_errors.append(
                    f"node {node.node_id!r} names unknown region "
                    f"{node.region_id!r}"
                )
            if node.hidden_shape != (self.d_model,):
                formula_errors.append(
                    f"node {node.node_id!r} hidden_shape must be "
                    f"({self.d_model},), got {node.hidden_shape!r}"
                )
            _validate_shape(
                node.selector_read_shape,
                f"node {node.node_id!r} selector_read_shape",
                formula_errors,
            )
            _validate_shape(
                node.state_shape,
                f"node {node.node_id!r} state_shape",
                formula_errors,
            )
            for name in (
                "input_norm",
                "ffn_norm",
                "aggregate",
                "update",
                "selector_read",
                "ffn_read",
                "node_compute",
                "emit",
            ):
                _validate_operation_config(
                    getattr(node, name),
                    f"node {node.node_id!r} {name}",
                    formula_errors,
                )
                _validate_known_reference_config(
                    name,
                    getattr(node, name),
                    f"node {node.node_id!r} {name}",
                    formula_errors,
                    state_update_type=node.update.get("type"),
                    semantic_context={
                        "d_model": self.d_model,
                        "state_shape": node.state_shape,
                        "selector_read_shape": node.selector_read_shape,
                    },
                )
            for norm_name in ("input_norm", "ffn_norm"):
                norm = getattr(node, norm_name)
                epsilon = norm.get("eps")
                normalized_epsilon = _finite_formula_number(epsilon)
                if normalized_epsilon is None or normalized_epsilon <= 0.0:
                    formula_errors.append(
                        f"node {node.node_id!r} {norm_name}.eps must be a "
                        "positive finite number with integer literals in the "
                        "JSON-safe range"
                    )
            update_type = node.update.get("type")
            if isinstance(update_type, str) and update_type in {
                "gdn",
                "attention_window",
            }:
                epsilon = node.update.get("norm_eps")
                normalized_epsilon = _finite_formula_number(epsilon)
                if normalized_epsilon is None or normalized_epsilon <= 0.0:
                    formula_errors.append(
                        f"node {node.node_id!r} update.norm_eps must be a "
                        "positive finite number with integer literals in the "
                        "JSON-safe range"
                    )
            if update_type == "none":
                if node.state_owner is not None:
                    topology_errors.append(
                        f"stateless node {node.node_id!r} must not declare "
                        "state_owner"
                    )
                if node.state_shape:
                    formula_errors.append(
                        f"stateless node {node.node_id!r} must use empty "
                        "state_shape"
                    )
            else:
                if node.state_owner != node.node_id:
                    topology_errors.append(
                        f"stateful node {node.node_id!r} must own its mutable "
                        "state; cross-node mutable state sharing is not part of "
                        "the standard Plan"
                    )
            _validate_optional_shape_field(
                node.aggregate,
                "output_shape",
                node.hidden_shape,
                f"node {node.node_id!r} aggregate",
                formula_errors,
            )
            _validate_optional_shape_field(
                node.update,
                "state_shape",
                node.state_shape,
                f"node {node.node_id!r} update",
                formula_errors,
            )
            _validate_optional_shape_field(
                node.selector_read,
                "output_shape",
                node.selector_read_shape,
                f"node {node.node_id!r} selector_read",
                formula_errors,
            )
            for config_name in ("ffn_read", "node_compute", "emit"):
                _validate_optional_shape_field(
                    getattr(node, config_name),
                    "output_shape",
                    node.hidden_shape,
                    f"node {node.node_id!r} {config_name}",
                    formula_errors,
                )
            for config_name in (
                "aggregate",
                "update",
                "selector_read",
                "ffn_read",
                "node_compute",
                "emit",
            ):
                config = getattr(node, config_name)
                if "d_model" in config and config["d_model"] != self.d_model:
                    formula_errors.append(
                        f"node {node.node_id!r} {config_name}.d_model must be "
                        f"{self.d_model}"
                    )

    def _validate_regions(
        self,
        node_set: Set[str],
        region_set: Set[str],
        topology_errors: _CategorizedValidationErrors,
        formula_errors: _CategorizedValidationErrors,
    ) -> None:
        declared_members: List[str] = []
        node_lookup = {node.node_id: node for node in self.nodes}
        for region in self.regions:
            if not region.node_ids:
                topology_errors.append(
                    f"region {region.region_id!r} must not be empty"
                )
            duplicate_members = _duplicates(region.node_ids)
            if duplicate_members:
                topology_errors.append(
                    f"region {region.region_id!r} repeats node IDs: "
                    + ", ".join(repr(item) for item in duplicate_members)
                )
            for node_id in region.node_ids:
                if node_id not in node_set:
                    topology_errors.append(
                        f"region {region.region_id!r} names unknown node "
                        f"{node_id!r}"
                    )
                elif node_lookup[node_id].region_id != region.region_id:
                    topology_errors.append(
                        f"node {node_id!r} names region "
                        f"{node_lookup[node_id].region_id!r}, but region "
                        f"{region.region_id!r} lists it"
                    )
            read_shapes = {
                node_lookup[node_id].selector_read_shape
                for node_id in region.node_ids
                if node_id in node_lookup
            }
            if len(read_shapes) > 1:
                formula_errors.append(
                    f"region {region.region_id!r} selector readouts must have "
                    "one common fixed shape"
                )
            declared_members.extend(region.node_ids)
            if region.profile not in _PROFILES:
                formula_errors.append(
                    f"region {region.region_id!r} has invalid profile "
                    f"{region.profile!r}"
                )
            if region.selector_timing not in _SELECTOR_TIMINGS:
                formula_errors.append(
                    f"region {region.region_id!r} has invalid selector timing "
                    f"{region.selector_timing!r}"
                )
            elif region.profile == "N" and region.selector_timing != "content":
                formula_errors.append(
                    f"region {region.region_id!r}: profile N only supports "
                    "content selector timing"
                )
            elif region.profile == "SD" and region.selector_timing == "post":
                formula_errors.append(
                    f"region {region.region_id!r}: standard SD does not "
                    "support post-update selection"
                )
            for node_id in region.node_ids:
                node = node_lookup.get(node_id)
                if node is None or _operation_schema(
                    "selector_read", node.selector_read
                ) is None:
                    continue
                read_type = node.selector_read.get("type")
                allowed_read_types = (
                    {"content", "content_norm", "content_linear"}
                    if region.selector_timing == "content"
                    else {
                        "content_state_linear",
                        "content_state_summary_linear",
                    }
                    if region.selector_timing in {"pre", "post"}
                    else set()
                )
                if read_type not in allowed_read_types:
                    formula_errors.append(
                        f"node {node_id!r} selector_read type {read_type!r} "
                        f"is incompatible with {region.selector_timing!r} "
                        "selector timing"
                    )
            if region.profile == "N":
                for node_id in region.node_ids:
                    node = node_lookup.get(node_id)
                    if node is not None and node.update.get("type") != "none":
                        formula_errors.append(
                            f"region {region.region_id!r}: profile N requires "
                            f"stateless node {node_id!r}"
                        )
            if (
                region.selector_timing == "content"
                and _declares_persistent_state(region.selector_context)
            ):
                formula_errors.append(
                    f"region {region.region_id!r}: content timing cannot read "
                    "persistent selector context"
                )
            if type(region.k_max) is not int or region.k_max < 1:
                formula_errors.append(
                    f"region {region.region_id!r} k_max must be a positive integer"
                )
            elif region.k_max > len(region.node_ids):
                topology_errors.append(
                    f"region {region.region_id!r} k_max exceeds its fixed size"
                )
            _validate_k_request(region, formula_errors)
            _validate_known_reference_config(
                "k_requested",
                region.k_requested,
                f"region {region.region_id!r} k_requested",
                formula_errors,
            )
            for name in (
                "score",
                "selector_context",
                "selector_history",
            ):
                _validate_operation_config(
                    getattr(region, name),
                    f"region {region.region_id!r} {name}",
                    formula_errors,
                )
                _validate_known_reference_config(
                    name,
                    getattr(region, name),
                    f"region {region.region_id!r} {name}",
                    formula_errors,
                )
            if region.score.get("type") == "fixed":
                values = region.score.get("values_by_node")
                if not isinstance(values, Mapping):
                    formula_errors.append(
                        f"region {region.region_id!r} fixed score must declare "
                        "values_by_node"
                    )
                else:
                    expected_nodes = set(region.node_ids)
                    actual_nodes = set(values)
                    if actual_nodes != expected_nodes:
                        formula_errors.append(
                            f"region {region.region_id!r} fixed score keys must "
                            "equal its static node IDs"
                        )
                    for node_id, value in values.items():
                        if _finite_formula_number(value) is None:
                            formula_errors.append(
                                f"region {region.region_id!r} fixed score for "
                                f"{node_id!r} must be finite numeric data with "
                                "integer literals in the JSON-safe range"
                            )
            duplicate_dependencies = _duplicates(region.control_dependencies)
            if duplicate_dependencies:
                topology_errors.append(
                    f"region {region.region_id!r} repeats control dependencies: "
                    + ", ".join(repr(item) for item in duplicate_dependencies)
                )
            for dependency in region.control_dependencies:
                if dependency not in region_set:
                    topology_errors.append(
                        f"region {region.region_id!r} names unknown control "
                        f"dependency {dependency!r}"
                    )
                if dependency == region.region_id:
                    topology_errors.append(
                        f"region {region.region_id!r} cannot depend on itself"
                    )
            forced = [
                node_lookup[node_id]
                for node_id in region.node_ids
                if node_id in node_lookup and node_lookup[node_id].forced_active
            ]
            if forced:
                if len(region.node_ids) != 1:
                    topology_errors.append(
                        f"forced-active node in region {region.region_id!r} "
                        "must use an independent singleton region"
                    )
                if region.k_max != 1 or not _request_is_exactly_one(
                    region.k_requested
                ):
                    formula_errors.append(
                        f"forced-active region {region.region_id!r} must request "
                        "exactly one active node"
                    )

        duplicate_memberships = _duplicates(declared_members)
        if duplicate_memberships:
            topology_errors.append(
                "nodes must belong to exactly one region; repeated memberships: "
                + ", ".join(repr(item) for item in duplicate_memberships)
            )
        missing_memberships = sorted(node_set - set(declared_members))
        if missing_memberships:
            topology_errors.append(
                "nodes missing from region partition: "
                + ", ".join(repr(item) for item in missing_memberships)
            )

    def _validate_edges(
        self,
        node_set: Set[str],
        topology_errors: _CategorizedValidationErrors,
    ) -> None:
        endpoints: List[Tuple[str, str]] = []
        region_of = {node.node_id: node.region_id for node in self.nodes}
        valid_pairs: List[Tuple[str, str]] = []
        for edge in self.edges:
            if edge.source not in node_set:
                topology_errors.append(
                    f"edge {edge.edge_id!r} has unknown source {edge.source!r}"
                )
            if edge.target not in node_set:
                topology_errors.append(
                    f"edge {edge.edge_id!r} has unknown target {edge.target!r}"
                )
            endpoints.append((edge.source, edge.target))
            if edge.source == edge.target:
                topology_errors.append(f"edge {edge.edge_id!r} is a self-loop")
            if edge.source in region_of and edge.target in region_of:
                if region_of[edge.source] == region_of[edge.target]:
                    topology_errors.append(
                        f"edge {edge.edge_id!r} connects nodes inside region "
                        f"{region_of[edge.source]!r}"
                    )
                valid_pairs.append((edge.source, edge.target))
        duplicate_edges = _duplicates(endpoints)
        if duplicate_edges:
            topology_errors.append(
                "duplicate parallel receiver edges: "
                + ", ".join(f"{source!r}->{target!r}" for source, target in duplicate_edges)
            )
        _, cyclic = _topological_order(sorted(node_set), valid_pairs)
        if cyclic:
            topology_errors.append(
                "receiver graph is cyclic; unresolved nodes: "
                + ", ".join(repr(item) for item in cyclic)
            )

    def _validate_boundaries_and_paths(
        self, node_set: Set[str], errors: _ValidationErrorSink
    ) -> None:
        incoming = {node_id: [] for node_id in node_set}
        outgoing = {node_id: [] for node_id in node_set}
        for edge in self.edges:
            if edge.source in node_set and edge.target in node_set:
                outgoing[edge.source].append(edge.target)
                incoming[edge.target].append(edge.source)
        derived_entries = {node for node, parents in incoming.items() if not parents}
        derived_terminals = {
            node for node, children in outgoing.items() if not children
        }
        declared_entries = set(self.entry_node_ids)
        declared_terminals = set(self.terminal_node_ids)
        if len(self.entry_node_ids) != len(declared_entries):
            errors.append("entry_node_ids contains duplicates")
        if len(self.terminal_node_ids) != len(declared_terminals):
            errors.append("terminal_node_ids contains duplicates")
        for node_id in self.entry_node_ids:
            if node_id not in node_set:
                errors.append(f"unknown entry node {node_id!r}")
        for node_id in self.terminal_node_ids:
            if node_id not in node_set:
                errors.append(f"unknown terminal node {node_id!r}")
        if declared_entries != derived_entries:
            errors.append(
                "entry_node_ids must equal nodes with no receiver parents; "
                f"expected {sorted(derived_entries)!r}"
            )
        if declared_terminals != derived_terminals:
            errors.append(
                "terminal_node_ids must equal nodes with no receiver children; "
                f"expected {sorted(derived_terminals)!r}"
            )
        reachable = _walk(declared_entries & node_set, outgoing)
        can_reach_terminal = _walk(declared_terminals & node_set, incoming)
        off_path = sorted(node_set - (reachable & can_reach_terminal))
        if off_path:
            errors.append(
                "every node must lie on a declared entry-to-terminal path; "
                "off-path nodes: " + ", ".join(repr(item) for item in off_path)
            )

    def _region_dependency_pairs(self) -> List[Tuple[str, str]]:
        region_of = {node.node_id: node.region_id for node in self.nodes}
        pairs: Set[Tuple[str, str]] = set()
        for edge in self.edges:
            if edge.source in region_of and edge.target in region_of:
                pairs.add((region_of[edge.source], region_of[edge.target]))
        for region in self.regions:
            for dependency in region.control_dependencies:
                pairs.add((dependency, region.region_id))
        if self.topology_kind == "hb":
            by_line: Dict[int, List[str]] = {}
            for region in self.regions:
                if type(region.line) is int and region.line >= 0:
                    by_line.setdefault(region.line, []).append(region.region_id)
            for line in sorted(by_line):
                previous = line - 1
                if previous not in by_line:
                    continue
                for source in by_line[previous]:
                    for target in by_line[line]:
                        pairs.add((source, target))
        return sorted(pairs)

    def _validate_region_graph(self, errors: _ValidationErrorSink) -> None:
        region_ids = [region.region_id for region in self.regions]
        valid = set(region_ids)
        pairs = [
            pair
            for pair in self._region_dependency_pairs()
            if pair[0] in valid and pair[1] in valid
        ]
        _, cyclic = _topological_order(region_ids, pairs)
        if cyclic:
            errors.append(
                "region dependency graph is cyclic; unresolved regions: "
                + ", ".join(repr(item) for item in cyclic)
            )

    def _validate_hb(
        self,
        topology_errors: _CategorizedValidationErrors,
    ) -> None:
        if self.topology_kind != "hb":
            for region in self.regions:
                if region.line is not None or region.phase is not None:
                    topology_errors.append(
                        f"general Plan region {region.region_id!r} must not "
                        "declare HB line/phase metadata"
                    )
            return
        by_line: Dict[int, List[RegionSpec]] = {}
        for region in self.regions:
            if type(region.line) is not int or region.line < 0:
                topology_errors.append(
                    f"HB region {region.region_id!r} must have a nonnegative "
                    "integer line"
                )
                continue
            if not isinstance(region.phase, str) or not region.phase:
                topology_errors.append(
                    f"HB region {region.region_id!r} must declare a phase"
                )
            by_line.setdefault(region.line, []).append(region)
        if not by_line:
            return
        maximum = max(by_line)
        expected = set(range(maximum + 1))
        missing = sorted(expected - set(by_line))
        if missing:
            topology_errors.append(
                f"HB lines must be contiguous from 0; missing {missing!r}"
            )
        for line, regions in sorted(by_line.items()):
            phases = {
                region.phase
                if isinstance(region.phase, str)
                else repr(region.phase)
                for region in regions
            }
            if len(phases) != 1:
                topology_errors.append(
                    f"all HB regions on line {line} must share one phase"
                )
        region_lookup = {region.region_id: region for region in self.regions}
        node_lookup = {node.node_id: node for node in self.nodes}
        node_line = {
            node.node_id: region_lookup[node.region_id].line
            for node in self.nodes
            if (
                node.region_id in region_lookup
                and type(region_lookup[node.region_id].line) is int
                and region_lookup[node.region_id].line >= 0
            )
        }
        for edge in self.edges:
            source_line = node_line.get(edge.source)
            target_line = node_line.get(edge.target)
            if (
                source_line is not None
                and target_line is not None
                and source_line >= target_line
            ):
                topology_errors.append(
                    f"HB edge {edge.edge_id!r} must point to a deeper line"
                )
            if not isinstance(edge.label, str) or edge.label not in _HB_EDGE_LABELS:
                topology_errors.append(
                    f"HB edge {edge.edge_id!r} has invalid source label "
                    f"{edge.label!r}"
                )
        for region in self.regions:
            for dependency in region.control_dependencies:
                other = region_lookup.get(dependency)
                if (
                    other is None
                    or type(other.line) is not int
                    or type(region.line) is not int
                ):
                    continue
                if other.line >= region.line:
                    topology_errors.append(
                        f"HB control dependency {dependency!r} -> "
                        f"{region.region_id!r} must come from a shallower line"
                    )
        first_line_nodes = {
            node_id for node_id, line in node_line.items() if line == 0
        }
        last_line_nodes = {
            node_id for node_id, line in node_line.items() if line == maximum
        }
        if set(self.entry_node_ids) != first_line_nodes:
            topology_errors.append("HB entry receivers must be exactly line 0")
        if set(self.terminal_node_ids) != last_line_nodes:
            topology_errors.append(
                "HB terminal receivers must be exactly the final line"
            )


def _validate_identifier(
    kind: str, value: object, errors: _ValidationErrorSink
) -> None:
    errors.extend(_stable_id_errors(kind, value))


def _validate_unique_ids(
    kind: str, values: Sequence[str], errors: _ValidationErrorSink
) -> None:
    duplicates = _duplicates(values)
    if duplicates:
        errors.append(
            f"duplicate {kind} IDs: "
            + ", ".join(repr(item) for item in duplicates)
        )


def _duplicates(values: Sequence[Any]) -> List[Any]:
    seen: Set[Any] = set()
    duplicates: Set[Any] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def _validate_shape(
    shape: Tuple[int, ...], name: str, errors: _ValidationErrorSink
) -> None:
    for dimension in shape:
        if type(dimension) is not int or dimension <= 0:
            errors.append(f"{name} dimensions must be positive integers")
            return


def _validate_operation_config(
    config: Mapping[str, JsonValue], name: str, errors: _ValidationErrorSink
) -> None:
    operation_type = config.get("type")
    if not isinstance(operation_type, str) or not operation_type:
        errors.append(f"{name} must declare a nonempty string 'type'")
    if operation_type == "custom":
        formula = config.get("formula")
        if not isinstance(formula, str) or not formula.strip():
            errors.append(f"{name} custom operation must declare its formula")


def _validate_known_reference_config(
    operation_field: str,
    config: Mapping[str, JsonValue],
    name: str,
    errors: _ValidationErrorSink,
    *,
    state_update_type: Optional[str] = None,
    semantic_context: Optional[Mapping[str, JsonValue]] = None,
) -> None:
    """Apply the exact eager schema without narrowing generic Plan syntax."""

    schema = _operation_schema(operation_field, config)
    operation_type = _normalized_operation_type(config.get("type"))
    formula_id = config.get("formula_id")
    if not isinstance(operation_type, str):
        return
    if formula_id is not None and not isinstance(formula_id, str):
        return
    field_schemas = {
        key: value
        for key, value in REFERENCE_OPERATION_CONFIG_SCHEMAS.items()
        if key[0] == operation_field
    }
    registered_types = {key[1] for key in field_schemas}
    registered_formula_bindings = {
        key for key in REFERENCE_OPERATION_CONFIG_SCHEMAS if key[2] == formula_id
    }

    # A genuinely custom operation carries its own explicit formula and is a
    # valid generic Plan that this eager executor may reject as unsupported.
    # In contrast, a registered operation type with a missing/unknown ID, or
    # any registered ID dispatched through the wrong field/type, is a malformed
    # formula declaration and must fail the Plan gate.
    should_validate_reference = schema is not None or (
        operation_type != "custom"
        and (
            operation_type in registered_types
            or bool(registered_formula_bindings)
        )
    )
    if not should_validate_reference:
        return
    try:
        validate_reference_operation_config(
            operation_field,
            config,
            state_update_type=state_update_type,
            semantic_context=semantic_context,
        )
    except ReferenceOperationConfigError as exc:
        errors.append(f"{name}: {exc}")


def _validate_optional_shape_field(
    config: Mapping[str, JsonValue],
    field_name: str,
    expected: Tuple[int, ...],
    context: str,
    errors: _ValidationErrorSink,
) -> None:
    if field_name not in config:
        return
    raw = config[field_name]
    if not isinstance(raw, tuple) or any(type(item) is not int for item in raw):
        errors.append(f"{context}.{field_name} must be a JSON integer array")
        return
    if tuple(raw) != expected:
        errors.append(
            f"{context}.{field_name} must be {list(expected)!r}, "
            f"got {list(raw)!r}"
        )


def _declares_persistent_state(config: Mapping[str, JsonValue]) -> bool:
    return bool(
        config.get("uses_persistent_state", False)
        or config.get("uses_selector_history", False)
    )


def _request_is_exactly_one(config: Mapping[str, JsonValue]) -> bool:
    request_type = config.get("type")
    return request_type == "fixed" and config.get("value") == 1


def _validate_k_request(
    region: RegionSpec, errors: _ValidationErrorSink
) -> None:
    config = region.k_requested
    _validate_operation_config(
        config, f"region {region.region_id!r} k_requested", errors
    )
    request_type = config.get("type")
    if request_type == "fixed":
        value = config.get("value")
        if (
            type(value) is not int
            or type(region.k_max) is not int
            or not (1 <= value <= region.k_max)
        ):
            errors.append(
                f"region {region.region_id!r} fixed K must be an integer in "
                f"[1, {region.k_max}]"
            )
        elif region.k_max <= len(region.node_ids) and value != region.k_max:
            errors.append(
                f"region {region.region_id!r} fixed K must equal "
                f"k_max={region.k_max} in core-v1"
            )
    elif request_type == "input":
        if config.get("field") != "requested_k":
            errors.append(
                f"region {region.region_id!r} input K field must be "
                "'requested_k'"
            )
        minimum = config.get("minimum")
        maximum = config.get("maximum")
        if type(minimum) is not int or minimum != 1:
            errors.append(
                f"region {region.region_id!r} input K minimum must be 1"
            )
        if type(maximum) is not int or maximum != region.k_max:
            errors.append(
                f"region {region.region_id!r} input K maximum must equal "
                f"k_max={region.k_max}"
            )
    else:
        errors.append(
            f"region {region.region_id!r} k_requested type must be fixed or "
            "input"
        )


def _topological_order(
    identifiers: Sequence[str], pairs: Sequence[Tuple[str, str]]
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    identifier_set = set(identifiers)
    successors: Dict[str, Set[str]] = {
        identifier: set() for identifier in identifier_set
    }
    indegree = {identifier: 0 for identifier in identifier_set}
    for source, target in pairs:
        if source not in identifier_set or target not in identifier_set:
            continue
        if target in successors[source]:
            continue
        successors[source].add(target)
        indegree[target] += 1
    ready = [identifier for identifier, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    order: List[str] = []
    while ready:
        source = heapq.heappop(ready)
        order.append(source)
        for target in sorted(successors[source]):
            indegree[target] -= 1
            if indegree[target] == 0:
                heapq.heappush(ready, target)
    cyclic = tuple(sorted(identifier for identifier, degree in indegree.items() if degree))
    return tuple(order), cyclic


def _walk(starts: Set[str], adjacency: Mapping[str, Sequence[str]]) -> Set[str]:
    visited: Set[str] = set()
    pending = list(starts)
    while pending:
        item = pending.pop()
        if item in visited:
            continue
        visited.add(item)
        pending.extend(adjacency.get(item, ()))
    return visited


def _reject_non_scalar_json_strings(value: JsonValue, *, path: str) -> None:
    """Defensively reject text for which canonical UTF-8 is undefined."""

    if isinstance(value, str):
        _require_unicode_scalar_sequence(value, path=path)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} JSON object keys must be strings")
            _require_unicode_scalar_sequence(
                key, path=f"{path} JSON object key {key!r}"
            )
            _reject_non_scalar_json_strings(item, path=f"{path}[{key!r}]")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_non_scalar_json_strings(item, path=f"{path}[{index}]")


def _canonical_json_text(value: JsonValue) -> str:
    _reject_non_scalar_json_strings(value, path="canonical JSON root")
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class ConcreteBinding:
    """Concrete dtypes chosen for every role in one logical Plan.

    A logical declaration of ``runtime`` is resolved here.  A concrete dtype
    already required by the logical Plan is repeated and must match.  This
    object contains no device, parameters, state, or executor choice.
    """

    dtype_roles: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dtype_roles",
            _freeze_config_if_possible(self.dtype_roles),
        )

    def validate_for(self, plan: Plan) -> "ConcreteBinding":
        plan.validate()
        errors: List[str] = []
        if not isinstance(self.dtype_roles, FrozenConfig):
            errors.append("concrete binding dtype_roles must be a JSON object")
            raise PlanValidationError(
                errors, failure_codes=("binding.invalid",)
            )
        for role, dtype in self.dtype_roles.items():
            if not isinstance(role, str) or not role:
                errors.append("concrete binding role names must be nonempty strings")
            if not isinstance(dtype, str):
                errors.append(
                    f"concrete dtype role {role!r} must be a string, got "
                    f"{type(dtype).__name__}"
                )
        if errors:
            raise PlanValidationError(
                errors, failure_codes=("binding.invalid",)
            )
        logical_roles = set(plan.dtype_roles)
        bound_roles = set(self.dtype_roles)
        if bound_roles != logical_roles:
            errors.append(
                "concrete binding roles must exactly match logical Plan roles; "
                f"expected {sorted(logical_roles)!r}"
            )
        for role, dtype in self.dtype_roles.items():
            if not isinstance(dtype, str) or dtype not in _CONCRETE_DTYPES:
                errors.append(
                    f"concrete dtype role {role!r} must not be symbolic, got "
                    f"{dtype!r}"
                )
                continue
            declared = plan.dtype_roles.get(role)
            if declared not in {None, "runtime", dtype}:
                errors.append(
                    f"concrete dtype {dtype!r} for role {role!r} conflicts "
                    f"with logical declaration {declared!r}"
                )
        if errors:
            raise PlanValidationError(
                errors, failure_codes=("binding.invalid",)
            )
        return self

    def canonical_dict(self) -> Dict[str, JsonValue]:
        return {"dtype_roles": _thaw_json(self.dtype_roles)}


@dataclass(frozen=True)
class TypedPlan:
    """A logical Plan paired with a validated concrete dtype binding."""

    logical_plan: Plan
    binding: ConcreteBinding

    def validate(self) -> "TypedPlan":
        self.binding.validate_for(self.logical_plan)
        return self

    def logical_hash(self) -> str:
        return self.logical_plan.canonical_hash()

    def canonical_dict(self) -> Dict[str, JsonValue]:
        self.validate()
        return {
            "schema_version": "1",
            "logical_plan_hash": self.logical_hash(),
            "binding": self.binding.canonical_dict(),
        }

    def canonical_json(self) -> str:
        return _canonical_json_text(self.canonical_dict())

    def canonical_bytes(self) -> bytes:
        """Return the ``tide-plan-json-v1`` typed-Plan UTF-8 bytes."""

        return self.canonical_json().encode("utf-8")

    def canonical_hash(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def typed_hash(self) -> str:
        """Explicit alias for the binding-sensitive canonical hash."""

        return self.canonical_hash()


def bind_dtypes(plan: Plan, **roles: str) -> TypedPlan:
    """Create and validate a concrete typed view without changing Plan hash."""

    return TypedPlan(plan, ConcreteBinding(roles)).validate()


__all__ = [
    "ConcreteBinding",
    "EdgeSpec",
    "FrozenConfig",
    "NodeSpec",
    "PLAN_CANONICALIZER_ID",
    "Plan",
    "PlanValidationError",
    "RegionSpec",
    "TypedPlan",
    "bind_dtypes",
    "validate_stable_id",
]

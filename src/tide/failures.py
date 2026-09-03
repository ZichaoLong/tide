"""Stable failure envelopes for eager-reference equivalence fixtures.

The semantic contract compares phase and code, never implementation-specific
exception class names, tracebacks, or messages.  A few eager-reference
exceptions identify a unique contract code.  Coarser exceptions require the
caller that knows the validation stage to supply the code explicitly; this
module deliberately does not infer a code from exception text.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Iterable, Mapping
from types import MappingProxyType
from typing import Any, Dict, Generic, Optional, Tuple, TypeVar, Union


FAILURE_SCHEMA = "tide.failure.v1"

_FAILURE_CODES = {
    "artifact": {
        "artifact.integrity",
        "artifact.schema",
    },
    "plan": {
        "plan.schema",
        "plan.topology",
        "plan.formula",
    },
    "binding": {"binding.invalid"},
    "capability": {"capability.unsupported"},
    "input": {
        "input.schema",
        "input.mask",
        "input.position",
    },
    "event": {"input.requested_k"},
    "state": {
        "state.schema",
        "state.owner_alias",
    },
    "execution": {
        "execution.local_operation",
        "execution.empty_terminal",
    },
    "checkpoint": {
        "checkpoint.integrity",
        "checkpoint.schema",
        "checkpoint.compatibility",
        "checkpoint.commit",
    },
    "runtime": {
        "runtime.configuration",
        "runtime.unavailable",
    },
}

# Public and immutable so fixture tooling can enumerate the complete v1
# vocabulary without duplicating it.
FAILURE_CODES_BY_PHASE = MappingProxyType(
    {phase: frozenset(codes) for phase, codes in _FAILURE_CODES.items()}
)
_PHASE_BY_CODE = {
    code: phase
    for phase, codes in FAILURE_CODES_BY_PHASE.items()
    for code in codes
}


class FailureEnvelopeError(ValueError):
    """A failure envelope or exception-to-code request is invalid."""


class _UnmappedContractException(TypeError):
    """Internal marker used to preserve unexpected exceptions in capture."""


class FailureEnvelopeMismatch(AssertionError):
    """Two valid failure envelopes differ in schema, phase, or codes."""

    def __init__(
        self, expected: "FailureEnvelope", actual: "FailureEnvelope"
    ) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(
            "failure envelopes differ: "
            f"expected={expected.canonical_json()} "
            f"actual={actual.canonical_json()}"
        )


def _reject_duplicate_json_keys(
    pairs: Iterable[Tuple[str, Any]],
) -> Dict[str, Any]:
    """Build one JSON object while rejecting parser-order ambiguity."""

    record: Dict[str, Any] = {}
    for key, value in pairs:
        if key in record:
            raise FailureEnvelopeError(
                f"duplicate failure-envelope JSON key: {key!r}"
            )
        record[key] = value
    return record


@dataclasses.dataclass(frozen=True)
class FailureEnvelope:
    """One normalized ``tide.failure.v1`` expected-failure record."""

    error_schema: str
    phase: str
    codes: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.error_schema != FAILURE_SCHEMA:
            raise FailureEnvelopeError(
                f"error_schema must be exactly {FAILURE_SCHEMA!r}"
            )
        if not isinstance(self.phase, str) or self.phase not in FAILURE_CODES_BY_PHASE:
            raise FailureEnvelopeError(f"unknown failure phase: {self.phase!r}")
        if isinstance(self.codes, (str, bytes)):
            raise FailureEnvelopeError("codes must be a nonempty array of strings")
        try:
            supplied_codes = tuple(self.codes)
        except TypeError as exc:
            raise FailureEnvelopeError(
                "codes must be a nonempty array of strings"
            ) from exc
        if not supplied_codes:
            raise FailureEnvelopeError("codes must not be empty")
        for code in supplied_codes:
            if not isinstance(code, str):
                raise FailureEnvelopeError("every failure code must be a string")
            expected_phase = _PHASE_BY_CODE.get(code)
            if expected_phase is None:
                raise FailureEnvelopeError(f"unknown failure code: {code!r}")
            if expected_phase != self.phase:
                raise FailureEnvelopeError(
                    f"failure code {code!r} belongs to phase "
                    f"{expected_phase!r}, not {self.phase!r}"
                )
        object.__setattr__(self, "codes", tuple(sorted(set(supplied_codes))))

    @classmethod
    def create(
        cls, phase: str, codes: Union[str, Iterable[str]]
    ) -> "FailureEnvelope":
        """Construct an envelope while supplying the fixed schema implicitly."""

        if isinstance(codes, str):
            normalized = (codes,)
        else:
            try:
                normalized = tuple(codes)
            except TypeError as exc:
                raise FailureEnvelopeError(
                    "codes must be a string or iterable of strings"
                ) from exc
        return cls(FAILURE_SCHEMA, phase, normalized)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "FailureEnvelope":
        """Validate and decode the exact JSON object shape used by fixtures."""

        if not isinstance(record, Mapping):
            raise FailureEnvelopeError("failure envelope must be a JSON object")
        expected_keys = {"error_schema", "phase", "codes"}
        if set(record) != expected_keys:
            raise FailureEnvelopeError(
                "failure envelope keys must be exactly "
                "'error_schema', 'phase', and 'codes'"
            )
        codes = record["codes"]
        if not isinstance(codes, list):
            raise FailureEnvelopeError("serialized codes must be a JSON array")
        return cls(record["error_schema"], record["phase"], tuple(codes))

    @classmethod
    def from_json(cls, text: str) -> "FailureEnvelope":
        """Decode a JSON failure envelope and validate its complete schema."""

        if not isinstance(text, str):
            raise FailureEnvelopeError("serialized failure envelope must be text")
        try:
            record = json.loads(
                text,
                object_pairs_hook=_reject_duplicate_json_keys,
            )
        except (TypeError, ValueError) as exc:
            raise FailureEnvelopeError("invalid failure-envelope JSON") from exc
        return cls.from_dict(record)

    def to_dict(self) -> Dict[str, Any]:
        """Return the normalized JSON-compatible fixture record."""

        return {
            "error_schema": self.error_schema,
            "phase": self.phase,
            "codes": list(self.codes),
        }

    def canonical_json(self) -> str:
        """Return deterministic UTF-8-compatible JSON text for diagnostics."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def compare_failure_envelopes(
    expected: FailureEnvelope, actual: FailureEnvelope
) -> None:
    """Raise :class:`FailureEnvelopeMismatch` unless both records are equal."""

    if not isinstance(expected, FailureEnvelope) or not isinstance(
        actual, FailureEnvelope
    ):
        raise TypeError("failure-envelope comparison requires validated envelopes")
    if expected != actual:
        raise FailureEnvelopeMismatch(expected, actual)


_RequestedCodes = Optional[Union[str, Iterable[str]]]


def _requested_code_tuple(codes: _RequestedCodes) -> Optional[Tuple[str, ...]]:
    if codes is None:
        return None
    if isinstance(codes, str):
        return (codes,)
    try:
        result = tuple(codes)
    except TypeError as exc:
        raise FailureEnvelopeError("codes must be a string or iterable of strings") from exc
    if not result:
        raise FailureEnvelopeError("explicit exception mapping codes must not be empty")
    if any(not isinstance(code, str) for code in result):
        raise FailureEnvelopeError("every explicit exception mapping code must be a string")
    return result


def _fixed_exception_envelope(
    fixed_code: str, requested_codes: _RequestedCodes
) -> FailureEnvelope:
    supplied = _requested_code_tuple(requested_codes)
    if supplied is not None and tuple(sorted(set(supplied))) != (fixed_code,):
        raise FailureEnvelopeError(
            f"this exception maps uniquely to {fixed_code!r}"
        )
    return FailureEnvelope.create(_PHASE_BY_CODE[fixed_code], fixed_code)


def _coarse_exception_envelope(
    exception_name: str,
    requested_codes: _RequestedCodes,
    allowed_codes: Iterable[str],
) -> FailureEnvelope:
    supplied = _requested_code_tuple(requested_codes)
    if supplied is None:
        raise FailureEnvelopeError(
            f"{exception_name} is too coarse for tide.failure.v1; "
            "the caller must supply codes from the known validation stage"
        )
    allowed = frozenset(allowed_codes)
    invalid = sorted(
        code for code in supplied if not isinstance(code, str) or code not in allowed
    )
    if invalid:
        raise FailureEnvelopeError(
            f"codes {invalid!r} are not valid for {exception_name}"
        )
    phases = {_PHASE_BY_CODE[code] for code in supplied}
    if len(phases) != 1:
        raise FailureEnvelopeError(
            "one failure envelope cannot combine codes from different phases"
        )
    return FailureEnvelope.create(next(iter(phases)), supplied)


def _diagnosed_exception_envelope(
    exception_name: str,
    diagnosed_codes: Iterable[str],
    requested_codes: _RequestedCodes,
) -> FailureEnvelope:
    diagnosed = tuple(sorted(set(diagnosed_codes)))
    if not diagnosed:
        raise FailureEnvelopeError(
            f"{exception_name} does not carry structured failure codes"
        )
    supplied = _requested_code_tuple(requested_codes)
    if supplied is not None and tuple(sorted(set(supplied))) != diagnosed:
        raise FailureEnvelopeError(
            f"explicit mapping does not match {exception_name} diagnostics"
        )
    phases = {_PHASE_BY_CODE.get(code) for code in diagnosed}
    if None in phases or len(phases) != 1:
        raise FailureEnvelopeError(
            f"{exception_name} carries invalid cross-phase diagnostics"
        )
    return FailureEnvelope.create(next(iter(phases)), diagnosed)  # type: ignore[arg-type]


def _defined_by(error: BaseException, module_name: str) -> bool:
    return any(base.__module__ == module_name for base in type(error).__mro__)


def failure_envelope_from_exception(
    error: BaseException, *, codes: _RequestedCodes = None
) -> FailureEnvelope:
    """Map one known eager-reference exception to a stable envelope.

    ``UnsupportedPlanError``, ``DynamicReachabilityError``, local-operation
    failures, and the two runtime exceptions have unique mappings.  A current
    ``PlanValidationError`` carries its validator-produced codes; legacy/coarse
    Plan errors, ``ExecutionContractError``, and ``CheckpointError`` require
    explicit ``codes`` because their messages are not a stable API and those
    classes can span more than one semantic validation stage.
    """

    # The runtime and Plan exception modules are framework-neutral.  Dispatch
    # them before considering Torch-backed executor/artifact exceptions so a
    # runtime configuration failure remains usable on a torch-less host.
    from .runtime import RuntimeConfigurationError, RuntimeUnavailableError

    if isinstance(error, RuntimeConfigurationError):
        return _fixed_exception_envelope("runtime.configuration", codes)
    if isinstance(error, RuntimeUnavailableError):
        return _fixed_exception_envelope("runtime.unavailable", codes)

    from .plan import PlanValidationError

    if isinstance(error, PlanValidationError):
        diagnosed = getattr(error, "failure_codes", ())
        if diagnosed:
            return _diagnosed_exception_envelope(
                "PlanValidationError", diagnosed, codes
            )
        return _coarse_exception_envelope(
            "PlanValidationError",
            codes,
            {
                "plan.schema",
                "plan.topology",
                "plan.formula",
                "binding.invalid",
            },
        )
    if _defined_by(error, "tide.engine"):
        from .engine import (
            DynamicReachabilityError,
            ExecutionContractError,
            LocalOperationError,
            UnsupportedPlanError,
        )

        if isinstance(error, UnsupportedPlanError):
            return _fixed_exception_envelope("capability.unsupported", codes)
        if isinstance(error, DynamicReachabilityError):
            return _fixed_exception_envelope("execution.empty_terminal", codes)
        if isinstance(error, LocalOperationError):
            return _fixed_exception_envelope("execution.local_operation", codes)
        if not isinstance(error, ExecutionContractError):
            raise _UnmappedContractException(
                f"{type(error).__name__} is not a mapped eager-reference contract exception"
            )
        return _coarse_exception_envelope(
            "ExecutionContractError",
            codes,
            {
                "binding.invalid",
                "input.schema",
                "input.mask",
                "input.position",
                "input.requested_k",
                "state.schema",
                "state.owner_alias",
                "execution.local_operation",
            },
        )
    if _defined_by(error, "tide.fixtures"):
        from .fixtures import FixtureError

        if not isinstance(error, FixtureError):
            raise _UnmappedContractException(
                f"{type(error).__name__} is not a mapped eager-reference contract exception"
            )
        supplied = _requested_code_tuple(codes)
        if supplied is not None and tuple(sorted(set(supplied))) != error.envelope.codes:
            raise FailureEnvelopeError(
                "explicit fixture mapping does not match its validated envelope"
            )
        return error.envelope

    if _defined_by(error, "tide.checkpoint"):
        from .checkpoint import CheckpointError

        if not isinstance(error, CheckpointError):
            raise _UnmappedContractException(
                f"{type(error).__name__} is not a mapped eager-reference contract exception"
            )
        return _coarse_exception_envelope(
            "CheckpointError",
            codes,
            FAILURE_CODES_BY_PHASE["checkpoint"],
        )
    raise _UnmappedContractException(
        f"{type(error).__name__} is not a mapped eager-reference contract exception"
    )


T = TypeVar("T")


@dataclasses.dataclass(frozen=True)
class ExecutionSucceeded(Generic[T]):
    """Successful result returned by :func:`capture_execution`."""

    value: T


@dataclasses.dataclass(frozen=True)
class ExecutionFailed:
    """Stable failure result returned by :func:`capture_execution`."""

    envelope: FailureEnvelope


ExecutionCapture = Union[ExecutionSucceeded[T], ExecutionFailed]


def capture_execution(
    operation: Callable[[], T], *, codes: _RequestedCodes = None
) -> ExecutionCapture[T]:
    """Run a nullary operation and capture only known contract failures.

    Unexpected exceptions propagate, preserving implementation bugs as test
    failures.  For a coarse known exception, the same explicit ``codes`` rule
    as :func:`failure_envelope_from_exception` applies.
    """

    if not callable(operation):
        raise TypeError("operation must be callable")
    try:
        return ExecutionSucceeded(operation())
    except BaseException as error:
        try:
            envelope = failure_envelope_from_exception(error, codes=codes)
        except _UnmappedContractException:
            raise error
        return ExecutionFailed(envelope)


__all__ = [
    "ExecutionCapture",
    "ExecutionFailed",
    "ExecutionSucceeded",
    "FAILURE_CODES_BY_PHASE",
    "FAILURE_SCHEMA",
    "FailureEnvelope",
    "FailureEnvelopeError",
    "FailureEnvelopeMismatch",
    "capture_execution",
    "compare_failure_envelopes",
    "failure_envelope_from_exception",
]

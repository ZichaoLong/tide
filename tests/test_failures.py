from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

from tide.builders import build_singleton
from tide.checkpoint import CheckpointError
from tide.engine import (
    DynamicReachabilityError,
    ExecutionContractError,
    LocalOperationError,
    SettleGraph,
    UnsupportedPlanError,
)
from tide.failures import (
    ExecutionFailed,
    ExecutionSucceeded,
    FAILURE_CODES_BY_PHASE,
    FAILURE_SCHEMA,
    FailureEnvelope,
    FailureEnvelopeError,
    FailureEnvelopeMismatch,
    capture_execution,
    compare_failure_envelopes,
    failure_envelope_from_exception,
)
from tide.plan import PlanValidationError, bind_dtypes
from tide.runtime import RuntimeConfigurationError, RuntimeUnavailableError


_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"


class FailureEnvelopeSchemaTests(unittest.TestCase):
    def test_every_v1_phase_and_code_is_constructible(self) -> None:
        expected = {
            "artifact": {"artifact.integrity", "artifact.schema"},
            "plan": {"plan.schema", "plan.topology", "plan.formula"},
            "binding": {"binding.invalid"},
            "capability": {"capability.unsupported"},
            "input": {"input.schema", "input.mask", "input.position"},
            "event": {"input.requested_k"},
            "state": {"state.schema", "state.owner_alias"},
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
            "runtime": {"runtime.configuration", "runtime.unavailable"},
        }
        self.assertEqual(
            {phase: set(codes) for phase, codes in FAILURE_CODES_BY_PHASE.items()},
            expected,
        )
        for phase, codes in expected.items():
            for code in codes:
                with self.subTest(phase=phase, code=code):
                    envelope = FailureEnvelope.create(phase, code)
                    self.assertEqual(envelope.error_schema, FAILURE_SCHEMA)
                    self.assertEqual(envelope.phase, phase)
                    self.assertEqual(envelope.codes, (code,))

    def test_codes_are_deduplicated_and_sorted(self) -> None:
        envelope = FailureEnvelope(
            FAILURE_SCHEMA,
            "plan",
            ("plan.topology", "plan.formula", "plan.topology"),
        )
        self.assertEqual(envelope.codes, ("plan.formula", "plan.topology"))

    def test_serialization_round_trip_is_exact_and_canonical(self) -> None:
        envelope = FailureEnvelope.create(
            "checkpoint", ["checkpoint.schema", "checkpoint.integrity"]
        )
        record = envelope.to_dict()
        self.assertEqual(
            record,
            {
                "error_schema": FAILURE_SCHEMA,
                "phase": "checkpoint",
                "codes": ["checkpoint.integrity", "checkpoint.schema"],
            },
        )
        self.assertEqual(FailureEnvelope.from_dict(record), envelope)
        self.assertEqual(
            FailureEnvelope.from_json(envelope.canonical_json()), envelope
        )
        self.assertEqual(
            envelope.canonical_json(),
            '{"codes":["checkpoint.integrity","checkpoint.schema"],'
            '"error_schema":"tide.failure.v1","phase":"checkpoint"}',
        )

    def test_invalid_schema_phase_code_and_serialized_shape_are_rejected(self) -> None:
        invalid_constructors = (
            lambda: FailureEnvelope("tide.failure.v2", "plan", ("plan.schema",)),
            lambda: FailureEnvelope(FAILURE_SCHEMA, "unknown", ("plan.schema",)),
            lambda: FailureEnvelope(FAILURE_SCHEMA, "plan", ()),
            lambda: FailureEnvelope(FAILURE_SCHEMA, "plan", "plan.schema"),
            lambda: FailureEnvelope(FAILURE_SCHEMA, "plan", ("plan.unknown",)),
            lambda: FailureEnvelope(FAILURE_SCHEMA, "plan", ("input.schema",)),
            lambda: FailureEnvelope(FAILURE_SCHEMA, "plan", (1,)),
            lambda: FailureEnvelope.create("plan", 1),  # type: ignore[arg-type]
            lambda: FailureEnvelope.from_dict(
                {
                    "error_schema": FAILURE_SCHEMA,
                    "phase": "plan",
                    "codes": ("plan.schema",),
                }
            ),
            lambda: FailureEnvelope.from_dict(
                {
                    "error_schema": FAILURE_SCHEMA,
                    "phase": "plan",
                    "codes": ["plan.schema"],
                    "extra": True,
                }
            ),
            lambda: FailureEnvelope.from_json("[]"),
            lambda: FailureEnvelope.from_json("not JSON"),
        )
        for index, constructor in enumerate(invalid_constructors):
            with self.subTest(index=index), self.assertRaises(FailureEnvelopeError):
                constructor()

    def test_duplicate_serialized_object_keys_are_rejected(self) -> None:
        duplicate_schema = (
            '{"error_schema":"tide.failure.v0",'
            '"error_schema":"tide.failure.v1",'
            '"phase":"input","codes":["input.schema"]}'
        )
        with self.assertRaises(FailureEnvelopeError):
            FailureEnvelope.from_json(duplicate_schema)

    def test_comparison_uses_schema_phase_and_complete_code_set(self) -> None:
        expected = FailureEnvelope.create("plan", "plan.schema")
        compare_failure_envelopes(
            expected, FailureEnvelope.create("plan", ["plan.schema", "plan.schema"])
        )
        for actual in (
            FailureEnvelope.create("plan", "plan.formula"),
            FailureEnvelope.create("input", "input.schema"),
        ):
            with self.subTest(actual=actual), self.assertRaises(
                FailureEnvelopeMismatch
            ):
                compare_failure_envelopes(expected, actual)
        with self.assertRaises(TypeError):
            compare_failure_envelopes(expected, expected.to_dict())  # type: ignore[arg-type]


class ExceptionMappingTests(unittest.TestCase):
    def test_fixed_exception_mappings(self) -> None:
        cases = (
            (
                UnsupportedPlanError("executor detail A"),
                "capability",
                "capability.unsupported",
            ),
            (
                DynamicReachabilityError("terminal detail A"),
                "execution",
                "execution.empty_terminal",
            ),
            (
                LocalOperationError("local formula detail A"),
                "execution",
                "execution.local_operation",
            ),
            (
                RuntimeConfigurationError("request detail A"),
                "runtime",
                "runtime.configuration",
            ),
            (
                RuntimeUnavailableError("probe detail A"),
                "runtime",
                "runtime.unavailable",
            ),
        )
        for error, phase, code in cases:
            with self.subTest(error=type(error).__name__):
                self.assertEqual(
                    failure_envelope_from_exception(error),
                    FailureEnvelope.create(phase, code),
                )

    def test_coarse_exception_mappings_require_explicit_codes(self) -> None:
        cases = (
            (PlanValidationError(["detail"]), "plan.formula"),
            (ExecutionContractError("detail"), "input.mask"),
            (CheckpointError("detail"), "checkpoint.compatibility"),
        )
        for error, code in cases:
            with self.subTest(error=type(error).__name__):
                with self.assertRaises(FailureEnvelopeError):
                    failure_envelope_from_exception(error)
                envelope = failure_envelope_from_exception(error, codes=code)
                self.assertEqual(envelope.codes, (code,))

    def test_plan_validation_can_report_multiple_codes_in_one_phase(self) -> None:
        envelope = failure_envelope_from_exception(
            PlanValidationError(["implementation text is irrelevant"]),
            codes=["plan.topology", "plan.formula", "plan.topology"],
        )
        self.assertEqual(envelope.phase, "plan")
        self.assertEqual(envelope.codes, ("plan.formula", "plan.topology"))

    def test_exception_text_changes_do_not_change_envelope(self) -> None:
        first = failure_envelope_from_exception(
            PlanValidationError(["first implementation message"]),
            codes="plan.schema",
        )
        second = failure_envelope_from_exception(
            PlanValidationError(["entirely different wording"]),
            codes="plan.schema",
        )
        self.assertEqual(first, second)

    def test_wrong_explicit_mapping_is_rejected(self) -> None:
        cases = (
            (UnsupportedPlanError("detail"), "execution.local_operation"),
            (PlanValidationError(["detail"]), "input.schema"),
            (ExecutionContractError("detail"), "checkpoint.schema"),
            (CheckpointError("detail"), "plan.schema"),
        )
        for error, code in cases:
            with self.subTest(error=type(error).__name__, code=code), self.assertRaises(
                FailureEnvelopeError
            ):
                failure_envelope_from_exception(error, codes=code)
        with self.assertRaises(FailureEnvelopeError):
            failure_envelope_from_exception(
                PlanValidationError(["detail"]),
                codes=["plan.schema", "binding.invalid"],
            )
        with self.assertRaises(TypeError):
            failure_envelope_from_exception(ValueError("not a contract exception"))


class CaptureExecutionTests(unittest.TestCase):
    def test_success_has_an_explicit_result_type(self) -> None:
        result = capture_execution(lambda: {"answer": 42})
        self.assertIsInstance(result, ExecutionSucceeded)
        assert isinstance(result, ExecutionSucceeded)
        self.assertEqual(result.value, {"answer": 42})

    def test_failure_has_an_explicit_result_type(self) -> None:
        def fail() -> None:
            raise ExecutionContractError("wording is not inspected")

        result = capture_execution(fail, codes="state.schema")
        self.assertIsInstance(result, ExecutionFailed)
        assert isinstance(result, ExecutionFailed)
        self.assertEqual(
            result.envelope, FailureEnvelope.create("state", "state.schema")
        )

    def test_local_operation_failure_has_a_fixed_capture_mapping(self) -> None:
        def fail() -> None:
            raise LocalOperationError("implementation wording is not inspected")

        result = capture_execution(fail)
        self.assertIsInstance(result, ExecutionFailed)
        assert isinstance(result, ExecutionFailed)
        self.assertEqual(
            result.envelope,
            FailureEnvelope.create("execution", "execution.local_operation"),
        )

    def test_low_precision_typed_plan_construction_is_a_capability_failure(
        self,
    ) -> None:
        logical = build_singleton(d_model=2)
        for dtype in ("float16", "bfloat16"):
            with self.subTest(dtype=dtype):
                typed = bind_dtypes(
                    logical,
                    hidden=dtype,
                    parameter=dtype,
                    state=dtype,
                    readout=dtype,
                )
                result = capture_execution(lambda: SettleGraph(typed))
                self.assertIsInstance(result, ExecutionFailed)
                assert isinstance(result, ExecutionFailed)
                self.assertEqual(
                    result.envelope,
                    FailureEnvelope.create(
                        "capability", "capability.unsupported"
                    ),
                )

    def test_coarse_capture_still_requires_code(self) -> None:
        def fail() -> None:
            raise CheckpointError("detail")

        with self.assertRaises(FailureEnvelopeError):
            capture_execution(fail)

    def test_unexpected_exception_propagates(self) -> None:
        sentinel = KeyError("implementation bug")

        def fail() -> None:
            raise sentinel

        with self.assertRaises(KeyError) as raised:
            capture_execution(fail)
        self.assertIs(raised.exception, sentinel)


class TorchlessFailureMappingTests(unittest.TestCase):
    def test_runtime_failure_mapping_does_not_import_torch(self) -> None:
        script = r"""
import sys

from tide.failures import (
    ExecutionFailed,
    capture_execution,
    failure_envelope_from_exception,
)
from tide.runtime import RuntimeConfigurationError, RuntimeUnavailableError

cases = (
    (RuntimeConfigurationError("invalid request"), "runtime.configuration"),
    (RuntimeUnavailableError("failed probe"), "runtime.unavailable"),
)
for error, expected_code in cases:
    envelope = failure_envelope_from_exception(error)
    assert envelope.phase == "runtime"
    assert envelope.codes == (expected_code,)

    def fail(error=error):
        raise error

    captured = capture_execution(fail)
    assert isinstance(captured, ExecutionFailed)
    assert captured.envelope == envelope

assert "torch" not in sys.modules
"""
        with tempfile.TemporaryDirectory() as temporary_root:
            blocker_root = pathlib.Path(temporary_root)
            (blocker_root / "torch.py").write_text(
                "raise RuntimeError('torch must stay lazy')\n", encoding="utf-8"
            )
            environment = dict(os.environ)
            environment["PYTHONPATH"] = os.pathsep.join(
                (str(blocker_root), str(_SOURCE_ROOT))
            )
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            environment["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
            completed = subprocess.run(
                [sys.executable, "-c", script],
                cwd=str(_REPOSITORY_ROOT),
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(
            completed.returncode,
            0,
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )


if __name__ == "__main__":
    unittest.main()

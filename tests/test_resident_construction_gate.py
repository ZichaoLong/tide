"""Construction witnesses cannot become numerical/profile qualification."""
from contextlib import ExitStack, redirect_stderr
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import profile_device_control as profile
import verify_device_control as gate


class ConstructionGate(unittest.TestCase):
    check = "weighted-budget-construction"

    def run_witness(self, exit_code, expected_marker=True):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            build, out = root / "build", root / "out"
            build.mkdir()
            name, dtypes = gate.CHECKS[self.check]
            self.assertEqual(dtypes, ("float32",))
            (build / name).touch()
            (build / "control-build.json").write_text(json.dumps(dict(
                backend="npu", component_sha256="component", core=dict(cpp_source_sha256="core"),
                binary_sha256={name: "binary"})))
            for field, value in (("component_hash", "component"), ("source_hash", "core"),
                                 ("digest", "binary"), ("source_state", ("fixed", ""))):
                stack.enter_context(patch.object(gate, field, return_value=value))

            def execute(command, **kwargs):
                self.assertEqual(command, [str(build / name), "--device=npu:0", "--dtype=float32"])
                text = gate.MARKERS[self.check] if expected_marker else "unrelated capacity refusal"
                kwargs["stdout"].write(text + "\n")
                return subprocess.CompletedProcess(command, exit_code)

            stack.enter_context(patch.object(gate.subprocess, "run", side_effect=execute))
            stack.enter_context(patch.object(sys, "argv", ["gate", "--build-dir", str(build),
                "--output-dir", str(out), "--device", "npu:0", "--checks", self.check]))
            if exit_code == 0 and expected_marker:
                gate.main()
            else:
                with self.assertRaises(RuntimeError):
                    gate.main()
            return json.loads((out / "result.json").read_text())

    def test_success_is_only_a_construction_witness(self):
        report = self.run_witness(0)
        self.assertEqual(report["state"], "passed")
        case, = report["cases"]
        self.assertEqual(case["evidence_kind"], "construction")
        self.assertIn("no numerical execution", case["scope"])
        self.assertEqual(case["exit_code"], 0)

    def test_printed_marker_cannot_promote_nonzero_exit(self):
        report = self.run_witness(2)
        self.assertEqual(report["state"], "failed")
        self.assertEqual(report["cases"][0]["exit_code"], 2)

    def test_unrelated_refusal_cannot_satisfy_the_witness(self):
        report = self.run_witness(0, expected_marker=False)
        self.assertEqual(report["state"], "failed")

    def test_profile_refuses_before_reading_build_or_launching_tools(self):
        stderr = io.StringIO()
        with patch.object(sys, "argv", ["profile", "--build-dir", "missing-build",
                "--output-dir", "unused-output", "--device", "npu:0", "--check", self.check]), \
                patch.object(profile.subprocess, "run") as launch, \
                patch.object(profile.Path, "read_text") as read, redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as stopped:
                profile.main()
        self.assertEqual(stopped.exception.code, 2)
        self.assertIn("construction-only", stderr.getvalue())
        launch.assert_not_called()
        read.assert_not_called()


if __name__ == "__main__":
    unittest.main()

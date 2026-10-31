"""Independent component failures remain failures while later cases can run."""
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import verify_device_control as gate


class GateFailures(unittest.TestCase):
    def check_inventory(self, keep_going, timeout=False):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            build, out = root / "build", root / "result"
            build.mkdir()
            names = [gate.CHECKS[name][0] for name in ("control", "queue")]
            for name in names:
                (build / name).touch()
            (build / "control-build.json").write_text(json.dumps(dict(
                backend="cuda", component_sha256="component",
                core=dict(cpp_source_sha256="core"), binary_sha256={name: "binary" for name in names})))
            for name, value in (("component_hash", "component"), ("source_hash", "core"),
                                ("digest", "binary"), ("source_state", ("commit", ""))):
                stack.enter_context(patch.object(gate, name, return_value=value))

            def execute(command, **kwargs):
                if Path(command[0]).name == names[0]:
                    if timeout:
                        raise subprocess.TimeoutExpired(command, 120)
                    kwargs["stdout"].write("original control failure\n")
                    return subprocess.CompletedProcess(command, 2)
                kwargs["stdout"].write(gate.MARKERS["queue"] + "\n")
                return subprocess.CompletedProcess(command, 0)

            stack.enter_context(patch.object(gate.subprocess, "run", side_effect=execute))
            argv = ["gate", "--build-dir", str(build), "--output-dir", str(out),
                    "--device", "cuda:0", "--checks", "control", "queue"]
            if keep_going:
                argv.append("--keep-going")
            stack.enter_context(patch.object(sys, "argv", argv))
            with self.assertRaises(RuntimeError):
                gate.main()
            result = json.loads((out / "result.json").read_text())
            self.assertEqual(result["state"], "failed")
            self.assertEqual(result["cases"][0]["exit_code"], 124 if timeout else 2)
            self.assertEqual([row["state"] for row in result["cases"]],
                             ["failed", "passed"] if keep_going else ["failed"])

    def test_default_retains_fail_fast(self):
        self.check_inventory(False)

    def test_keep_going_never_promotes_failure(self):
        self.check_inventory(True)

    def test_timeout_is_failed_and_later_inventory_runs(self):
        self.check_inventory(True, timeout=True)


if __name__ == "__main__":
    unittest.main()

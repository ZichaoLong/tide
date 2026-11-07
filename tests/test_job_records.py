"""Lifecycle tests use tiny CPU subprocesses, no systemd or device allocation."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from tide_lab.supervisor import atomic_json, file_hash, finalize, identity_alive, read_json, source_inventory, verify_source


ROOT = Path(__file__).resolve().parents[1]


class JobLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "frozen"
        (self.source / "tide_lab").mkdir(parents=True)
        shutil.copyfile(ROOT / "tide_lab/supervisor.py", self.source / "tide_lab/supervisor.py")
        atomic_json(self.source / ".tide-source.json", {
            "schema": "tide-frozen-source-v1", "commit": "a" * 40,
            "content": source_inventory(self.source)})
        self.output = self.root / "job"
        self.output.mkdir()
        self.environment = self.root / "env.sh"
        self.environment.write_text(f"export PATH={Path(sys.executable).parent}:\"$PATH\"\n")

    def tearDown(self):
        self.temporary.cleanup()

    def spec(self, command, timeout=10):
        value = {
            "schema": "tide-job-v1", "job_id": "unit-test", "source": {
                "path": str(self.source), "commit": "a" * 40,
                "tree_sha256": read_json(self.source / ".tide-source.json")["content"]["tree_sha256"]},
            "device": "cpu", "environment_script": {"path": str(self.environment), "sha256": file_hash(self.environment)},
            "environment": {"PYTHONDONTWRITEBYTECODE": "1"}, "command": command,
            "resources": {"timeout_seconds": timeout}, "inputs": [], "queue_helper": None}
        record = self.output / "job.json"
        atomic_json(record, value)
        return record

    def launch(self, record):
        return subprocess.Popen([sys.executable, str(self.source / "tide_lab/supervisor.py"),
                                 "run", "--record", str(record)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def wait_worker(self, process):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            state = read_json(self.output / "worker-state.json", {})
            if state.get("state") == "running":
                return state
            if process.poll() is not None:
                self.fail(f"supervisor exited before worker ran: {process.communicate()}")
            time.sleep(.02)
        self.fail("worker did not start")

    def test_argv_is_not_shell_text_and_success_is_inner_exit(self):
        literal = "literal $HOME; $(touch SHOULD_NOT_EXIST)"
        record = self.spec([sys.executable, "-c", "import sys; print(sys.argv[1])", literal])
        process = self.launch(record)
        stdout, stderr = process.communicate(timeout=10)
        self.assertEqual(process.returncode, 0, stderr)
        self.assertIn(literal, stdout)
        self.assertFalse((self.source / "SHOULD_NOT_EXIST").exists())
        state = read_json(self.output / "state.json")
        self.assertEqual(state["state"], "completed")
        self.assertEqual(state["workload_exit_code"], 0)
        self.assertTrue(state["termination_verified"])

    def test_nonzero_workload_is_preserved(self):
        process = self.launch(self.spec([sys.executable, "-c", "raise SystemExit(7)"]))
        _, stderr = process.communicate(timeout=10)
        self.assertEqual(process.returncode, 7, stderr)
        state = read_json(self.output / "state.json")
        self.assertEqual(state["state"], "failed")
        self.assertEqual(state["workload_exit_code"], 7)

    def test_changed_source_is_rejected_before_workload(self):
        record = self.spec([sys.executable, "-c", "raise SystemExit(0)"])
        (self.source / "new-input.json").write_text("{}")
        process = self.launch(record)
        process.communicate(timeout=10)
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse((self.output / "worker-state.json").exists())
        self.assertIn("frozen source differs", read_json(self.output / "state.json")["error"])

    def test_signal_stop_terminates_descendant_and_retains_state(self):
        process = self.launch(self.spec([sys.executable, "-c", "import time; time.sleep(60)"]))
        worker = self.wait_worker(process)
        atomic_json(self.output / "stop-request.json", {"reason": "unit test"})
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=10)
        self.assertFalse(identity_alive(worker["process"]))
        state = read_json(self.output / "state.json")
        self.assertEqual(state["state"], "stopped")
        self.assertTrue(state["termination_verified"])

    def test_bounded_timeout_is_distinct_from_failure(self):
        process = self.launch(self.spec([sys.executable, "-c", "import time; time.sleep(60)"], timeout=1))
        process.communicate(timeout=10)
        self.assertEqual(process.returncode, 124)
        state = read_json(self.output / "state.json")
        self.assertEqual(state["state"], "timed_out")
        self.assertTrue(state["termination_verified"])

    def test_external_symlink_cannot_claim_frozen_inputs(self):
        (self.source / "mutable").symlink_to(self.environment)
        with self.assertRaisesRegex(ValueError, "external or broken symlink"):
            verify_source(self.source)

    def test_systemd_timeout_cannot_be_reported_as_completed(self):
        record = self.spec([sys.executable, "-c", "pass"])
        atomic_json(self.output / "state.json", {"state": "completed", "control_group": "/test"})
        with mock.patch.dict(os.environ, {"SERVICE_RESULT": "timeout", "EXIT_CODE": "killed", "EXIT_STATUS": "9"}):
            with mock.patch("tide_lab.supervisor.cgroup_processes", return_value=[]):
                finalize(record)
        state = read_json(self.output / "state.json")
        self.assertEqual(state["state"], "timed_out")
        self.assertTrue(state["termination_verified"])

    def test_systemd_oom_after_uncatchable_exit_has_terminal_record(self):
        record = self.spec([sys.executable, "-c", "pass"])
        atomic_json(self.output / "state.json", {"state": "running", "control_group": "/test"})
        with mock.patch.dict(os.environ, {"SERVICE_RESULT": "oom-kill", "EXIT_CODE": "killed", "EXIT_STATUS": "9"}):
            with mock.patch("tide_lab.supervisor.cgroup_processes", return_value=[]):
                finalize(record)
        state = read_json(self.output / "state.json")
        self.assertEqual(state["state"], "failed")
        self.assertEqual(state["systemd"]["SERVICE_RESULT"], "oom-kill")


if __name__ == "__main__":
    unittest.main()

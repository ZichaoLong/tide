"""Real process lifecycle checks, independent of Torch or accelerator hardware."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/container_job.py"


class ContainerJobTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "source"
        self.repo.mkdir()
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        subprocess.run(["git", "-C", str(self.repo), "-c", "user.name=Test", "-c",
                        "user.email=test@localhost", "commit", "-qm", "fixture", "--allow-empty"], check=True)
        self.out = self.root / "job"

    def tearDown(self):
        if (self.out / "status.json").exists():
            self.call("stop", check=False)
        self.temp.cleanup()

    def call(self, action, *args, check=True):
        return subprocess.run([sys.executable, str(SCRIPT), action, "--output-dir", str(self.out), *args],
                              capture_output=True, text=True, check=check, timeout=20)

    def start(self, code, timeout=5):
        return self.call("start", "--cwd", str(self.repo), "--timeout", str(timeout),
                         "--grace", ".1", "--", sys.executable, "-c", code)

    def terminal(self):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            record = json.loads(self.call("status").stdout)
            if record["state"] in {"passed", "failed", "cancelled"}:
                self.assertEqual(record["observed_live_workload_pids"], [])
                return record
            time.sleep(.05)
        self.fail("no terminal record")

    def test_submitter_exit_and_success(self):
        self.start("import time; time.sleep(.5); print('completed', flush=True)")
        self.assertEqual(self.terminal()["state"], "passed")
        self.assertEqual((self.out / "task.log").read_text().strip(), "completed")

    def test_failure_preserves_exit_code(self):
        self.start("raise SystemExit(7)")
        self.assertEqual(self.terminal()["exit_code"], 7)

    def test_timeout_kills_term_ignoring_descendants(self):
        code = ("import subprocess,sys,time,signal; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "subprocess.Popen([sys.executable,'-c',"
                "'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)']); "
                "time.sleep(30)")
        self.start(code, timeout=.5)
        record = self.terminal()
        self.assertEqual(record["state"], "failed")
        self.assertEqual(record["exit_code"], 124)

    def test_stop_checks_whole_process_group(self):
        self.start("import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); time.sleep(30)")
        time.sleep(.15)
        record = json.loads(self.call("stop").stdout)
        self.assertEqual(record["state"], "cancelled")
        self.assertEqual(record["observed_live_workload_pids"], [])
        self.call("stop")

    def test_no_output_reuse_or_dirty_source(self):
        self.start("print('first')")
        self.terminal()
        previous = (self.out / "status.json").read_bytes()
        with self.assertRaises(subprocess.CalledProcessError):
            self.start("print('second')")
        self.assertEqual((self.out / "status.json").read_bytes(), previous)
        (self.repo / "unsaved").write_text("user change")
        self.assertNotEqual(self.call("start", "--cwd", str(self.repo), "--timeout", "5",
            "--", "true", check=False).returncode, 0)


if __name__ == "__main__":
    unittest.main()

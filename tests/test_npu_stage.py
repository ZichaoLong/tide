"""Safety-critical phase cleanup, exercised with real CPU descendants only."""
from pathlib import Path
import sys
import tempfile
import unittest

from scripts.npu_learning_stage import bounded_command, ChildFailure, ChildTimeout
from tide_lab.supervisor import group_processes


class PhaseCleanupTests(unittest.TestCase):
    def test_success_records_verified_exit(self):
        with tempfile.TemporaryDirectory() as root:
            record = {}
            bounded_command([sys.executable, '-c', 'print("finished")'],
                            Path(root)/'log', 3, record, lambda: None, .2)
            self.assertEqual(record['exit_code'], 0)
            self.assertTrue(record['termination_verified'])
            self.assertEqual(record['state'], 'completed')

    def test_timeout_kills_term_ignoring_grandchild(self):
        with tempfile.TemporaryDirectory() as root:
            child = "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)"
            parent = ('import subprocess,sys,time;'
                      f'subprocess.Popen([sys.executable,"-c",{child!r}]);time.sleep(30)')
            record = {}
            with self.assertRaises(ChildTimeout):
                bounded_command([sys.executable, '-c', parent], Path(root)/'log',
                                .5, record, lambda: None, .2)
            self.assertTrue(record['termination_verified'])
            self.assertEqual(group_processes(record['process_group']), [])
            self.assertEqual(record['state'], 'failed')

    def test_outer_zero_does_not_hide_live_child(self):
        with tempfile.TemporaryDirectory() as root:
            parent = ('import subprocess,sys;'
                      'subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"])')
            record = {}
            with self.assertRaisesRegex(ChildFailure, 'live descendants'):
                bounded_command([sys.executable, '-c', parent], Path(root)/'log',
                                3, record, lambda: None, .2)
            self.assertEqual(record['exit_code'], 0)
            self.assertEqual(record['state'], 'failed')
            self.assertTrue(record['termination_verified'])


if __name__ == '__main__':
    unittest.main()

"""Lean subprocess lifecycle cases; run without pytest's Torch conftest."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from measurement_lanes import node_memory
from measurement_group import run_group


@unittest.skipUnless(shutil.which('numactl') and Path('/proc/self/numa_maps').exists(),
                     'Linux NUMA tools required for real child lifecycle tests')
class ProcessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        available = set(os.sched_getaffinity(0))
        self.lanes = []
        for node, details in node_memory().items():
            local = available.intersection(details['cpus'])
            if local:
                self.lanes.append(dict(name='lane' + str(node), cpus=[min(local)],
                    memory_nodes=[node], memory_bytes=64*1024**2))
        if not self.lanes:
            self.skipTest('no allowed NUMA CPU')

    def case(self, lane, code, timeout=10):
        return dict(lane=lane, command=[sys.executable, '-c', code], cwd=str(self.root),
                    environment=dict(os.environ), timeout_seconds=timeout)

    def test_actual_pair_overlaps_and_finishes_empty(self):
        if len(self.lanes) < 2:
            self.skipTest('two NUMA nodes required')
        cases = []
        for lane in self.lanes[:2]:
            code = ('import time,json;from pathlib import Path;'
                    'start=time.monotonic();buffer=bytearray(2*1024**2);time.sleep(5.5);'
                    f'Path({lane["name"]!r}).write_text(json.dumps([start,time.monotonic()]))')
            cases.append(self.case(lane, code))
        result = run_group(cases, output=self.root/'pair', reserve_bytes=1024**2, interval=.1)
        times = [json.loads((self.root / lane['name']).read_text()) for lane in self.lanes[:2]]
        self.assertLess(max(t[0] for t in times), min(t[1] for t in times))
        self.assertEqual(result['state'], 'passed')
        self.assertTrue(all(not row['remaining_group_pids'] for row in result['cases']))
        self.assertTrue(all(any(sample['processes'] for sample in row['placement_samples'])
                            for row in result['cases']))

    def test_timeout_reaps_grandchild(self):
        code = ('import subprocess,sys,time;'
                'subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"]);'
                'time.sleep(60)')
        with self.assertRaises(TimeoutError):
            run_group([self.case(self.lanes[0], code, timeout=.3)],
                      output=self.root/'timeout', reserve_bytes=1024**2, interval=.1)
        result = json.loads((self.root/'timeout/result.json').read_text())
        self.assertEqual(result['state'], 'failed')
        self.assertEqual(result['cases'][0]['remaining_group_pids'], [])

    def test_exited_adopted_grandchild_is_reaped_before_leak_check(self):
        code = ('import os,time\n'
                'if os.fork()==0: os._exit(0)\n'
                'time.sleep(.2)\n'
                'os._exit(0)\n')
        result = run_group([self.case(self.lanes[0], code)],
            output=self.root/'zombie', reserve_bytes=1024**2, interval=.1)
        row = result['cases'][0]
        self.assertEqual(result['state'], 'passed')
        self.assertEqual([v['exit_code'] for v in row['reaped_descendants']], [0])
        self.assertEqual(row['remaining_group_pids'], [])

    def test_natural_helper_exit_has_a_bounded_unsignalled_grace(self):
        code = ('import os,time\n'
                'if os.fork()==0:\n'
                ' time.sleep(.5)\n'
                ' os._exit(0)\n'
                'os._exit(0)\n')
        result = run_group([self.case(self.lanes[0], code)],
            output=self.root/'teardown', reserve_bytes=1024**2, interval=.1)
        row = result['cases'][0]
        self.assertEqual(result['state'], 'passed')
        self.assertGreater(row['natural_teardown_elapsed_seconds'], .1)
        self.assertLessEqual(row['natural_teardown_elapsed_seconds'], 2.)
        self.assertEqual([v['exit_code'] for v in row['reaped_descendants']], [0])

    def test_live_helper_leak_still_fails_and_is_cleaned(self):
        code = ('import os,time\n'
                'if os.fork()==0: time.sleep(60)\n'
                'os._exit(0)\n')
        with self.assertRaisesRegex(RuntimeError, 'remaining live group'):
            run_group([self.case(self.lanes[0], code)], output=self.root/'leak',
                      reserve_bytes=1024**2, interval=.1, teardown_seconds=.2)
        result = json.loads((self.root/'leak/result.json').read_text())
        self.assertEqual(result['state'], 'failed')
        self.assertTrue(result['cases'][0]['remaining_after_parent_exit'])
        self.assertEqual(result['cases'][0]['remaining_group_pids'], [])

    def test_failed_adopted_descendant_is_not_hidden_by_parent_success(self):
        code = ('import os,time\n'
                'if os.fork()==0: os._exit(7)\n'
                'time.sleep(.2)\n'
                'os._exit(0)\n')
        with self.assertRaisesRegex(RuntimeError, 'adopted descendant failed'):
            run_group([self.case(self.lanes[0], code)], output=self.root/'orphan-failure',
                      reserve_bytes=1024**2, interval=.1)
        result = json.loads((self.root/'orphan-failure/result.json').read_text())
        self.assertEqual(result['state'], 'failed')
        self.assertEqual([v['exit_code'] for v in result['cases'][0]['reaped_descendants']], [7])
        self.assertEqual(result['cases'][0]['remaining_group_pids'], [])

    def test_failed_companion_cancels_other_owned_group(self):
        if len(self.lanes) < 2:
            self.skipTest('two NUMA nodes required')
        cases = [self.case(self.lanes[0], 'import sys;sys.exit(7)'),
                 self.case(self.lanes[1], 'import time;time.sleep(60)')]
        with self.assertRaises(RuntimeError):
            run_group(cases, output=self.root/'failed', reserve_bytes=1024**2, interval=.1)
        result = json.loads((self.root/'failed/result.json').read_text())
        self.assertEqual(result['state'], 'failed')
        self.assertEqual(result['cases'][0]['exit_code'], 7)
        self.assertEqual(result['cases'][1]['state'], 'cancelled')
        self.assertTrue(all(not row['remaining_group_pids'] for row in result['cases']))

    def test_rss_breach_fails_and_cleans_up(self):
        lane = dict(self.lanes[0], memory_bytes=24*1024**2)
        code = 'import time;buffer=bytearray(40*1024**2);time.sleep(60)'
        with self.assertRaises(MemoryError):
            run_group([self.case(lane, code)], output=self.root/'memory',
                      reserve_bytes=1024**2, interval=.1)
        result = json.loads((self.root/'memory/result.json').read_text())
        self.assertEqual(result['cases'][0]['remaining_group_pids'], [])


if __name__ == '__main__':
    unittest.main()

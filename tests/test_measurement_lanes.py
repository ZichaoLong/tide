"""Resource admission and real process cleanup, without Torch or an accelerator."""
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from measurement_lanes import admit, check_placement, node_memory, process_placement
from measurement_group import run_group


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.host = dict(effective_cpu_ids=list(range(16)), cpu_budget=8,
                         memory_budget_bytes=1000)
        self.nodes = {0: dict(cpus=list(range(8)), available_estimate_bytes=600),
                      1: dict(cpus=list(range(8, 16)), available_estimate_bytes=600)}
        self.lanes = [dict(name='cpu', cpus=[0, 1, 2], memory_nodes=[0], memory_bytes=300),
                      dict(name='npu', cpus=[8, 9, 10], memory_nodes=[1], memory_bytes=400)]

    def run_admission(self):
        return admit(self.lanes, reserve_bytes=100, host=self.host, nodes=self.nodes)

    def test_combined_budget_and_single_reserve(self):
        result = self.run_admission()
        self.assertEqual(result['reserved_memory_bytes'], 800)
        self.assertEqual(result['reserved_cpu_count'], 8)

    def test_two_individually_fitting_lanes_can_exceed_shared_memory(self):
        for lane in self.lanes:
            lane['memory_bytes'] = 500
            admit([lane], reserve_bytes=100, host=self.host, nodes=self.nodes)
        with self.assertRaises(MemoryError):
            self.run_admission()

    def test_cpu_mask_count_includes_controller(self):
        self.lanes[1]['cpus'].append(11)
        with self.assertRaisesRegex(ValueError, 'combined CPU'):
            self.run_admission()

    def test_disjoint_cpus_do_not_make_shared_memory_nodes_valid(self):
        self.lanes[1].update(cpus=[3, 4], memory_nodes=[0])
        with self.assertRaisesRegex(ValueError, 'overlap'):
            self.run_admission()

    def test_rejects_nonlocal_and_unavailable_cpus(self):
        for cpus in ([7], [16]):
            self.lanes[1]['cpus'] = cpus
            with self.assertRaises(ValueError):
                self.run_admission()

    def test_node_shortage_cannot_borrow_other_lane_memory(self):
        self.nodes[1]['available_estimate_bytes'] = 399
        with self.assertRaisesRegex(MemoryError, 'selected-node'):
            self.run_admission()

    def test_invalid_and_duplicate_requests(self):
        original = copy.deepcopy(self.lanes)
        for key, value in [('name', '../bad'), ('name', 'cpu'), ('memory_bytes', True),
                           ('memory_nodes', [2]), ('cpus', []), ('cpus', [8, 8])]:
            self.lanes = copy.deepcopy(original)
            self.lanes[1][key] = value
            with self.assertRaises(ValueError):
                self.run_admission()

    def test_thread_and_anonymous_memory_escape_are_rejected(self):
        lane = self.lanes[0]
        good = dict(thread_cpu_masks=[[0, 1], [2]], anonymous_pages_by_node={0: 10})
        check_placement(good, lane)
        for bad in [dict(good, thread_cpu_masks=[[0, 3]]),
                    dict(good, anonymous_pages_by_node={1: 1})]:
            with self.assertRaises(RuntimeError):
                check_placement(bad, lane)

    def test_shared_file_pages_excluded_but_private_mapping_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); task = root / '42/task/42'; task.mkdir(parents=True)
            (task / 'status').write_text('Cpus_allowed_list:\t0-2\n')
            (root / '42/numa_maps').write_text(
                '100 bind:0 file=/lib.so anon=1 N1=10\n'
                '200 bind:0 heap anon=4 N0=4\n'
                '300 bind:0 anon=2 N1=2\n')
            observed = process_placement(42, root)
            self.assertEqual(observed['anonymous_pages_by_node'], {0: 4, 1: 2})
            with self.assertRaises(RuntimeError):
                check_placement(observed, self.lanes[0])


class ProcExitTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.task = self.root / '42/task/42'
        self.task.mkdir(parents=True)
        (self.task / 'status').write_text('Cpus_allowed_list:\t0-2\n')
        (self.root / '42/numa_maps').write_text('200 bind:0 heap anon=4 N0=4\n')

    def test_disappearing_process_during_task_enumeration(self):
        for error in (FileNotFoundError, ProcessLookupError):
            with self.subTest(error=error), patch.object(Path, 'iterdir', side_effect=error):
                self.assertIsNone(process_placement(42, self.root))

    def test_disappearing_thread_or_process_during_read(self):
        original = Path.read_text
        for name in ('status', 'numa_maps'):
            for error in (FileNotFoundError, ProcessLookupError):
                def read(path, *args, **kwargs):
                    if path.name == name:
                        raise error()
                    return original(path, *args, **kwargs)
                with self.subTest(name=name, error=error), patch.object(Path, 'read_text', read):
                    observed = process_placement(42, self.root)
                    if name == 'numa_maps':
                        self.assertIsNone(observed)
                    else:
                        self.assertEqual(observed['thread_cpu_masks'], [])
                        self.assertEqual(observed['anonymous_pages_by_node'], {0: 4})

    def test_unrelated_proc_errors_are_not_hidden(self):
        for method in ('iterdir', 'read_text'):
            for error in (PermissionError, OSError):
                with self.subTest(method=method, error=error), \
                        patch.object(Path, method, side_effect=error):
                    with self.assertRaises(error):
                        process_placement(42, self.root)


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

"""Independent timeline witnesses: overlap, uncorrelated nodes and bad exports."""
import hashlib
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from inspect_cuda_timeline import inspect


class TimelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "trace.sqlite"
        self.pid = 7 << 24
        with sqlite3.connect(self.path) as db:
            db.executescript("""
                create table StringIds(id integer primary key, value text);
                create table CUPTI_ACTIVITY_KIND_KERNEL(start integer, end integer,
                    graphNodeId integer, shortName integer, streamId integer,
                    correlationId integer, globalPid integer, deviceId integer);
                create table CUPTI_ACTIVITY_KIND_RUNTIME(nameId integer,
                    correlationId integer, globalTid integer);
                insert into StringIds values(1, 'tide_closure'), (2, 'ordinary'),
                    (3, 'cudaGraphLaunch_v10000'), (4, 'cudaStreamSynchronize');
            """)

    def rows(self, rows):
        with sqlite3.connect(self.path) as db:
            db.executemany("insert into CUPTI_ACTIVITY_KIND_KERNEL values(?,?,?,?,?,?,?,?)",
                           [(*row, self.pid, 0) for row in rows])

    def test_overlap_is_union_and_missing_correlations_are_not_guessed(self):
        self.rows([(1000, 6000, 11, 1, 1, 70), (4000, 8000, 12, 1, 2, 99),
                   (10000, 13000, 11, 1, 1, 99), (20000, 30000, 0, 2, 1, 80)])
        with sqlite3.connect(self.path) as db:
            db.executemany("insert into CUPTI_ACTIVITY_KIND_RUNTIME values(?,?,?)",
                           [(3, 70, self.pid+1), (4, 70, self.pid+2),
                            (3, 99, (8 << 24)+1)])
        before = hashlib.sha256(self.path.read_bytes()).hexdigest()
        result = inspect(self.path)
        self.assertEqual(result["kernel_activities"], 4)
        self.assertEqual(result["graph_node_kernel_activities"], 3)
        self.assertEqual(result["distinct_executed_graph_nodes"], 2)
        self.assertEqual(result["graph_node_kernel_union_us"], 10)
        self.assertEqual(result["host_correlated_graph_activities"], 1)
        self.assertEqual(result["graph_launch_calls"], 1)
        self.assertEqual(result["kernels"][0]["summed_duration_us"], 12)
        self.assertEqual(len(result["focus_sample"]), 3)
        self.assertEqual(result["database_sha256"], before)
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), before)

    def test_non_graph_kernels_do_not_become_resident(self):
        self.rows([(0, 5000, 0, 2, 1, 70)])
        result = inspect(self.path)
        self.assertEqual(result["graph_node_kernel_activities"], 0)
        self.assertEqual(result["graph_node_kernel_union_us"], 0)
        self.assertEqual(result["graph_launch_calls"], 0)
        self.assertEqual(result["focus_sample"], [])

    def test_empty_activity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no actual CUDA"):
            inspect(self.path)

    def test_reversed_interval_is_rejected(self):
        self.rows([(4, 3, 11, 1, 1, 70)])
        with self.assertRaisesRegex(ValueError, "interval"):
            inspect(self.path)

    def test_unresolved_kernel_name_is_rejected(self):
        self.rows([(0, 3, 11, 99, 1, 70)])
        with self.assertRaisesRegex(ValueError, "no recorded name"):
            inspect(self.path)

    def test_duplicate_node_ids_from_other_processes_are_not_conflated(self):
        self.rows([(0, 3, 11, 1, 1, 70), (5, 9, 11, 1, 1, 70)])
        with sqlite3.connect(self.path) as db:
            db.execute("update CUPTI_ACTIVITY_KIND_KERNEL set globalPid=? where start=5",
                       (8 << 24,))
        with self.assertRaisesRegex(ValueError, "single-process"):
            inspect(self.path)

    def test_multiple_devices_need_separate_analysis(self):
        self.rows([(0, 3, 11, 1, 1, 70), (5, 9, 12, 1, 1, 71)])
        with sqlite3.connect(self.path) as db:
            db.execute("update CUPTI_ACTIVITY_KIND_KERNEL set deviceId=1 where start=5")
        with self.assertRaisesRegex(ValueError, "single-device"):
            inspect(self.path)


if __name__ == "__main__":
    unittest.main()

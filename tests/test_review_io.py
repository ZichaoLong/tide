import json
from pathlib import Path
import tempfile
import unittest

from scripts.review_io import GRAPH_SCHEMA, write_review_json


class ReviewIsolationTests(unittest.TestCase):
    def test_shadowed_future_run_path_is_rejected_before_directory_creation(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            future = root/'runs/future-candidate/future-run/state.json'
            with self.assertRaisesRegex(ValueError, 'research/map.json'):
                write_review_json(future, dict(schema=GRAPH_SCHEMA, nodes=[]), root=root)
            self.assertFalse((root/'runs').exists())
            with self.assertRaisesRegex(ValueError, 'workload paths are read-only'):
                write_review_json(future, dict(schema='review'), root=root)
            self.assertFalse((root/'runs').exists())

    def test_map_and_evidence_allowed_but_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            graph = dict(schema=GRAPH_SCHEMA, nodes=[])
            write_review_json('research/map.json', graph, root=root)
            self.assertEqual(json.loads((root/'research/map.json').read_text()), graph)
            write_review_json('docs/evidence/review.json', dict(state='checked'), root=root)
            (root/'runs').mkdir()
            (root/'docs/evidence/escape.json').symlink_to(root/'runs/state.json')
            with self.assertRaises(ValueError):
                write_review_json('docs/evidence/escape.json', dict(state='checked'), root=root)
            self.assertFalse((root/'runs/state.json').exists())


if __name__ == '__main__':
    unittest.main()

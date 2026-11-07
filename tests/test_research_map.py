"""Protect the distinction between planned graph nodes and actual evidence."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from render_map import validate


class ResearchEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT/'research/map.json').read_text())

    def test_pending_result_cannot_support_a_claim(self):
        edge = next(e for e in self.data['edges'] if e['source'] == 'R1')
        edge['kind'] = 'supports'
        with self.assertRaisesRegex(ValueError, 'actual scoped evidence'):
            validate(self.data)

    def test_actual_evidence_needs_scope_and_source(self):
        result = next(n for n in self.data['nodes'] if n['id'] == 'R1')
        result['status'] = 'scoped'
        with self.assertRaisesRegex(ValueError, 'sources and scope'):
            validate(self.data)

    def test_deleted_endpoint_fails(self):
        self.data['nodes'] = [n for n in self.data['nodes'] if n['id'] != 'B0']
        with self.assertRaisesRegex(ValueError, 'unknown endpoint'):
            validate(self.data)

    def test_prerequisite_cycles_fail(self):
        edge = copy.deepcopy(self.data['edges'][0])
        edge.update(id='cycle', source='E0', target='B0')
        self.data['edges'].append(edge)
        with self.assertRaisesRegex(ValueError, 'cycle'):
            validate(self.data)

    def test_unsafe_source_url_fails(self):
        evidence = next(n for n in self.data['nodes'] if n['kind'] == 'evidence')
        evidence['sources'][0]['url'] = 'javascript:alert(1)'
        with self.assertRaisesRegex(ValueError, 'public HTTPS'):
            validate(self.data)


if __name__ == '__main__':
    unittest.main()

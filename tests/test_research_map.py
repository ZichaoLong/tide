"""Protect the distinction between planned graph nodes and actual evidence."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from render_map import RUN_STATUSES, validate


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

    def test_execution_states_do_not_create_scientific_support(self):
        node = next(n for n in self.data['nodes'] if n['id'] == 'E3')
        for status in RUN_STATUSES:
            with self.subTest(status=status):
                node['status'] = status
                validate(self.data)
        edge = copy.deepcopy(self.data['edges'][0])
        edge.update(id='execution-is-not-support', source='E3', target='G0', kind='supports')
        self.data['edges'].append(edge)
        with self.assertRaisesRegex(ValueError, 'actual scoped evidence'):
            validate(self.data)

    def test_completed_run_does_not_fill_pending_result(self):
        node = next(n for n in self.data['nodes'] if n['id'] == 'R1')
        node['runs'] = [{'id':'example-completed', 'status':'completed',
                         'config':'configs/example.json', 'record':'runs/example/result.json'}]
        validate(self.data)
        self.assertEqual(node['status'], 'awaiting')
        next(e for e in self.data['edges'] if e['source'] == 'R1')['kind'] = 'supports'
        with self.assertRaisesRegex(ValueError, 'actual scoped evidence'):
            validate(self.data)

    def test_pending_result_cannot_be_reviewed_transfer_advice(self):
        edge = next(e for e in self.data['edges'] if e['source'] == 'R1' and e['kind'] == 'transfer')
        edge['status'] = 'documented'
        with self.assertRaisesRegex(ValueError, 'actual scoped evidence'):
            validate(self.data)

    def test_reviewed_result_requires_explanatory_limits(self):
        node = next(n for n in self.data['nodes'] if n['id'] == 'R1')
        node.update(status='scoped', scope='Small task, one seed only.',
                    sources=copy.deepcopy(next(n for n in self.data['nodes'] if n['id']=='F0')['sources']))
        with self.assertRaisesRegex(ValueError, 'reviewed result needs comparisons'):
            validate(self.data)

    def test_private_instance_cannot_be_a_public_source(self):
        evidence = next(n for n in self.data['nodes'] if n['kind'] == 'evidence')
        evidence['sources'][0]['url'] = 'https://127.0.0.1:7860/'
        with self.assertRaisesRegex(ValueError, 'public HTTPS'):
            validate(self.data)

    def test_run_records_require_explicit_state_and_relative_paths(self):
        node = next(n for n in self.data['nodes'] if n['id'] == 'E3')
        node['runs'] = [{'id':'example', 'status':'running',
                         'config':'configs/example.json', 'record':'https://private.example/run'}]
        with self.assertRaisesRegex(ValueError, 'repository-relative path'):
            validate(self.data)

    def test_first_batch_studies_have_explicit_two_routes(self):
        studies = {n.get('study_id') for n in self.data['nodes'] if n.get('study_id')}
        self.assertEqual(studies, {'TL-A0','TL-A1','TL-A2','TL-B0','TL-B1','TL-C'})
        titles = {n['title'] for n in self.data['nodes']}
        self.assertTrue({'从零训练完整 TIDE','预训练模型生长'} <= titles)


if __name__ == '__main__':
    unittest.main()

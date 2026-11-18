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
        # Synthetic examples keep validation independent of live research
        # decisions. Never load archived evidence just to run these checks.
        source = [{'label': 'Synthetic source', 'url': 'https://example.org/fixture'}]
        def node(key, kind, status, *, group='current', y=100):
            value = dict(id=key, kind=kind, status=status, view_group=group,
                         x=30, y=y, title=key, summary='Synthetic fixture',
                         question='Fixture?', decision='No scientific decision.',
                         budget='Not executable.')
            if status == 'scoped':
                value.update(scope='Synthetic fixture only.', sources=copy.deepcopy(source))
                if kind == 'result':
                    value.update(comparisons='Fixture.', counterevidence='Fixture.',
                                 unresolved='Fixture.', transfer_conditions='Fixture.')
            return value
        self.data = dict(
            schema='tide-research-map-prototype-v1', width=600, height=700,
            view_groups=[dict(id='current', y=0, height=320),
                         dict(id='history', y=350, height=350)],
            nodes=[node('G0', 'goal', 'target'), node('B0', 'baseline', 'unselected'),
                   node('E0', 'experiment', 'planned'), node('E3', 'experiment', 'planned'),
                   node('F0', 'evidence', 'scoped'), node('R2', 'result', 'awaiting'),
                   node('R17', 'result', 'scoped', group='history', y=400)],
            edges=[dict(id='prerequisite', source='B0', target='E0',
                        kind='precondition', status='planned'),
                   dict(id='future-transfer', source='R2', target='G0',
                        kind='transfer', status='planned')])

    def pending_result(self):
        # Real studies can finish. Keep a hypothetical pending fixture rather
        # than requiring the human graph to retain an unfilled result forever.
        node = next(n for n in self.data['nodes'] if n['id'] == 'R2')
        node['status'] = 'awaiting'
        for key in ('scope', 'sources', 'comparisons', 'counterevidence', 'unresolved', 'transfer_conditions'):
            node.pop(key, None)
        for edge in self.data['edges']:
            if edge['source'] == node['id'] and edge['kind'] == 'transfer':
                edge['status'] = 'planned'
        return node

    def test_pending_result_cannot_support_a_claim(self):
        self.pending_result()
        edge = next(e for e in self.data['edges'] if e['source'] == 'R2')
        edge['kind'] = 'supports'
        with self.assertRaisesRegex(ValueError, 'actual scoped evidence'):
            validate(self.data)

    def test_actual_evidence_needs_scope_and_source(self):
        result = self.pending_result()
        result['status'] = 'scoped'
        with self.assertRaisesRegex(ValueError, 'sources and scope'):
            validate(self.data)

    def test_deleted_endpoint_fails(self):
        self.data['nodes'] = [n for n in self.data['nodes'] if n['id'] != 'B0']
        with self.assertRaisesRegex(ValueError, 'unknown endpoint'):
            validate(self.data)

    def test_history_evidence_cannot_disappear_outside_its_view(self):
        node = next(n for n in self.data['nodes'] if n['id'] == 'R17')
        node['view_group'] = 'current'
        with self.assertRaisesRegex(ValueError, 'outside group bounds'):
            validate(self.data)

    def test_unknown_view_group_fails(self):
        self.data['view_groups'][0]['id'] = 'missing'
        with self.assertRaisesRegex(ValueError, 'unknown or duplicate group'):
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
        node = self.pending_result()
        node['runs'] = [{'id':'example-completed', 'status':'completed',
                         'config':'configs/example.json', 'record':'runs/example/result.json'}]
        validate(self.data)
        self.assertEqual(node['status'], 'awaiting')
        next(e for e in self.data['edges'] if e['source'] == 'R2')['kind'] = 'supports'
        with self.assertRaisesRegex(ValueError, 'actual scoped evidence'):
            validate(self.data)

    def test_pending_result_cannot_be_reviewed_transfer_advice(self):
        self.pending_result()
        edge = next(e for e in self.data['edges'] if e['source'] == 'R2')
        edge['kind'] = 'transfer'
        edge['status'] = 'documented'
        with self.assertRaisesRegex(ValueError, 'actual scoped evidence'):
            validate(self.data)

    def test_reviewed_result_requires_explanatory_limits(self):
        node = self.pending_result()
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

    def test_current_map_is_valid_without_loading_an_archive(self):
        validate(json.loads((ROOT/'research/map.json').read_text()))


if __name__ == '__main__':
    unittest.main()

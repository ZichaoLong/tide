"""Regression coverage for terminal failures omitted by audit-only summaries."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from measurement_outcomes import reconcile


class OutcomeTests(unittest.TestCase):
    def attempt(self, **changes):
        return dict(dict(run='first', cell=0, state='failed', finished='2026-10-09',
                         model_executed=False, error=''), **changes)

    def test_pre_admission_failure_without_audit_is_retained(self):
        row = self.attempt(error='MemoryError: lane exceeds selected-node memory estimate: cpu')
        report = reconcile(2, [1], [row])
        self.assertEqual(report['category_counts'], {'cpu-memory-admission': 1, 'accepted': 1})
        self.assertEqual(report['retained_failures'][0]['run'], 'first')

    def test_later_acceptance_does_not_erase_failure_history(self):
        row = self.attempt(queue_status='timed_out')
        report = reconcile(1, [0], [row, row, self.attempt(run='second', state='passed')])
        self.assertEqual(report['category_counts'], {'accepted': 1})
        self.assertEqual(len(report['retained_failures']), 1)

    def test_complete_over_bound_is_not_accepted(self):
        row = self.attempt(model_executed=True, completed_over_bound=True)
        report = reconcile(1, [], [row])
        self.assertEqual(report['accepted_cells'], [])
        self.assertEqual(report['category_counts'], {'completed-over-bound': 1})

    def test_successful_receipt_cannot_replace_domain_acceptance(self):
        report = reconcile(1, [], [self.attempt(state='passed')])
        self.assertEqual(report['accepted_cells'], [])
        self.assertEqual(report['cell_outcomes'][0]['category'], 'not-measured')

    def test_unknown_error_is_explicit(self):
        report = reconcile(1, [], [self.attempt(error='permission denied')])
        self.assertEqual(report['category_counts'], {'unclassified-failure': 1})

    def test_conflicting_terminal_receipts_fail(self):
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            reconcile(1, [], [self.attempt(), self.attempt(state='passed')])

    def test_bound_flag_requires_completed_failed_work(self):
        with self.assertRaises(ValueError):
            reconcile(1, [], [self.attempt(completed_over_bound=True)])


if __name__ == '__main__':
    unittest.main()

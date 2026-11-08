"""Only the diagnosed time-cap failure authorizes this finite continuation."""
import unittest
from scripts.complete_capability_budget import budget_failure


class CompletionPolicyTests(unittest.TestCase):
    def test_only_exact_declared_budget_failure_is_eligible(self):
        self.assertTrue(budget_failure(dict(state='failed',error="TimeoutError('fixed per-run training budget exhausted')")))
        for result in [dict(state='completed'),dict(state='interrupted'),
                       dict(state='failed',error='FloatingPointError(nonfinite)'),
                       dict(state='failed',error='AssertionError(resume mismatch)'),
                       dict(state='failed',error='TimeoutError(queue expired)')]:
            self.assertFalse(budget_failure(result))


if __name__=='__main__':unittest.main()

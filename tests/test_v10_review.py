import unittest
from scripts.review_direct_v10 import application_state


class V10ReviewStates(unittest.TestCase):
    def test_worker_success_cannot_overwrite_training_failure(self):
        self.assertEqual(application_state('completed','failed','failed'),'failed')
        self.assertEqual(application_state('completed','budget_truncated','budget_truncated'),'budget_truncated')
        self.assertEqual(application_state('completed',None,'failed'),'failed')
        self.assertEqual(application_state('completed','running','running'),'unchecked')

    def test_actual_completion_live_and_queued_remain_distinct(self):
        self.assertEqual(application_state('completed','completed','completed'),'completed')
        self.assertEqual(application_state('running','running','running'),'running')
        self.assertEqual(application_state('waiting_resources',None,None),'waiting_resources')
        self.assertEqual(application_state('failed',None,None),'failed')


if __name__=='__main__':unittest.main()

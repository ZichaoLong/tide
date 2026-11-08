"""Resolved pilots must fit joint update/token/time caps, not just throughput."""
import json
from pathlib import Path
import unittest
from scripts.run_language_route import choose_case,forecast


class LanguageBudgetTests(unittest.TestCase):
    def setUp(self):self.plan=json.loads(Path('configs/studies/language/initial.json').read_text())
    def case(self,batch,seconds,length=32):
        r=dict(mean_step_seconds=seconds,valid_tokens_per_update=batch*length,
               evaluation_preview=dict(seconds=1.,tokens=256))
        return dict(batch=batch,length=length,calibrations={'dense':r,'sparse':r})
    def test_fastest_raw_batch_must_still_fit_minimum_updates(self):
        selected=choose_case([self.case(32,1),self.case(256,50)],self.plan,1000)
        self.assertEqual(selected['batch'],32)
        self.assertGreaterEqual(selected['steps'],128)
        self.assertLessEqual(selected['steps']*32*32,1000000)
        self.assertLessEqual(selected['forecast_seconds'],1000)
    def test_tiny_case_without_minimum_tokens_is_rejected(self):
        self.assertIsNone(choose_case([self.case(1,1)],self.plan,3000))
    def test_whole_pair_and_not_only_each_member_must_fit(self):
        self.assertIsNone(choose_case([self.case(32,3)],self.plan,1000))
    def test_validation_startup_and_headroom_count(self):
        r=self.case(32,1)['calibrations']['dense']
        self.assertGreater(forecast(r,128),128+60)


if __name__=='__main__':unittest.main()

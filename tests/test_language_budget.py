"""Resolved pilots must fit joint update/token/time caps, not just throughput."""
import json
import copy
from pathlib import Path
import tempfile
import unittest
from scripts.run_language_route import choose_case,forecast
from tide_lab.language_reuse import validate_recipe,verify_source_files


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

    def test_named_small_exposure_can_train_without_relabelling_old_admission(self):
        old=json.loads(Path('configs/studies/language/fc-pilot-v2.json').read_text())
        new=json.loads(Path('configs/studies/language/fc-explore-v3.json').read_text())
        case=self.case(8,20)
        case['exact_exposures']=[int(32503*(i+1)/128) for i in range(128)]
        self.assertIsNone(choose_case([case],old,7140))
        selected=choose_case([case],new,7140)
        self.assertEqual((selected['batch'],selected['steps']),(8,128))
        self.assertLess(selected['exact_exposures'][-1],old['training']['minimum_valid_tokens'])
        validate_recipe(old,new)

    def test_reused_checks_do_not_admit_changed_optimizer_or_tolerance(self):
        old=json.loads(Path('configs/studies/language/fc-pilot-v2.json').read_text())
        new=json.loads(Path('configs/studies/language/fc-explore-v3.json').read_text())
        for section,key in [('training','lr_a'),('checks','core_atol'),('core','width')]:
            changed=copy.deepcopy(new);changed[section][key]*=2
            with self.assertRaises(ValueError):validate_recipe(old,changed)

    def test_reused_source_rejects_changed_training_file(self):
        from tide_lab.records import digest
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'tide_lab').mkdir();(root/'scripts').mkdir()
            names=['tide_lab/language_training.py','scripts/language_stage.py']
            for name in names:(root/name).write_text('original')
            identity={'content':{'files':[dict(path=n,sha256=digest(root/n)) for n in names]}}
            verify_source_files(root,identity)
            (root/names[0]).write_text('changed')
            with self.assertRaises(ValueError):verify_source_files(root,identity)


if __name__=='__main__':unittest.main()

import json
import tempfile
import unittest
from pathlib import Path
from scripts.control_hb_v12_repair import failure_class


class RepairAdmissionTests(unittest.TestCase):
    def test_numerical_comparison_or_nonfinite_is_not_a_solver_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'runs/v12-main-example/train';p.mkdir(parents=True)
            for error in ('nonfinite next-token loss','causal intervention failed','forward tolerance failed'):
                (p/'result.json').write_text(json.dumps(dict(state='failed',error=error)))
                self.assertIsNone(failure_class(root,'example'))
            (p/'result.json').write_text(json.dumps(dict(state='failed',error='allocation feasibility residual 3e-5 > 2e-5')))
            self.assertEqual(failure_class(root,'example'),'quad_feasibility')

    def test_disk_permission_error_is_not_disappearance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'artifacts/jobs/v12-main-example';p.mkdir(parents=True)
            (p/'job.log').write_text("du: cannot access 'progress.json.tmp': Permission denied")
            self.assertIsNone(failure_class(root,'example'))
            (p/'job.log').write_text("du: cannot access 'progress.json.tmp': No such file or directory")
            self.assertEqual(failure_class(root,'example'),'disk_scan_disappearance')


if __name__=='__main__':unittest.main()

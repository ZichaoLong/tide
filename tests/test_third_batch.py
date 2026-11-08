"""Hold-out leakage and diagnostic state changes would invalidate this comparison."""
import copy
import json
from pathlib import Path
import random
import unittest
from unittest.mock import patch


class ThirdBatchChecks(unittest.TestCase):
    def test_frozen_holdout_is_unique_disjoint_and_reproducible(self):
        from tide_lab.heldout import make_dataset, validate_dataset
        data=json.loads(Path('research/data/hb-test2048-v1.json').read_text())
        check=validate_dataset(data)
        self.assertEqual(check['samples'],2048)
        self.assertEqual(check['old_test_overlap'],0)
        self.assertEqual(make_dataset(data['task']),data)
        duplicate=copy.deepcopy(data);duplicate['examples'][1]=dict(duplicate['examples'][0],index=1)
        with self.assertRaisesRegex(ValueError,'invariant'):
            validate_dataset(duplicate)
        from tide_lab.tasks import RetrievalTask
        task=RetrievalTask(data['task']);x,y=task.batch('test',256,0,20261107)
        leaked=copy.deepcopy(data);leaked['examples'][0]=dict(index=0,tokens=x[0].tolist(),target=y[0].item())
        with self.assertRaisesRegex(ValueError,'invariant'):
            validate_dataset(leaked)

    def test_branch_diagnostic_restores_rng_modes_and_weights(self):
        import torch
        from tide_lab.train import diagnostic_snapshot, cpu_tree
        from tide_lab.tasks import RetrievalTask
        from scripts.check_resume import equal
        class Branch(torch.nn.Module):
            def __init__(self):
                super().__init__();self.head=torch.nn.Linear(1,8);self.observe=True
            def forward(self,x,branch_enabled=True):
                # Random draws model an observation consuming global RNG.
                random.random();torch.rand(1)
                return self.head(x[:,-1:].float())+(0.1 if branch_enabled else 0.)
        model=Branch().train();task=RetrievalTask()
        config=dict(model=dict(kind='growth',branch='tide'),learning_gate=.25,
                    evaluation=dict(samples=4,batch_size=2,seed=20261107))
        before=dict(model=cpu_tree(model.state_dict()),rng=torch.get_rng_state(),python=random.getstate())
        row=diagnostic_snapshot(model,task,config,torch.device('cpu'),40)
        equal(before,dict(model=cpu_tree(model.state_dict()),rng=torch.get_rng_state(),python=random.getstate()))
        self.assertTrue(model.training and model.observe)
        self.assertEqual(len(row['paired_branch']['per_example']),4)
        self.assertEqual(row['original_learning_gate'],.25)

    def test_compute_cap_and_common_deadline_both_bind(self):
        from scripts.bounded_workload import remaining_seconds
        with patch('time.monotonic',return_value=120),patch('time.time',return_value=1000):
            self.assertEqual(remaining_seconds(900,1100,100),100)
            self.assertEqual(remaining_seconds(900,5000,100),880)
            self.assertLess(remaining_seconds(900,900,100),0)


if __name__=='__main__':unittest.main()

"""Observation changes must preserve training state and nonfinite protection."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.check_resume import equal


class CadenceTests(unittest.TestCase):
    def test_core5_observation_and_fresh_resume_preserve_complete_state(self):
        self.check_observation_and_resume({})

    def test_hb_sparse_observation_and_resume_preserve_complete_state(self):
        self.check_observation_and_resume(dict(topology='hb',send='sparse'))

    def check_observation_and_resume(self, model_options):
        import torch
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            config=dict(schema='tide-lab-run-v1',study_id='cadence-check',seed=11,
                device='cpu',dtype='float32',threads=1,
                task=dict(n_keys=4,n_values=4,pairs=2,delay=0),
                model=dict(kind='core5',width=4,depth=1,region_width=4,temperature=.25),
                training=dict(steps=5,batch_size=2,lr=.001,eps=1e-6,weight_decay=.01,
                    log_every=1,eval_every=5,checkpoint_every=2,max_seconds=120,data_seed=1011),
                evaluation=dict(samples=2,batch_size=2,seed=20261107,test=False))
            config['model'].update(model_options)
            def run(label,c,extra=()):
                path=root/(label+'.json');path.write_text(json.dumps(c))
                # Resumed run must use the same byte-for-byte config.
                if label=='resumed':path=root/'sampled.json'
                result=subprocess.run([sys.executable,'-m','tide_lab.train','--config',str(path),
                    '--output',str(root/label),*extra],capture_output=True,text=True,timeout=150)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                return torch.load(root/label/'checkpoint.pt',map_location='cpu',weights_only=False)
            all_steps=run('all',config)
            sampled_config=copy.deepcopy(config);sampled_config['training']['log_every']=3
            sampled_config['evaluation']['diagnostic_steps']=[3]
            sampled=run('sampled',sampled_config)
            run('split',sampled_config,('--stop-after','2'))
            resumed=run('resumed',sampled_config,('--resume',str(root/'split/checkpoint.pt')))
            for part in ('model','optimizer','step','rng','python_rng','accelerator_rng','sequence_state'):
                equal(all_steps[part],sampled[part],part)
            equal(sampled,resumed)
            events=[json.loads(line) for line in (root/'sampled/metrics.jsonl').read_text().splitlines()]
            self.assertEqual([e['step'] for e in events if 'train/loss' in e['metrics']],[1,3,5])
            a=json.loads((root/'all/result.json').read_text())['measurement']
            b=json.loads((root/'sampled/result.json').read_text())['measurement']
            self.assertEqual(a['work_totals'],b['work_totals'])
            self.assertEqual((b['steps_counted'],b['observed_steps'],b['unobserved_steps']),(5,3,2))
            diagnostic=json.loads((root/'sampled/diagnostic-step3.json').read_text())
            self.assertEqual(diagnostic['step'],3)

    def test_nonfinite_unobserved_step_still_fails(self):
        with tempfile.TemporaryDirectory() as name:
            script='''
import argparse,json
from pathlib import Path
import torch
from tide_lab import train
from tide_lab.tasks import NoMemoryClassifier
class Poison(NoMemoryClassifier):
    calls=0
    def forward(self,x):
        value=super().forward(x)
        if self.training:
            self.calls+=1
            if self.calls==2:return value*float('nan')
        return value
train.make_model=lambda c,t,a:Poison(t.vocab_size,t.value_vocab_size)
root=Path(__import__('sys').argv[1])
c=dict(schema='tide-lab-run-v1',study_id='cadence-nonfinite',seed=11,device='cpu',threads=1,
       model=dict(kind='no_memory'),task=dict(pairs=2,delay=0),
       training=dict(steps=5,batch_size=2,lr=.001,log_every=3,eval_every=5,checkpoint_every=5,max_seconds=60,data_seed=1),
       evaluation=dict(samples=2,batch_size=2,seed=3,test=False))
(root/'config.json').write_text(json.dumps(c))
code=train.execute(argparse.Namespace(config=str(root/'config.json'),output=str(root/'run'),
       run_id=None,asset_path=None,resume=None,init_from=None,stop_after=None))
assert code==1
r=json.loads((root/'run/result.json').read_text())
assert r['state']=='failed' and r['steps']==1 and r['error_type']=='FloatingPointError',r
'''
            result=subprocess.run([sys.executable,'-c',script,name],capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__=='__main__':unittest.main()

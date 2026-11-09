"""Application transfer rejects recipe drift and preserves optimizer/cursor state."""
import copy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
import torch
from tide_lab.language_completion import import_checkpoint
from tide_lab.records import digest


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=torch.nn.Embedding(5,2)
        self.head=torch.nn.Linear(2,5,bias=False)
        self.head.weight=self.embedding.weight
        self.encoder=SimpleNamespace(model_id='new',binding=SimpleNamespace(graph=SimpleNamespace(identity='graph')))


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);model=Model();opt=torch.optim.AdamW(model.parameters())
        model.embedding.weight.square().sum().backward();opt.step()
        ident=dict(route='a',kind='tide-dense',depth=1,seed=11,batch=8,length=32,
                   schedule_steps=512,data_lock_sha256='data')
        plan=dict(core=dict(width=2,support_policy='strict'),training=dict(lr=.003))
        self.plan=copy.deepcopy(plan);self.plan['core']['support_policy']='positive-floor-v1'
        state=dict(schema='tide-language-checkpoint-v1',identity=ident,step=32,valid_tokens=8171,
                   model=model.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),python_rng=random.getstate())
        torch.save(state,self.root/'state.pt')
        (self.root/'manifest.json').write_text(json.dumps(dict(identity=ident,config=plan)))
        self.cfg=dict(checkpoint=str(self.root/'state.pt'),sha256=digest(self.root/'state.pt'),
            parent_manifest=str(self.root/'manifest.json'),parent_manifest_sha256=digest(self.root/'manifest.json'),
            step=32,format='language-checkpoint',original_initial_validation_nll=8.3)
        self.expected=state

    def call(self):
        m=Model();o=torch.optim.AdamW(m.parameters())
        result=import_checkpoint(self.cfg,self.plan,m,o,'data',torch.device('cpu'),'tide-dense',11,8,32,1,512)
        return result,m,o

    def test_transfer_preserves_parameters_optimizer_and_cursor(self):
        (step,tokens,record),m,o=self.call()
        self.assertEqual((step,tokens),(32,8171));self.assertFalse(record['exact_cross_identity_resume_claim'])
        self.assertTrue(torch.equal(m.embedding.weight,self.expected['model']['embedding.weight']))
        self.assertIs(m.embedding.weight,m.head.weight)
        self.assertTrue(torch.equal(next(iter(o.state.values()))['exp_avg'],self.expected['optimizer']['state'][0]['exp_avg']))
        self.assertTrue(torch.equal(torch.get_rng_state(),self.expected['rng']))

    def test_modified_checkpoint_is_rejected(self):
        self.cfg['sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'checkpoint changed'):self.call()

    def test_recipe_change_is_rejected(self):
        self.plan['training']['lr']=.1
        with self.assertRaisesRegex(ValueError,'recipe'):self.call()

    def test_wrong_target_data_is_rejected(self):
        m=Model();o=torch.optim.AdamW(m.parameters())
        with self.assertRaisesRegex(ValueError,'data_lock'):
            import_checkpoint(self.cfg,self.plan,m,o,'other-data',torch.device('cpu'),'tide-dense',11,8,32,1,512)

if __name__=='__main__':unittest.main()

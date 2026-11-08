"""Protect task independence and the exact scope of local intervention."""
import json
from pathlib import Path
import unittest
import tempfile
from types import SimpleNamespace
import torch
from tide_lab.capability import dataset, split_for, blend_last, statistics, normalize, preservation_gate


class CapabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=json.loads(Path('configs/studies/fourth/b-preserve.json').read_text())
        cls.data=dataset(cls.plan)

    def test_content_disjoint_and_answers_independently_decodable(self):
        seen=set()
        for group,rows in self.data.items():
            family,split=group.split('/')
            self.assertEqual(len(rows),self.plan['data'][split+'_per_task'])
            for row in rows:
                self.assertNotIn(row['content_key'],seen);seen.add(row['content_key'])
                self.assertEqual(split_for(row['content_key']),split)
                expected=(dict(row['content']['pairs'])[row['content']['query']] if family=='lookup'
                          else row['content']['words'][-1])
                self.assertEqual(row['answer'],expected)
        self.assertEqual(self.data,dataset(self.plan))

    def test_intervention_changes_only_declared_position_without_mutation(self):
        y=torch.arange(24,dtype=torch.float32).reshape(2,3,4);old=y.clone();p=y[:,-1]+2
        self.assertIs(blend_last(y,p,0),y)
        new=blend_last(y,p,.1)
        self.assertTrue(torch.equal(y,old));self.assertTrue(torch.equal(new[:,:-1],y[:,:-1]))
        self.assertTrue(torch.allclose(new[:,-1]-y[:,-1],torch.full((2,4),.2)))
        with self.assertRaises(ValueError):blend_last(y,p,-.1)

    def test_normalizer_uses_training_statistics_and_nonzero_gate(self):
        torch.manual_seed(41);x=torch.randn(20,6);y=torch.randn(20,6)
        stats=statistics(x,y);nx,ny=normalize(x,y,stats)
        self.assertTrue(torch.allclose(nx.mean(0),torch.zeros(6),atol=1e-6))
        self.assertAlmostEqual(float(ny.square().mean()),1,places=6)
        summary=dict(accuracy_drop=0,newly_wrong=0,samples=64,answer_nll_increase=0)
        self.assertFalse(preservation_gate(summary,0,self.plan['gates']))
        self.assertTrue(preservation_gate(summary,.01,self.plan['gates']))
        summary['newly_wrong']=2
        self.assertFalse(preservation_gate(summary,.01,self.plan['gates']))

    def test_new_loss_and_public_core_receive_gradients(self):
        from scripts.capability_stage import mechanisms
        from tide_lab.records import digest
        torch.set_num_threads(1)
        class Sink:
            def log(self,*args):pass
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);plan=dict(self.plan,device='cpu')
            (root/'plan.json').write_text(json.dumps(plan))
            torch.manual_seed(13);x=torch.randn(12,32);y=torch.sin(x)+.1*x.square()
            cache=dict(plan_sha256=digest(root/'plan.json'),train=dict(x=x,y=y),
                       validation={'fixture':dict(x=x[:4],y=y[:4])},stats=statistics(x,y))
            torch.save(cache,root/'cache.pt')
            result=mechanisms(SimpleNamespace(plan=root/'plan.json',cache=root/'cache.pt'),
                              plan,torch.device('cpu'),Sink())
            self.assertEqual(result['state'],'passed')
            for checks in result['checks'].values():
                self.assertTrue(checks['readout_gradient_strict'])
                self.assertTrue(checks['inner_gradient_nonzero'])


if __name__=='__main__':unittest.main()

import copy
import unittest
import torch
from torch.nn import functional as F
from tide_lab.moe import Top2Experts,MoETransformer,objective_backward,parameter_inventory
from tide_lab.moe_config import config,screening_configs
from tide_lab.campaign_runtime import optimizer_for


class MoETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_bucketed_reference_vjp_empty_experts_and_repeat_tokens(self):
        torch.manual_seed(72)
        for dtype,atol,rtol in ((torch.float64,1e-11,1e-10),(torch.float32,1e-6,1e-5)):
            m=Top2Experts(16,24,20,2).to(dtype)
            r=copy.deepcopy(m);r.implementation='reference'
            x=torch.randn(2,4,16,dtype=dtype,requires_grad=True);xr=x.detach().clone().requires_grad_()
            mask=torch.tensor([[1,1,1,1],[1,0,0,0]],dtype=torch.bool)
            m.valid_mask=r.valid_mask=mask
            y=m(x);yr=r(xr)
            torch.testing.assert_close(y,yr,atol=atol,rtol=rtol)
            v=torch.randn_like(y);(y*v).sum().backward();(yr*v).sum().backward()
            torch.testing.assert_close(x.grad,xr.grad,atol=atol,rtol=rtol)
            for p,q in zip(m.parameters(),r.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=atol,rtol=rtol)
            self.assertEqual(int(m.stats['counts'].sum()),10)
            self.assertTrue(bool((m.stats['counts']==0).any()))
            self.assertEqual(m.stats['indices'].shape,(8,2))
            self.assertTrue(bool((m.stats['indices'][:,0]!=m.stats['indices'][:,1]).all()))
            self.assertLess(m.physical_rows,2*8*2)

    def test_padding_exclusion_and_global_aux_accumulation(self):
        torch.manual_seed(97)
        m=MoETransformer(width=16,layers=2,heads=4,hidden=24,experts=8,vocab_size=32).double()
        other=copy.deepcopy(m)
        ids=torch.randint(0,32,(4,5));labels=torch.randint(0,32,(4,5));labels[1,3:]=-100;labels[3,2:]=-100
        b=dict(input_ids=ids,labels=labels);c=config('small-e20',.001,.01)
        total,work,n=objective_backward(m,[b],c)
        micros=[{k:v[i:i+1] for k,v in b.items()} for i in range(4)]
        tot2,w2,n2=objective_backward(other,micros,c)
        self.assertEqual((n,n2),(15,15));self.assertAlmostEqual(total,tot2,places=12)
        for key in ('aux/balance','aux/z_loss','train/objective'):self.assertAlmostEqual(work[key],w2[key],places=12)
        for p,q in zip(m.parameters(),other.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=1e-11,rtol=1e-9)
        self.assertGreater(w2['work/aux_prepass_expert_rows'],0)
        # An ignored suffix cannot affect any valid target's logits or router statistics.
        changed=ids.clone();changed[labels==-100]=(changed[labels==-100]+7)%32
        with torch.no_grad():
            y=m(ids,valid_mask=labels!=-100);aux=m.auxiliary()
            z=m(changed,valid_mask=labels!=-100);aux2=m.auxiliary()
        torch.testing.assert_close(y[labels!=-100],z[labels!=-100],rtol=0,atol=0)
        for p,q in zip(aux,aux2):torch.testing.assert_close(p,q,atol=1e-12,rtol=1e-12)

    def test_alias_optimizer_bias_decay_and_routing_gradients(self):
        m=MoETransformer(width=16,layers=2,hidden=24,experts=8,vocab_size=32)
        c=config('small-e20',.001);opt=optimizer_for(m,c)
        self.assertIs(m.embedding.weight,m.head.weight)
        self.assertEqual(sum(len(g['params']) for g in opt.param_groups),len(list(m.parameters())))
        for group in opt.param_groups:
            for block in m.blocks:
                if any(p is block.ffn.b1 or p is block.ffn.b2 for p in group['params']):self.assertEqual(group['weight_decay'],0)
        b=dict(input_ids=torch.randint(0,32,(2,8)),labels=torch.randint(0,32,(2,8)))
        objective_backward(m,[b],c)
        self.assertTrue(all(float(block.ffn.router.weight.grad.abs().sum())>0 for block in m.blocks))
        before=m.embedding.weight.detach().clone();opt.step()
        self.assertFalse(torch.equal(before,m.embedding.weight))

    def test_inventory_and_grid(self):
        rows=screening_configs();self.assertEqual(len(rows),30)
        self.assertEqual(len({c['id'] for c in rows}),30)
        m=MoETransformer(width=16,layers=3,hidden=24,experts=8,vocab_size=32)
        self.assertEqual(sum(p.numel() for p in m.parameters()),parameter_inventory(16,3,24,8,vocab_size=32)['total'])
        self.assertEqual([parameter_inventory(*args)['total'] for args in ((192,6,384,20),(192,6,384,64),(256,12,480,20),(256,12,480,64))],
            [19471744,58602880,63443968,193729024])


if __name__=='__main__':unittest.main()

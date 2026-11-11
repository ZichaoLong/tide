import copy
import json
from pathlib import Path
import unittest
import torch
from torch.nn import functional as F
from tide_lab.direct_semantic_ops import allocation,dual_recurrence,causal_conv,sorted_send,weighted_softmax,norm
from tide_lab.direct_semantic import Aggregation,DirectSemanticLM
from tide_lab.direct_semantic_memory import Memory


def serial(k,v,q,logg,write,delta,old,log_input):
    state=k.new_zeros(*k.shape[:-2],v.shape[-1],k.shape[-1]);rows=[]
    for t in range(k.shape[-2]):
        pre=state*(logg[...,t]-log_input[...,t]).exp()[...,None,None]
        retained=state*logg[...,t].exp()[...,None,None]
        val=v[...,t,:]
        if delta:val=val-(retained@k[...,t,:,None]).squeeze(-1)
        state=retained+write[...,t,None,None]*val.unsqueeze(-1)*k[...,t,None,:]
        rows.append(((pre if old else state)@q[...,t,:,None]).squeeze(-1))
    return torch.stack(rows,-2)


class DirectMath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_allocation_formula_and_implicit_vjp(self):
        dtype=torch.float64
        t=torch.tensor([[2.,1.,100.]],dtype=dtype,requires_grad=True)
        mu=torch.tensor([[1.,1.,0.]],dtype=dtype,requires_grad=True)
        b=torch.tensor([[1.4]],dtype=dtype,requires_grad=True)
        c,res=allocation(t,mu,b,b,2.,'quad')
        torch.testing.assert_close(c,torch.tensor([[.95,.45,0.]],dtype=dtype),atol=1e-12,rtol=1e-12)
        gt,gb=torch.autograd.grad((c*torch.tensor([[1.,3.,7.]],dtype=dtype)).sum(),(t,b))
        torch.testing.assert_close(gt,torch.tensor([[-.5,.5,0.]],dtype=dtype),atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(gb,torch.tensor([[2.]],dtype=dtype))
        for kind,alpha in [('quad',1.),('bern',1.),('ent',1.),('ent',1.5),('ent',2.)]:
            x=torch.tensor([[.1,.2,-.1]],dtype=dtype,requires_grad=True)
            cap=torch.tensor([[.7,.8,.9]],dtype=dtype,requires_grad=True)
            target=torch.tensor([[.8]],dtype=dtype,requires_grad=True)
            self.assertTrue(torch.autograd.gradcheck(lambda x,cap,b:allocation(x,cap,b,b,1.,kind,alpha)[0],(x,cap,target),atol=1e-5,rtol=1e-4))
        x=torch.tensor([[.1,.2]],dtype=dtype);cap=torch.ones_like(x);b=torch.ones((1,1),dtype=dtype)
        out,_=allocation(x,cap,b,b,.5,'ent',1.)
        torch.testing.assert_close(out,torch.softmax(x/.5,-1),atol=1e-12,rtol=1e-12)
        out,_=allocation(x,cap,b,b,.5,'bern',1.)
        expected=torch.sigmoid((x[:,0]-x[:,1])/(2*.5))
        torch.testing.assert_close(out[:,0],expected,atol=1e-12,rtol=1e-12)

    def test_allocation_zero_singleton_and_ties(self):
        score=torch.tensor([[1.,1.,-1.],[0.,-2.,-3.],[4.,5.,6.]])
        mu=torch.tensor([[1.,1.,1.],[.2,.8,0.],[0.,0.,0.]])
        out=sorted_send(score,mu,1)
        torch.testing.assert_close(out,torch.tensor([[1.,0.,0.],[.2,0.,0.],[0.,0.,0.]]))
        caps=torch.tensor([[.2,.3]],requires_grad=True);t=torch.tensor([[-99.,99.]],requires_grad=True)
        b=caps.sum(-1,keepdim=True);c,_=allocation(t,caps,b,torch.ones_like(b),1.,'ent',1.)
        torch.testing.assert_close(c,caps);c.sum().backward()
        torch.testing.assert_close(caps.grad,torch.ones_like(caps));torch.testing.assert_close(t.grad,torch.zeros_like(t))

    def test_recurrence_forward_old_read_and_gradient(self):
        torch.manual_seed(23)
        for delta in (False,True):
            for old in (False,True):
                inputs=[(torch.randn(2,7,d,dtype=torch.float64)*.2).requires_grad_() for d in (3,4,3)]
                log_input=(-torch.rand(2,7,dtype=torch.float64)*.4).requires_grad_()
                write=torch.rand(2,7,dtype=torch.float64,requires_grad=True)
                k,v,q=inputs;logg=log_input-.13
                a=dual_recurrence(k,v,q,logg,write,delta=delta,old=old,log_input_gate=log_input)
                b=serial(k,v,q,logg,write,delta,old,log_input)
                torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-12)
                weight=torch.randn_like(a);params=(*inputs,log_input,write)
                ga=torch.autograd.grad((a*weight).sum(),params,retain_graph=True)
                gb=torch.autograd.grad((b*weight).sum(),params)
                for x,y in zip(ga,gb):torch.testing.assert_close(x,y,rtol=1e-9,atol=1e-11)

    def test_fixed_time_conv_weak_history(self):
        z=torch.tensor([[[[4.],[5.],[6.]]]],dtype=torch.float64)
        w=torch.tensor([[[1.,.01,1.]]],dtype=torch.float64,requires_grad=True)
        coeff=torch.tensor([[[1.],[2.],[3.]]],dtype=torch.float64);bias=torch.zeros(1,1,dtype=torch.float64)
        y=causal_conv(z,w,coeff,bias,2,2)
        self.assertAlmostEqual(float(y[0,0,2,0]),18.1)
        y[0,0,2,0].backward();self.assertEqual(float(w.grad[0,0,1]),10.)
        wrong_residue=causal_conv(z,w.detach(),coeff,bias,2,1)
        self.assertAlmostEqual(float(wrong_residue[0,0,2,0]),6.15)

    def test_source_sharing_against_scalar_sum_and_vjp(self):
        torch.manual_seed(18)
        for mode in ('region','node','source','adapted'):
            m=Aggregation(2,3,2,mode,1,[(1,0),(2,1)],[0,1]).double()
            z=torch.randn(2,1,2,3,dtype=torch.float64,requires_grad=True)
            mass=torch.tensor([[[.2,.1]],[[.5,.9]]],dtype=torch.float64,requires_grad=True)
            edges=torch.tensor([[1.,1.],[0.,1.]],dtype=torch.float64)
            actual,r=m(z,mass,edges,1.);rows=[]
            for v in range(2):
                num=0;den=0
                for src in range(2):
                    if not edges[v,src]:continue
                    if mode=='region':a,b,e=m.a,m.b,m.e[src]
                    elif mode=='node':a,b,e=m.a[v],m.b[v],m.e[v,src]
                    elif mode=='source':a,b,e=m.a[src],m.b[src],m.e[src]
                    else:a,b,e=m.a+m.la[v,src]@m.ra[v,src],m.b+m.lb[v,src]@m.rb[v,src],m.e[v,src]
                    f=F.linear(F.silu(F.linear(norm(z[src]),a)+e),b)
                    num=num+mass[src].unsqueeze(-1)*(z[src]+f);den=den+mass[src]
                rows.append(num/den.unsqueeze(-1))
            reference=torch.stack(rows)
            torch.testing.assert_close(actual,reference,rtol=1e-12,atol=1e-12)
            ga=torch.autograd.grad(actual.square().sum(),(z,mass,m.a),retain_graph=True)
            gb=torch.autograd.grad(reference.square().sum(),(z,mass,m.a))
            for x,y in zip(ga,gb):torch.testing.assert_close(x,y,rtol=1e-10,atol=1e-11)

    def test_kv_snapshot_window_and_full_weight(self):
        spec=dict(width=4,summary_width=2,ffn_width=8,adapter_rank=2,heads=1,memory_read='sum',
            window=5,beta_short=.1,epsilon=1e-5,full_frequencies=[[]],read_frequencies=[],
            position_origin=1,position_scale=2,distance_slopes=[0.],alpha_att=1.,alpha_ff=0.)
        torch.manual_seed(14);m=Memory(1,spec,2).double()
        u=torch.randn(1,1,4,4,dtype=torch.float64);r=torch.tensor([[[.1,0.,.2,.7]]],dtype=torch.float64)
        x=norm(u);p=m.current(x);kv=F.linear(x,m.base)+m.adapter_out(m.adapter_in(x));k,v=kv.chunk(2,-1);q=m.query(x)
        for read in ('0','-','+'):
            out=m(u,r,read)
            for t in range(4):
                ids=[i for i in range(t+1) if 2*(t-i)<5 and r[0,0,i]>0]
                selected=[] if read=='0' else [i for i in ids if read=='+' or i<t]
                for a,beta in enumerate((.1,0.)):
                    weights=torch.stack([r[0,0,i]*torch.exp(r.new_tensor(-beta*2*(t-i))) for i in selected]) if selected else r.new_zeros(0)
                    mass=weights.sum();value=sum((ww*p[0,0,i] for i,ww in zip(selected,weights)),p.new_zeros(2))/mass.clamp_min(1)
                    torch.testing.assert_close(out['memory'][a][0,0,t],value,atol=1e-12,rtol=1e-12)
                ww=torch.stack([r[0,0,i] for i in ids]);logits=torch.stack([(q[0,0,t]*k[0,0,i]).sum()/2+ww[j].log() for j,i in enumerate(ids)])
                att=torch.softmax(logits,0);val=sum((aa*v[0,0,i] for i,aa in zip(ids,att)),v.new_zeros(4))*ww.sum().clamp_max(1)
                expected=u[0,0,t]+F.linear(val,m.output.weight[0])
                torch.testing.assert_close(out['content'][0,0,t],expected,atol=1e-12,rtol=1e-12)

    def test_positive_support_is_not_numerical_zero(self):
        a=torch.tensor([[1.,1.,0.]]);s=torch.tensor([[0.,-200.,100.]])
        p=weighted_softmax(s,a,a>0)
        self.assertGreater(float(p[0,1]),0);self.assertEqual(float(p[0,2]),0)
        torch.testing.assert_close(p.sum(-1),torch.ones(1))

    def test_role_initialization_survives_local_ablation(self):
        root=Path(__file__).resolve().parents[1]/'configs/studies/language/direct-v8/local'
        a=json.loads((root/'fc-anchor-s11.json').read_text())
        b=json.loads((root/'fc-heads2-s11.json').read_text())
        ma=DirectSemanticLM(a['direct']['plan']);mb=DirectSemanticLM(b['direct']['plan'])
        self.assertIs(ma.embedding.weight,ma.head.weight)
        for i in range(4):
            self.assertTrue(torch.equal(ma.memories[i].ffgate.weight,mb.memories[i].ffgate.weight))
            self.assertTrue(torch.equal(ma.selectors[i].local.a.weight,mb.selectors[i].local.a.weight))

if __name__=='__main__':unittest.main()

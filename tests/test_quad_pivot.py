"""Independent KKT/reference and regression checks for collapsed interiors."""
import math
import unittest
import torch
from tide_lab.direct_semantic_ops import allocation


def reference(scores,caps,budget,tau):
    """Scalar FP64 monotone root, independent of the vector active-set solver."""
    if budget<=0:return [0.]*len(scores)
    if budget>=math.fsum(caps):return list(caps)
    offset=max(s for s,m in zip(scores,caps) if m>0)
    t=[s-offset for s in scores]
    low=min(s-tau*m for s,m in zip(t,caps) if m>0)
    high=max(s for s,m in zip(t,caps) if m>0)
    for _ in range(100):
        mid=(low+high)/2
        mass=math.fsum(min(m,max(0.,(s-mid)/tau)) for s,m in zip(t,caps))
        if mass>budget:low=mid
        else:high=mid
    lam=(low+high)/2
    return [min(m,max(0.,(s-lam)/tau)) for s,m in zip(t,caps)]


def witness(dtype=torch.float32):
    # Eligible coordinates24,25,26,55,56 from the captured v11 failure;
    # removing zero-cap coordinates does not change this regional problem.
    t=torch.tensor([[1.4316635131835938,7.896038055419922,105.17596435546875,
                     73.19136810302734,-0.10172100365161896]],dtype=dtype)
    mu=torch.tensor([[.49997571110725403,1.,.5000242590904236,1.,1.]],dtype=dtype)
    return t,mu,torch.tensor([[1.5]],dtype=dtype),.12027693222849974


def stress_cases():
    g=torch.Generator().manual_seed(20261116)
    for offset in (0.,1e4,1e6):
        for tau in (.001,.12,1.,100.):
            t=torch.randn(32,64,generator=g)*30+offset
            mu=torch.rand(32,64,generator=g)
            mu=torch.where(torch.rand(32,64,generator=g)<.4,mu,torch.zeros_like(mu))
            mu[:,0]=1
            # Include almost-saturated budgets, empty/full rows and ordinary
            # interior roots without importing the old implementation.
            b=mu.sum(-1,keepdim=True)*torch.rand(32,1,generator=g)
            b[0]=0;b[1]=mu[1].sum();b[2]=min(1.5,float(mu[2].sum()))
            yield offset,tau,t,mu,b


class QuadPivotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_captured_cap_collapse_forward_and_vjp(self):
        t,mu,b,tau=witness();t.requires_grad_();mu.requires_grad_();b.requires_grad_()
        c,res=allocation(t,mu,b,b,tau)
        want=torch.tensor([reference(t[0].tolist(),mu[0].tolist(),b.item(),tau)])
        torch.testing.assert_close(c,want,atol=1e-6,rtol=1e-5)
        self.assertLessEqual(float(res.max()),2e-5)
        self.assertTrue(0<float(c[0,3])<1)
        g=torch.tensor([[1.,2.,3.,7.,11.]])
        dt,dm,db=torch.autograd.grad((c*g).sum(),(t,mu,b))
        torch.testing.assert_close(dt,torch.zeros_like(dt),atol=0,rtol=0)
        torch.testing.assert_close(dm,torch.tensor([[0.,0.,-4.,0.,0.]]),atol=0,rtol=0)
        torch.testing.assert_close(db,torch.tensor([[7.]]),atol=0,rtol=0)

    def test_zero_collapse_and_empty(self):
        t=torch.tensor([[1e6,0.],[9.,-4.]])
        mu=torch.tensor([[1.,1.],[0.,0.]])
        b=torch.tensor([[1e-5],[0.]])
        c,res=allocation(t,mu,b,b,.1)
        torch.testing.assert_close(c,torch.tensor([[1e-5,0.],[0.,0.]]),atol=1e-10,rtol=1e-6)
        self.assertLessEqual(float(res.max()),2e-5)

    def test_scalar_reference_stress(self):
        for offset,tau,t,mu,b in stress_cases():
            with self.subTest(offset=offset,tau=tau):
                c,res=allocation(t,mu,b,b,tau)
                want=torch.tensor([reference(s,m,q[0],tau) for s,m,q in zip(t.tolist(),mu.tolist(),b.tolist())])
                self.assertTrue(bool(torch.isfinite(c).all()))
                self.assertTrue(bool(((c>=0)&(c<=mu)).all()))
                self.assertLessEqual(float(res.max()),2e-5)
                torch.testing.assert_close(c,want,atol=1e-6,rtol=1e-5)

    def test_interior_gradcheck_and_interval_boundaries(self):
        t=torch.tensor([[.4,.2,-2.]],dtype=torch.float64,requires_grad=True)
        mu=torch.tensor([[.7,.8,1.]],dtype=torch.float64,requires_grad=True)
        b=torch.tensor([[.6]],dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda x,m,q:allocation(x,m,q,q,1.)[0],(t,mu,b)))
        lo=torch.tensor([[.1]],dtype=torch.float64);hi=torch.tensor([[1.]],dtype=torch.float64)
        c,_=allocation(t,mu,lo,hi,1.)
        torch.testing.assert_close(c,torch.tensor([[.4,.2,0.]],dtype=torch.float64),atol=0,rtol=0)


if __name__=='__main__':unittest.main()

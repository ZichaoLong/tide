"""Allocation §4.1/§6/S.2: independent Decimal optimization and map derivatives."""
from decimal import Decimal, localcontext
import pytest
import torch
from tidegraph.weighted.entropy_allocation import allocate
from tidegraph.weighted.regularizers import Regularizer
from tidegraph.weighted import allocate_quad

RULES=[("bern",None),("ent",1.),("ent",1.000001),("ent",1.5),("ent",2.)]


def decimal_reference(scores,caps,lo,hi,tau,kind,alpha):
    """80-digit scalar bisection on a fixed broad bracket; no Tensor solver."""
    with localcontext() as ctx:
        ctx.prec=80
        d=lambda x:Decimal.from_float(float(x))
        t=list(map(d,scores));mu=list(map(d,caps));low,high,temp=map(d,(lo,hi,tau))
        shape=d(alpha) if alpha is not None else None
        def response(lam):
            result=[]
            for s,m in zip(t,mu):
                if m==0:result.append(Decimal(0));continue
                x=(s-lam)/temp
                if kind=="bern":c=m/(1+(-x).exp())
                elif shape==1:c=min(m,(x-1).exp())
                else:
                    base=max(Decimal(0),(shape-1)*x+1/shape)
                    c=min(m,base**(1/(shape-1)))
                result.append(c)
            return result
        if sum(mu)==0 or high==0:return [0.]*len(mu)
        if low==sum(mu):return list(map(float,mu))
        values=response(Decimal(0));total=sum(values)
        if low<=total<=high:return list(map(float,values))
        target=low if total<low else high
        left=min(t)-1000;right=max(t)+1000
        for _ in range(300):
            mid=(left+right)/2
            if sum(response(mid))>target:left=mid
            else:right=mid
        return list(map(float,response((left+right)/2)))


@pytest.mark.parametrize("kind,alpha",RULES)
def test_entropy_decimal_optima_and_score_shift(dtype,kind,alpha):
    scores=torch.tensor([.4,-.7,1.1,-.2],dtype=dtype)
    caps=torch.tensor([.7,.9,.25,0.],dtype=dtype)
    tau=scores.new_tensor(.8)
    for low,high in ((.2,1.),(1.4,1.7),(.6,.6),(.2,.4),(0.,3.)):
        a=allocate(scores,caps,low,high,tau,regularizer=kind,ent_alpha=alpha)
        expected=scores.new_tensor(decimal_reference(scores,caps,scores.new_tensor(low),scores.new_tensor(high),tau,kind,alpha))
        torch.testing.assert_close(a.controls,expected,atol=1e-6 if dtype==torch.float32 else 1e-11,rtol=1e-5 if dtype==torch.float32 else 1e-9)
        assert float(a.residual)<1e-5
    target=scores.new_tensor(.8)
    a=allocate(scores,caps,target,target,tau,regularizer=kind,ent_alpha=alpha)
    shifted=allocate(scores+10,caps,target,target,tau,regularizer=kind,ent_alpha=alpha)
    torch.testing.assert_close(a.controls,shifted.controls)


@pytest.mark.parametrize("kind,alpha",RULES)
def test_entropy_exact_map_vjp_and_temperature(dtype,kind,alpha):
    args=(torch.tensor([.2,-.1,.4],dtype=dtype,requires_grad=True),
          torch.tensor([.7,.8,.6],dtype=dtype,requires_grad=True),
          torch.tensor(.9,dtype=dtype,requires_grad=True),torch.tensor(.8,dtype=dtype,requires_grad=True))
    t,mu,total,tau=args
    result=allocate(t,mu,total,total,tau,regularizer=kind,ent_alpha=alpha)
    rule=Regularizer(kind,alpha);c=result.controls;free=(c>0)&(c<mu)
    # Independent implicit Jacobian from optimal occupancies (not bracket q).
    if kind=="bern":
        p=c/mu;a=mu*p*(1-p)/tau;e=p
    else:
        a=torch.where(free,c.pow(2-alpha)/tau,torch.zeros_like(c));e=((c==mu)&(mu>0)).to(dtype)
    expected=torch.diag(a)-torch.outer(a,a)/a.sum()
    rows=torch.stack([torch.autograd.grad(v,t,retain_graph=True)[0] for v in c])
    torch.testing.assert_close(rows,expected)
    root=t.new_tensor([.3,-.5,.7]);mean=(a*root).sum()/a.sum()
    gc=torch.autograd.grad((root*c).sum(),(mu,total,tau),retain_graph=True)
    torch.testing.assert_close(gc[0],e*(root-mean));torch.testing.assert_close(gc[1],mean)
    if dtype==torch.float64:
        assert torch.autograd.gradcheck(lambda s,m,b,temp:allocate(s,m,b,b,temp,regularizer=kind,ent_alpha=alpha).controls,
                                       args,eps=1e-6,atol=2e-6,rtol=2e-5)


def test_shannon_softmax_nonunit_caps_and_ent_two_shift(dtype):
    t=torch.tensor([.2,-.1,.4],dtype=dtype,requires_grad=True);tau=torch.tensor(.8,dtype=dtype,requires_grad=True)
    mu=t.new_tensor([.8,.7,.9]);one=t.new_tensor(1.)
    shannon=allocate(t,mu,one,one,tau,regularizer="ent",ent_alpha=1).controls
    torch.testing.assert_close(shannon,(t/tau).softmax(0))
    # Caps need not equal one; touching/exceeding them changes the map.
    capped=allocate(t,mu.new_tensor([.1,.7,.9]),one,one,tau,regularizer="ent",ent_alpha=1).controls
    assert capped[0]==mu.new_tensor(.1) and not torch.allclose(capped,shannon)
    for lo,hi in ((.2,2.),(.6,.6)):
        ent=allocate(t,mu,lo,hi,tau,regularizer="ent",ent_alpha=2).controls
        shifted=allocate_quad(t+tau/2,mu,lo,hi,tau).controls
        torch.testing.assert_close(ent,shifted)
        a=torch.autograd.grad(ent.square().sum(),(t,tau),retain_graph=True)
        b=torch.autograd.grad(shifted.square().sum(),(t,tau),retain_graph=True)
        for x,y in zip(a,b):torch.testing.assert_close(x,y)


@pytest.mark.parametrize("kind,alpha",RULES)
def test_entropy_singletons_zero_caps_and_branch_boundaries(dtype,kind,alpha):
    t=torch.tensor([0.,0.,0.],dtype=dtype,requires_grad=True)
    mu=torch.tensor([.4,.7,0.],dtype=dtype,requires_grad=True)
    zero=allocate(t,mu,0.,0.,regularizer=kind,ent_alpha=alpha)
    assert not zero.controls.requires_grad and zero.constraint=="zero"
    full=allocate(t,mu,mu.sum(),2.,regularizer=kind,ent_alpha=alpha)
    assert full.controls is mu and full.constraint=="caps"
    c=allocate(t[:1],mu[:1],.3,.3,regularizer=kind,ent_alpha=alpha).controls
    torch.testing.assert_close(c,t.new_tensor([.3]))
    torch.testing.assert_close(torch.autograd.grad(c.sum(),t)[0],torch.zeros_like(t))


def test_positive_support_underflow_and_saturated_bernoulli_slope(dtype):
    t=torch.tensor([-10000.,0.],dtype=dtype)
    for kind,alpha in (("bern",None),("ent",1.)):
        with pytest.raises(ArithmeticError,match="support underflowed"):
            allocate(t,torch.ones_like(t),0.,2.,regularizer=kind,ent_alpha=alpha)
    # Bernoulli has an interior derivative even when sigmoid rounds to one.
    x=torch.tensor([20. if dtype==torch.float32 else 40.],dtype=dtype,requires_grad=True)
    mu=torch.ones_like(x)
    c=allocate(x,mu,0.,2.,regularizer="bern").controls
    grad=torch.autograd.grad(c.sum(),x)[0]
    assert grad>0
    torch.testing.assert_close(grad,torch.exp(-x)/(1+torch.exp(-x)).square())


def test_entropy_configuration_shape_is_fixed():
    for kind,alpha in (("invalid",None),("bern",1.5),("ent",None),("ent",.9),("ent",2.1),("ent",torch.tensor(1.5))):
        with pytest.raises(ValueError):Regularizer(kind,alpha)


def test_quad_configuration_identity_remains_compatible():
    from tidegraph.weighted.config import RegionalConfig,identity
    # Recorded from the immutable pre-entropy implementation 442e8b2.
    assert identity("weighted-regional-total-v1",RegionalConfig()) == (
        "weighted-regional-total-v1:333b0d2be7b05cc31f6ee536f7a36d5ba50ddd6841cfb0bd8f68774a6ceb902d")
    for kwargs in (dict(regularizer="ent"),dict(regularizer="bern",ent_alpha=1.5)):
        with pytest.raises(ValueError):RegionalConfig(**kwargs)
    a=RegionalConfig(regularizer="ent",ent_alpha=1.)
    b=RegionalConfig(regularizer="ent",ent_alpha=1.000001)
    assert identity("weighted-regional-total-v1",a)!=identity("weighted-regional-total-v1",b)

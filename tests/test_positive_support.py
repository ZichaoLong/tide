"""TL-REQ-003 actual row, independent closed-form derivatives and storage edges."""
from decimal import Decimal, localcontext
import json
from pathlib import Path
import pytest
import torch
from tidegraph.weighted import allocate, allocate_batch, observe_support, RegionalConfig
from tidegraph.weighted.support_policy import FLOOR
from tidegraph.weighted.stable_mass import scaled_mean, mass_attention
from test_positive_support_graph import device  # Explicit CPU/NPU target fixture.


def row(device="cpu", dtype=torch.float32):
    data = json.loads((Path(__file__).parent/"data/tl_req_003_shannon_row.json").read_text())
    make = lambda v: torch.tensor(v, dtype=dtype, device=device, requires_grad=True)
    return make(data["scores"]), make(data["caps"]), make(data["target"]), make(data["temperature"])


def decimal_face(scores, caps, target, tau, root):
    """Actual row: last cap saturated; verify this face without a Tensor solver."""
    with localcontext() as ctx:
        ctx.prec = 80
        d = lambda x: Decimal.from_float(float(x))
        s, m, b, t, g = list(map(d,scores)), list(map(d,caps)), d(target), d(tau), list(map(d,root))
        exp = [((v-s[0])/t).exp() for v in s[:3]]
        c = [(b-m[3])*v/sum(exp) for v in exp]+[m[3]]
        lam = s[0]-t*(1+c[0].ln())
        assert all(0 < c[i] < m[i] for i in range(3))
        assert ((s[3]-lam)/t-1).exp() >= m[3]
        a = [v/t for v in c[:3]]+[Decimal(0)]
        mean = sum(v*w for v,w in zip(a,g))/sum(a)
        ds = [v*(w-mean) for v,w in zip(a,g)]
        dc = [Decimal(0)]*3+[g[3]-mean]
        dt = sum(-(s[i]-lam)/t*ds[i] for i in range(4))
        return [float(v) for v in c], [float(v) for v in ds], [float(v) for v in dc], float(mean), float(dt)


def call(args, policy="strict", batched=False):
    s,c,b,t = args
    kwargs = dict(regularizer="ent", ent_alpha=1., support_policy=policy)
    return (allocate_batch([s],[c],[b],[b],t,**kwargs)[0] if batched else allocate(s,c,b,b,t,**kwargs))


@pytest.mark.parametrize("batched", [False, True])
def test_actual_row_strict_failure_and_floor(batched,device):
    args = row(device)
    with pytest.raises(ArithmeticError, match="support underflowed"):
        call(args, batched=batched)
    with observe_support() as observed:
        result = call(args, FLOOR, batched)
    expected, *_ = decimal_face(*args, [0,0,0,0])
    assert result.controls[2].item() == 2.**-149
    assert result.support_repairs.tolist() == [False,False,True,False]
    assert bool((result.controls > 0).all())
    torch.testing.assert_close(result.controls, result.controls.new_tensor(expected), atol=1e-6, rtol=1e-5)
    assert observed.metrics()["repaired_coordinates"] == 1
    assert result.backward == "ent-positive-floor-v1"


@pytest.mark.parametrize("batched", [False, True])
def test_unfloored_response_vjp_retains_large_cotangent(batched,device):
    args = row(device)
    root = args[0].new_tensor([0,0,1e30,0])
    _, ds, dc, db, dt = decimal_face(*args, root)
    actual = torch.autograd.grad(call(args,FLOOR,batched).controls, args, root)
    for got, expected in zip(actual, (ds,dc,db,dt)):
        # Zero absolute allowance makes tiny derivatives observable; a blanket
        # atol would accept the lost root or the derivative of floored c.
        torch.testing.assert_close(got, got.new_tensor(expected), atol=0, rtol=1e-5)


def test_independent_rows_shared_temperature_none_zero_and_versions():
    a,b = row(),row()
    tau = a[3]
    results = allocate_batch([a[0],b[0]],[a[1],b[1]],[a[2],b[2]],[a[2],b[2]],tau,
                             regularizer="ent",ent_alpha=1.,support_policy=FLOOR)
    roots = (a[0],b[0],tau)
    grads = torch.autograd.grad(results[0].controls.sum(), roots, retain_graph=True, allow_unused=True)
    assert grads[1] is None
    zeros = torch.autograd.grad(results[0].controls, roots, torch.zeros(4), retain_graph=True, allow_unused=True)
    assert zeros[1] is None and torch.equal(zeros[0],torch.zeros_like(a[0])) and zeros[2].item()==0
    loss = sum(r.controls.square().sum() for r in results)
    combined = torch.autograd.grad(loss,tau,retain_graph=True)[0]
    separate = sum(torch.autograd.grad(r.controls.square().sum(),tau,retain_graph=True)[0] for r in results)
    torch.testing.assert_close(combined,separate,atol=0,rtol=0)
    with torch.no_grad(): a[0].add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        torch.autograd.grad(loss,tau)


def test_zero_caps_endpoints_and_weak_mass_refusal():
    s = torch.tensor([-1000.,0.],requires_grad=True)
    c = torch.tensor([0.,1.],requires_grad=True)
    options = dict(regularizer="ent",ent_alpha=1.,support_policy=FLOOR)
    assert allocate(s,c,.5,.5,**options).controls[0] == 0
    assert not allocate(s,c,0.,0.,**options).controls.requires_grad
    assert allocate(s,c,c.sum(),2.,**options).controls is c
    with pytest.raises(ArithmeticError,match="mass perturbation"):
        allocate(s.new_full((2,),-1000.),torch.ones(2),0.,2.,**options)
    with pytest.raises(ValueError,match="requires Shannon"):
        allocate(s,c,.5,.5,support_policy=FLOOR)
    with pytest.raises(ValueError,match="Shannon"):
        RegionalConfig(support_policy=FLOOR)


def test_scaled_singleton_mean_and_unrepresentable_public_adjoint(device):
    mass = torch.tensor([2.**-149],device=device,requires_grad=True)
    values = torch.tensor([[.25]],device=device,requires_grad=True)
    y = scaled_mean(mass,values)
    assert y.item() == .25
    dm,dv = torch.autograd.grad(y.sum(),(mass,values))
    assert dm.item()==0 and dv.item()==1
    mass = torch.tensor([2.**-149,2.**-149],device=device,requires_grad=True)
    values = torch.tensor([[0.],[1.]],device=device,requires_grad=True)
    y = scaled_mean(mass,values)
    assert y.item()==.5
    with pytest.raises(ArithmeticError,match="requested dtype range"):
        y.sum().backward()


def test_mass_attention_fuses_log_weight_adjoint(device):
    w = torch.tensor([2.**-149,2.**-148],device=device,requires_grad=True)
    v = torch.tensor([[1.],[3.]],device=device,requires_grad=True)
    logits = torch.zeros(2,device=device,requires_grad=True)
    result = mass_attention(logits,v,w,w.sum())
    assert result.item() == 7*2.**-149
    dw = torch.autograd.grad(result.sum(),w)[0]
    torch.testing.assert_close(dw,w.new_tensor([1.,3.]),atol=1e-6,rtol=1e-5)


def test_normal_scale_mean_and_attention_independent_gradcheck():
    make = lambda x: torch.tensor(x,dtype=torch.float64,requires_grad=True)
    weights, values, logits = make([.1,.2,.3]),make([[1.,2.],[3.,-1.],[.4,.8]]),make([.2,-.3,.5])
    assert torch.autograd.gradcheck(scaled_mean,(weights,values))
    assert torch.autograd.gradcheck(lambda l,v,w:mass_attention(l,v,w,w.sum()),(logits,values,weights))


@pytest.mark.parametrize("bounds",[(0.,4.),(.9,3.),(0.,.2),(.8,.8)])
def test_ordinary_vjp_matches_original_map_and_shift(bounds):
    make=lambda v: torch.tensor(v,dtype=torch.float64,requires_grad=True)
    args=[make([-.6,.1,.4]),make([.7,.8,.9]),make(bounds[0]),make(bounds[1]),make(.4)]
    shift=make(.1)
    options=dict(regularizer="ent",ent_alpha=1.,score_shift=shift)
    a=allocate(*args,**options); b=allocate(*args,**options,support_policy=FLOOR)
    torch.testing.assert_close(a.controls,b.controls,atol=0,rtol=0)
    root=make([.7,-.3,.2])
    expected=torch.autograd.grad(a.controls,(*args,shift),root,allow_unused=True)
    actual=torch.autograd.grad(b.controls,(*args,shift),root,allow_unused=True)
    for x,y in zip(expected,actual):
        if x is None or y is None:
            assert x is y
        else:
            torch.testing.assert_close(x,y,atol=1e-10,rtol=1e-8)


def test_fp64_range_counterexample_retains_unfloored_derivative():
    make=lambda x:torch.tensor(x,dtype=torch.float64,requires_grad=True)
    s,c,b,t=make([-200.,0.,0.,0.]),make([1.]*4),make(2.),make(.25)
    with pytest.raises(ArithmeticError,match="support underflowed"):
        call((s,c,b,t))
    result=call((s,c,b,t),FLOOR)
    assert result.controls[0].item() == 2.**-1074
    root=make([1e300,0.,0.,0.])
    actual=torch.autograd.grad(result.controls,s,root)[0]
    with localcontext() as ctx:
        ctx.prec=80
        e=Decimal(-800).exp()
        controls=[2*e/(e+3)]+[2/(e+3)]*3
        g=Decimal.from_float(1e300)
        mean=controls[0]*g/2
        ds=[v*((g if i==0 else 0)-mean)/Decimal(".25") for i,v in enumerate(controls)]
    torch.testing.assert_close(actual,make([float(v) for v in ds]),atol=0,rtol=1e-8)


def test_rejection_keeps_row_snapshot():
    from tidegraph.weighted.allocation_failure import AllocationFailure
    s,c,b,t=row()
    with pytest.raises(AllocationFailure) as caught:
        allocate_batch([s,s.new_full((3,),-1000.),s.new_full((4,),-1000.)],
            [c,torch.ones(3),c],[b,0.,0.],[b,3.,4.],t,
            regularizer="ent",ent_alpha=1.,support_policy=FLOOR)
    assert caught.value.row == 2  # first width-four group maps to original row.
    assert caught.value.details["support_policy"] == FLOOR
    assert caught.value.record()["tensors"]["scores"]["values"] == [-1000.]*4


def test_attention_small_mass_large_root(device):
    delta=2.**-149
    w=torch.tensor([delta,delta],device=device,requires_grad=True)
    v=torch.tensor([[1.],[3.]],device=device,requires_grad=True)
    l=torch.tensor([-4.,0.],device=device,requires_grad=True)
    value=mass_attention(l,v,w,w.sum())
    actual=torch.autograd.grad(value,(l,v,w),w.new_tensor([1e30]))
    wd,vd,ld=[x.detach().cpu().double().requires_grad_() for x in (w,v,l)]
    expected=(ld+wd.log()).softmax(0)@vd*wd.sum()
    grads=torch.autograd.grad(expected,(ld,vd,wd),torch.tensor([1e30],dtype=torch.float64))
    for a,b in zip(actual,grads):
        torch.testing.assert_close(a.cpu(),b.float(),atol=0,rtol=1e-5)

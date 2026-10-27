"""Allocation §6.2: exact common-shift null direction on active mass constraints.

No tolerance-based gradient clipping: eta is an independent formal argument.
"""
import pytest
import torch
from tidegraph.weighted import allocate
from test_weighted_kv_formulas import close


RULES=[("quad",None),("bern",None),("ent",1.),("ent",1.5)]


def evaluate(backend,s,mu,lo,hi,tau,rule,alpha,eta):
    if backend=="python":
        return allocate(s,mu,lo,hi,tau,regularizer=rule,ent_alpha=alpha,score_shift=eta).controls
    import _tide_native as core
    return core.weighted_allocate_shifted(s,mu,s.new_tensor(lo) if not isinstance(lo,torch.Tensor) else lo,
        s.new_tensor(hi) if not isinstance(hi,torch.Tensor) else hi,tau,rule,alpha,eta)


@pytest.mark.parametrize("backend",["python","native"])
@pytest.mark.parametrize("rule,alpha",RULES)
@pytest.mark.parametrize("branch",["lower","upper","equality","slack"])
def test_formal_threshold_derivative_on_mass_constraint(dtype,backend,rule,alpha,branch):
    s=torch.tensor([-2.,-1.] if branch=="lower" else [2.,3.] if branch=="upper" else [.2,.3],dtype=dtype,requires_grad=True)
    mu=torch.tensor([.7,.8],dtype=dtype,requires_grad=True);eta=s.new_tensor(.1,requires_grad=True);tau=s.new_tensor(1.)
    lo,hi=(.9,2.) if branch=="lower" else (0.,.6) if branch=="upper" else (.65,.65) if branch=="equality" else (0.,2.)
    c=evaluate(backend,s,mu,lo,hi,tau,rule,alpha,eta)
    gs,ge=torch.autograd.grad(c@s.new_tensor([.23,-.47]),(s,eta),retain_graph=True)
    if branch!="slack":assert ge.item()==0.
    else:close(ge,-gs.sum())
    if dtype==torch.float64:
        eps=1e-5
        plus=evaluate(backend,s.detach(),mu.detach(),lo,hi,tau,rule,alpha,eta.detach()+eps)
        minus=evaluate(backend,s.detach(),mu.detach(),lo,hi,tau,rule,alpha,eta.detach()-eps)
        close(((plus-minus)/(2*eps))@s.new_tensor([.23,-.47]),ge)


@pytest.mark.parametrize("backend",["python","native"])
@pytest.mark.parametrize("rule,alpha",RULES)
def test_common_cotangent_offset_and_bound_dependency(dtype,backend,rule,alpha):
    s=torch.tensor([.2,.3,.1],dtype=dtype,requires_grad=True);mu=s.new_tensor([.7,.8,.6],requires_grad=True)
    eta=s.new_tensor(.1,requires_grad=True);tau=s.new_tensor(.8,requires_grad=True)
    bound=.7+.1*eta
    c=evaluate(backend,s,mu,bound,bound,tau,rule,alpha,eta)
    close(torch.autograd.grad(c.sum(),eta,retain_graph=True)[0],eta.new_tensor(.1))
    # Adding a constant cotangent on a fixed-sum map only changes dL/dB.
    g=s.new_tensor([.25,.5,.75]);roots=(s,mu,tau)
    a=torch.autograd.grad(c,roots,g,retain_graph=True)
    b=torch.autograd.grad(c,roots,g+4096.,retain_graph=True)
    for x,y in zip(a,b):assert torch.equal(x,y)
    with torch.no_grad():eta.add_(.01)
    with pytest.raises(RuntimeError,match="modified by an inplace operation"):c.sum().backward()


@pytest.mark.parametrize("backend",["python","native"])
@pytest.mark.parametrize("rule,alpha",RULES)
def test_zero_cap_cotangent_does_not_choose_the_center(dtype,backend,rule,alpha):
    s=torch.tensor([0.,.1,.3],dtype=dtype,requires_grad=True);mu=s.new_tensor([0.,.7,.8])
    c=evaluate(backend,s,mu,.7,.7,s.new_tensor(1.),rule,alpha,s.new_tensor(0.))
    a=torch.autograd.grad(c,s,s.new_tensor([0.,.2,.4]),retain_graph=True)[0]
    b=torch.autograd.grad(c,s,s.new_tensor([1e20,.2,.4]),retain_graph=True)[0]
    assert torch.equal(a,b)

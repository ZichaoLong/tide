"""Native primitives independently checked against equation/vertex anchors."""
import math
import pytest
import torch
from tidegraph.weighted import attention_shares,total_zero,hard_selection
from test_weighted_kv_formulas import close


def test_native_attention_zero_base_right_derivative(dtype):
    import _tide_native as core
    s=torch.tensor([0.,math.log(2),math.log(3)],dtype=dtype,requires_grad=True)
    a=torch.tensor([1.,0.,1.],dtype=dtype,requires_grad=True);tau=s.new_tensor(1.,requires_grad=True)
    p=core.weighted_attention_shares(s,a,tau)
    close(p,s.new_tensor([.25,0.,.75]))
    ds,da,dt=torch.autograd.grad(p[1],(s,a,tau),retain_graph=True)
    close(ds,torch.zeros_like(s));close(da,s.new_tensor([0.,.5,0.]));close(dt,s.new_zeros(()))
    g=torch.tensor([.7,-.2,.3],dtype=dtype)
    expected=attention_shares(s,a,tau)
    for x,y in zip(torch.autograd.grad(p,(s,a,tau),g),torch.autograd.grad(expected,(s,a,tau),g)):close(x,y)


def test_native_sorted_fractional_fill_and_hard_ties(dtype):
    import _tide_native as core
    s=torch.tensor([1.,1.,1.,0.],dtype=dtype,requires_grad=True)
    mu=torch.tensor([.3,.3,.3,.3],dtype=dtype,requires_grad=True);b=s.new_tensor(1.,requires_grad=True)
    c=core.weighted_total_zero(s,mu,b,[3,2,1,0]);close(c,s.new_tensor([.3,.3,.3,.1]))
    ds,dm,db=torch.autograd.grad(c[3],(s,mu,b),allow_unused=True)
    assert ds is None;close(dm,s.new_tensor([-1.,-1.,-1.,0.]));close(db,s.new_tensor(1.))
    close(core.weighted_hard_selection(s,mu,1,[3,2,1,0]),s.new_tensor([0.,0.,.3,0.]))
    close(core.weighted_hard_selection(-s,mu,2,[3,2,1,0]),s.new_tensor([0.,0.,0.,.3]))
    close(core.weighted_proportional(mu,1),s.new_tensor([.25]*4))


def test_native_full_caps_are_exact_not_a_spurious_partial(dtype):
    import _tide_native as core
    s=torch.tensor([1.,3.,2.],dtype=dtype,requires_grad=True)
    mu=torch.tensor([.1,.3,.6],dtype=dtype,requires_grad=True);b=mu.detach().sum().requires_grad_()
    c=core.weighted_total_zero(s,mu,b,[0,1,2]);assert torch.equal(c,mu)
    gm,gb=torch.autograd.grad(c.sum(),(mu,b));close(gm,torch.ones_like(mu));close(gb,b.new_zeros(()))


@pytest.mark.parametrize("function",["attention","zero","hard"])
def test_native_sending_input_versions_and_support(dtype,function):
    import _tide_native as core
    s=torch.tensor([1.,.4],dtype=dtype,requires_grad=True);mu=torch.ones_like(s,requires_grad=True)
    result=(core.weighted_attention_shares(s,mu,s.new_tensor(1.)) if function=="attention" else
            core.weighted_total_zero(s,mu,s.new_tensor(.6),[0,1]) if function=="zero" else
            core.weighted_hard_selection(s,mu,1,[0,1]))
    with torch.no_grad():s.add_(.1)
    with pytest.raises(RuntimeError,match="modified by an inplace operation"):result.sum().backward()
    with pytest.raises(RuntimeError,match="support underflowed"):
        core.weighted_attention_shares(s.new_tensor([-10000.,0.]),mu,s.new_tensor(1.))

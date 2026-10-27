"""Fixed-time convolution §2 scalar example and §3 event-shift counterexample."""
import pytest
import torch
from tidegraph.weighted.convolution import ConvCache,FixedTimeConv


def example(dtype):
    c=FixedTimeConv(1,3,2,generator=torch.Generator().manual_seed(2),dtype=dtype)
    with torch.no_grad():
        c.weight.copy_(torch.tensor([[1.],[2.],[3.]],dtype=dtype));c.bias.zero_()
    return c


def test_fixed_time_textbook_weak_event_and_snapshot_equality(dtype):
    conv=example(dtype);ref=torch.zeros(1,dtype=dtype)
    cache=conv.write(ConvCache.empty(ref),0,ref.new_tensor(1.),ref.new_tensor([4.]))
    for epsilon in (.2,.01,0.):
        weight=ref.new_tensor(epsilon,requires_grad=True)
        inserted=conv.write(cache,2,weight,ref.new_tensor([5.]))
        value=conv(inserted,4,ref.new_tensor([6.]))
        torch.testing.assert_close(value,ref.new_tensor([18+10*epsilon]))
        if epsilon:
            torch.testing.assert_close(torch.autograd.grad(value.sum(),weight)[0],ref.new_tensor(10.))
        else:
            assert inserted is cache
        now=conv.write(inserted,4,ref.new_tensor(.3),ref.new_tensor([6.]))
        torch.testing.assert_close(conv(now,4,ref.new_tensor([6.])),value)
        assert now.times.tolist()==[0,2,4] if epsilon else now.times.tolist()==[0,4]
    # Event compaction would move the time-0 entry to lag 1 and return 14.
    torch.testing.assert_close(conv(cache,4,ref.new_tensor([6.])),ref.new_tensor([18.]))


def test_sparse_residues_large_int64_and_exact_expired_cache(dtype):
    conv=example(dtype);ref=torch.zeros(1,dtype=dtype);clock=2**54+3
    cache=ConvCache.empty(ref)
    for time,value in ((clock,2.),(clock+1,99.),(clock+2,3.)):
        cache=conv.write(cache,time,ref.new_tensor(1.),ref.new_tensor([value]))
    torch.testing.assert_close(conv(cache,clock+4,ref.new_tensor([1.])),ref.new_tensor([13.]))
    idle=conv.write(cache,clock+20,ref.new_tensor(0.),ref.new_tensor([5.]))
    assert idle is cache and idle.times[0].item()==clock
    torch.testing.assert_close(conv(idle,clock+20,ref.new_tensor([1.])),ref.new_tensor([1.]))
    _,effective=conv.effective(idle,clock+20)
    assert effective.count_nonzero()==0 and len(idle.times)==3
    slots,_=conv.effective(idle,2**63-1)
    assert slots.tolist()==list(range(2**63-5,2**63))


@pytest.mark.parametrize("size",[1,4])
def test_fixed_time_equals_dense_causal_convolution_and_vjp(dtype,size):
    c=FixedTimeConv(3,size,5,generator=torch.Generator().manual_seed(5),dtype=dtype)
    x=torch.linspace(-.7,.8,18,dtype=dtype).reshape(6,3).requires_grad_()
    cache=ConvCache.empty(x[0]);actual=[]
    for i,z in enumerate(x):
        actual.append(c(cache,5*i+1,z));cache=c.write(cache,5*i+1,z.new_tensor(1.),z)
    # Standard grouped conv1d uses oldest-to-current kernels; reverse our lags.
    expected=torch.nn.functional.conv1d(torch.nn.functional.pad(x.t().unsqueeze(0),(size-1,0)),
             c.weight.flip(0).t().unsqueeze(1),c.bias,groups=3)[0].t()
    actual=torch.stack(actual)
    torch.testing.assert_close(actual,expected)
    ga=torch.autograd.grad(actual.square().sum(),(x,c.weight,c.bias),retain_graph=True)
    ge=torch.autograd.grad(expected.square().sum(),(x,c.weight,c.bias))
    for a,b in zip(ga,ge):torch.testing.assert_close(a,b)

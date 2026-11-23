import os

import pytest
import torch

from tide.allocation import allocate
from tide.region import region_control


@pytest.fixture(scope="module")
def device():
    name = os.environ.get("TIDE_STABLE_DEVICE", "cpu")
    if name.startswith("npu"):
        import torch_npu
        torch.npu.set_device(name)
    return torch.device(name)


@pytest.mark.parametrize("delta", [5.7220458984375e-6, 2.**-20, 0.])
def test_weak_regional_remainder_and_received_derivative(delta, device):
    s = torch.tensor([34.98263168334961, -205.1953582763672], device=device)
    w = torch.tensor([.5-delta/2, .5+delta/2], device=device, requires_grad=True)
    out, zero = region_control(s[:,None,None], w[:,None,None], torch.zeros(2, dtype=torch.bool, device=device),
                               1., "quad", "mixed", return_constant_zero=True)
    assert out.flatten().tolist() == [1-delta, delta]
    assert zero.flatten().tolist() == [False, delta == 0]
    assert torch.autograd.grad(out[1].sum(), w)[0].tolist() == ([-2., 0.] if delta else [0.,0.])


@pytest.mark.parametrize("scores,w,cap,k,expected,score_grad", [
    ([2.,2.**-26],[.5,.5],[1.,1.],1.,[1.,0.],[0.,0.]),
    ([-1.,-3.],[1.,2.**-27],[1.,2.**-26],1.,[1.,0.],[0.,0.]),
    ([2.,2.],[1.,2.**-27],[1.,2.**-26],2.,[1.,2.**-26],[0.,0.]),
    ([2.,0.],[.5,.5],[1.,1.],1.,[1.,0.],[0.,0.]),
])
def test_early_return_total_source_and_exact_tie(device,scores,w,cap,k,expected,score_grad):
    scores = torch.tensor(scores, device=device, requires_grad=True)
    upper = torch.tensor(cap, device=device)
    weights = torch.tensor(w, device=device)
    out = allocate(scores, weights, upper, 0., upper, k)
    assert out.tolist() == expected
    assert torch.autograd.grad(out[1],scores)[0].tolist() == score_grad


def test_real_interior_rounded_to_cap_uses_intermediate_face(device):
    # x0 = 1 - 2^-26 rounds to 1. Locally x0 = k - upper1.
    caps = torch.tensor([1.,2.**-26], device=device, requires_grad=True)
    k = torch.tensor(1., device=device, requires_grad=True)
    out = allocate(caps.new_tensor([2.,2.]), caps.new_tensor([1.,2.**-27]), caps, 0., caps, k)
    assert out.tolist() == [1.,2.**-26]
    cap_grad, k_grad = torch.autograd.grad(out[0],(caps,k))
    assert cap_grad.tolist() == [0.,-1.] and k_grad.item() == 1.


def test_three_node_remainder_and_excess_floor(device):
    delta = 5.7220458984375e-6
    caps = torch.tensor([.5,.5-delta,1.], device=device, requires_grad=True)
    out = allocate(caps.new_tensor([35.,20.,-205.]), caps, caps, 0., caps, 1.)
    assert out.tolist() == [.5,.5-delta,delta]
    assert torch.autograd.grad(out[2],caps)[0].tolist() == [-1.,-1.,0.]
    floors = caps.new_tensor([1.,2.**-26])
    with pytest.raises(ValueError, match="floors exceed"):
        allocate(floors, floors, floors, floors, floors, 1.)


def test_positive_interior_rounds_to_zero_without_constant_lower_proof():
    unit = torch.finfo(torch.float32).tiny * torch.finfo(torch.float32).eps
    scores = torch.tensor([3.,-1.,-1.],requires_grad=True)
    upper = torch.tensor([2*unit,1.,1.])
    out, lower = allocate(scores, torch.full_like(upper,unit), upper, 0., upper, 1., return_lower_face=True)
    assert out.tolist() == [2*unit,0.,0.] and lower.tolist() == [False,False,False]
    assert torch.autograd.grad(out[1],scores)[0].tolist() == [0.,.5,-.5]


def test_third_sum_component_controls_true_received_face(device):
    w = torch.tensor([.5, 2.**-26, 2.**-100], device=device, requires_grad=True)
    s = torch.tensor([.5, 2.**-26, 0.], device=device, requires_grad=True)
    cap = torch.ones_like(s)
    out, lower = allocate(s,w,cap,0.,cap,1.,return_lower_face=True)
    # All three scores are interior once the missing receipt is distributed.
    assert out[2].item() == torch.tensor(2.**-100/3).item() and not lower.any().item()
    gw, = torch.autograd.grad(out.sum(),w)
    assert gw.tolist() == [1.,1.,1.]


def test_capped_score_does_not_distort_gap_between_two_interiors(device):
    s = torch.tensor([-86.10834503173828, -85.5880126953125, 12.589069366455078],
                     device=device, requires_grad=True)
    w = s.new_tensor([.5782507658004761, .4217492341995239, .125])
    upper = s.new_tensor([1., .8434984683990479, .25])
    out = allocate(s, w, upper, s.new_tensor([0., 0., .0625]), upper, 2.)
    # The capped third node leaves .75; the two free weights differ by s0-s1.
    assert out.tolist() == [.11483383178710938, .6351661682128906, .25]
    assert torch.autograd.grad(out[0], s)[0].tolist() == [.5, -.5, 0.]

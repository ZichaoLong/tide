"""Device-sensitive sending and typed-copy truth anchors, independent of schedules."""
import torch
from tidegraph.qualification_checks import compare_finite, assert_placement
from tidegraph.weighted.sending import total_zero, hard_selection
from tidegraph.weighted.placement_tensors import mapper


def check(device):
    # IDs cannot be sorted after a float cast: these values differ below FP64 ULP.
    ids = [2**54+7, 2**54+5, 2**54+6]
    s = torch.tensor([1., 1., 1.], device=device, requires_grad=True)
    mu = torch.tensor([.25, .5, .75], device=device, requires_grad=True)
    total = torch.tensor(.625, device=device, requires_grad=True)
    c = total_zero(s, mu, total, ids)
    compare_finite(torch.tensor([0., .5, .125]), c)
    assert_placement(c, device)
    bars = torch.tensor([2., 3., 5.], device=device)
    gs, gm, gt = torch.autograd.grad((c*bars).sum(), (s, mu, total), allow_unused=True)
    assert gs is None
    compare_finite(torch.tensor([0., -2., 0.]), gm)
    compare_finite(torch.tensor(5.), gt)
    c = hard_selection(-s, mu, 2, ids)
    compare_finite(torch.tensor([0., .5, 0.]), c)
    gs, gm = torch.autograd.grad((c*bars).sum(), (s, mu), allow_unused=True)
    assert gs is None
    compare_finite(torch.tensor([0., 3., 0.]), gm)
    source = [torch.tensor([.25, .5], requires_grad=True) for _ in range(3)]
    clock = torch.tensor([2**54+7, -1], dtype=torch.int64)
    move = mapper([*source, source[0], clock], device, torch.float32)
    assert move(source[0]) is move(source[0])
    compare_finite(clock, move(clock))
    assert_placement(move(clock), device)
    loss = move(source[0]).sum()*3+move(source[1]).sum()*0
    ga, gb, gc = torch.autograd.grad(loss, source, allow_unused=True)
    compare_finite(torch.tensor([3., 3.]), ga)
    compare_finite(torch.zeros(2), gb)
    assert gc is None
    return ["stable-large-int64-score-ties", "fractional-fill-literal-VJP",
            "nonpositive-hard-fallback-literal-VJP", "packed-transfer-None-zero-aliases-int64"]

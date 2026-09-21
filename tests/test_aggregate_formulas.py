import math
import pytest
import torch
from tidegraph import Continuation, External, Graph, Node, PortLayout, Region
from tidegraph.compare import equivalent
from tidegraph.frontier import run as frontier
from tidegraph.native import Native
from tidegraph.ops import Model
from tidegraph.reference import run


KINDS = ["sum", "mean", "weighted_mean", "active_softmax", "all_softmax"]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("implementation", ["reference", "python", "native", "native-serial"])
def test_analytic_source_normalization_vjp_absence_and_zero(dtype, kind, implementation):
    g = Graph((Node(0, aggregation=kind),), (), (Region(1),), (0, 0, 0), (0,), PortLayout((), (), (2, 0, 1), (0,)))
    m = Model(g, width=2, dtype=dtype)
    with torch.no_grad():
        for i, scale in enumerate(m.input_scale):
            scale.fill_(i+2)
        for slot, mass in enumerate((2, 3, 5)):
            if kind == "weighted_mean":
                m.nodes[0].extra[f"agg_mass_{slot}"].fill_(math.log(math.expm1(mass)))
            elif "softmax" in kind:
                m.nodes[0].extra[f"agg_logit_{slot}"].fill_(math.log(mass))
    x = torch.tensor([1., -2.], dtype=dtype, requires_grad=True)
    zero = torch.zeros(2, dtype=dtype, requires_grad=True)
    other = torch.ones(2, dtype=dtype, requires_grad=True)
    xs = [External(0, 0, 0, 0, x), External(0, 1, 0, 0, zero),
          External(1, 0, 0, 0, other), External(1, 1, 0, 0, other)]
    q = Continuation(g.identity, 2)
    if implementation in {"reference", "python"}:
        result = (run if implementation == "reference" else frontier)(g, m, q, xs, 1, sealed_until=1)
    else:
        result = Native(g, m, packed=implementation == "native", workers=3).run(q, xs, 1, sealed_until=1)
    event = result.trace[0]; h = event["content"]
    denominator = 10 if kind == "all_softmax" else 7
    c0, c2 = (1., 1.) if kind == "sum" else (.5, .5) if kind == "mean" else (2/denominator, 5/denominator)
    expected = x.detach() * (2*c2)
    equivalent(h, expected)
    assert list(event["contributions"]) == [0, 2]
    equivalent(event["contributions"][0], zero.detach())
    equivalent(event["contributions"][2], expected)
    u = x.new_tensor([0.3, -0.4])
    variables = [x, zero, other, *m.input_scale]
    names = [f"agg_{'mass' if kind == 'weighted_mean' else 'logit'}_{i}" for i in range(3)]
    parameters = [m.nodes[0].extra[name] for name in names] if kind not in {"sum", "mean"} else []
    dx, dz, do, ds0, ds1, ds2, *dp = torch.autograd.grad((h*u).sum(), variables + parameters, allow_unused=True)
    equivalent(dx, u * (2*c2)); equivalent(dz, u * (3*c0))
    assert do is None and ds2 is None
    equivalent(ds0, (u*x.detach()).sum()*c2); equivalent(ds1, u.new_zeros(()))
    for slot, gradient in enumerate(dp):
        if slot == 1 and kind != "all_softmax":
            assert gradient is None
            continue
        y = x.detach()*2 if slot == 2 else torch.zeros_like(x)
        if kind == "weighted_mean":
            expected_grad = ((y-expected)*u).sum() * (1-math.exp(-(2, 3, 5)[slot])) / denominator
        else:
            expected_grad = ((y-expected)*u).sum() * (2, 3, 5)[slot] / denominator
        equivalent(gradient, expected_grad)


@pytest.mark.parametrize("kind", ["weighted_mean", "active_softmax", "all_softmax"])
@pytest.mark.parametrize("implementation", ["python", "native"])
def test_absent_source_optimizer_behavior(dtype, kind, implementation):
    def update(reference):
        g = Graph((Node(0, aggregation=kind),), (), (Region(1),), (0, 0), (0,))
        m = Model(g, width=2, dtype=dtype)
        optimizer = torch.optim.AdamW(m.parameters(), lr=0.01, weight_decay=0.1)
        xs = [External(0, 0, 0, 0, torch.ones(2, dtype=dtype))]; q = Continuation(g.identity, 1)
        engine = lambda: Native(g, m, packed=True).run(q, xs, 1, sealed_until=1)
        result = run(g, m, q, xs, 1, sealed_until=1) if reference else frontier(
            g, m, q, xs, 1, sealed_until=1) if implementation == "python" else engine()
        absent = m.nodes[0].extra["agg_mass_1" if kind == "weighted_mean" else "agg_logit_1"]
        before = absent.detach().clone()
        result.outputs[0][-1].sum().backward()
        assert (absent.grad is None) == (kind != "all_softmax")
        assert m.input_scale[1].grad is None
        optimizer.step()
        assert (absent in optimizer.state) == (kind == "all_softmax")
        assert torch.equal(before, absent) == (kind != "all_softmax")
        return m.state_dict(), optimizer.state_dict()
    equivalent(update(True), update(False))

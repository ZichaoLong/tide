"""Independent CPU autograd/optimizer anchors for the consumer boundary."""
from types import SimpleNamespace
import pytest
import torch
from test_online_consumer import ROOT
from tools.online_bench.resident_loss import head_loss, embedding_gradient, ConsumerOptimizer


def test_compact_loss_does_not_evaluate_absent_poison():
    torch.manual_seed(7)
    values = torch.randn(5,4)*.1
    values[1] = float("nan"); values[3] = float("inf")
    mask = torch.tensor([True,False,True,False,True])
    coords = torch.tensor([[0,2,9,1,0,0],[0,0,0,0,0,0],[1,2,18,1,0,0],[0,0,0,0,0,0],[0,2,27,1,0,0]])
    w = torch.randn(7,4)*.1
    window = SimpleNamespace(values=values,valid=mask,coordinates=coords)
    with torch.no_grad():
        loss, root, dw, count = head_loss(window,w,stride=9,denominator=6,backward=True)
    x = values[mask].clone().requires_grad_(); h = w.clone().requires_grad_()
    labels = torch.tensor([14%7,24%7,28%7])
    oracle = torch.nn.functional.cross_entropy(x @ h.t(),labels,reduction="sum")/6
    dx, dh = torch.autograd.grad(oracle,(x,h))
    torch.testing.assert_close(loss,oracle)
    torch.testing.assert_close(root[mask],dx)
    torch.testing.assert_close(dw,dh)
    assert count == 3 and not root[~mask].any()
    window.valid.zero_()
    assert head_loss(window,w,stride=9,denominator=6,backward=True) == (None,None,None,0)


def test_embedding_repeated_ids_and_detached_boundary_leaves():
    embedding = torch.zeros(7,4)
    coords = torch.tensor([[0,4,0,0,0,0],[0,4,7,0,0,1],[1,4,0,0,0,0],[0,4,0,1,5,0],[0,4,0,0,0,0]])
    values = torch.tensor([[1.]*4,[2.]*4,[0.]*4,[float("inf")]*4,[float("nan")]*4])
    b = SimpleNamespace(coordinates=coords,values=values,valid=torch.ones(5,dtype=torch.bool),
                        connected=torch.tensor([True,True,True,True,False]))
    got = embedding_gradient([b],embedding)
    leaf = embedding.clone().requires_grad_()
    ref = torch.nn.functional.embedding(torch.tensor([0,0,3]),leaf)
    grad, = torch.autograd.grad(ref,leaf,values[:3])
    torch.testing.assert_close(got,grad)
    b.connected.zero_()
    assert embedding_gradient([b],embedding) is None
    b.connected[2] = True
    assert torch.equal(embedding_gradient([b],embedding),torch.zeros_like(embedding))


@pytest.mark.parametrize("kind",["sgd","adamw"])
def test_consumer_optimizer_matches_torch_and_refuses_before_commit(kind):
    initial = torch.tensor([[.5,-.25],[.125,.0625]])
    payloads = (initial.clone(),initial.clone())
    leaves = [x.clone().requires_grad_() for x in payloads]
    options = dict(lr=.0001,weight_decay=.001,foreach=False)
    oracle = (torch.optim.SGD(leaves,momentum=.25,**options) if kind=="sgd" else
              torch.optim.AdamW(leaves,eps=1e-6,**options))
    owner = ConsumerOptimizer(*payloads,kind)
    with torch.no_grad():
        for step in range(4):
            grads = [torch.full_like(initial,.125*(step-1)),None if step<2 else torch.zeros_like(initial)]
            before = [x.clone() for x in payloads]
            owner.prepare(grads)
            for x,b in zip(payloads,before):
                assert torch.equal(x,b)
            for x,g in zip(leaves,grads):x.grad=g
            oracle.step(); owner.commit()
            for x,y in zip(payloads,leaves):torch.testing.assert_close(x,y,rtol=1e-6,atol=1e-7)
        before = [x.clone() for x in (*payloads,*owner.masters)]
        with pytest.raises(RuntimeError,match="nonfinite"):
            owner.prepare((torch.full_like(initial,float("inf")),torch.zeros_like(initial)))
        for x,y in zip((*payloads,*owner.masters),before):assert torch.equal(x,y)
        assert owner.proposal is None

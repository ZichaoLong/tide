"""Bank-sized Attention transpose against independent smooth formulas."""
import pytest
import torch
from tidegraph.compare import equivalent
from tidegraph.weighted.block_attention import attention


def test_block_attention_literal_first_order(dtype):
    q=torch.randn(2,2,3,2,dtype=dtype,requires_grad=True)
    k=torch.randn(2,2,5,2,dtype=dtype,requires_grad=True)
    v=torch.randn(2,2,5,3,dtype=dtype,requires_grad=True)
    weights=torch.tensor([[[.3,.6,0,0,0],[.2,.4,.8,0,0],[.1,.7,.3,.2,0]],
                          [[.4,0,0,0,0],[.2,.5,0,0,0],[0,0,0,0,0]]],dtype=dtype,requires_grad=True)
    actual=attention(q,k,v,weights)
    owners=[]
    for b in range(2):
        heads=[]
        for h in range(2):
            rows=[]
            for i in range(3):
                used=weights[b,i]>0
                if used.any():
                    logits=k[b,h,used]@q[b,h,i]/2**.5+weights[b,i,used].log()
                    rows.append(logits.softmax(0)@v[b,h,used])
                else:
                    rows.append(torch.zeros(3,dtype=dtype))
            heads.append(torch.stack(rows))
        owners.append(torch.stack(heads))
    expected=torch.stack(owners)
    equivalent(actual,expected)
    bar=torch.sin(expected.detach())
    equivalent(torch.autograd.grad(actual,(q,k,v,weights),bar,retain_graph=True),
               torch.autograd.grad(expected,(q,k,v,weights),bar,retain_graph=True))
    with pytest.raises(ValueError,match='first-order'):
        torch.autograd.grad(actual.sum(),q,create_graph=True)

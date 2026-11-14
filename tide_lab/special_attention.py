"""Explicit attention variants for the bounded HB/KV experiment.

No foundation execution code. The fused variant implements the real-arithmetic
weighted softmax with a key-bias coordinate. It does NOT apply the historical
normal-tiny probability/decay floors; that numerical-policy difference is
explicit, checked and retained. Exact zero key masks and their zero boundary
VJP remain. No silent dtype conversion or CPU device fallback.
"""
import math
import torch
from torch.nn import functional as F


def weighted_attention(q,k,v,w,*,rate=0.,stride=1,implementation='math'):
    """[..., L, d] and [..., L] receipt; inclusive causal, no window cutoff.

    Keys carry log(w_j)+rate*stride*j; the omitted -rate*stride*i is a
    row-constant in softmax. Extra Q/K coordinates preserve the w gradient
    even on fusion kernels without a differentiable additive-bias input.
    """
    l,d=q.shape[-2:];dv=v.shape[-1]
    if k.shape[-2:]!=(l,d) or v.shape[-2]!=l or w.shape!=q.shape[:-1]:
        raise ValueError('specialized attention requires aligned self-attention')
    pos=torch.arange(l,device=q.device)
    valid=(pos[:,None]>=pos[None,:]) & (w>0).unsqueeze(-2)
    logw=w.clamp_min(torch.finfo(w.dtype).tiny).log()
    bias=logw+rate*stride*pos.to(w.dtype)
    if implementation=='math':
        logits=q@k.transpose(-1,-2)/math.sqrt(d)+bias.unsqueeze(-2)
        nonempty=valid.any(-1,keepdim=True)
        logits=logits.masked_fill(~valid,float('-inf'))
        prob=torch.where(nonempty,logits,torch.zeros_like(logits)).softmax(-1)
        prob=torch.where(valid,prob,torch.zeros_like(prob))
        return prob@v
    if implementation!='npu_fusion_augmented':raise ValueError('unknown explicit attention implementation')
    if q.device.type!='npu':raise ValueError('npu_fusion_augmented requires an NPU')
    import torch_npu
    padded=max(math.ceil((d+1)/16)*16,math.ceil(dv/16)*16)
    qa=F.pad(torch.cat((q,torch.full_like(q[...,:1],math.sqrt(d))),-1),(0,padded-d-1))
    ka=F.pad(torch.cat((k,bias.unsqueeze(-1)),-1),(0,padded-d-1))
    va=F.pad(v,(0,padded-dv))
    batch=math.prod(q.shape[:-2])
    out=torch_npu.npu_fusion_attention(qa.reshape(batch,1,l,padded),ka.reshape(batch,1,l,padded),
        va.reshape(batch,1,l,padded),1,'BNSD',atten_mask=(~valid).reshape(batch,1,l,l),
        scale=1/math.sqrt(d),keep_prob=1.,inner_precise=2,sparse_mode=0)[0]
    # The kernel's invalid-row behavior is checked, and an explicit zero
    # output is also part of the public experiment function.
    out=out.reshape(*q.shape[:-1],padded)[...,:dv]
    return torch.where(valid.any(-1,keepdim=True),out,torch.zeros_like(out))

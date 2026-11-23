"""Independent rational projection and analytic first derivative for the r24 replay."""
from fractions import Fraction as F
from functools import lru_cache

import torch


@lru_cache(maxsize=4096)
def exact_row(data, n):
    s, w, mu, lo, hi = ([F(v) for v in data[i:i+n]] for i in range(0,5*n,n))
    k = F(data[-1])
    capacity = sum(hi)
    high, hs = (k, "budget") if k <= capacity else (capacity, "cap")
    received, rs = (F(1), "unit") if sum(w) >= 1 else (sum(w), "w")
    lower, ls = (received,rs) if received <= high else (high,hs)
    if sum(lo) >= lower:
        lower, ls = sum(lo), "floor"

    def response(multiplier):
        return [max(l,min(h,v-multiplier)) for l,h,v in zip(lo,hi,s)]

    c = response(0)
    initial = sum(c)
    source = "slack"
    if initial < lower or initial > high:
        target, source = (lower,ls) if initial < lower else (high,hs)
        if target == sum(lo):
            c = lo
        elif target == capacity:
            c = hi
        elif n == 2:
            first = max(lo[0],target-hi[1],min(hi[0],target-lo[1],(target+s[0]-s[1])/2))
            c = [first,target-first]
        else:
            # Independently locate the affine segment of the monotone response.
            stops = sorted({v-b for v,l,h in zip(s,lo,hi) for b in (l,h)})
            for left,right in zip(stops,stops[1:]):
                start,end = sum(response(left)),sum(response(right))
                if start >= target >= end:
                    multiplier = left if start == end else left+(right-left)*(start-target)/(start-end)
                    c = response(multiplier)
                    break
        assert sum(c) == target
    lf = [v==l and l!=u for v,l,u in zip(c,lo,hi)]
    uf = [v==u and l!=u for v,l,u in zip(c,lo,hi)]
    return [float(v) for v in c] + lf + uf + [source==x for x in ("slack","budget","cap","w","floor")]


class Projection(torch.autograd.Function):
    @staticmethod
    def forward(ctx,s,w,mu,lo,hi,k):
        n = s.shape[-1]
        rows = torch.cat((s,w,mu,lo,hi,k),-1).detach().cpu().reshape(-1,5*n+1)
        rows = s.new_tensor([exact_row(tuple(row),n) for row in rows.tolist()])
        c,lf,uf,flags = rows.reshape(*s.shape[:-1],3*n+5).split((n,n,n,5),-1)
        lf,uf = lf.bool(),uf.bool()
        fixed = lo==hi
        ctx.save_for_backward(lf,uf,fixed,fixed & (lo==mu),flags.bool())
        ctx.mark_non_differentiable(lf)
        return c,lf

    @staticmethod
    def backward(ctx,grad,unused):
        lf,uf,fixed,fixed_mu,flags = ctx.saved_tensors
        slack,budget,cap,w,floor = flags.split(1,-1)
        inside = ~(lf|uf|fixed)
        # Evaluate the mathematical average independently from the base scaled VJP.
        work = grad.detach().cpu().double()
        free = inside.cpu()
        average = (work*free).sum(-1,keepdim=True)/free.sum(-1,keepdim=True).clamp_min(1)
        mean = torch.where(slack.cpu(),0.,average).to(grad)
        delta = grad-mean
        gs = torch.where(inside,delta,0.)
        gw = torch.where(w,mean,0.).expand_as(grad)
        gm = torch.where(fixed_mu,delta,0.)
        gl = torch.where(lf|(fixed & ~fixed_mu),delta,0.) + torch.where(floor,mean,0.)
        gu = torch.where(uf,delta,0.) + torch.where(cap,mean,0.)
        gk = torch.where(budget,mean,0.)
        return gs,gw,gm,gl,gu,gk


def allocate(scores,w,mu,lower,upper,k,kind="quad",*,return_lower_face=False,**kwargs):
    assert kind == "quad"
    args = torch.broadcast_tensors(*(torch.as_tensor(v,dtype=scores.dtype,device=scores.device)
                                     for v in (scores,w,mu,lower,upper)))
    k = torch.as_tensor(k,dtype=scores.dtype,device=scores.device)
    if k.ndim == args[0].ndim and k.shape[-1]==1:
        k = k.squeeze(-1)
    out,lf = Projection.apply(*args,k.expand(args[0].shape[:-1]).unsqueeze(-1))
    return (out,lf) if return_lower_face else out

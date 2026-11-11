"""Independent static-graph math primitives for the v8 screening implementation.

Equations: pinned core-5 regional note §§4/6/S; no foundation execution imports.
Normal-tiny flooring is a declared finite-precision exploration policy, NOT
mathematical sparsity. Exact-zero branches and fixed-order ties stay distinct.
"""
import math
from contextvars import ContextVar
import torch
from torch import nn
from torch.nn import functional as F

FLOORS=ContextVar('direct_semantic_floor_observations',default=None)


def observe_floor(logx,valid):
    bucket=FLOORS.get()
    if bucket is not None:
        bucket.append(((logx<math.log(torch.finfo(logx.dtype).tiny))&valid).sum().detach())


def norm(x,eps=1e-5):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+eps)


class NodeLinear(nn.Module):
    """Independent node matrices; [node,...,in] -> [node,...,out]."""
    def __init__(self,n,di,do,bias=False):
        super().__init__();self.weight=nn.Parameter(torch.randn(n,do,di)/math.sqrt(di))
        self.bias=nn.Parameter(torch.zeros(n,do)) if bias else None

    def forward(self,x):
        y=torch.bmm(x.reshape(x.shape[0],-1,x.shape[-1]),self.weight.transpose(1,2))
        y=y.reshape(*x.shape[:-1],self.weight.shape[1])
        if self.bias is not None:y=y+self.bias.reshape(x.shape[0],*([1]*(x.ndim-2)),-1)
        return y


def positive_exp(logx):
    """FP policy: normal-tiny floor, zero derivative below floor; finite inputs."""
    return logx.clamp_min(math.log(torch.finfo(logx.dtype).tiny)).exp()


def weighted_softmax(scores,weights,mask):
    """Stable positive-support softmax on eligible coordinates, explicit empty row.

    mask broadcasts to scores. Zero weights have declared zero boundary VJP.
    Floor and renormalize only eligible probabilities; never activate masked rows.
    """
    valid=mask & (weights>0)
    logw=weights.clamp_min(torch.finfo(scores.dtype).tiny).log()
    logits=(scores+logw).masked_fill(~valid,float('-inf'))
    nonempty=valid.any(-1,keepdim=True)
    logits=torch.where(nonempty,logits,torch.zeros_like(logits))
    logp=logits-torch.logsumexp(logits,-1,keepdim=True)
    observe_floor(logp,valid)
    prob=torch.where(valid,positive_exp(logp),torch.zeros_like(logp))
    denom=prob.sum(-1,keepdim=True)
    return prob/torch.where(denom>0,denom,torch.ones_like(denom))


def response(x,mu,tau,kind,alpha):
    if kind=='quad':return torch.minimum(mu,(x/tau).clamp_min(0))
    if kind=='bern':return torch.where(mu>0,positive_exp(mu.clamp_min(torch.finfo(mu.dtype).tiny).log()+F.logsigmoid(x/tau)),torch.zeros_like(mu)).minimum(mu)
    if kind=='ent' and alpha==1:
        return torch.where(mu>0,positive_exp(torch.minimum(x/tau-1,mu.clamp_min(torch.finfo(mu.dtype).tiny).log())),torch.zeros_like(mu)).minimum(mu)
    if kind=='ent':return torch.minimum(mu,((alpha-1)*x/tau+1/alpha).clamp_min(0).pow(1/(alpha-1)))
    raise ValueError('unknown allocation regularizer')


class _Allocation(torch.autograd.Function):
    @staticmethod
    def forward(ctx,t,mu,lower,upper,tau,kind,alpha):
        total=mu.sum(-1,keepdim=True);base=response(t,mu,tau,kind,alpha);bs=base.sum(-1,keepdim=True)
        # Equal limits take the lower input once. Both B arguments may alias.
        low=(bs<lower)|(lower==upper);high=(bs>upper)&~low;bound=low|high
        target=torch.where(low,lower,torch.where(high,upper,bs))
        q=(target/torch.where(total>0,total,torch.ones_like(total))).clamp(1e-7,1-1e-7)
        z=q*mu
        if kind=='quad':inverse=tau*z
        elif kind=='bern':inverse=tau*(q.log()-torch.log1p(-q))
        elif alpha==1:inverse=tau*(1+z.clamp_min(torch.finfo(t.dtype).tiny).log())
        else:inverse=tau*(z.pow(alpha-1)-1/alpha)/(alpha-1)
        points=t-inverse
        lo=points.masked_fill(mu<=0,float('inf')).amin(-1,keepdim=True)
        hi=points.masked_fill(mu<=0,float('-inf')).amax(-1,keepdim=True)
        lo=torch.where(total>0,lo,torch.zeros_like(lo));hi=torch.where(total>0,hi,torch.zeros_like(hi))
        for _ in range(48 if t.dtype==torch.float64 else 28):
            mid=(lo+hi)*.5;s=response(t-mid,mu,tau,kind,alpha).sum(-1,keepdim=True)
            lo=torch.where(s>target,mid,lo);hi=torch.where(s>target,hi,mid)
        lam=(lo+hi)*.5
        c=torch.where(bound,response(t-lam,mu,tau,kind,alpha),base)
        full=(target>=total)&(total>0);empty=(target<=0)|(total<=0)
        c=torch.where(full,mu,torch.where(empty,torch.zeros_like(c),c))
        inside=(c>0)&(c<mu)
        if kind=='bern':
            p=c/torch.where(mu>0,mu,torch.ones_like(mu));a=mu*p*(1-p)/tau;e=p
        else:
            a=inside.to(t.dtype)/tau if kind=='quad' else torch.where(inside,c.clamp_min(torch.finfo(c.dtype).tiny).pow(2-alpha)/tau,torch.zeros_like(c))
            e=((c>=mu)&(mu>0)).to(t.dtype)
        a=torch.where(full|empty,torch.zeros_like(a),a)
        e=torch.where(full,torch.ones_like(e),torch.where(empty,torch.zeros_like(e),e))
        ctx.save_for_backward(a,e,bound&~full&~empty,low,high)
        residual=torch.where(bound,(c.sum(-1,keepdim=True)-target).abs(),torch.zeros_like(target))
        ctx.mark_non_differentiable(residual)
        return c,residual

    @staticmethod
    def backward(ctx,g,unused):
        a,e,bound,low,high=ctx.saved_tensors;den=a.sum(-1,keepdim=True)
        mean=torch.where(bound&(den>0),(a*g).sum(-1,keepdim=True)/torch.where(den>0,den,torch.ones_like(den)),torch.zeros_like(den))
        return a*(g-mean),e*(g-mean),torch.where(low,mean,torch.zeros_like(mean)),torch.where(high,mean,torch.zeros_like(mean)),None,None,None


def allocation(t,mu,lower,upper,tau=.25,kind='quad',alpha=1.):
    c,residual=_Allocation.apply(t,mu,lower,upper,tau,kind,alpha)
    bucket=FLOORS.get()
    if bucket is not None and (kind=='bern' or (kind=='ent' and alpha==1)):
        bucket.append(((c<=torch.finfo(c.dtype).tiny*1.001)&(mu>0)).sum().detach())
    return c,residual


def sorted_send(score,mu,k,*,total=None):
    order=score.masked_fill(mu<=0,float('-inf')).argsort(dim=-1,descending=True,stable=True)
    caps=mu.gather(-1,order)
    if total is not None:
        before=caps.cumsum(-1)-caps;values=torch.minimum(caps,(total-before).clamp_min(0))
    else:
        ranked=score.gather(-1,order);indices=torch.arange(score.shape[-1],device=score.device)
        keep=(indices<k)&(ranked>0)&(caps>0)
        # All nonpositive: select the highest effective candidate, fixed node tie.
        keep=keep|((indices==0)&~keep.any(-1,keepdim=True)&(caps>0))
        values=torch.where(keep,caps,torch.zeros_like(caps))
    return torch.zeros_like(mu).scatter(-1,order,values)


def time_weights(w,stride,rate=0.,window=0,read='+'):
    l=w.shape[-1];position=torch.arange(l,device=w.device);dt=(position[:,None]-position[None,:])*stride
    mask=(dt>=0 if read=='+' else dt>0)
    if window:mask=mask&(dt<window)
    if read=='0':mask=torch.zeros_like(mask)
    log_decay=-rate*dt.clamp_min(0).to(w.dtype)
    observe_floor(log_decay,mask)
    decay=positive_exp(log_decay)
    weights=w.unsqueeze(-2)*decay*mask
    return weights,mask,dt


def summary(w,p,stride,rate,read='+',window=0):
    output=[];amount=[]
    for beta in (rate,0.):
        a,_,_=time_weights(w,stride,beta,window,read);mass=a.sum(-1)
        output.append((a@p)/mass.clamp_min(1).unsqueeze(-1));amount.append(mass)
    return output,amount


def causal_conv(z,w,coeff,bias,stride,conv_stride):
    """Current projection unweighted; historical slots contain w*z at fixed times."""
    result=z*coeff[:,0,None,None,:]+bias[:,None,None,:]
    for tap in range(1,coeff.shape[1]):
        logical=tap*conv_stride
        if logical%stride:continue # no event occupies that other clock residue
        offset=logical//stride
        if offset>=z.shape[2]:continue
        history=z[:,:,:-offset]*w[:,:,:-offset,None]
        result=result+F.pad(history,(0,0,offset,0))*coeff[:,tap,None,None,:]
    return result


def dual_recurrence(k,v,q,log_gate,write,*,delta,old=False,log_input_gate=None):
    """Exact real-arithmetic expansion of H'=g H + e k^T, batched in time.

    Inputs [...,time,key/value], gates [...,time]. Delta innovations solve
    (I+strict-lower(beta*decay*K*K^T))*e=beta*v. A finite nilpotent doubling
    polynomial avoids backend-specific triangular solvers. It is verified
    against an independent serial matrix recurrence, including VJPs.
    """
    l=k.shape[-2];idx=torch.arange(l,device=k.device);lower=idx[:,None]>idx[None,:]
    prefix=log_gate.cumsum(-1)
    exponent=prefix.unsqueeze(-1)-prefix.unsqueeze(-2)
    observe_floor(exponent,idx[:,None]>=idx[None,:])
    decay=positive_exp(exponent.clamp_max(0))* (idx[:,None]>=idx[None,:])
    innovation=write.unsqueeze(-1)*v
    if delta:
        power=-(write.unsqueeze(-1)*decay*(k@k.transpose(-1,-2)))*lower
        # Strict lower power is nilpotent. No inverse and no divide by gate.
        span=1
        while span<l:
            innovation=innovation+power@innovation
            span*=2
            if span<l:power=power@power
    if old:
        if log_input_gate is None:raise ValueError('old read requires the current nonpassive gate')
        decay=positive_exp((exponent-log_input_gate.unsqueeze(-1)).clamp_max(0))*lower
    return (decay*(q@k.transpose(-1,-2)))@innovation

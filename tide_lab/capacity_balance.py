"""V11 experimental candidate-aware load bias for the specialized HB model.

This is a changed routing function, not an equivalent executor or a new
allocation solver. The existing caps, quad VJP and strict feasibility threshold
remain unchanged. Biases use training receipt/Full observations, never labels
or evaluation feedback. Dense physical execution remains explicit.
"""
import torch
from .direct_specialized import SpecializedSemanticLM


def opportunity_counts(received, full, valid):
    present=received & valid[None,:,:]
    count=present.sum(0).clamp_min(1)
    q=(present.to(torch.float32)/count[None,:,:]).sum((1,2))
    f=(full & valid[None,:,:]).sum((1,2)).float()
    return q,f,present


def bias_step(bias,q,f,scale,rate,max_abs):
    eligible=q>0
    target=q/q.sum().clamp_min(1e-30)
    actual=f/f.sum().clamp_min(1e-30)
    gap=target-actual
    direction=torch.where(gap.abs()>.05*target,gap.sign(),torch.zeros_like(gap))*eligible
    # Remove the irrelevant common offset within currently eligible units.
    direction=(direction-(direction.sum()/eligible.sum().clamp_min(1)))*eligible
    return (bias+rate*scale.clamp(.1,10.)*direction).clamp(-max_abs,max_abs)


class CapacityBalancedLM(SpecializedSemanticLM):
    def __init__(self,*args,routing_balance,**kwargs):
        super().__init__(*args,**kwargs)
        self.balance=dict(routing_balance)
        if self.balance['kind']!='candidate-load-bias-v1' or self.balance['rate']<0:
            raise ValueError('unknown route intervention')
        self._collect=False;self._valid=None;self._observations={};self._raw={}
        for j,nodes in enumerate(self.members):
            n=len(nodes)
            for key in ('bias','q_ema','f_ema','receipt_total','full_total'):
                self.register_buffer(f'balance_{key}_{j}',torch.zeros(n,dtype=torch.int64 if key.endswith('_total') else torch.float32))
            self.selectors[j].local.register_forward_hook(self._score_hook(j))
            self.selectors[j].register_forward_hook(self._route_hook(j))
        self.register_buffer('balance_updates',torch.zeros((),dtype=torch.int64))

    def _score_hook(self,j):
        def hook(module,args,score):
            if self._collect:self._raw[j]=score.detach()
            return score+getattr(self,f'balance_bias_{j}')[:,None,None]
        return hook

    def _route_hook(self,j):
        def hook(module,args,out):
            if not self._collect:return
            r=args[0];q,f,present=opportunity_counts(r>0,out[0]>0,self._valid)
            raw=self._raw.pop(j);count=present.sum(0).clamp_min(1)
            mean=(raw*present).sum(0)/count
            sq=((raw-mean[None,:,:]).square()*present).sum()
            values=(q,f,present.sum((1,2)).float(),sq,present.sum().float())
            self._observations[j]=values if j not in self._observations else tuple(a+b for a,b in zip(self._observations[j],values))
        return hook

    def begin_update(self):
        self._collect=True;self._observations={};self._raw={}

    def set_training_mask(self,valid):self._valid=valid

    @torch.no_grad()
    def observe_update(self,work,tokens):
        values=super().observe_update(work,tokens);self._collect=False
        if len(self._observations)!=len(self.members):raise ValueError('incomplete route training observations')
        self.balance_updates.add_(1)
        for j,(q,f,receipt,sq,count) in self._observations.items():
            qe=getattr(self,f'balance_q_ema_{j}');fe=getattr(self,f'balance_f_ema_{j}')
            qn=q/q.sum().clamp_min(1);fn=f/f.sum().clamp_min(1)
            qe.mul_(.9).add_(qn,alpha=.1);fe.mul_(.9).add_(fn,alpha=.1)
            getattr(self,f'balance_receipt_total_{j}').add_(receipt.to(torch.int64))
            getattr(self,f'balance_full_total_{j}').add_(f.to(torch.int64))
            bias=getattr(self,f'balance_bias_{j}');scale=(sq/count.clamp_min(1)).sqrt()
            if tokens>=self.balance['start_tokens']:
                # No currently absent node can be activated by this controller.
                eligible=q>0
                bias.copy_(bias_step(bias,qe*eligible,fe*eligible,scale,self.balance['rate'],self.balance['max_abs_bias']))
            if not bool(torch.isfinite(bias).all()):raise FloatingPointError('route bias nonfinite')
            values.update({f'balance/region{j}/zero_full':float((f==0).sum()),
                f'balance/region{j}/effective_units':float(torch.exp(-(fn*fn.clamp_min(1e-30).log()).sum())),
                f'balance/region{j}/max_share':float(fn.max()),f'balance/region{j}/bias_abs_max':float(bias.abs().max()),
                f'balance/region{j}/score_std':float(scale)})
        self._observations={};return values

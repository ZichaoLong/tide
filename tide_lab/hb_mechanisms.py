"""V12 experiment-owned HB variants, not a foundation executor.

Private histories and the fixed local graph remain independent of weight tying.
Centered quad is a finite-precision policy for the same fixed-total function;
top2, sharing and content/readout variants are explicitly different models.
"""
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from .direct_semantic import Selector
from .direct_semantic_ops import NodeLinear, allocation
from .direct_specialized import SpecializedSemanticLM


class SharedNodeLinear(nn.Module):
    """One regional matrix plus optional independent node low-rank corrections."""
    def __init__(self, original, rank, seed, role):
        super().__init__()
        n,do,di=original.weight.shape
        self.weight=nn.Parameter(original.weight[0].detach().clone())
        self.private_parameters=0
        if rank:
            self.down=NodeLinear(n,di,rank);self.up=NodeLinear(n,rank,do)
            for name,p in self.named_parameters():
                if name=='weight':continue
                key=f'v12-shared-adapter:{seed}:{role}:{name}:{tuple(p.shape)}'
                g=torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')%(2**63-1))
                with torch.no_grad():p.copy_(torch.randn(p.shape,generator=g,device=p.device)*(.01 if name.startswith('up') else 1/math.sqrt(di)))
            self.private_parameters=sum(p.numel() for mod in (self.down,self.up) for p in mod.parameters())//n
        self.shared_parameters=self.weight.numel()

    def forward(self,x):
        y=F.linear(x,self.weight)
        return y+self.up(self.down(x)) if hasattr(self,'down') else y


def smooth_top2(score,caps,budget,tau):
    """Hard support, differentiable capped two-way split; absent nodes stay zero.

    Selected cap sum may be below the original regional budget. This variant
    explicitly uses min(original budget, selected cap sum), and records that
    changed budget. It is not the core-5 regional optimizer or a STE.
    """
    if score.shape[-1]<2:raise ValueError('top2 needs at least two node slots')
    index=score.masked_fill(caps<=0,float('-inf')).argsort(dim=-1,descending=True,stable=True)[...,:2]
    s=score.gather(-1,index);mu=caps.gather(-1,index)
    target=torch.minimum(budget,mu.sum(-1,keepdim=True))
    lo=(target[...,0]-mu[...,1]).clamp_min(0)
    hi=torch.minimum(mu[...,0],target[...,0])
    # Bounded smooth logits avoid FP32 sigmoid rounding a valid interior to a
    # cap. This changes the score response and is declared in the config.
    p=torch.sigmoid(8*torch.tanh((s[...,0]-s[...,1])/(8*tau)))
    first=lo+(hi-lo)*p
    pair=torch.stack((first,target[...,0]-first),-1)
    c=torch.zeros_like(caps).scatter(-1,index,pair)
    residual=(c.sum(-1,keepdim=True)-target).abs()
    return c,residual


class MechanismSelector(Selector):
    def regional_allocation(self,score,caps,budget):
        policy=self.mechanism
        if policy.get('route')=='top2' and self.progress>=2*2**20:
            return smooth_top2(score,caps,budget,self.s['allocation_temperature'])
        if policy.get('center_quad',True):
            # Translation invariance applies to a fixed-total solve. Detaching
            # the common origin does not change the real-arithmetic score VJP.
            origin=score.masked_fill(caps<=0,float('-inf')).amax(-1,keepdim=True)
            origin=torch.where(caps.sum(-1,keepdim=True)>0,origin,torch.zeros_like(origin))
            score=score-origin.detach()
        return allocation(score,caps,budget,budget,self.s['allocation_temperature'],self.s['regularizer'])


class HBMechanismLM(SpecializedSemanticLM):
    def __init__(self,*args,mechanism,**kwargs):
        super().__init__(*args,**kwargs)
        self.mechanism=dict(mechanism)
        allowed={'center_quad','share','rank','route','read_content','readout','residual','position'}
        if set(self.mechanism)-allowed:raise ValueError('unknown HB mechanism option')
        if self.mechanism.get('share','none') not in ('none','ffn','full'):raise ValueError('sharing mode')
        if self.mechanism.get('route','quad') not in ('quad','top2'):raise ValueError('route mode')
        if self.mechanism.get('readout','tied_mean') not in ('tied_mean','untied_mean','ports4'):raise ValueError('readout mode')
        seed=kwargs.get('seed',11);d=self.width
        for j,(m,s) in enumerate(zip(self.memories,self.selectors)):
            if s.s['selection_mode']!='regional_total' or s.s['regularizer']!='quad':raise ValueError('v12 requires fixed-total quad parent')
            s.__class__=MechanismSelector;s.mechanism=self.mechanism
            roles=[]
            if self.mechanism.get('share','none')!='none':roles+=['ffgate','ffup','ffdown']
            if self.mechanism.get('share')=='full':roles+=['query','output']
            for role in roles:
                setattr(m,role,SharedNodeLinear(getattr(m,role),self.mechanism.get('rank',0),seed,f'memories.{j}.{role}'))
            if self.mechanism.get('read_content'):
                # Deterministic private projection; fixed gain, no hidden loss.
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed+10100+j)
                    m.read_content=NodeLinear(len(self.members[j]),2*m.s['summary_width'],d)
        mode=self.mechanism.get('readout','tied_mean')
        if mode!='tied_mean':
            ports=4 if mode=='ports4' else 1
            if len(self.plan['graph']['outputs'])%ports:raise ValueError('equal output ports required')
            old=self.head
            self.head=nn.Linear(ports*d,old.out_features)
            with torch.no_grad():
                self.head.weight.copy_(old.weight.detach().repeat(1,ports));self.head.bias.copy_(old.bias)
        if self.mechanism.get('position'):
            pos=torch.arange(1024,dtype=torch.float32)[:,None]
            phase=pos*torch.exp(torch.arange(0,d,2,dtype=torch.float32)*(-math.log(10000)/d))
            encoding=torch.zeros(1024,d);encoding[:,0::2]=phase.sin();encoding[:,1::2]=phase.cos()
            self.register_buffer('input_position',encoding*(math.sqrt(2)*.02))
        self._recount()

    def _recount(self):
        self.mechanism_counts=[]
        for j,m in enumerate(self.memories):
            n=len(self.members[j]);pf=sf=0
            full_roles=['query','output','ffgate','ffup','ffdown']
            if hasattr(m,'read_content'):full_roles.append('read_content')
            full_modules=[getattr(m,k) for k in full_roles]
            for mod in full_modules:
                if isinstance(mod,SharedNodeLinear):pf+=mod.private_parameters;sf+=mod.shared_parameters
                else:pf+=sum(p.numel() for p in mod.parameters())//n
            full_ids={id(p) for mod in full_modules for p in mod.parameters()}
            receive=sum(p.numel() for p in m.parameters() if id(p) not in full_ids and p is not m.base)//n
            receive+=sum(p.numel() for p in self.selectors[j].parameters())//n
            shared=m.base.numel()+self.aggregators[j].a.numel()+self.aggregators[j].b.numel()
            self.mechanism_counts.append((pf,sf,receive,shared,self.aggregators[j].e.shape[-1]))
        self.shared_lm_count=sum(p.numel() for p in {id(p):p for mod in (self.embedding,self.head) for p in mod.parameters()}.values())
        self.total_parameter_count=sum(p.numel() for p in self.parameters())

    def parameter_uses(self,j,received,full,w):
        pf,sf,pr,sr,pe=self.mechanism_counts[j]
        return (full.sum()*pf+full.any(0).sum()*sf,received.sum()*pr,
                received.any(0).sum()*sr+(w>0).sum()*pe)

    def embed(self,ids):
        x=super().embed(ids)
        if hasattr(self,'input_position'):
            if ids.shape[1]>len(self.input_position):raise ValueError('position capacity')
            x=x+self.input_position[:ids.shape[1]].to(x.dtype)
        return x

    def kv(self,j,u,r):
        m=super().kv(j,u,r)
        if self.mechanism.get('read_content'):
            m['content']=m['content']+.1*self.memories[j].read_content(torch.cat(m['memory'],-1))
        return m

    def readout(self,values,mass,embedding):
        if self.mechanism.get('readout')=='ports4':
            outputs=self.plan['graph']['outputs'];w=torch.stack([mass[v] for v in outputs])
            z=torch.stack([values[v] for v in outputs]);n=len(outputs)//4
            pieces=(w.unsqueeze(-1)*z).reshape(4,n,*z.shape[1:]).sum(1)
            pieces=pieces/w.sum(0).clamp_min(1)[None,:,:,None]
            h=pieces.permute(1,2,0,3).flatten(-2)
        else:h=super().readout(values,mass,embedding)
        if self.mechanism.get('residual'):h=h+embedding
        return h

    def gradient_metrics(self):
        groups={'read':[],'ffn':[],'router':[]};zero=nodes=0
        for name,p in self.named_parameters():
            group='router' if name.startswith('selectors.') else 'read' if '.read_' in name else 'ffn' if any('.'+r+'.' in name for r in ('ffup','ffgate','ffdown')) else None
            if group and p.grad is not None:groups[group].append(p.grad.detach().square().sum())
            if '.read_q.weight' in name and p.grad is not None:
                v=p.grad.detach().flatten(1).square().sum(1);zero+=(v==0).sum();nodes+=v.numel()
        out={f'gradient/{k}_l2':float(torch.stack(v).sum().sqrt()) if v else 0. for k,v in groups.items()}
        if nodes:out['gradient/read_q_zero_node_fraction']=float(zero)/nodes
        return out

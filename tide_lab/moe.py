"""Experiment-owned, dropless top-2 decoder MoE. No TIDE execution core.

The optimized expert path buckets loads by powers of two and batches GEMMs.
One small counts download/layer builds packing metadata. There is no per-expert
matrix-multiply loop, capacity factor, token dropping, or hidden dtype fallback.
The separate scalar-expert reference is for tiny independent checks only.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .language import SmallTransformer


def parameter_inventory(width,layers,hidden,experts,k=2,vocab_size=4096):
    shared=vocab_size*width+layers*(4*width**2+8*width)+2*width+vocab_size
    one=2*width*hidden+hidden+width
    router=layers*width*experts
    return dict(shared=shared,router=router,per_expert=one,
                total=shared+router+layers*experts*one,
                active=shared+router+layers*k*one,expert_fraction=k/experts)


class Top2Experts(nn.Module):
    def __init__(self,width,hidden,experts,layers,implementation='bucketed_bmm'):
        super().__init__()
        if experts<2 or implementation not in ('bucketed_bmm','reference'):
            raise ValueError('unsupported expert count/implementation')
        self.width=width;self.hidden=hidden;self.experts=experts;self.k=2
        self.implementation=implementation
        self.router=nn.Linear(width,experts,bias=False)
        self.w1=nn.Parameter(torch.empty(experts,width,hidden))
        self.b1=nn.Parameter(torch.zeros(experts,hidden))
        self.w2=nn.Parameter(torch.empty(experts,hidden,width))
        self.b2=nn.Parameter(torch.zeros(experts,width))
        nn.init.normal_(self.router.weight,std=.02)
        nn.init.normal_(self.w1,std=.02)
        nn.init.normal_(self.w2,std=.02/math.sqrt(2*layers))
        self.valid_mask=None;self.stats=None;self.physical_rows=0

    def reference(self,x,indices,weights):
        # Deliberately independent implementation: one dense expert at a time,
        # then explicit token/expert masks. Unselected experts have zero VJP.
        y=torch.zeros_like(x)
        for i in range(self.experts):
            z=F.gelu(x@self.w1[i]+self.b1[i])@self.w2[i]+self.b2[i]
            a=((indices==i).to(weights.dtype)*weights).sum(-1)
            y=y+z*a[:,None]
        self.physical_rows=len(x)*self.experts
        return y

    def bucketed(self,x,indices,weights):
        e=self.experts;k=self.k;assignments=indices.flatten()
        order=torch.argsort(assignments,stable=True)
        counts=torch.bincount(assignments,minlength=e)
        # Explicit bounded host synchronization. Shapes and buckets are routing
        # metadata; the CPU reference never supplies the execution schedule.
        counts_cpu=counts.detach().cpu().tolist()
        starts=[];offset=0;buckets={}
        for i,n in enumerate(counts_cpu):
            starts.append(offset);offset+=n
            if n:
                capacity=1<<(n-1).bit_length()
                buckets.setdefault(capacity,[]).append(i)
        packed=x.index_select(0,order//k)
        result=torch.zeros_like(packed);physical=0
        for capacity,experts in sorted(buckets.items()):
            # Build only integer metadata on host; all values/GEMMs stay on x.device.
            ids=torch.tensor(experts,device=x.device,dtype=torch.long)
            start=torch.tensor([starts[i] for i in experts],device=x.device,dtype=torch.long)
            rank=start[:,None]+torch.arange(capacity,device=x.device)[None,:]
            # Padding can read any in-bounds row; only declared real rows are
            # gathered into the result, so its forward and VJP are exactly absent.
            padded=packed.index_select(0,rank.clamp_max(len(packed)-1).flatten()).reshape(len(experts),capacity,self.width)
            h=F.gelu(torch.bmm(padded,self.w1.index_select(0,ids))+self.b1.index_select(0,ids)[:,None,:])
            y=torch.bmm(h,self.w2.index_select(0,ids))+self.b2.index_select(0,ids)[:,None,:]
            gather_cpu=torch.cat([torch.arange(counts_cpu[i])+j*capacity for j,i in enumerate(experts)])
            dest_cpu=torch.cat([torch.arange(counts_cpu[i])+starts[i] for i in experts])
            real=y.flatten(0,1).index_select(0,gather_cpu.to(x.device))
            result.index_copy_(0,dest_cpu.to(x.device),real)
            physical+=len(experts)*capacity
        inverse=torch.empty_like(order).scatter_(0,order,torch.arange(len(order),device=x.device))
        self.physical_rows=physical
        return (result.index_select(0,inverse).reshape(len(x),k,self.width)*weights[:,:,None]).sum(1)

    def forward(self,x):
        shape=x.shape;flat=x.reshape(-1,self.width)
        # Actual campaign is FP32. FP64 is retained only for tiny CPU formula/VJP checks.
        routing_dtype=torch.float64 if flat.dtype==torch.float64 else torch.float32
        logits=F.linear(flat.to(routing_dtype),self.router.weight.to(routing_dtype))
        p=logits.softmax(-1)
        _,indices=torch.topk(logits,self.k,dim=-1,sorted=True)
        # Softmax over selected logits equals renormalizing their full-softmax
        # probabilities, without dividing by potentially underflowed weights.
        weights=logits.gather(1,indices).softmax(-1).to(flat.dtype)
        mask=(self.valid_mask.reshape(-1) if self.valid_mask is not None else
              torch.ones(len(flat),device=flat.device,dtype=torch.bool))
        valid=mask.to(p.dtype)
        counts=torch.zeros(self.experts,device=x.device,dtype=p.dtype).scatter_add_(0,indices.flatten(),
            valid[:,None].expand(-1,self.k).reshape(-1)).detach()
        self.stats=dict(counts=counts,p_sum=(p*valid[:,None]).sum(0),
            z_sum=(logits.logsumexp(-1).square()*valid).sum(),valid=valid.sum(),
            indices=indices.detach(),weights=weights.detach())
        y=self.reference(flat,indices,weights) if self.implementation=='reference' else self.bucketed(flat,indices,weights)
        return y.reshape(shape)


class MoETransformer(SmallTransformer):
    def __init__(self,seed=11,width=192,layers=6,heads=4,hidden=384,experts=20,vocab_size=4096,
                 position_scale=math.sqrt(2)*.02,attention_backend='eager',expert_implementation='bucketed_bmm'):
        super().__init__(seed=seed,width=width,layers=layers,heads=heads,ffn=hidden,vocab_size=vocab_size,
            position_scale=position_scale,attention_backend=attention_backend,initialization='gpt2')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+1973)
            for block in self.blocks:
                block.ffn=Top2Experts(width,hidden,experts,layers,expert_implementation)
        self.inventory=parameter_inventory(width,layers,hidden,experts,vocab_size=vocab_size)
        self.observe=False

    def no_weight_decay(self):
        return {f'blocks.{i}.ffn.{name}' for i in range(len(self.blocks)) for name in ('b1','b2')}

    def forward(self,x,valid_mask=None,**kwargs):
        for block in self.blocks:block.ffn.valid_mask=valid_mask
        y=super().forward(x)
        positions=x.numel();l=len(self.blocks);e=self.blocks[0].ffn.experts
        self.last_metrics={'work/prediction_positions':positions,
            'work/expert_assignments':positions*l*2,'work/expert_slots':positions*l*e,
            'work/physical_expert_rows':sum(b.ffn.physical_rows for b in self.blocks),
            'work/active_parameter_uses':self.inventory['active']*positions}
        return y

    def auxiliary(self,global_counts=None,denominator=None):
        balance=[];z=[]
        for i,b in enumerate(self.blocks):
            s=b.ffn.stats;n=s['valid'] if denominator is None else denominator
            f=(s['counts']/(2*n) if global_counts is None else global_counts[i]/(2*n))
            balance.append(b.ffn.experts*(f*(s['p_sum']/n)).sum())
            z.append(s['z_sum']/n)
        return torch.stack(balance).mean(),torch.stack(z).mean()

    def router_counts(self):return [b.ffn.stats['counts'].clone() for b in self.blocks]


def objective_backward(model,batches,config):
    """Global-batch Switch loss, invariant to physical microbatch splitting.

    Frequencies are stop-gradient, measured by a deterministic no-grad pass
    when accumulation is needed. The extra forward is explicit physical work.
    No dropout or batch-dependent normalization is used in this architecture.
    This helper accumulates gradients; it never takes an optimizer step.
    """
    count=sum(int((b['labels']!=-100).sum()) for b in batches)
    if count<=0:raise ValueError('empty MoE logical batch')
    frequencies=None;prepass_work={};work={};counts=None;probability=None
    if len(batches)>1:
        with torch.no_grad():
            for b in batches:
                model(b['input_ids'],valid_mask=b['labels']!=-100)
                values=model.router_counts()
                frequencies=values if frequencies is None else [a+v for a,v in zip(frequencies,values)]
                for k,v in model.last_metrics.items():prepass_work[k]=prepass_work.get(k,0)+v
    total=balance_total=z_total=0.
    for b in batches:
        logits=model(b['input_ids'],valid_mask=b['labels']!=-100)
        ce=F.cross_entropy(logits.flatten(0,1),b['labels'].flatten(),ignore_index=-100,reduction='sum')/count
        balance,z=model.auxiliary(frequencies,count)
        loss=ce+config['training']['balance_coefficient']*balance+config['training']['z_coefficient']*z
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite MoE task/auxiliary objective')
        loss.backward();total+=float(ce.detach());balance_total+=float(balance.detach());z_total+=float(z.detach())
        for k,v in model.last_metrics.items():work[k]=work.get(k,0)+v
        values=model.router_counts();counts=values if counts is None else [a+v for a,v in zip(counts,values)]
        values=[b.ffn.stats['p_sum'].detach() for b in model.blocks]
        probability=values if probability is None else [a+v for a,v in zip(probability,values)]
    # Reduce telemetry once per logical update, not per expert GEMM.
    summaries=[]
    for c,p in zip(counts,probability):
        f=c/(2*count);q=p/count
        summaries.append(torch.stack([(c==0).sum(),f.max(),-(f*f.clamp_min(1e-30).log()).sum(),
                                      -(q*q.clamp_min(1e-30).log()).sum()]))
    stats=torch.stack(summaries).detach().cpu().tolist()
    for i,row in enumerate(stats):
        for key,value in zip(('cold_experts','max_load_fraction','assignment_entropy','router_entropy'),row):
            work[f'router/layer{i}/{key}']=value
    if getattr(model,'observe',False):
        for i,c in enumerate(torch.stack(counts).cpu().tolist()):
            for e,v in enumerate(c):work[f'router/layer{i}/expert{e}_fraction']=v/(2*count)
    work.update({'aux/balance':balance_total,'aux/z_loss':z_total,
        'train/objective':total+config['training']['balance_coefficient']*balance_total+config['training']['z_coefficient']*z_total,
        'work/aux_prepass_expert_rows':prepass_work.get('work/physical_expert_rows',0),
        'work/aux_prepass_positions':prepass_work.get('work/prediction_positions',0),
        'activity/active_parameters_per_position':model.inventory['active'],
        'activity/expert_fraction':model.inventory['expert_fraction']})
    return total,work,count

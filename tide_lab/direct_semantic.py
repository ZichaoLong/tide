"""Bounded direct local-candidate screening on exported FC/HB time grids.

Independent mathematical implementation; no foundation execution imports.
All Full rows are physically computed, then explicitly masked. This is NOT a
sparse executor or the foundation's public continuation/return contract.
"""
import math
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from .direct_semantic_ops import NodeLinear,norm,allocation,weighted_softmax,sorted_send,FLOORS
from .direct_semantic_memory import Memory


def selected(spec,key,default):return default if spec.get(key) is None else spec[key]


class Encoder(nn.Module):
    def __init__(self,n,di,hidden,out):
        super().__init__();self.a=NodeLinear(n,di,hidden,True);self.b=NodeLinear(n,hidden,out,True)
    def forward(self,x):return norm(self.b(F.silu(self.a(x))))


class Scalar(nn.Module):
    def __init__(self,n,di,hidden):
        super().__init__();self.a=NodeLinear(n,di,hidden,True);self.b=NodeLinear(n,hidden,1,True)
    def forward(self,x):return self.b(F.silu(self.a(x))).squeeze(-1)


class Selector(nn.Module):
    def __init__(self,n,width,summary_width,spec):
        super().__init__();self.s=spec;self.n=n
        d=width;ds=summary_width;hidden=spec['hidden_width'];sel=selected(spec,'selection_width',hidden)
        reg=selected(spec,'region_control_width',hidden);flat=d+4*ds+3;self.sel=sel
        mode=spec['selection_mode'];self.regional=mode in ('regional_total','regional_total_zero','regional_threshold','regional_threshold_hard')
        if mode!='proportional':
            if spec['local_score']=='cat':self.local=Scalar(n,flat,hidden)
            else:
                self.score_enc=nn.ModuleList([Encoder(n,i,hidden,sel) for i in (d,2*ds,3)])
                self.match_gate=NodeLinear(n,3*sel,hidden,True);self.match_up=NodeLinear(n,3*sel,hidden,True)
                self.match_out=NodeLinear(n,hidden,1,False)
                self.match_bias=nn.Parameter(torch.zeros(n,1,1))
        if self.regional and spec['region_control_mode']=='context':
            if spec['region_evidence']=='cat':self.evidence=NodeLinear(n,flat,reg,True)
            else:
                self.reg_enc=nn.ModuleList([Encoder(n,i,hidden,sel) for i in (d,2*ds,3)])
                self.reg_combine=NodeLinear(n,4*sel,reg,True)
            self.reg_a=nn.Linear(reg+2,hidden);self.reg_b=nn.Linear(hidden,1)
            self._init(self.reg_a);self._init(self.reg_b)
        if not self.regional:
            if spec['send_gain']=='context':self.gain=Scalar(n,flat,hidden)
            elif spec.get('gain_logit') is not None:self.gain_logit=nn.Parameter(torch.full((n,1,1),spec['gain_logit']))
        if mode!='proportional' and spec['region_score']=='att':
            self.loc=NodeLinear(n,d,sel,True)
            self.key=(Encoder(n,2*ds,hidden,sel) if spec['region_key']=='content' else NodeLinear(n,2*ds+2,sel,True))
            self.query_in=nn.Linear(sel,sel);self.query_out=nn.Linear(sel,sel)
            self._init(self.query_in);self._init(self.query_out)

    @staticmethod
    def _init(module):nn.init.normal_(module.weight,std=1/math.sqrt(module.in_features));nn.init.zeros_(module.bias)

    def forward(self,r,m,present=None):
        s=self.s;mode=s['selection_mode'];w=r.clamp(0,1);present=r>0 if present is None else present
        mem=torch.cat(m['memory'],-1)
        stat=torch.stack((r.clamp_min(0).log1p(),m['mass'][0].log1p(),m['mass'][1].log1p()),-1)
        z=(m['x'],mem,stat)
        flat=torch.cat((m['x'],stat[...,:1],m['memory'][0],m['memory'][1],stat[...,1:],
                        m['p']*m['memory'][0],m['p']*m['memory'][1]),-1)
        if self.regional:mu=(s['gamma_max']*w).clamp_max(1)
        else:
            if hasattr(self,'gain'):gamma=1+(s['gamma_max']-1)*torch.sigmoid(self.gain(flat))
            elif hasattr(self,'gain_logit'):gamma=1+(s['gamma_max']-1)*torch.sigmoid(self.gain_logit)
            else:gamma=selected(s,'fixed_gain',s['gamma_max'])
            mu=(gamma*r.clamp_min(0)).clamp_max(1)
        # Work on [...,node] regional rows; all other tensors retain node first.
        caps=mu.movedim(0,-1);wr=w.movedim(0,-1);total=caps.sum(-1,keepdim=True)
        zero=total.new_zeros(())
        if mode=='proportional':
            c=caps*total.clamp_max(s['k'])/torch.where(total>0,total,torch.ones_like(total))
            return c.movedim(-1,0),zero
        if hasattr(self,'local'):score=self.local(flat)
        else:
            encoded=[e(t) for e,t in zip(self.score_enc,z)];xx,mm,_=encoded
            f=torch.cat(encoded,-1)
            score=selected(s,'alpha_match',1.)*(xx*mm).sum(-1)/math.sqrt(self.sel)
            score=self.match_bias+score+selected(s,'alpha_res',1.)*self.match_out(F.silu(self.match_gate(f))*self.match_up(f)).squeeze(-1)
        if s['region_score']=='att':
            key=self.key(mem if s['region_key']=='content' else torch.cat((mem,stat[...,1:]),-1))
            weight=w if self.regional or s['query_weight']=='received' else mu
            avg=(weight.unsqueeze(-1)*self.loc(m['x'])).sum(0)/weight.sum(0).clamp_min(1).unsqueeze(-1)
            query=self.query_out(F.silu(self.query_in(avg)))
            score=selected(s,'alpha_loc',1.)*score+selected(s,'alpha_sel',1.)*(key*query.unsqueeze(0)).sum(-1)/math.sqrt(self.sel)
        score=score.movedim(0,-1)
        if self.regional:
            recv=wr.sum(-1,keepdim=True);bottom=recv.clamp_max(1);top=total.clamp_max(s['k'])
            if s['region_control_mode']=='context':
                if hasattr(self,'evidence'):e=norm(self.evidence(flat))
                else:
                    ec=[mod(t) for mod,t in zip(self.reg_enc,z)]
                    e=norm(self.reg_combine(torch.cat((*ec,ec[0]*ec[1]),-1)))
                avg=(w.unsqueeze(-1)*e).sum(0)/w.sum(0).clamp_min(1).unsqueeze(-1)
                control=self.reg_b(F.silu(self.reg_a(torch.cat((avg,recv.log1p(),total.log1p()),-1))))
            else:control=None
            if mode in ('regional_total','regional_total_zero'):
                fraction=torch.sigmoid(control) if control is not None else s['total_fraction']
                budget=bottom+fraction*(top-bottom)
                if mode=='regional_total_zero':c=sorted_send(score,caps,s['k'],total=budget);residual=zero
                else:c,residual=allocation(score,caps,budget,budget,s['allocation_temperature'],s['regularizer'],selected(s,'ent_alpha',1.))
            else:
                t=score-(control if control is not None else s['eta'])
                if mode=='regional_threshold_hard':c=sorted_send(t,caps,s['k']);residual=zero
                else:c,residual=allocation(t,caps,bottom,torch.full_like(bottom,s['k']),s['allocation_temperature'],s['regularizer'],selected(s,'ent_alpha',1.))
        elif mode in ('constrained','hard_selection'):
            t=score-selected(s,'eta',0.)
            if mode=='hard_selection':c=sorted_send(t,caps,s['k']);residual=zero
            else:c,residual=allocation(t,caps,total.clamp_max(1),torch.full_like(total,s['k']),s['allocation_temperature'],s['regularizer'],selected(s,'ent_alpha',1.))
        else:
            base=selected(s,'allocation_base','uniform' if mode=='unit_attention' else 'mu')
            a=(present.movedim(0,-1).to(score.dtype) if base=='uniform' else wr if base=='received' else caps)
            prob=weighted_softmax(score/selected(s,'allocation_temperature',1.),a,present.movedim(0,-1))
            c=prob*(caps if mode=='direct_attention' else total.clamp_max(1) if mode=='pooled_attention' else 1)
            residual=zero
        return c.movedim(-1,0),residual.amax()


class Aggregation(nn.Module):
    def __init__(self,n,d,hidden,mode,rank,connections,classes):
        super().__init__();self.mode=mode;self.n=n;self.sources=connections
        count=len(classes);self.classes=classes
        if mode in ('region','adapted'):
            self.a=nn.Parameter(torch.randn(hidden,d)/math.sqrt(d));self.b=nn.Parameter(torch.randn(d,hidden)/math.sqrt(hidden))
        elif mode=='source':self.a=nn.Parameter(torch.randn(count,hidden,d)/math.sqrt(d));self.b=nn.Parameter(torch.randn(count,d,hidden)/math.sqrt(hidden))
        elif mode=='node':self.a=nn.Parameter(torch.randn(n,hidden,d)/math.sqrt(d));self.b=nn.Parameter(torch.randn(n,d,hidden)/math.sqrt(hidden))
        else:raise ValueError('unknown source sharing')
        if mode in ('node','adapted'):self.e=nn.Parameter(torch.zeros(n,count,hidden))
        else:self.e=nn.Parameter(torch.zeros(count,hidden))
        if mode=='adapted':
            self.la=nn.Parameter(torch.randn(n,count,hidden,rank)*.01);self.ra=nn.Parameter(torch.randn(n,count,rank,d)/math.sqrt(d))
            self.lb=nn.Parameter(torch.randn(n,count,d,rank)*.01);self.rb=nn.Parameter(torch.randn(n,count,rank,hidden)/math.sqrt(hidden))

    def forward(self,z,mass,incidence,alpha):
        # z/mass: source slots,B,L,d. A class can occur in several slots.
        x=norm(z);n,b,l,d=z.shape;class_index=[self.classes.index(c) for _,c in self.sources]
        if self.mode=='region':f=F.linear(F.silu(F.linear(x,self.a)+self.e[class_index,None,None,:]),self.b)
        elif self.mode=='source':
            a=self.a[class_index];bb=self.b[class_index]
            hidden=torch.bmm(x.reshape(n,-1,d),a.transpose(1,2))+self.e[class_index,None,:]
            f=torch.bmm(F.silu(hidden),bb.transpose(1,2)).reshape(n,b,l,d)
        else:
            xn=x.unsqueeze(0).expand(self.n,-1,-1,-1,-1)
            if self.mode=='node':
                hidden=torch.bmm(xn.reshape(self.n,-1,d),self.a.transpose(1,2)).reshape(self.n,n,b,l,-1)
                hidden=F.silu(hidden+self.e[:,class_index,None,None,:])
                f=torch.bmm(hidden.reshape(self.n,-1,hidden.shape[-1]),self.b.transpose(1,2)).reshape(self.n,n,b,l,d)
            else:
                a=self.a+self.la@self.ra;bb=self.b+self.lb@self.rb
                a=a[:,class_index];bb=bb[:,class_index]
                hidden=torch.matmul(a[:,:,None,None],xn.unsqueeze(-1)).squeeze(-1)
                hidden=F.silu(hidden+self.e[:,class_index,None,None,:])
                f=torch.matmul(bb[:,:,None,None],hidden.unsqueeze(-1)).squeeze(-1)
        weight=mass.unsqueeze(0)*incidence[:,:,None,None]
        received=weight.sum(1);den=torch.where(received>0,received,torch.ones_like(received))
        fraction=weight/den.unsqueeze(1)
        content=z.unsqueeze(0)+alpha*(f.unsqueeze(0) if f.ndim==4 else f)
        return (fraction.unsqueeze(-1)*content).sum(1),received


class DirectSemanticLM(nn.Module):
    def __init__(self,plan,seed=11,vocab_size=4096):
        super().__init__();self.plan=plan;g=plan['graph'];d=plan['width'];self.width=d;self.last_metrics={}
        self.members=[[v for v,j in enumerate(g['node_regions']) if j==region] for region in range(g['regions'])]
        self.order=sorted(range(g['regions']),key=lambda j:g['ranks'][j]);self.connections=[]
        self.memories=nn.ModuleList();self.selectors=nn.ModuleList();self.aggregators=nn.ModuleList()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);self.embedding=nn.Embedding(vocab_size,d);nn.init.normal_(self.embedding.weight,std=.02)
            for region,nodes in enumerate(self.members):
                # Limited static specialization: one local memory specification per region.
                specs=[plan['memories'][v] for v in nodes]
                if any(s!=specs[0] for s in specs):raise ValueError('heterogeneous memory specifications within one region unsupported')
                slots=[];pairs=[]
                for e,(src,dst,delay) in enumerate(g['edges']):
                    if dst not in nodes:continue
                    if g['ranks'][g['node_regions'][dst]]-g['ranks'][g['node_regions'][src]]!=delay:
                        raise ValueError('nonaligned edge outside static all-position contract')
                    slot=(src,plan['classes']['edges'][e])
                    if slot not in slots:slots.append(slot)
                    pairs.append((nodes.index(dst),slots.index(slot)))
                for i,dst in enumerate(g['inputs']):
                    if dst not in nodes:continue
                    slot=(-1,plan['classes']['inputs'][i])
                    if slot not in slots:slots.append(slot)
                    pairs.append((nodes.index(dst),slots.index(slot)))
                if not slots:raise ValueError('unreachable static region')
                incidence=torch.zeros(len(nodes),len(slots))
                for dst,src in pairs:incidence[dst,src]+=1
                self.register_buffer('incidence_'+str(region),incidence)
                self.connections.append(slots)
                self.aggregators.append(Aggregation(len(nodes),d,plan['source_hidden_width'],plan['source_binding']['mode'],
                    plan['source_binding'].get('rank'),slots,sorted({c for _,c in slots})))
                self.memories.append(Memory(len(nodes),specs[0],plan['stride']))
                self.selectors.append(Selector(len(nodes),d,specs[0]['summary_width'],plan['regions'][region]))
            self.head=nn.Linear(d,vocab_size);self.head.weight=self.embedding.weight;nn.init.zeros_(self.head.bias)
            self._initialize_roles(seed)

    @torch.no_grad()
    def _initialize_roles(self,seed):
        # Common same-shape roles retain the same initial tensor across an
        # ablation even when another module adds/removes parameters before it.
        for name,p in self.named_parameters():
            key=f'direct-v8-role-init:{seed}:{name}:{tuple(p.shape)}'
            generator=torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')%(2**63-1))
            leaf=name.rsplit('.',1)[-1]
            if leaf in ('output_scale','skip'):p.fill_(1)
            elif leaf in ('bias','conv_bias','dt_bias','rate_log','match_bias','e','gain_logit'):
                if leaf!='gain_logit':p.zero_()
            else:
                std=.02 if name=='embedding.weight' else .01 if leaf in ('la','lb') else 1/math.sqrt(p.shape[1] if leaf=='conv' else p.shape[-1])
                if '.adapter_out.' in name:std*=.1
                p.copy_(torch.randn(p.shape,generator=generator,dtype=p.dtype)*std)

    def forward(self,ids):
        observations=[];token=FLOORS.set(observations)
        try:return self._forward(ids,observations)
        finally:FLOORS.reset(token)

    def _forward(self,ids,floors):
        embedding=self.embedding(ids);b,l,d=embedding.shape;one=embedding.new_ones(b,l)
        values={-1:embedding};mass={-1:one};counts=[];residuals=[]
        for j in self.order:
            slots=self.connections[j];z=torch.stack([values[src] for src,_ in slots]);w=torch.stack([mass[src] for src,_ in slots])
            u,r=self.aggregators[j](z,w,getattr(self,'incidence_'+str(j)),self.plan['alpha_agg'])
            memory=self.memories[j](u,r,self.plan['regions'][j]['read_mode'])
            c,error=self.selectors[j](r,memory)
            content=torch.where((c>0).unsqueeze(-1),memory['content'],torch.zeros_like(memory['content']))
            for index,node in enumerate(self.members[j]):values[node]=content[index];mass[node]=c[index]
            counts.append(torch.stack(((r>0).sum(),(c>0).sum(),c.sum(),(c==0).sum())).detach())
            residuals.append(error.detach())
        outputs=self.plan['graph']['outputs'];w=torch.stack([mass[v] for v in outputs])
        hidden=(w.unsqueeze(-1)*torch.stack([values[v] for v in outputs])).sum(0)/w.sum(0).clamp_min(1).unsqueeze(-1)
        # One aggregate observation transfer, not one synchronization per operator.
        floor_total=torch.stack(floors).sum() if floors else hidden.new_zeros(())
        observed=torch.cat((torch.stack(counts).flatten(),torch.stack(residuals),floor_total.reshape(1))).detach().cpu().tolist()
        nr=len(self.members);candidates=sum(observed[4*i] for i in range(nr));active=sum(observed[4*i+1] for i in range(nr))
        max_error=max(observed[4*nr:5*nr]);tol=1e-10 if hidden.dtype==torch.float64 else 2e-5
        if not math.isfinite(max_error) or max_error>tol:raise RuntimeError(f'allocation feasibility residual {max_error} > {tol}')
        self.last_metrics={'work/candidates':candidates,'work/state_updates':candidates,'work/read_events':candidates,
            'work/full':active,'work/physical_full_rows':self.plan['graph']['nodes']*b*l,
            'activity/full_fraction':active/max(candidates,1),'activity/send_strength':sum(observed[4*i+2] for i in range(nr)),
            'solver/max_feasibility_residual':max_error,'work/max_full_batch':len(self.members[0])*b*l}
        self.last_metrics['numerical/positive_floor_events']=observed[-1]
        for j in range(nr):
            self.last_metrics[f'region/{j}/candidates']=observed[4*j]
            self.last_metrics[f'region/{j}/full']=observed[4*j+1]
        return self.head(hidden)

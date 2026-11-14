"""Bounded HB/KV/qry specialization, retaining reference parameter roles.

Independent experiment implementation. Static neighbor packing, regional
unbind and fused weighted attention; no foundation executor imports. Same
state_dict parameter names allow explicit same-weight comparison. Historical
normal-tiny attention flooring is not claimed for the new attention policy.
"""
import math
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .direct_semantic import DirectSemanticLM
from .direct_semantic_ops import norm,FLOORS,positive_exp
from .special_attention import weighted_attention
from .ordered_divide import ordered_divide


class SpecializedSemanticLM(DirectSemanticLM):
    def __init__(self,plan,seed=11,vocab_size=4096,*,attention='math',ffn_checkpoint=False,
                 aggregation='neighbors',activity_controller=None,aggregate_vjp='native'):
        super().__init__(plan,seed,vocab_size)
        if aggregate_vjp not in ('native','ordered-v1'):
            raise ValueError('unknown aggregate VJP policy')
        if aggregate_vjp=='ordered-v1' and aggregation!='neighbors':
            raise ValueError('ordered-v1 is qualified only for static neighbors aggregation')
        self.aggregate_vjp=aggregate_vjp
        self.attention=attention;self.ffn_checkpoint=ffn_checkpoint;self.aggregation=aggregation
        self._decay_cache={};self.role_counts=[];self.slot_degrees=[]
        self.activity_controller=dict(activity_controller or {})
        self.activity_state=dict(ema=None,scale=1.,updates=0,last_tokens=0)
        if self.light_regions or plan['source_binding']['mode']!='region':
            raise ValueError('specialization requires ordinary region-shared HB')
        g=plan['graph']
        for j,nodes in enumerate(self.members):
            m=self.memories[j];s=m.s;sel=self.selectors[j].s
            if (m.family!='kv' or s['memory_read']!='qry' or s['heads']!=1 or
                any(s['full_frequencies']) or s['read_frequencies'] or any(s['distance_slopes']) or
                sel['read_mode']!='+' or sel['region_score']!='ind' or sel['local_score']!='cat' or
                sel['selection_mode'] not in ('regional_total','pooled_attention') or
                sel['selection_mode']=='regional_total' and sel['region_control_mode']!='fixed'):
                raise ValueError('unsupported local semantics for bounded specialization')
            slots=self.connections[j];indices={pair:i for i,pair in enumerate(slots)}
            neighbors=[[] for _ in nodes];local={v:i for i,v in enumerate(nodes)}
            for e,(src,dst,delay) in enumerate(g['edges']):
                if dst in local:neighbors[local[dst]].append(indices[(src,plan['classes']['edges'][e])])
            for i,dst in enumerate(g['inputs']):
                if dst in local:neighbors[local[dst]].append(indices[(-1,plan['classes']['inputs'][i])])
            if not neighbors or len({len(v) for v in neighbors})!=1:
                raise ValueError('specialization needs a uniform static indegree within each region')
            idx=[i for row in neighbors for i in row]
            self.register_buffer('_neighbor_'+str(j),torch.tensor(idx,dtype=torch.long),persistent=False)
            self.register_buffer('_classes_'+str(j),torch.tensor([self.aggregators[j].classes.index(c) for _,c in slots],dtype=torch.long),persistent=False)
            degree=[sum(row.count(i) for row in neighbors) for i in range(len(slots))]
            self.register_buffer('_degrees_'+str(j),torch.tensor(degree,dtype=torch.float32),persistent=False)
            self.slot_degrees.append(len(neighbors[0]))
            if len({c for _,c in slots})!=len(slots):raise ValueError('active class accounting requires unique slot classes')
            full=sum(p.numel() for module in (m.query,m.output,m.ffgate,m.ffup,m.ffdown) for p in module.parameters())//len(nodes)
            receive=(sum(p.numel() for p in m.parameters())-m.base.numel())//len(nodes)-full
            receive+=sum(p.numel() for p in self.selectors[j].parameters())//len(nodes)
            shared=m.base.numel()+self.aggregators[j].a.numel()+self.aggregators[j].b.numel()
            self.role_counts.append((full,receive,shared,self.aggregators[j].e.shape[-1]))
        self.shared_lm_count=self.embedding.weight.numel()+self.head.bias.numel()
        self.total_parameter_count=sum(p.numel() for p in self.parameters())

    def get_extra_state(self):
        return dict(schema='tide-v10-activity-state-v1',**self.activity_state)

    def set_extra_state(self,state):
        if state['schema']!='tide-v10-activity-state-v1':raise ValueError('wrong activity-controller state')
        self.activity_state={k:v for k,v in state.items() if k!='schema'}

    def set_progress(self,tokens):
        super().set_progress(tokens)
        if self.activity_controller and tokens>=self.activity_controller.get('start_tokens',2*2**20):
            for j,s in enumerate(self.selectors):
                s.s['allocation_temperature']=max(.005,min(32.,self.plan['regions'][j]['allocation_temperature']*self.activity_state['scale']))

    def observe_update(self,work,tokens):
        """Training-statistics-only controller, once per optimizer update.

        No validation feedback, no change to the quad formula or strength
        budget. State is checkpointed; target is measured Full/all-node slots.
        """
        if not self.activity_controller:return {}
        st=self.activity_state;st['updates']+=1;st['last_tokens']=int(tokens)
        if tokens>=self.activity_controller.get('start_tokens',2*2**20):
            ratio=work['work/full']/max(1,work['work/node_slots'])
            st['ema']=ratio if st['ema'] is None else .9*st['ema']+.1*ratio
            if st['updates']%8==0:
                error=math.log(self.activity_controller['target']/max(st['ema'],1e-8))
                st['scale']=max(.002,min(128.,st['scale']*math.exp(.3*max(-.5,min(.5,error)))))
        return {'activity/controller_tau_scale':st['scale'],'activity/controller_ema':st['ema'] or 0.}

    def aggregate(self,j,z,w,incidence):
        if self.aggregation=='reference':return self.aggregators[j](z,w,incidence,self.plan['alpha_agg'])
        if self.aggregation!='neighbors':raise ValueError('unknown aggregation implementation')
        a=self.aggregators[j];n,b,l,d=z.shape
        e=a.e.index_select(0,getattr(self,'_classes_'+str(j)))[:,None,None,:]
        f=F.linear(F.silu(F.linear(norm(z),a.a)+e),a.b)
        weighted=w.unsqueeze(-1)*(z+self.plan['alpha_agg']*f)
        idx=getattr(self,'_neighbor_'+str(j));rnodes=len(self.members[j]);degree=self.slot_degrees[j]
        received=w.index_select(0,idx).reshape(rnodes,degree,b,l).sum(1)
        value=weighted.index_select(0,idx).reshape(rnodes,degree,b,l,d).sum(1)
        denominator=torch.where(received>0,received,torch.ones_like(received)).unsqueeze(-1)
        value=(ordered_divide(value,denominator) if self.aggregate_vjp=='ordered-v1' else value/denominator)
        return value,received

    def kv(self,j,u,r):
        m=self.memories[j];s=m.s;n,b,l,d=u.shape;stride=self.plan['stride']
        if s['window'] and s['window']<(l-1)*stride+1:
            raise ValueError('this specialization does not truncate the local history window')
        w=r.clamp(0,1);x=norm(u,s['epsilon']);p=m.current(x)
        k,v=(F.linear(x,m.base)+m.adapter_out(m.adapter_in(x))).chunk(2,-1)
        # Mass needs only a shared LxL decay operator, never [node,B,L,L].
        key=(str(x.device),x.dtype,l,s['beta_short'],stride)
        if key not in self._decay_cache:
            position=torch.arange(l,device=x.device)
            dt=(position[:,None]-position[None,:])*stride
            self._decay_cache[key]=positive_exp(-s['beta_short']*dt.clamp_min(0).to(x.dtype))*(dt>=0)
        masses=[F.linear(w,self._decay_cache[key]),w.cumsum(-1)]
        q=m.read_q(x);rk=m.read_k(k);rv=m.read_v(v)
        values=[weighted_attention(q,rk,rv,w,rate=rate,stride=stride,implementation=self.attention)*mass.clamp_max(1).unsqueeze(-1)
                for rate,mass in zip((s['beta_short'],0.),masses)]
        full=weighted_attention(m.query(x),k,v,w,implementation=self.attention)*masses[1].clamp_max(1).unsqueeze(-1)
        content=u+s['alpha_att']*m.output(full)
        def ffn(y):
            f=norm(y,s['epsilon'])
            return m.ffdown(F.silu(m.ffgate(f))*m.ffup(f))
        extra=(checkpoint(ffn,content,use_reentrant=False,preserve_rng_state=False)
               if self.ffn_checkpoint and torch.is_grad_enabled() else ffn(content))
        content=content+s['alpha_ff']*extra
        return dict(x=x,p=p,memory=values,mass=masses,content=content)

    def _forward(self,ids,floors):
        embedding=self.embedding(ids);b,l,d=embedding.shape;one=embedding.new_ones(b,l)
        values={-1:embedding};mass={-1:one};counts=[];errors=[];extra=[]
        for j in self.order:
            slots=self.connections[j]
            z=torch.stack([values[src] for src,_ in slots]);w=torch.stack([mass[src] for src,_ in slots])
            u,r=self.aggregate(j,z,w,getattr(self,'incidence_'+str(j)))
            memory=self.kv(j,u,r);c,error=self.selectors[j](r,memory)
            content=torch.where((c>0).unsqueeze(-1),memory['content'],torch.zeros_like(memory['content']))
            # One UnbindBackward per tensor instead of one SelectBackward per node.
            for node,value,strength in zip(self.members[j],content.unbind(0),c.unbind(0)):
                values[node]=value;mass[node]=strength
            received=r>0;full=c>0;C=received.sum();A=full.sum()
            hist=received.to(r.dtype).cumsum(-1)
            pf,pr,ps,pe=self.role_counts[j]
            unsat=self.selectors[j].last_unsaturated
            counts.append(torch.stack((C,A,c.sum(),(c==0).sum(),error,
                unsat if unsat is not None else r.new_zeros(()),self.selectors[j].last_support_errors)).detach())
            extra.append(torch.stack((A*pf,C*pr,received.any(0).sum()*ps+(w>0).sum()*pe,
                ((w>0)*getattr(self,'_degrees_'+str(j))[:,None,None]).sum(),
                (full*hist).sum(),2*(received*hist).sum(),
                (((c>0)&(c<(self.selectors[j].s['gamma_max']*r.clamp(0,1)).clamp_max(1))).sum(0)>1).sum())).detach())
        outputs=self.plan['graph']['outputs'];w=torch.stack([mass[v] for v in outputs])
        hidden=(w.unsqueeze(-1)*torch.stack([values[v] for v in outputs])).sum(0)/w.sum(0).clamp_min(1).unsqueeze(-1)
        floor_total=torch.stack(floors).sum() if floors else hidden.new_zeros(())
        observation=torch.cat((torch.stack(counts).flatten(),torch.stack(extra).sum(0),floor_total.reshape(1))).detach().cpu().tolist()
        nr=len(self.members);rows=[observation[7*i:7*i+7] for i in range(nr)]
        candidates=sum(x[0] for x in rows);active=sum(x[1] for x in rows);error=max(x[4] for x in rows)
        support=sum(x[6] for x in rows);tol=1e-10 if hidden.dtype==torch.float64 else 2e-5
        if not math.isfinite(error) or error>tol:raise RuntimeError(f'allocation feasibility residual {error} > {tol}')
        if support:raise RuntimeError(f'mathematically positive sending lost support: {support} coordinates')
        pf,pr,ps,messages,fpairs,rpairs,router_rows=observation[7*nr:7*nr+7];ps+=self.shared_lm_count*b*l
        slots=self.plan['graph']['nodes']*b*l
        self.last_metrics={'work/candidates':candidates,'work/state_updates':candidates,'work/read_events':candidates,
            'work/full':active,'work/physical_full_rows':slots,'work/node_slots':slots,'work/body_candidates':candidates,
            'work/body_full':active,'work/body_node_slots':slots,'work/boundary_full':0.,'work/boundary_candidates':0.,
            'work/physical_light_rows':0.,'work/unsaturated':sum(x[5] for x in rows),'work/max_full_batch':max(map(len,self.members))*b*l,
            'activity/full_fraction':active/max(candidates,1),'activity/full_all_fraction':active/slots,
            'activity/receipt_fraction':candidates/slots,'activity/body_full_all_fraction':active/slots,
            'activity/body_receipt_fraction':candidates/slots,'activity/send_strength':sum(x[2] for x in rows),
            'solver/max_feasibility_residual':error,'discrete/positive_support_errors':support,
            'numerical/positive_floor_events':observation[-1],
            'numerical/attention_probability_floor_applied':0.,
            'work/full_parameter_uses':pf,'work/receive_parameter_uses':pr,'work/shared_parameter_uses':ps,
            'work/active_parameter_uses':pf+pr+ps,'work/dense_parameter_uses':self.total_parameter_count*b*l,
            'work/prediction_positions':b*l,'work/message_edges_including_input':messages,
            'work/logical_full_attention_pairs':fpairs,'work/logical_read_attention_pairs':rpairs,
            'work/dense_causal_attention_pairs':3*slots*(l+1)/2,
            'work/physical_attention_pair_upper_bound':3*slots*l,
            'work/router_multi_unsaturated_rows':router_rows,
            'activity/active_parameters_per_position':(pf+pr+ps)/(b*l)}
        for j,row in zip(self.order,rows):
            self.last_metrics.update({f'region/{j}/candidates':row[0],f'region/{j}/full':row[1],f'region/{j}/unsaturated':row[5]})
        return self.head(hidden)

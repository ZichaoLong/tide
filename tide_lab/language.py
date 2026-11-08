"""Application-owned all-token heads and a single pretrained MLP replacement.

Graph scheduling, local functions, private state and VJP stay in tidegraph.
The vectorized readout is task-side pooling of public output payloads; checks
compare its values and gradients with the public scalar readout.
"""
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .core5 import Core5Encoder


def sequence_readout(encoder,result,features):
    from tidegraph.weighted import Payload
    batch,length,width=features.shape
    ports=len(encoder.binding.graph.outputs)
    zero=features.new_zeros(width+1)
    slots=[zero]*(batch*length*ports)
    seen=set()
    for b,time,port,value in result.outputs:
        if not 0<=b<batch or not 0<=port<ports:raise ValueError('output batch/port outside task')
        delta=time-encoder.spec.rank(encoder.binding.graph.outputs[port])
        position,remainder=divmod(delta,encoder.spec.stride)
        if remainder or not 0<=position<length:raise ValueError('output clock outside declared sequence')
        key=(b*length+position)*ports+port
        if key in seen:raise ValueError('duplicate task output coordinate')
        seen.add(key);slots[key]=value
    payload=Payload.decode(torch.stack(slots).reshape(batch,length,ports,width+1))
    mass,numeric=payload.numerical()
    return numeric.sum(2)/mass.sum(2).clamp_min(1).unsqueeze(-1)


class SequenceCore5(Core5Encoder):
    def forward(self,features,*,implementation='eager',chunk_size=None):
        result=self.run_features(features,implementation=implementation,chunk_size=chunk_size,trace=self.observe)
        s=result.stats;candidates=s.get('candidate_events',0);full=s.get('selected_events',0)
        self.last_metrics={'work/candidates':candidates,'work/state_updates':candidates,
            'work/read_events':candidates,'work/full':full,'work/messages':s.get('visited_edges',0),
            'work/source_rows':s.get('source_rows',0),'activity/full_fraction':full/max(candidates,1),
            **{'work/'+k:s.get(k,0) for k in ('max_full_batch','max_state_batch','max_state_sequence','max_region_batch','region_blocks')}}
        return sequence_readout(self,result,features)


class TideLanguageModel(nn.Module):
    def __init__(self,config,seed=11,vocab_size=4096):
        super().__init__();options=dict(config)
        width=options.get('width',32)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.embedding=nn.Embedding(vocab_size,width)
            nn.init.normal_(self.embedding.weight,std=.02)
            self.encoder=SequenceCore5(seed=seed,**options)
            self.head=nn.Linear(width,vocab_size)
            self.head.weight=self.embedding.weight
            nn.init.zeros_(self.head.bias)
        self.last_metrics={}

    def forward(self,x,*,intervention=None,implementation='eager',chunk_size=None):
        features=self.embedding(x)
        hidden=self.encoder(features,implementation=implementation,chunk_size=chunk_size)
        self.last_metrics=dict(self.encoder.last_metrics)
        if intervention=='zero_core':hidden=torch.zeros_like(hidden)
        elif intervention is not None:raise ValueError('unknown language intervention')
        return self.head(hidden)


class CausalBlock(nn.Module):
    def __init__(self,width,heads,ffn):
        super().__init__();self.heads=heads
        self.norm1=nn.LayerNorm(width);self.norm2=nn.LayerNorm(width)
        self.qkv=nn.Linear(width,3*width);self.out=nn.Linear(width,width)
        self.ffn=nn.Sequential(nn.Linear(width,ffn),nn.GELU(),nn.Linear(ffn,width))

    def forward(self,x):
        b,l,d=x.shape
        q,k,v=self.qkv(self.norm1(x)).reshape(b,l,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind(0)
        scores=q@k.transpose(-1,-2)/math.sqrt(d//self.heads)
        mask=torch.ones(l,l,dtype=torch.bool,device=x.device).triu(1)
        attention=scores.masked_fill(mask,float('-inf')).softmax(-1)
        x=x+self.out((attention@v).transpose(1,2).reshape(b,l,d))
        return x+self.ffn(self.norm2(x))


class SmallTransformer(nn.Module):
    def __init__(self,seed=11,vocab_size=4096,width=96,layers=3,heads=4,ffn=384):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.embedding=nn.Embedding(vocab_size,width)
            nn.init.normal_(self.embedding.weight,std=.02)
            self.blocks=nn.ModuleList([CausalBlock(width,heads,ffn) for _ in range(layers)])
            self.norm=nn.LayerNorm(width);self.head=nn.Linear(width,vocab_size)
            self.head.weight=self.embedding.weight;nn.init.zeros_(self.head.bias)
        position=torch.arange(1024,dtype=torch.float32)[:,None]
        phase=position*torch.exp(torch.arange(0,width,2,dtype=torch.float32)*(-math.log(10000)/width))
        encoding=torch.zeros(1024,width);encoding[:,0::2]=phase.sin();encoding[:,1::2]=phase.cos()
        self.register_buffer('position',encoding,persistent=True);self.last_metrics={}

    def forward(self,x,**kwargs):
        if x.shape[1]>len(self.position):raise ValueError('declared positional capacity exceeded')
        h=self.embedding(x)+self.position[:x.shape[1]].to(dtype=self.embedding.weight.dtype)
        for block in self.blocks:h=block(h)
        return self.head(self.norm(h))


class LocalReplacement(nn.Module):
    def __init__(self,kind,core_config,stats,seed=11,dimension=896):
        super().__init__();self.kind=kind;self.last_metrics={}
        for key,value in stats.items():self.register_buffer(key,value.detach().clone())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            if kind.startswith('tide'):
                width=core_config['width'];self.input=nn.Linear(dimension,width,bias=False)
                self.core=SequenceCore5(seed=seed,**core_config)
                self.output=nn.Linear(width,dimension,bias=False)
            elif kind=='mlp':
                self.input=nn.Linear(dimension,330,bias=False);self.core=nn.GELU()
                self.output=nn.Linear(330,dimension,bias=False)
            else:raise ValueError('unknown local replacement')
            nn.init.zeros_(self.output.weight)

    def forward(self,x):
        h=self.input((x-self.x_mean)/self.x_scale)
        if self.kind.startswith('tide'):h=torch.tanh(h)
        h=self.core(h)
        if self.kind.startswith('tide'):self.last_metrics=dict(self.core.last_metrics)
        return self.output(h)*self.y_scale+self.y_mean


def load_qwen(asset,device):
    from transformers import AutoModelForCausalLM
    from .growth import verify_checkpoint
    verify_checkpoint(Path(asset))
    model=AutoModelForCausalLM.from_pretrained(asset,local_files_only=True,trust_remote_code=False,
        torch_dtype=torch.float32,attn_implementation='eager').to(device).eval()
    model.requires_grad_(False)
    if model.config.model_type!='qwen2' or model.config.hidden_size!=896:raise ValueError('wrong fixed Qwen')
    return model


class FrozenPrefixTail(nn.Module):
    """Exact frozen-prefix factorization for teacher-forced final-MLP training.

    Prefix preparation cost is recorded separately. These timings are not
    whole-model inference speed; that is checked with the actual replacement.
    """
    def __init__(self,lm,kind,core_config,stats,seed=11):
        super().__init__();self.kind=kind
        self.norm=lm.model.norm;self.head=lm.lm_head
        self.selected=(lm.model.layers[-1].mlp if kind=='original' else
                       LocalReplacement(kind,core_config,stats,seed))
        self.norm.requires_grad_(False);self.head.requires_grad_(False)
        self.selected.requires_grad_(True);self.last_metrics={}

    def predicted_mlp(self,x):
        value=self.selected(x)
        self.last_metrics=dict(getattr(self.selected,'last_metrics',{}))
        return value

    def hidden(self,batch,alpha=1.):
        if alpha==0:
            self.last_metrics={}
            return batch['residual']+batch['teacher']
        prediction=self.predicted_mlp(batch['x'])
        output=prediction if alpha==1 else batch['teacher']+alpha*(prediction-batch['teacher'])
        return batch['residual']+output

    def logits(self,batch,alpha=1.):return self.head(self.norm(self.hidden(batch,alpha)))

    def ce(self,batch,alpha=1.,chunk=128):
        """Bound vocabulary workspace; recompute head only during backward."""
        from torch.utils.checkpoint import checkpoint
        h=self.hidden(batch,alpha).flatten(0,1);targets=batch['labels'].flatten()
        total=h.new_zeros(())
        def part(x,y):return F.cross_entropy(self.head(self.norm(x)),y,ignore_index=-100,reduction='sum')
        for start in range(0,len(h),chunk):
            x,y=h[start:start+chunk],targets[start:start+chunk]
            total=total+(checkpoint(part,x,y,use_reentrant=False,preserve_rng_state=False)
                         if torch.is_grad_enabled() and x.requires_grad else part(x,y))
        return total/(targets!=-100).sum().clamp_min(1)

    @torch.no_grad()
    def token_nll(self,batch,alpha=1.,chunk=128):
        h=self.hidden(batch,alpha).flatten(0,1);targets=batch['labels'].flatten();values=[]
        for start in range(0,len(h),chunk):
            logits=self.head(self.norm(h[start:start+chunk]))
            values.append(F.cross_entropy(logits,targets[start:start+chunk],ignore_index=-100,reduction='none'))
        return torch.cat(values).reshape(batch['labels'].shape)


def trainable_inventory(model):
    aliases={}
    for name,p in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(p),[]).append(name)
    return dict(total_parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        shared_groups=[v for v in aliases.values() if len(v)>1])


class IntegratedMLP(nn.Module):
    """One real module replacement; optional inference-only hidden-prefix replay."""
    def __init__(self,original,replacement,alpha=1.,replay=False):
        super().__init__();self.original=original;self.replacement=replacement
        self.alpha=alpha;self.replay=replay;self.history=None;self.original_calls=0

    def forward(self,x):
        if self.alpha==0:
            self.original_calls+=1;return self.original(x)
        if self.replay:
            if torch.is_grad_enabled():raise ValueError('prefix replay wrapper is inference-only')
            self.history=x if self.history is None else torch.cat([self.history,x],dim=1)
            prediction=self.replacement(self.history)[:,-x.shape[1]:]
        else:prediction=self.replacement(x)
        if self.alpha==1:return prediction
        self.original_calls+=1;original=self.original(x)
        return original+self.alpha*(prediction-original)

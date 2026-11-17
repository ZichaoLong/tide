"""Region-window evaluators for the current strictly layered FC/HB instances."""
from dataclasses import dataclass, fields, is_dataclass
import torch
from torch import nn

from .config import ModelConfig
from .topology import build_topology
from .reception import Reception
from .region import Description, RegionNetwork, region_control
from .kv import KVMemory
from .parameters import NodeOwners


def _number(keys):
    lookup = {}
    return [lookup.setdefault(key,len(lookup)) for key in keys]


def detach_state(value):
    if isinstance(value,torch.Tensor):
        return value.detach()
    if is_dataclass(value):
        return type(value)(**{f.name:detach_state(getattr(value,f.name)) for f in fields(value)})
    if isinstance(value,dict):
        return {k:detach_state(v) for k,v in value.items()}
    if isinstance(value,tuple) and hasattr(value,"_fields"):
        return type(value)(*(detach_state(v) for v in value))
    if isinstance(value,(tuple,list)):
        return type(value)(detach_state(v) for v in value)
    return value


@dataclass
class LayerOutput:
    values: torch.Tensor
    strengths: torch.Tensor


@dataclass
class ModelState:
    position: int
    memories: tuple
    pending: tuple


@dataclass
class Result:
    values: torch.Tensor  # [output node, B, position/logical tick, d]
    strengths: torch.Tensor
    times: torch.Tensor
    state: ModelState
    work: dict | None = None

    def readout(self):
        total = self.strengths.sum(0)
        denominator = torch.where(total>1,total,torch.ones_like(total))
        return (self.values*self.strengths[...,None]).sum(0)/denominator[...,None]


class TideModel(nn.Module):
    def __init__(self,config:ModelConfig):
        super().__init__()
        self.config = config
        self.graph = graph = build_topology(config.topology)
        n,d = len(graph.nodes),config.d
        self.settle = graph.config.timing == "settle"
        self.kinds = ["S" if config.memory=="S" or config.memory=="X" and node.role=="H" else "K"
                      for node in graph.nodes]
        gkeys = [(node.layer,node.role,self.kinds[node.id]) for node in graph.nodes]
        if config.sharing=="region_shared":
            gkeys = [(node.region,key) for node,key in zip(graph.nodes,gkeys)]
        groups = None if config.sharing=="independent" else _number(gkeys)
        sources = [set(graph.edges[e].source_class for e in graph.incoming[v.id]) |
                   ({"ext"} if v.layer==0 else set()) for v in graph.nodes]
        self.receive = Reception(sources,d,config.aggregation,groups,config.adapter_rank,config.attention_backend)
        scored = [v.id for v in graph.nodes if not (v.role=="H" and graph.config.partition=="singleton")]
        self.description_ids = {v:i for i,v in enumerate(scored)}
        self.describe = Description(len(scored),d,None if groups is None else _number([gkeys[v] for v in scored]),config.adapter_rank)
        self.memories = nn.ModuleDict()
        self.memory_ids = {}
        for kind in ("K","S"):
            members = [v.id for v in graph.nodes if self.kinds[v.id]==kind]
            if not members:
                continue
            self.memory_ids[kind] = {v:i for i,v in enumerate(members)}
            kgroups = None if groups is None else _number([gkeys[v] for v in members])
            args = (len(members),d,kgroups,config.adapter_rank,graph.period)
            kwargs = dict(settle=self.settle,backend=config.attention_backend,chunk=config.attention_chunk)
            if kind=="K":
                kvkeys = [(graph.nodes[v].layer,graph.nodes[v].region) if config.sharing=="region_shared"
                          else (graph.nodes[v].layer,tuple(sorted(set(graph.nodes[u].role for u in graph.regions[graph.nodes[v].region]))))
                          for v in members]
                kv_groups = None if groups is None else _number(kvkeys)
                self.memories[kind] = KVMemory(*args,**kwargs,kv_groups=kv_groups)
                self.memories[kind].unit_weights = graph.config.kind=="fc"
            else:
                from .ssd_memory import SSDMemory
                self.memories[kind] = SSDMemory(*args,**kwargs)
        rkeys = [(graph.region_ranks[r],tuple(sorted(set(graph.nodes[v].role for v in nodes))))
                 for r,nodes in enumerate(graph.regions)]
        fixed = [all(graph.nodes[v].role=="H" for v in nodes) and graph.config.partition=="singleton"
                 for nodes in graph.regions]
        region_keys = [rkeys[r] if config.sharing=="layer_shared" else r for r in range(len(graph.regions)) if not fixed[r]]
        labels = _number(region_keys)
        iterator = iter(labels)
        self.region_group = [-1 if skip else next(iterator) for skip in fixed]
        self.regional = nn.ModuleList(RegionNetwork(d,config.attention_backend) for _ in set(labels))
        self.inputs = nn.ModuleDict({role:nn.Linear(config.input_dim or d,d,bias=False)
                                    for role in sorted(set(graph.nodes[v].role for v in graph.input_nodes))})
        self.layer_nodes = [tuple(v.id for v in graph.nodes if v.layer==l) for l in range(len(graph.widths))]
        self.layer_regions = [tuple(r for r,vs in enumerate(graph.regions) if graph.nodes[vs[0]].layer==l)
                              for l in range(len(graph.widths))]
        self._routes = self._make_routes()

    def _make_routes(self):
        routes = []
        labels = self.receive.embedding.labels
        local = {v:i for nodes in self.layer_nodes for i,v in enumerate(nodes)}
        for nodes in self.graph.regions:
            width = max(1,max(len(self.graph.incoming[v]) for v in nodes))
            classes = torch.zeros((len(nodes),width),dtype=torch.long)
            banks = {}
            for row,v in enumerate(nodes):
                if self.graph.nodes[v].layer==0:
                    classes[row,0] = labels["ext"]
                for slot,eid in enumerate(self.graph.incoming[v]):
                    edge = self.graph.edges[eid]
                    layer = self.graph.nodes[edge.src].layer
                    if layer not in banks:
                        banks[layer] = (torch.zeros((len(nodes),width),dtype=torch.long),
                                        torch.zeros((len(nodes),width),dtype=torch.bool))
                    banks[layer][0][row,slot] = local[edge.src]
                    banks[layer][1][row,slot] = True
                    classes[row,slot] = labels[edge.source_class]
                if self.graph.incoming[v]:
                    classes[row,len(self.graph.incoming[v]):] = classes[row,0]
            routes.append((classes,banks))
        return routes

    def _incoming(self,region,layers,pending,embedded,b,length,device,inject=True):
        nodes = self.graph.regions[region]
        classes,banks = self._routes[region]
        if self.graph.nodes[nodes[0]].layer==0:
            content = torch.stack([embedded[self.graph.nodes[v].role] for v in nodes])[:,:,:,None,:]
            mass = content.new_ones((len(nodes),b,length,1))
            if not inject:
                mass = torch.zeros_like(mass)
            if not self.settle:
                valid = torch.arange(length,device=device)%self.graph.period==0
                mass = mass*valid[None,None,:,None]
            return content,mass,classes.to(device)
        content = embedded["D"].new_zeros((len(nodes),b,length,classes.shape[1],self.config.d))
        mass = content[...,0]*0
        for layer,(indices,mask) in banks.items():
            source = layers[layer]
            if not self.settle:
                previous = pending[layer] if pending else LayerOutput(torch.zeros_like(source.values[:,:,:1]),
                                                                      torch.zeros_like(source.strengths[:,:,:1]))
                values = torch.cat((previous.values,source.values[:,:,:-1]),2)
                strengths = torch.cat((previous.strengths,source.strengths[:,:,:-1]),2)
            else:
                values,strengths = source.values,source.strengths
            ids,valid = indices.to(device),mask.to(device)
            content = content + values[ids].permute(0,2,3,1,4)*valid[:,None,None,:,None]
            mass = mass + strengths[ids].permute(0,2,3,1)*valid[:,None,None,:]
        return content,mass,classes.to(device)

    def forward(self,x,state=None,*,record_work=False):
        return self._run(x,state,record_work=record_work)

    def flush(self,state,*,record_work=False):
        """Finish a finite Timed input stream without inventing new messages.

        Returns enough whole input periods to drain every strictly layered path.
        Settle has already returned every output and does not need this call.
        The returned state's clock advances through these empty periods.
        """
        if self.settle or state is None or not state.pending:
            raise ValueError("flush requires a continued Timed stream")
        periods = (len(self.layer_nodes)-1+self.graph.period-1)//self.graph.period
        sample = state.pending[0].values
        x = sample.new_zeros((sample.shape[1],periods,self.config.input_dim or self.config.d))
        return self._run(x,state,record_work=record_work,inject=False)

    def _run(self,x,state=None,*,record_work=False,inject=True):
        if x.dtype not in (torch.float32,torch.float64):
            raise TypeError("models use FP32; FP64 is available for accuracy verification")
        if x.ndim!=3 or x.shape[-1]!=(self.config.input_dim or self.config.d) or x.shape[1]==0:
            raise ValueError("input must be nonempty [B,L,input_dim]")
        b,positions,_ = x.shape
        start = 0 if state is None else state.position
        if type(start) is not int or start < 0:
            raise OverflowError("state.position must be a nonnegative Python int")
        maximum = ((start+positions-1)*self.graph.period+len(self.layer_nodes) if self.settle else
                   (start+positions)*self.graph.period)
        if maximum > torch.iinfo(torch.int64).max:
            raise OverflowError("continuation logical time exceeds int64")
        length = positions if self.settle else positions*self.graph.period
        embedded = {role:projection(x) for role,projection in self.inputs.items()}
        if not self.settle:
            loc = torch.arange(positions,device=x.device)*self.graph.period
            embedded = {role:value.new_zeros((b,length,self.config.d)).index_copy(1,loc,value)
                        for role,value in embedded.items()}
        layers,newmem = [],[]
        work = []
        for layer,regions in enumerate(self.layer_regions):
            values,controls = [],[]
            for region in regions:
                members = self.graph.regions[region]
                owners = NodeOwners(torch.tensor(members,device=x.device),members)
                contents,masses,classes = self._incoming(region,layers,() if state is None else state.pending,
                                                         embedded,b,length,x.device,inject)
                received,u = self.receive(contents,masses,classes,owners)
                # Unit FC ingress and the received lower budget imply every
                # node writes unit weight. Use that exact reachable-state identity.
                mass = (torch.ones_like(received) if self.graph.config.kind=="fc" else
                        torch.where(received>=1,torch.ones_like(received),received.clamp_min(0)))
                base = torch.arange(length,device=x.device,dtype=torch.int64)
                time = ((base+start)*self.graph.period+layer+1 if self.settle else
                        base+start*self.graph.period+1)
                times = time[None,None].expand(len(members),b,length)
                desc = [u.new_zeros((len(members),b,length,self.config.d//4)) for _ in range(3)]
                desc += [u.new_zeros((len(members),b,length)) for _ in range(2)]
                contexts,saved = {},{}
                for kind in self.memories:
                    idx = [i for i,v in enumerate(members) if self.kinds[v]==kind]
                    if not idx:
                        continue
                    sub = torch.tensor(idx,device=x.device)
                    possible = tuple(self.memory_ids[kind][members[i]] for i in idx)
                    mids = NodeOwners(torch.tensor(possible,device=x.device),possible)
                    old = None if state is None else state.memories[region][kind]
                    parts,context,updated = self.memories[kind].prepare(u[sub],mass[sub],times[sub],mids,old)
                    desc = [a.index_copy(0,sub,z.to(a.dtype)) for a,z in zip(desc,parts)]
                    contexts[kind] = (sub,context)
                    saved[kind] = updated
                roles = torch.tensor([self.graph.nodes[v].role=="H" for v in members],device=x.device)
                if bool(roles.all()) and self.graph.config.partition=="singleton":
                    control = torch.where(received>=0.5,torch.ones_like(received),2*received)
                else:
                    possible = tuple(self.description_ids[v] for v in members)
                    description_owners = NodeOwners(torch.tensor(possible,device=x.device),possible)
                    z,score = self.describe(u,received,desc,description_owners)
                    score = self.regional[self.region_group[region]](z,score,mass)
                    control = region_control(score,received,roles,self.config.budget or max(1,self.graph.config.n//32),
                                             self.config.allocation,self.graph.config.partition)
                full = torch.zeros_like(u)
                if record_work:
                    work.append(torch.stack(((received>0).sum(),(control>0).sum(),(masses>0).sum(),
                                             torch.as_tensor(received.numel(),device=x.device))))
                for kind,(sub,context) in contexts.items():
                    full = full.index_copy(0,sub,self.memories[kind].finish(context,control[sub]).to(full.dtype))
                values.append(full); controls.append(control); newmem.append(saved)
            layers.append(LayerOutput(torch.cat(values,0),torch.cat(controls,0)))
        pending = tuple(LayerOutput(v.values[:,:,-1:],v.strengths[:,:,-1:]) for v in layers) if not self.settle else ()
        final = layers[-1]
        nextstate = ModelState(start+positions,tuple(newmem),pending)
        measured = None
        if record_work:
            measured = dict(zip(("candidates", "full", "positive_messages", "candidate_slots"),
                                torch.stack(work).sum(0).tolist()))
        return Result(final.values,final.strengths,time,nextstate,measured)

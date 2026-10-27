"""Architecture §2–4 and allocation §2–3 local bindings, upstream 0770593.

Generic graph scheduling is outside this module. Each input atom participates
once even when several physical sources have the same parameter class.
"""
import torch
from ..aggregate import AggregateProgram, AggregateResult
from ..full import FullProgram, FullResult
from ..history import History
from ..readout import ReadProgram
from ..region import RegionProgram, Selection
from .entropy_allocation import allocate
from .config import TOTAL, HARD, ATTENTION
from .sending import attention_send, proportional, total_zero, hard_selection
from .tensors import Payload, Received, norm, parameter


class SourceTransform(torch.nn.Module):
    def __init__(self, a, b, embedding):
        super().__init__()
        self.a, self.b, self.embedding = a, b, embedding

    def forward(self, z, epsilon):
        return self.b@torch.nn.functional.silu(self.a@norm(z,epsilon)+self.embedding)


class WeightedAggregate(AggregateProgram):
    def __init__(self, profile, classes, transforms, alpha, epsilon, hidden_width):
        super().__init__()
        self.profile, self.classes = profile, classes
        self.transforms = torch.nn.ModuleDict({str(k): v for k,v in transforms.items()})
        self.alpha, self.epsilon = alpha, epsilon
        self.hidden_width = hidden_width

    def step(self, w, request):
        masses, terms = [], []
        for source in request.sources:
            p = Payload.decode(source.atom.value)
            mass = p.strength.clamp_min(0)
            z = p.content
            if self.alpha:
                z = z+self.alpha*self.transforms[str(self.classes.classify(source.atom))](z,self.epsilon)
            masses.append(mass); terms.append(mass*z)
        mass = torch.stack(masses).sum()
        u = torch.stack(terms).sum(0)/mass if mass > 0 else terms[0].new_zeros(terms[0].shape)
        return AggregateResult(Received(mass,u).encode(), {})


class ScalarHead(torch.nn.Module):
    def __init__(self, width, hidden, generator, dtype):
        super().__init__()
        self.matrix = parameter((hidden,width),generator,dtype)
        self.hidden_bias = parameter((hidden,),generator,dtype)
        self.vector = parameter((hidden,),generator,dtype)
        self.bias = parameter((),generator,dtype)

    def forward(self, value):
        return self.bias+(torch.nn.functional.silu(value@self.matrix.t()+self.hidden_bias)*self.vector).sum(-1)


class Describe(ReadProgram):
    def __init__(self, profile, memory_config, region_config, generator, dtype):
        super().__init__()
        self.profile, self.memory_config, self.config = profile, memory_config, region_config
        width = memory_config.width+4*memory_config.summary_width+3
        self.score = ScalarHead(width,region_config.hidden_width,generator,dtype)
        self.field_layout = {"w": (), "mu": (), "score": ()}
        if region_config.regional:
            self.field_layout["evidence"] = (region_config.region_control_width,)
        if region_config.region_control_mode == "context":
            self.evidence_weight = parameter((region_config.region_control_width,width),generator,dtype)
            self.evidence_bias = parameter((region_config.region_control_width,),generator,dtype)
        if region_config.send_gain == "context":
            self.gain = ScalarHead(width,region_config.hidden_width,generator,dtype)
        elif region_config.gain_logit is not None:
            self.gain_logit = torch.nn.Parameter(torch.tensor(region_config.gain_logit,dtype=dtype))

    def features(self, memory, state, time, h):
        short, long, ws, wl = memory.mem(state,time,h.content)
        current = memory.current(h.content)
        return torch.cat((norm(h.content,self.memory_config.epsilon),h.mass.log1p().reshape(1),
                          short,long,ws.log1p().reshape(1),wl.log1p().reshape(1),current*short,current*long))

    def step(self, weights, request):
        h, memory = Received.decode(request.content.value), weights.kernel
        # The 0 mode binds the actual initial sequence. None is only the
        # engine's mode-restricted request marker, never a mathematical state.
        state = memory.initial(weights) if request.state is None else request.state
        features = self.features(memory,state,request.time,h)
        received = h.mass.clamp(max=1)
        if self.config.regional:
            mu = (self.config.gamma_max*received).clamp(max=1)
        else:
            if self.config.send_gain == "context":
                gamma = 1+(self.config.gamma_max-1)*self.gain(features).sigmoid()
            elif self.config.gain_logit is not None:
                gamma = 1+(self.config.gamma_max-1)*self.gain_logit.sigmoid()
            else:
                gamma = self.config.fixed_gain
            mu = (gamma*h.mass).clamp(max=1)
        result = {"w": received, "mu": mu, "score": self.score(features)}
        if self.config.regional:
            result["evidence"] = (norm(self.evidence_weight@features+self.evidence_bias,self.memory_config.epsilon)
                if self.config.region_control_mode == "context" else features.new_zeros(self.config.region_control_width))
        return result


class RegionalTotal(RegionProgram):
    """Weighted region binding; historical class name remains for API compatibility."""
    def __init__(self, profile, config, generator, dtype):
        super().__init__()
        self.profile, self.config = profile, config
        if config.region_control_mode == "context":
            self.head = ScalarHead(config.region_control_width+2,config.hidden_width,generator,dtype)

    def validate_weights(self, layout):
        if (layout.spec.budget != self.config.capacity(len(layout.members))
                or not layout.spec.observe_all):
            raise ValueError("weighted region requires the declared k/K and observe_all")

    def validate_history(self, history, layout):
        if history.scalars or history.node_maps or history.tensors:
            raise ValueError("weighted region history is a singleton")

    def total(self, w, mu, evidence):
        recv, mass = w.sum(), mu.sum()
        low, high = recv.clamp(max=1), mass.clamp(max=self.config.k)
        fraction = (self.context(w,mu,evidence).sigmoid() if self.config.region_control_mode == "context"
                    else recv.new_tensor(self.config.total_fraction))
        return low+fraction*(high-low)

    def context(self, w, mu, evidence):
        recv, mass = w.sum(), mu.sum()
        z = (w.unsqueeze(-1)*evidence).sum(0)/recv.clamp(min=1)
        return self.head(torch.cat((z,recv.log1p().reshape(1),mass.log1p().reshape(1))))

    def step(self, r):
        nodes, descriptors = zip(*r.candidates)
        fields = {name: torch.stack([d[name] for d in descriptors]) for name in descriptors[0]}
        c, mode = self.config, self.config.selection_mode
        w, mu, scores = (fields[n] for n in ("w","mu","score"))
        if mode in TOTAL:
            low = high = self.total(w,mu,fields["evidence"])
            result = total_zero(scores,mu,low,nodes) if mode == "regional_total_zero" else None
        elif mode in ATTENTION:
            result = attention_send(scores,mu,w,mode=mode,base=c.allocation_base,temperature=c.allocation_temperature)
        elif mode == "proportional":
            result = proportional(mu,c.k)
        else:
            eta = self.context(w,mu,fields["evidence"]) if c.region_control_mode == "context" else c.eta
            scores = scores-eta
            result = hard_selection(scores,mu,c.k,nodes) if mode in HARD else None
            low = (w.sum() if c.regional else mu.sum()).clamp_max(1)
            high = c.k
        if result is None:
            result = allocate(scores,mu,low,high,c.allocation_temperature,
                              regularizer=c.regularizer,ent_alpha=c.ent_alpha).controls
        controls = dict(zip(nodes,result.unbind()))
        return Selection({v for v,c in controls.items() if c.detach() > 0}, controls, History(last_time=r.time))


class WeightedFull(FullProgram):
    def __init__(self, profile, content):
        super().__init__()
        self.profile, self.content = profile, content

    def step(self, weights, request, slots, mode, zeta):
        if mode != "hard":
            raise ValueError("weighted Full has its own exact VJP; legacy Emit mode must be hard")
        h = Received.decode(request.content.value)
        g = self.content(weights.kernel,request.comparison,request.time,h.content)
        payload = Payload(request.control,g).encode()
        return FullResult(payload,{slot: payload for slot in range(slots)})

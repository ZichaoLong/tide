"""Architecture §5 concrete model binding; all graph schedulers stay independent."""
from dataclasses import dataclass
import torch
from ..graph import Graph, Node, Region
from ..ops import Model
from ..source_class import SourceClasses
from .config import KVConfig, RegionalConfig, finite, identity, positive_int, region_profile
from .kv import KVContent, KVMemory
from .recurrent_config import RecurrentConfig
from .recurrent import BasicContent, BasicMemory
from .gated_delta import GatedDeltaContent, GatedDeltaMemory
from .ssd import SSDContent, SSDMemory
from .programs import Describe, RegionalTotal, SourceTransform, WeightedAggregate, WeightedFull
from .tensors import parameter


MEMORY_TYPES = {"kv": (KVMemory, KVContent), "affine": (BasicMemory, BasicContent),
                "delta": (BasicMemory, BasicContent), "gated_delta": (GatedDeltaMemory, GatedDeltaContent),
                "ssd": (SSDMemory, SSDContent)}


def memory_family(config):
    return "kv" if isinstance(config, KVConfig) else config.memory_family


def memory_profile(config):
    return identity("weighted-"+memory_family(config)+"-v1", config)


class WeightedModel(Model):
    def validate_binding(self, graph):
        if graph.identity != self.graph_identity:
            raise ValueError("weighted graph/model binding mismatch")
        for name in ("input_scale", "agg_scale", "edge_scale", "output_scale"):
            if any(p.requires_grad or not torch.equal(p,torch.ones_like(p)) for p in getattr(self,name)):
                raise ValueError("weighted payload transport must use fixed unit scales")
        for v,(node,spec) in enumerate(zip(self.nodes,graph.nodes)):
            c,r = node.kernel.config,self.regions[spec.region].config
            agg = node.aggregate_program
            expected = (memory_profile(c),identity("weighted-describe-v1",c,r),
                        identity("weighted-full-v1",c),
                        identity("weighted-aggregate-v1",agg.classes,agg.alpha,agg.hidden_width,c.epsilon))
            if ((spec.memory,spec.readout,spec.emission,spec.aggregation) != expected
                    or node.read_program.memory_config != c or node.read_program.config != r
                    or node.full_program.content.config != c or agg.epsilon != c.epsilon):
                raise ValueError("weighted function configuration changed without a new graph binding")
        if any(spec.selector != region_profile(program.config)
               for spec,program in zip(graph.regions,self.regions)):
            raise ValueError("weighted region configuration changed without a new graph binding")

    def validate_checkpoint_weights(self, actual):
        for name, value in actual.items():
            if name.split(".")[0] in {"input_scale","agg_scale","edge_scale","output_scale"}:
                if not torch.equal(value,torch.ones_like(value)):
                    raise ValueError("checkpoint changes weighted payload unit transport")


@dataclass(frozen=True)
class Binding:
    graph: Graph
    model: WeightedModel
    memories: tuple[KVConfig | RecurrentConfig, ...]
    regions: tuple[RegionalConfig, ...]
    source_classes: SourceClasses


def bind(topology, *, memories=KVConfig(), regions=RegionalConfig(), source_classes=None,
         alpha_agg=0., source_hidden_width=4, dtype=torch.float64, device="cpu", seed=7):
    """Bind physical topology/ports to weighted memory; replace old local programs.

    KV bases and source A/B matrices share within each region. KV adapters,
    summaries, Describe/Full parameters and state are private per node. Source
    embeddings share within (region,class). Callers may explicitly tie same-shape
    Parameters afterwards; checkpoint preflight preserves that alias topology.
    """
    if dtype not in {torch.float32,torch.float64}:
        raise ValueError("weighted binding currently requires FP32/FP64")
    if topology.origins:
        raise ValueError("weighted physical topology cannot contain legacy input-origin adapters")
    finite("alpha_agg",alpha_agg,0); positive_int("source_hidden_width",source_hidden_width)
    memory = (memories,)*len(topology.nodes) if isinstance(memories,(KVConfig,RecurrentConfig)) else tuple(memories)
    region = (regions,)*len(topology.regions) if isinstance(regions,RegionalConfig) else tuple(regions)
    if (len(memory) != len(topology.nodes) or len(region) != len(topology.regions)
            or any(not isinstance(x,(KVConfig,RecurrentConfig)) for x in memory)
            or any(not isinstance(x,RegionalConfig) for x in region)
            or len({c.width for c in memory}) != 1):
        raise ValueError("weighted binding node/region/width domain mismatch")
    classes = source_classes or SourceClasses((0,)*len(topology.edges),(0,)*len(topology.inputs))
    classes.validate(topology)
    generator = torch.Generator().manual_seed(seed)
    width = memory[0].width
    region_ids = [region_profile(c) for c in region]
    rp = {j: RegionalTotal(region_ids[j],c,generator,dtype) for j,c in enumerate(region)}
    bases = [parameter((2*width,width),generator,dtype) for _ in region]
    source_a = [parameter((source_hidden_width,width),generator,dtype) for _ in region]
    source_b = [parameter((width,source_hidden_width),generator,dtype) for _ in region]
    embeddings, states, readers, aggregates, fulls, nodes = {}, {}, {}, {}, {}, []
    for v,(spec,c) in enumerate(zip(topology.nodes,memory)):
        rc, j = region[spec.region], spec.region
        sid = memory_profile(c)
        rid = identity("weighted-describe-v1",c,rc)
        aid = identity("weighted-aggregate-v1",classes,alpha_agg,source_hidden_width,c.epsilon)
        fid = identity("weighted-full-v1",c)
        state_type, content_type = MEMORY_TYPES[memory_family(c)]
        states[v] = (state_type(c,sid,bases[j],generator,dtype) if isinstance(c,KVConfig)
                     else state_type(c,sid,generator,dtype))
        readers[v] = Describe(rid,c,rc,generator,dtype)
        labels = {classes.inputs[p] for p,n in enumerate(topology.inputs) if n == v}
        labels |= {classes.edges[e] for e,edge in enumerate(topology.edges) if edge.target == v}
        transforms = {}
        for label in sorted(labels):
            key = (j,label)
            if key not in embeddings:
                embeddings[key] = parameter((source_hidden_width,),generator,dtype)
            transforms[label] = SourceTransform(source_a[j],source_b[j],embeddings[key])
        aggregates[v] = WeightedAggregate(aid,classes,transforms,alpha_agg,c.epsilon,source_hidden_width)
        fulls[v] = WeightedFull(fid,content_type(c,generator,dtype))
        nodes.append(Node(j,memory=sid,readout=rid,aggregation=aid,emission=fid))
    specs = tuple(Region(c.capacity(len(layout.members)),observe_all=True,count_priority=False,
                         read_mode={"0":"content","-":"old","+":"proposal"}[c.read_mode],
                         selector=region_ids[j]) for j,(layout,c) in enumerate(zip(topology.region_layouts,region)))
    # Physical source slots remain distinct even when parameter classes coincide.
    graph = Graph(tuple(nodes),topology.edges,specs,topology.inputs,topology.outputs,topology.ports)
    model = WeightedModel(graph,width=width+1,seed=seed,dtype=dtype,device=device,
                          state_programs=states,read_programs=readers,aggregate_programs=aggregates,
                          full_programs=fulls,region_programs=rp)
    # Legacy scalar-profile parameters have no role in this binding.
    for node in model.nodes:
        for name in ("decay","weight","bias","read"):
            old = node._parameters.pop(name)
            node.register_buffer(name,torch.zeros_like(old))
    for name in ("input_scale","agg_scale","edge_scale","output_scale"):
        for scale in getattr(model,name):
            scale.requires_grad_(False)
            with torch.no_grad():
                scale.fill_(1)
    model.validate_binding(graph)
    for j,p in enumerate(model.regions):
        p.validate_weights(graph.region_layouts[j])
    return Binding(graph,model,memory,region,classes)

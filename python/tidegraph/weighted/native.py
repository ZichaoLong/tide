"""Tensor-preserving adapter; execution and formulas are wholly C++ owned."""
from dataclasses import asdict
from ..native_model import encode_records
from .kv import KVMemory
from .programs import Describe, WeightedAggregate, WeightedFull, RegionalTotal


def region_config(core, config):
    result = core.WeightedRegionalConfig()
    for name in ("hidden_width","region_control_width","k","gamma_max",
                 "allocation_temperature","region_control_mode"):
        setattr(result,name,getattr(config,name))
    if config.total_fraction is not None:
        result.total_fraction = config.total_fraction
    result.validate()
    return result


def encode(core, graph, model):
    model.validate_binding(graph)
    for node,spec in zip(model.nodes,graph.nodes):
        for name,kind,profile in (("kernel",KVMemory,spec.memory),("read_program",Describe,spec.readout),
                                  ("aggregate_program",WeightedAggregate,spec.aggregation),
                                  ("full_program",WeightedFull,spec.emission)):
            program = getattr(node,name)
            if type(program) is not kind or program.profile != profile:
                raise ValueError("weighted native program binding mismatch")
    if any(type(r) is not RegionalTotal for r in model.regions):
        raise ValueError("weighted native region binding mismatch")
    g,m = encode_records(core,graph,model)
    nodes = list(m.nodes)
    for i,(weight,pyweight) in enumerate(zip(nodes,model.nodes)):
        weight.extra = dict(pyweight.named_parameters(remove_duplicate=False))
        config = core.WeightedKVConfig()
        for name,value in asdict(pyweight.kernel.config).items():
            setattr(config,name,value)
        config.validate()
        region = region_config(core,model.regions[graph.nodes[i].region].config)
        agg = pyweight.aggregate_program
        core.configure_weighted_node(weight,config,region,agg.classes.edges,agg.classes.inputs,agg.alpha)
    m.nodes = nodes
    regions = list(m.regions)
    for weight,program in zip(regions,model.regions):
        core.configure_weighted_region(weight,region_config(core,program.config))
    m.regions = regions
    return g,m

"""Named finite accelerator cases; each retains the full configured graph."""
from dataclasses import replace
from tidegraph import GraphConfig, Edge


def cases(implementation, dtype="float32"):
    for family in ("pdg", "timed-dag", "settle"):
        topology = dict(kind="ring", size=4) if family == "pdg" else dict(kind="diamond")
        topology.update(module=dict(full="swiglu"), node_overrides={
            "0":dict(memory="linear"), "1":dict(memory="ssm"),
            "2":dict(memory="delta"), "3":dict(memory="attention")})
        config = GraphConfig.from_dict(dict(schema_version=1, family=family, topology=topology,
            model=dict(width=4, dtype=dtype, scale_init=.25),
            execution=dict(implementation=implementation, mode="hst")))
        yield f"mixed-{family}", config
        yield f"batched-{family}", replace(config, execution=replace(config.execution,
            full_autograd="batched", aggregate_autograd="batched"))
    for family, topology, schedule in (("pdg","ring","ring"), ("pdg","self_loop","self_loop"),
                                       ("timed-dag","chain","chain"), ("timed-dag","diamond","diamond")):
        yield "specialized-" + schedule, GraphConfig.from_dict(dict(schema_version=1, family=family,
            topology=dict(kind=topology, size=1 if topology == "self_loop" else 4,
                          module=dict(memory="ssm",full="swiglu")),
            model=dict(width=4,dtype=dtype,scale_init=.25), execution=dict(implementation=implementation,
                schedule=schedule, packed=implementation == "native" or family != "pdg", mode="softp")))
    from tidegraph.fiber_pool import PROFILES as fiber_profiles
    from tidegraph.lh_full import PROFILES as full_profiles
    def configured(module, *, regions=None, overrides=None, execution=None):
        return GraphConfig.from_dict(dict(schema_version=1, family="timed-dag",
            topology=dict(kind="diamond", budget=1, module=module, region_overrides=regions,
                          node_overrides=overrides), model=dict(width=4,dtype=dtype,scale_init=.25),
            execution=dict(implementation=implementation, mode="hst", **(execution or {}))))
    yield "identity-nodes", configured(dict(identity=True))
    for memory in ("delta-rule-v1", "lh-add-repeat-v1", *fiber_profiles):
        yield "memory-"+memory, configured(dict(memory=memory,query_heads=2,kv_heads=2))
    for full in full_profiles:
        yield "full-"+full, configured(dict(full=full))
    for aggregate in ("sum", "mean", "weighted_mean", "active_softmax", "all_softmax"):
        yield "aggregate-"+aggregate, configured(dict(aggregation=aggregate))
    for selector in ("positive-v1", "tensor-history-v1", "lh-count-affect-v1"):
        yield "region-"+selector, configured({},regions={"1":dict(selector=selector)})
    yield "next-control-blend", configured(dict(next_state="control-blend-v1"))
    yield "sparse-emission-reset", configured(dict(emission="slot_affine",emit_period=2,clear=True),
        overrides={"0":dict(emit_phases=[0,1]),"2":dict(emit_phases=[-2])})
    clocked = configured(dict(state_clock=dict(period=2,first=0,count=1)))
    yield "periodic-state-clock", replace(clocked, graph=replace(clocked.graph,
        edges=tuple(replace(edge, delay=2) for edge in clocked.graph.edges)))
    parallel = configured(dict(memory="ssm"))
    yield "parallel-edges-delay", replace(parallel, graph=replace(parallel.graph,
        edges=parallel.graph.edges+(Edge(0,1,2),)))
    if implementation == "native":
        yield "worker-streams", configured(dict(memory="ssm",full="swiglu"), execution=dict(
            workers=2,parallel_regions=True,compact_events=True,defer_state_release=True,
            packed_sources=True,batch_next=True,full_autograd="batched",aggregate_autograd="batched"))
        for name, value in (("attention_packing","single"), ("fiber_pooling","csr"),
                            ("fiber_cache","owned"), ("attention_layout","head")):
            yield "fiber-policy-"+name, configured(dict(memory=next(iter(fiber_profiles)),query_heads=2,kv_heads=2),
                execution={name:value, "workers":2})


def fixture_options(name, config):
    """Clocked workloads must declare valid external phases and a whole-period cut."""
    if name != "periodic-state-clock":
        return {}
    import torch
    from tidegraph import External
    generator = torch.Generator().manual_seed(19)
    return dict(inputs=[External(0, 0, position, 2*position,
                                torch.randn(config.width, generator=generator, dtype=getattr(torch, config.dtype)))
                        for position in range(2)], stop=8)

"""Named finite accelerator cases; each retains the full configured graph."""
from dataclasses import replace
from tidegraph import GraphConfig


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

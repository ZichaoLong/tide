"""FC/HB instance §1/§4.4: explicit input-position clock and source binding."""
from dataclasses import replace
import torch
from ..coordinates import integers
from ..source_class import SourceClasses
from .binding import bind
from .config import KVConfig, RegionalConfig, finite, positive_int
from .recurrent_config import RecurrentConfig
from .tensors import Payload


def bind_layered(spec, *, memory=KVConfig(), regions=RegionalConfig(), window_positions=0,
                 short_rate_per_position=.1, **kwargs):
    """Use T=D*H, beta=beta_bar/D, Pos(theta)=(theta-rank)/D.

    `spec` is an FC/HB ports topology. Original-time callers may instead use
    `bind` and give each KVConfig directly. Zero window denotes infinity.
    """
    if spec.binding != "ports":
        raise ValueError("layered weighted instance requires physical ports")
    positive_int("window_positions",window_positions,0)
    finite("short_rate_per_position",short_rate_per_position,0,True)
    window = window_positions*spec.stride
    positive_int("logical window",window,0)
    supplied = ((memory,)*len(spec.graph.nodes) if isinstance(memory,(KVConfig,RecurrentConfig)) else tuple(memory))
    if len(supplied) != len(spec.graph.nodes):
        raise ValueError("layered memory count mismatch")
    if window_positions and not any(isinstance(c,KVConfig) for c in supplied):
        raise ValueError("window_positions applies only to KV nodes")
    def at_node(v, c):
        if isinstance(c,KVConfig):
            return replace(c,window=window,beta_short=short_rate_per_position/spec.stride,
                           position_origin=spec.rank(v),position_scale=spec.stride)
        if not isinstance(c,RecurrentConfig):
            raise ValueError("invalid layered memory config")
        # Instance §4.3: fixed position slots, passive rates per position.
        return replace(c,lambda_short=short_rate_per_position/spec.stride,
                       matrix_rates=tuple(rate/spec.stride for rate in c.matrix_rates),
                       conv_stride=1 if c.basic else spec.stride)
    memories = tuple(at_node(v,c) for v,c in enumerate(supplied))
    # 0 names the external class; node v has a separate class v+1.
    classes = SourceClasses(tuple(e.source+1 for e in spec.graph.edges),(0,)*len(spec.graph.inputs))
    bound = bind(spec.graph,memories=memories,regions=regions,source_classes=classes,**kwargs)
    return replace(spec,graph=bound.graph),bound


def layered_inputs(spec, values, *, start_position=0, start_occurrence=None):
    """Instance unit-strength broadcast of [batch, position, content-width]."""
    integers("layered input start",start_position)
    if values.ndim != 3 or start_position < 0:
        raise ValueError("layered input requires [batch,position,width]")
    occurrence = start_position if start_occurrence is None else start_occurrence
    payload = Payload(torch.ones(values.shape[:-1],dtype=values.dtype,device=values.device),values).encode()
    return [row for batch in range(len(values)) for port in range(len(spec.graph.inputs))
            for row in spec.port_external(payload[batch],range(start_position,start_position+values.shape[1]),
                                          port=port,batch=batch,start_occurrence=occurrence)]


def readout(outputs, coordinates, reference, *, batch=0):
    """Architecture §4.4: Σm+z/max(1,Σm+), using actual output records only.

    Empty output and cancellation can both give zero; output existence remains
    in `outputs` and is never inferred from this numerical projection.
    """
    coordinates = tuple(coordinates)
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("ReadOut coordinates must be distinct")
    integers("ReadOut coordinates",batch,*(v for row in coordinates for v in row))
    if batch < 0 or any(len(row)!=2 or min(row)<0 for row in coordinates):
        raise ValueError("invalid ReadOut coordinates")
    wanted,seen = set(coordinates),set()
    masses,contents = [],[]
    for b,time,port,value in outputs:
        if b != batch or (time,port) not in wanted:
            continue
        if (time,port) in seen:
            raise ValueError("duplicate output record")
        seen.add((time,port))
        p = Payload.decode(value)
        mass,numeric = p.numerical();masses.append(mass);contents.append(numeric)
    if not masses:
        return torch.zeros_like(reference)
    return torch.stack(contents).sum(0)/torch.stack(masses).sum().clamp(min=1)

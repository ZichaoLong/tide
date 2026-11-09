"""Explicit numerical execution views; see docs/weighted-dense-query.md.

The model/parameter owners are unchanged. dense-v1 batches dense projections
across cache lengths, retaining structural attention branches in the VJP.
"""
import torch
from ..placement import _shallow_module
from .config import KVConfig
from .programs import Describe, WeightedFull


def validate(memory, region):
    if (not isinstance(memory, KVConfig) or memory.memory_read != "qry"
            or (region.local_score, region.region_score, region.selection_mode,
                region.region_control_mode, region.region_evidence, region.regularizer,
                region.support_policy) != ("cat", "ind", "regional_total", "context", "cat", "quad", "strict")):
        raise ValueError("dense-v1 requires KV/qry cat/ind regional_total context/cat quad strict")


def execution_view(model, mode):
    if mode not in {"legacy", "dense-v1"}:
        raise ValueError("unknown query_execution")
    if mode == "legacy":
        return model
    devices = {node.bias.device for node in model.nodes}
    if len(devices) != 1:
        raise ValueError("dense-v1 requires a single payload device")
    result, nodes = _shallow_module(model), []
    for weights in model.nodes:
        if type(weights.read_program) is not Describe or type(weights.full_program) is not WeightedFull:
            raise ValueError("dense-v1 requires installed weighted Read/Full before placement")
        validate(weights.kernel.config, weights.read_program.config)
        if weights.bias.dtype not in {torch.float32, torch.float64}:
            raise ValueError("dense-v1 requires float32 or float64 payload")
        node = _shallow_module(weights)
        node.read_program = _shallow_module(weights.read_program)
        node.full_program = _shallow_module(weights.full_program)
        node.read_program.query_execution = node.full_program.query_execution = mode
        nodes.append(node)
    result.nodes = torch.nn.ModuleList(nodes)
    return result


def dense(program):
    return getattr(program, "query_execution", "legacy") == "dense-v1"


def branches(memory, weights, requests, kind):
    """One detached metadata transfer per device, no per-row boolean indexing."""
    from ..region_metadata import positive
    from .numeric_time import passive_decay
    flags = []
    with torch.no_grad():
        for r in requests:
            state = r.state if kind == "read" else r.comparison
            state = memory.initial(weights) if state is None else state
            mask, age = memory.visible(state, r.time)
            writes = state.slots["weights"]
            flags.append(((writes > 0) & mask).any())
            if kind == "read":
                short = writes*passive_decay(age.clamp_min(0), memory.config.beta_short, writes)
                flags.append(((short > 0) & mask).any())
    values = positive(flags)
    n = 2 if kind == "read" else 1
    return [tuple(values[i:i+n]) for i in range(0, len(values), n)]

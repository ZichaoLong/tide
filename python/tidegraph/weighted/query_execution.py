"""Explicit numerical execution views; see docs/weighted-dense-query.md.

The model/parameter owners are unchanged. dense-v1 batches dense projections
across cache lengths, retaining structural attention branches in the VJP.
"""
import torch
from ..placement import _shallow_module
from .config import KVConfig
from .programs import Describe, WeightedFull
from .dense_describe import supported as describe_supported


def validate(memory, region, mode="dense-v1"):
    if mode == "packed-v2":
        if not isinstance(memory, KVConfig) or region.support_policy != "strict":
            raise ValueError("packed-v2 requires KV with strict positive support")
        return
    if (not isinstance(memory, KVConfig) or memory.memory_read != "qry"
            or (region.local_score, region.region_score, region.selection_mode,
                region.region_control_mode, region.region_evidence, region.regularizer,
                region.support_policy) != ("cat", "ind", "regional_total", "context", "cat", "quad", "strict")):
        raise ValueError("dense-v1 requires KV/qry cat/ind regional_total context/cat quad strict")


def execution_view(model, mode):
    if mode not in {"auto", "legacy", "dense-v1", "packed-v2"}:
        raise ValueError("unknown query_execution")
    if mode == "legacy" and not any(hasattr(n.read_program, "query_execution") or
            hasattr(n.read_program, "describe_execution") for n in model.nodes):
        return model
    devices = {node.bias.device for node in model.nodes}
    if mode not in {"auto", "legacy"} and len(devices) != 1:
        raise ValueError("dense-v1 requires a single payload device")
    result, nodes = _shallow_module(model), []
    for weights in model.nodes:
        selected = mode
        heads = "scalar"
        if mode == "auto":
            selected = "legacy"
            if (type(weights.read_program) is Describe and type(weights.full_program) is WeightedFull
                    and weights.bias.dtype in {torch.float32, torch.float64} and len(devices) == 1):
                if describe_supported(weights.read_program.config):
                    heads = "batched"
                if (isinstance(weights.kernel.config, KVConfig)
                        and getattr(weights.kernel, "support_policy", "strict") == "strict"
                        and weights.read_program.config.support_policy == "strict"):
                    selected = "packed-v2"
            if selected == "legacy" and heads == "scalar" and not (
                    hasattr(weights.read_program, "query_execution") or hasattr(weights.read_program, "describe_execution")):
                nodes.append(weights)
                continue
        if mode == "legacy" and type(weights.read_program) is not Describe:
            nodes.append(weights)
            continue
        if type(weights.read_program) is not Describe or type(weights.full_program) is not WeightedFull:
            raise ValueError("dense-v1 requires installed weighted Read/Full before placement")
        if selected != "legacy":
            if getattr(weights.kernel, "support_policy", "strict") != "strict":
                raise ValueError(f"{selected} requires strict positive support in the actual memory kernel")
            validate(weights.kernel.config, weights.read_program.config, selected)
            heads = "batched" if describe_supported(weights.read_program.config) else "scalar"
        if selected != "legacy" and weights.bias.dtype not in {torch.float32, torch.float64}:
            raise ValueError("dense-v1 requires float32 or float64 payload")
        node = _shallow_module(weights)
        node.read_program = _shallow_module(weights.read_program)
        node.full_program = _shallow_module(weights.full_program)
        node.read_program.query_execution = node.full_program.query_execution = selected
        node.read_program.describe_execution = heads
        nodes.append(node)
    result.nodes = torch.nn.ModuleList(nodes)
    return result


def dense(program):
    return getattr(program, "query_execution", "legacy") in {"dense-v1", "packed-v2"}


def packed(program):
    return getattr(program, "query_execution", "legacy") == "packed-v2"


def describe_mode(program):
    """Resolved descriptor implementation, separate from the memory query mode."""
    selected = getattr(program, "describe_execution", "batched" if dense(program) else "scalar")
    return selected if describe_supported(getattr(program, "config", None)) else "scalar"


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

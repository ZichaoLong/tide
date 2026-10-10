"""Batch one causal tick across independent samples; never crosses Next."""
from collections import defaultdict
import torch
from .state_vjp import supported as batched_state
from .read_vjp import supported as batched_read
from .packing import prepare_sequences


def prepare_tick(graph, model, q, events, stats):
    groups = defaultdict(list)
    for event in events:
        if "proposal" not in event:
            groups[event["node"]].append(event)
    for node, es in groups.items():
        weights = model.nodes[node]
        kernel = getattr(weights, "kernel", None)
        if not getattr(kernel, "sequence_contract", False) or not hasattr(kernel, "packed_sequence"):
            continue
        sequences = [((e["batch"], node), [e]) for e in es]
        region = graph.regions[graph.nodes[node].region]
        calls = prepare_sequences(weights, sequences, q, region.read_mode)
        stats["state_step_batch_calls"] = stats.get("state_step_batch_calls", 0) + calls
        stats["max_state_batch"] = max(stats.get("max_state_batch", 0), len(es))
        stats["state_steps"] += len(es)
        stats["read_calls"] = stats.get("read_calls", 0) + 1
        if torch.is_grad_enabled():
            key = "batched_state_events" if batched_state(kernel) else "semantic_state_replays"
            stats[key] = stats.get(key, 0) + len(es)
            key = "batched_read_events" if batched_read(weights.read_program) else "semantic_read_replays"
            stats[key] = stats.get(key, 0) + len(es)
        if not getattr(kernel, "joint_sequence", False):
            stats["state_scalar_batch_steps"] = stats.get("state_scalar_batch_steps", 0) + len(es)


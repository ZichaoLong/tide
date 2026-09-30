"""General positive-delay online region blocks with an independent scalar oracle.

This implementation owns host scheduling. Accelerator tensor execution does not
make it the separate, not-yet-qualified device-resident scheduler.
"""
from collections import defaultdict
import math
from .block_policy import validate
from .blocks import canonicalize, deliver, evaluate_block
from .greedy_plan import ClosurePlan
from .records import Result
from .validation import validate_window


def run(graph, model, initial, external, stop, *, sealed_until, mode="hard", zeta=1.0,
        trace=True, prefill=True, max_events=1000000, packed=True,
        full_autograd="replay", aggregate_autograd="replay"):
    policy = dict(packed=packed, full_autograd=full_autograd, aggregate_autograd=aggregate_autograd)
    validate(model, **policy)
    if mode not in {"hard", "hst", "softp"} or not math.isfinite(zeta):
        raise ValueError("invalid emit mode/zeta")
    if type(max_events) is not int or max_events < 1:
        raise ValueError("greedy requires positive live-fiber capacity")
    atoms, ledger = validate_window(graph, model, initial, external, stop, sealed_until)
    q = initial.fork()
    q.ledger = ledger
    fibers = defaultdict(list)
    for atom in list(q.pending) + atoms:
        fibers[atom.batch, atom.node, atom.time].append(atom)
    plan = ClosurePlan(graph)
    events, messages, outputs = [], [], []
    stats = dict(greedy_stages=0, greedy_relaxed_edges=0, region_blocks=0,
                 candidate_events=0, visited_edges=0, max_live_fibers=0)
    while True:
        stats["max_live_fibers"] = max(stats["max_live_fibers"], len(fibers))
        blocks, relaxed = plan.ready(fibers, stop, max_events)
        stats["greedy_relaxed_edges"] += relaxed
        if not blocks:
            break
        stats["greedy_stages"] += 1
        # All ready prefixes use the stage-start closure certificate. Kernels
        # preserve each region's causal state/history and batch the legal work.
        for region, frames in sorted(blocks.items()):
            block, counters = evaluate_block(graph, model, q, frames, fibers, mode=mode,
                                              zeta=zeta, prefill=prefill, **policy)
            for batch, _, time, nodes in frames:
                for node in nodes:
                    del fibers[batch, node, time]
            sent = []
            deliver(graph, model, block, fibers, sent, outputs)
            stats["visited_edges"] += len(sent)
            stats["region_blocks"] += 1
            stats["max_greedy_frames"] = max(stats.get("max_greedy_frames", 0), len(frames))
            if trace:
                events.extend(block)
                messages.extend(sent)
            for name, count in counters.items():
                stats[name] = max(stats.get(name, 0), count) if name.startswith("max_") else stats.get(name, 0) + count
            if len(fibers) > max_events:
                raise ValueError("greedy live-fiber capacity exceeded")
    q.pending = [atom for atoms in fibers.values() for atom in atoms if atom.kind == 1]
    q.cut = stop
    return canonicalize(graph, Result(q, events, outputs, messages, stats))

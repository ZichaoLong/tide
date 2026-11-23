"""The first finite pilot: main semantic families and explicit cost probes."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

from .io import atomic_json


def build_plan():
    from tide import ModelConfig
    tasks = []

    def tide(name, topology, **overrides):
        config = dict(topology=topology, d=64, memory="K", sharing="region_shared",
                      allocation="quad", budget=1, aggregation="source_residual",
                      attention_backend="auto", region_batch_size=1)
        config.update(overrides)
        config = ModelConfig.from_dict(config).to_dict()
        # Check topology-dependent legality before occupying a device.
        from tide import build_topology
        graph = build_topology(ModelConfig.from_dict(config).topology)
        return {"name": name, "model": {"family": "tide", "config": config},
                "graph_nodes": len(graph.nodes), "graph_regions": len(graph.regions)}

    def add(binding, backend="npu", batch=8, length=64, threads=8, suffix=""):
        task = dict(id=backend+"-"+binding["name"]+suffix, backend=backend, seed=20261119,
                    model=binding["model"], batch_size=batch, sequence_length=length,
                    cpu_threads=threads, train_seconds=420 if backend == "npu" else 240,
                    wall_seconds=900, max_steps=20001, stream_chunks=4, lr=3e-4,
                    warmup_steps=10, weight_decay=0.01, grad_clip=1.0, precision="fp32",
                    eval_batches=2, decode_steps=8, measurement_warmup_steps=3,
                    resume_check=True, estimated_output_bytes=2*1024**3)
        task.update({k:v for k,v in binding.items() if k.startswith("graph_")})
        tasks.append(task)
        return task

    fc = dict(kind="fc", n=8, layers=4)
    lattice = dict(kind="lattice", n=8, depth=2)
    exchange = dict(kind="exchange", n=8, depth=1, rounds=2)
    hub = dict(exchange, hubs=4)
    multi = dict(kind="multiscale", n=8, profile="compact-7", timing="settle")
    timed = dict(multi, timing="timed", hubs=4)
    bindings = [
        tide("fc-k-independent", fc, sharing="independent"),
        tide("fc-k-d128", fc, sharing="independent", d=128),
        tide("fc-s-shared", fc, memory="S"),
        tide("fc-k-incoming64", fc, aggregation="incoming_light_64"),
        tide("lattice-k", lattice), tide("lattice-s", lattice, memory="S"),
        tide("exchange-k", exchange), tide("exchange-k-independent", exchange, sharing="independent"),
        tide("exchange-x", hub, memory="X"), tide("exchange-s", hub, memory="S"),
        tide("exchange-x-layer-shared", hub, memory="X", sharing="layer_shared"),
        tide("exchange-x-independent", hub, memory="X", sharing="independent"),
        tide("exchange-x-incoming64", hub, memory="X", aggregation="incoming_light_64"),
        tide("exchange-x-incoming128", hub, memory="X", aggregation="incoming_light_128"),
        tide("exchange-x-incoming-wide", hub, memory="X", aggregation="incoming_wide"),
        tide("exchange-x-shannon", hub, memory="X", allocation="shannon"),
        tide("exchange-x-pool", hub, memory="X", allocation="pool"),
        tide("exchange-x-budget4", hub, memory="X", budget=4),
        tide("multiscale-k", multi),
        tide("multiscale-x-compact9", dict(multi, profile="compact-9", hubs=4), memory="X"),
        tide("timed-x-group", timed, memory="X"),
        tide("timed-s", dict(timed, hubs=0), memory="S"),
        tide("timed-x-singleton", dict(timed, partition="singleton"), memory="X"),
        tide("timed-x-mixed", dict(timed, partition="mixed"), memory="X"),
        tide("exchange-x-reverse", hub, memory="X_reverse"),
        tide("exchange-x-kv32", hub, memory="X", kv_width=32),
        tide("exchange-x-memory-shared", hub, memory="X", sharing="independent", memory_sharing="region_shared"),
        tide("exchange-x-d128", hub, memory="X", d=128),
    ]
    for binding in bindings:
        add(binding)
    baselines = [dict(name=f"transformer-d{d}-l{layers}", model=dict(family="transformer", d=d,
                     layers=layers, heads=d//32, ffn_width=ffn, window=128))
                 for d,layers,ffn in [(128,4,384),(256,6,768),(384,8,1024)]]
    for baseline in baselines:
        add(baseline)
    add(bindings[8], batch=16, length=128, suffix="-b16-l128")
    add(bindings[0], backend="cpu", batch=2, length=32, threads=8, suffix="-b2-l32-t8")
    add(bindings[0], backend="cpu", threads=40, suffix="-b8-l64-t40")
    add(bindings[8], backend="cpu", batch=2, length=32, threads=40, suffix="-b2-l32-t40")
    add(bindings[8], backend="cpu", threads=40, suffix="-b8-l64-t40")
    add(baselines[0], backend="cpu", threads=16, suffix="-t16")
    add(baselines[1], backend="cpu", threads=40, suffix="-t40")
    add(bindings[20], backend="cpu", batch=2, length=32, threads=40, suffix="-b2-l32")
    add(bindings[2], backend="cpu", threads=16, suffix="-t16")
    assert len(tasks) == 40 and len({t["id"] for t in tasks}) == 40
    return {"schema": "pilot-round-plan-v1", "round_id": "pilot-r1-20261119", "tasks": tasks,
            "duration_seconds": 3600, "dispatch_reserve_seconds": 300, "finish_reserve_seconds": 60,
            "term_grace_seconds": 20, "quota_reserve_bytes": 16*1024**3,
            "max_output_bytes": 640*1024**3,
            "purpose": "mechanism, cost, finite learning and continuation calibration; not a tuned quality ranking",
            "precision_policy": "FP32 parameters/arithmetic/optimizer; no AMP in first pilot",
            "data_policy": "WikiText103 upstream train, article hash holdout, train-only byte BPE8192",
            "timed_readout": "last tick before the next external input; no flush inserted between windows",
            "comparison_policy": "fixed seed/data sampler within matching workload; baselines are capacity references, not claimed exact parameter matches",
            "learning_rate_policy": "linear 10-step warmup then constant 3e-4; resumable without changing horizon",
            "retention": "latest application checkpoint per task; failed recovery reference and actual update; successful comparison payloads removed after metrics saved",
            "acceptance": {"execution": "finite training and measurable inference when budget permits",
                           "recovery": "exact saved data/data stream/RNG; original atol1e-5 rtol1e-4 for floating outputs, gradients, state and updates; integer metadata exact",
                           "limits": "finite numerical disagreement retained separately; causal/nonfinite/execution errors fail the affected task"}}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    plan=build_plan()
    atomic_json(args.output, plan)
    print(f"{len(plan['tasks'])} tasks: 32 NPU and 8 CPU -> {args.output}")


if __name__ == "__main__":
    main()

"""Independent Python event-attention fixture and device loss cotangents."""
from dataclasses import replace
import os
import torch
from tidegraph import GraphRuntime, ExecutionOptions, ExecutionPlacement, ResidentLimits
from resident_training_cases import configuration
from resident_cache_roots import roots, terms


def runtime(family, device, schedule="greedy", mode="hard", clear=False):
    cfg = configuration(family, aggregation="all_softmax", observe_all=False)
    nodes = tuple(replace(n, memory="attention", query_heads=2, kv_heads=1,
                          window=1 if i == 0 else 3, clear=clear and i % 2 == 0)
                  for i, n in enumerate(cfg.graph.nodes))
    cfg = replace(cfg, graph=replace(cfg.graph, nodes=nodes))
    if device == "cpu":
        r = GraphRuntime(cfg, device=device,
            options=ExecutionOptions(schedule="reference", packed=False, trace=True, mode=mode))
    else:
        r = GraphRuntime(cfg, device=device, native_library=os.environ["TIDE_BUILD_DIR"],
            resident_library=os.environ["TIDE_RESIDENT_LIBRARY"],
            options=ExecutionOptions(implementation="native", schedule=schedule, trace=True, mode=mode,
                placement=ExecutionPlacement(preset="resident"),
                resident_limits=ResidentLimits(queue=96, arrivals=128, outputs=128, trace=512,
                    workspace_bytes=256*1024*1024, kv_rows=8, kv_trace_rows=1024, attention_key_rows=2)))
    r.model.nodes[2].extra["attn_q"] = r.model.nodes[0].extra["attn_q"]
    r.model.nodes[2].extra["attn_v"] = r.model.nodes[0].extra["attn_k"]
    r.model.input_scale[1] = r.model.agg_scale[0]
    return r

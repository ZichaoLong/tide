"""Five same-fiber policies with independent Python scheduling/autograd."""
from dataclasses import replace
import os
import torch
from tidegraph import GraphRuntime, ExecutionOptions, ExecutionPlacement, ResidentLimits
from resident_training_cases import configuration
from resident_cache_roots import roots, terms


def runtime(family, device, schedule="greedy", mode="hard", clear=False, pooling="sum"):
    cfg = configuration(family, observe_all=False)
    nodes = tuple(replace(n, memory=f"lh-fiber-attention-{pooling}-repeat-v1",
                          query_heads=1 if i == 3 else 2, kv_heads=1 if i == 3 else 2,
                          clear=clear and i % 2 == 0)
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
                    workspace_bytes=256*1024*1024, kv_rows=48, kv_trace_rows=4096,
                    attention_chunk_rows=3, attention_key_rows=2)))
    with torch.no_grad():
        for n in r.model.nodes:
            eye = torch.eye(4, device=n.extra["fiber_qkv"].device)
            n.extra["fiber_qkv"].copy_(torch.cat((eye*.25, eye*.5, eye*.75), 1))
            n.extra["fiber_qkv_bias"].fill_(.015625)
            n.extra["fiber_out"].copy_(eye*.5)
            n.extra["fiber_out_bias"].fill_(.03125)
            n.extra["fiber_decay"].fill_(.03125)
            if "fiber_pool" in n.extra:
                p = n.extra["fiber_pool"]
                p.copy_(torch.arange(p.numel(), device=p.device) * .125 - .125)
    r.model.nodes[2].extra["fiber_qkv"] = r.model.nodes[0].extra["fiber_qkv"]
    r.model.nodes[2].extra["fiber_out"] = r.model.nodes[0].weight
    r.model.nodes[2].extra["fiber_decay"] = r.model.nodes[0].extra["fiber_decay"]
    r.model.input_scale[1] = r.model.agg_scale[0]
    return r

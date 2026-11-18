"""Rebuild the finite legal scale ladder and budget before real allocation."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import torch

from tide import ModelConfig, TideModel
from benchmarks.matching import parameter_memory_budget, match_transformers
from scripts.job import atomic


def scale_configs():
    common = dict(memory="K", aggregation="source_residual", allocation="quad", sharing="independent")
    result = {}
    for label, layers in (("fc-0p5b", 12), ("fc-1b", 24)):
        result[label] = ModelConfig.from_dict(dict(common, d=384,
            topology=dict(kind="fc", n=16, layers=layers)))
    for label, n, d, profile in (("hb-0p5b", 32, 128, "9"), ("hb-1b", 32, 256, "5-dd4"),
                                ("hb-9b", 32, 768, "5-dd4"), ("hb-17b", 64, 768, "5-dd4")):
        result[label] = ModelConfig.from_dict(dict(common, memory="X", d=d,
            topology=dict(kind="multiscale", profile=profile, n=n, hubs=4, timing="settle")))
    return result


def budget(config, batch, length, precision, mode):
    # Allocate only metadata and actual Parameter shapes, never model payloads.
    with torch.device("meta"):
        model = TideModel(config)
    count = sum(p.numel() for p in model.parameters())
    component = parameter_memory_budget(count, mode=mode, precision=precision)
    graph = model.graph
    k_nodes = model.kinds.count("K")
    s_nodes = model.kinds.count("S")
    slots = min(length, 128 if model.settle else 128*graph.period)
    kv_bytes = k_nodes*batch*slots*(8+4+2*config.d*(2 if precision=="bf16" else 4)+config.d)
    ssd_bytes = s_nodes*batch*(8+4*(config.d*32+2*(config.d//4)+2+4*(config.d+64))+4*8)
    # This is a disclosed planning allowance, not a measured allocation proof.
    ticks = length if model.settle else length*graph.period
    node_values = len(graph.nodes)*batch*ticks*config.d*4
    max_edges = max(len(graph.incoming[node.id]) for node in graph.nodes)
    max_region = max(map(len, graph.regions))
    incoming = max_region*batch*ticks*max(1,max_edges)*(config.d+1)*4
    largest_parameter = max(p.numel()*p.element_size() for p in model.parameters())
    allowance = (node_values*(36 if mode != "inference" else 8) + incoming*3 + largest_parameter*3)
    persistent = component["persistent_parameter_floor_bytes"]
    return dict(config=config.to_dict(), unique_parameters=count, nodes=len(graph.nodes),
        depth=len(graph.widths), edges=len(graph.edges), batch=batch, length=length,
        memory=component, bounded_state_bytes=kv_bytes+ssd_bytes,
        explicit_activation_workspace_allowance_bytes=allowance,
        allowance_method="node outputs x36 autograd/x8 inference, peak incoming x3, largest Parameter x3; estimate, not proof",
        planned_bytes=persistent+component["foreach_scratch_allowance_bytes"]+kv_bytes+ssd_bytes+allowance,
        single_64gib_parameter_floor_fits=persistent < 60*2**30,
        total_matches={key:value for key,value in match_transformers(len(graph.nodes),len(graph.widths),count,count).items()
                       if key in {"A","C"}},
        active_matches="require actual fixed-workload audit before B/D can be generated")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--write-configs")
    parser.add_argument("--batch",type=int,default=1)
    parser.add_argument("--length",type=int,default=8)
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists():
        parser.error("output already exists")
    source=os.environ.get("TIDE_SOURCE_COMMIT") or subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    record=dict(source=source, scope="planning only; unrun modes are not passes", observations=[])
    for name,config in scale_configs().items():
        if args.write_configs:
            path=Path(args.write_configs)/(name+".json")
            if path.exists():
                raise FileExistsError(path)
            atomic(path,config.to_dict())
        for precision in ("fp32","bf16"):
            for mode in ("inference","grad_forward","train"):
                row=budget(config,args.batch,args.length,precision,mode)
                record["observations"].append(dict(name=name,precision=precision,mode=mode,**row))
    atomic(output,record)
    print(json.dumps(dict(output=str(output),rows=len(record["observations"]))))


if __name__ == "__main__":
    main()

"""Rebuild the finite legal scale ladder and budget before real allocation."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import torch

from tide import ModelConfig, TideModel
from tide.parameters import NodeLinear, NodeVector
from tide.reception import SourceEmbedding
from tide.placement import _default_partitions
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


def _private_state_budget(model, batch, length, precision, *, steady=False):
    """Payload bounds per node, plus retained Timed pending-view storage.

    This accounts for the implementation's convolution cache capacity in logical
    time slots, including its padded entries, rather than assuming four events.
    It excludes autograd graphs and transient prepared/history banks.
    """
    if (type(batch) is not int or type(length) is not int or min(batch,length) < 1 or
            precision not in {"fp32","bf16"}):
        raise ValueError("positive batch/length and FP32/BF16 policy required")
    d, period = model.config.d, model.graph.period
    ticks = length if model.settle else length*period
    kv_slots = (128 if model.settle else 128*period)
    if not steady:
        kv_slots = min(kv_slots, ticks)
    conv_slots = 3*(period if model.settle else 1)+1
    if not steady:
        conv_slots = min(conv_slots, ticks)
    kv = batch*kv_slots*(8+4+2*d*(2 if precision == "bf16" else 4)+(d//4)*4)
    ssd = batch*(8+4*(d*32+2*(d//4)+2)+conv_slots*(8+4*(d+64)))
    pending_payload = 0 if model.settle else batch*(d+1)*4
    pending_retained = 0 if model.settle else batch*ticks*(d+1)*4
    return dict(K=kv, S=ssd, kv_slots=kv_slots, convolution_slots=conv_slots,
                pending_payload_per_node=pending_payload,
                pending_retained_storage_upper_per_node=pending_retained)


def _partition_budget(model, batch, length, precision, mode, partitions, layer_partitions, capacity_bytes):
    if model.graph.config.kind not in {"lattice", "exchange", "multiscale"}:
        raise ValueError("partition capacity planning supports HB layer placement")
    if type(partitions) is not int or partitions < 1 or capacity_bytes <= 0:
        raise ValueError("positive partition count and device capacity required")
    layout = _default_partitions(model, partitions) if layer_partitions is None else tuple(layer_partitions)
    if (len(layout) != len(model.layer_nodes) or any(type(index) is not int for index in layout)
            or tuple(sorted(layout)) != layout or
            set(layout) != set(range(partitions))):
        raise ValueError("invalid contiguous layer partition")
    node_partition = tuple(layout[node.layer] for node in model.graph.nodes)
    domains = {"receive": tuple(range(len(model.graph.nodes))),
               "describe": tuple(node for node, _ in sorted(model.description_ids.items(), key=lambda row: row[1]))}
    domains.update({"memories."+kind: tuple(node for node, _ in sorted(owners.items(), key=lambda row: row[1]))
                    for kind, owners in model.memory_ids.items()})
    counts, largest, components = [0]*partitions, [0]*partitions, [{} for _ in range(partitions)]
    handled = set()
    def add(name, parameter, owners_by_row):
        if id(parameter) in handled:
            raise ValueError("unexpected parameter alias across independently partitioned modules")
        handled.add(id(parameter))
        rows = [0]*partitions
        for partition in owners_by_row:
            rows[partition] += 1
        row_size = parameter.numel()//max(1, parameter.shape[0])
        if sum(rows) != parameter.shape[0]:
            raise AssertionError("static row ownership does not cover the actual Parameter")
        for partition, number in enumerate(rows):
            size = number*row_size
            counts[partition] += size
            largest[partition] = max(largest[partition], size*parameter.element_size())
            if size:
                components[partition][name] = size
    source_classes = [set(model.graph.edges[edge].source_class for edge in model.graph.incoming[node.id]) |
                      ({"ext"} if node.layer == 0 else set()) for node in model.graph.nodes]
    for path, module in model.named_modules():
        prefix = next((name for name in domains if path.startswith(name+".")), None)
        if isinstance(module, (NodeLinear, NodeVector)):
            nodes = domains[prefix]
            private_partitions = [node_partition[node] for node in nodes]
            groups = module._host_groups  # Construction-owned CPU metadata, even with meta weights.
            group_partitions = {}
            for group, partition in zip(groups, private_partitions):
                if group_partitions.setdefault(group, partition) != partition:
                    raise ValueError("shared base crosses proposed partition")
            ordered_groups = [group_partitions[group] for group in range(len(group_partitions))]
            for leaf, parameter in module.named_parameters(recurse=False):
                add(path+"."+leaf, parameter,
                    private_partitions if leaf in {"left", "right"} else ordered_groups)
        elif isinstance(module, SourceEmbedding):
            # Reconstruct only source-class slot metadata. Its lookup tensor can
            # be meta, and no floating payload or .to(cpu) is needed to inspect it.
            rows = {}
            receive_groups = model.receive.a._host_groups
            for node, names in enumerate(source_classes):
                for name in sorted(names):
                    key = receive_groups[node], name
                    partition = node_partition[node]
                    if rows.setdefault(key, partition) != partition:
                        raise ValueError("source embedding row crosses proposed partition")
            add(path+".weight", module.weight, list(rows.values()))
    regional = {}
    for region, group in enumerate(model.region_group):
        if group >= 0:
            partition = node_partition[model.graph.regions[region][0]]
            if regional.setdefault(group, partition) != partition:
                raise ValueError("regional network crosses proposed partition")
    for name, parameter in model.named_parameters():
        if id(parameter) in handled:
            continue
        if name.startswith("inputs."):
            partition = layout[0]
        elif name.startswith("regional."):
            partition = regional[int(name.split(".")[1])]
        else:
            raise ValueError("unclassified placed parameter: "+name)
        handled.add(id(parameter))
        counts[partition] += parameter.numel()
        largest[partition] = max(largest[partition], parameter.numel()*parameter.element_size())
        components[partition][name] = parameter.numel()
    total = sum(parameter.numel() for parameter in model.parameters())
    if sum(counts) != total:
        raise AssertionError("partition count differs from unique meta Parameters")
    cold, steady = (_private_state_budget(model, batch, length, precision, steady=kind) for kind in (False, True))
    rows = []
    for partition, parameters in enumerate(counts):
        nodes = [node.id for node in model.graph.nodes if node_partition[node.id] == partition]
        kinds = {kind: sum(model.kinds[node] == kind for node in nodes) for kind in ("K", "S")}
        memory = parameter_memory_budget(parameters, mode=mode, precision=precision)
        state_cold = sum(kinds[kind]*cold[kind] for kind in kinds)+len(nodes)*cold["pending_payload_per_node"]
        state_steady = sum(kinds[kind]*steady[kind] for kind in kinds)+len(nodes)*steady["pending_payload_per_node"]
        retained_pending = len(nodes)*steady["pending_retained_storage_upper_per_node"]
        payload_pending = len(nodes)*steady["pending_payload_per_node"]
        steady_retained = state_steady+retained_pending-payload_pending
        floor = memory["persistent_parameter_floor_bytes"]
        rows.append(dict(partition=partition, layers=[i for i, owner in enumerate(layout) if owner == partition],
                         nodes=len(nodes), memory_nodes=kinds, unique_parameters=parameters,
                         parameter_bytes=4*parameters, largest_parameter_shard_bytes=largest[partition],
                         memory=memory, cold_state_payload_bytes=state_cold,
                         steady_state_payload_bytes=state_steady,
                         steady_state_retained_upper_bytes=steady_retained,
                         device_capacity_bytes=capacity_bytes,
                         headroom_after_parameter_floor_and_cold_state_bytes=capacity_bytes-floor-state_cold,
                         headroom_after_parameter_floor_and_steady_state_bytes=capacity_bytes-floor-steady_retained,
                         parameter_components=components[partition]))
    return dict(scope="static exact unique parameter partition; state planning bounds; no tensor payload allocation or execution qualification",
                layer_partitions=list(layout), unique_parameters=total, precision=precision, mode=mode,
                batch=batch, input_length=length, partitions=rows,
                state_layout=dict(cold=cold, steady=steady),
                unbudgeted=["autograd/activations", "temporary prepared/history banks", "parameter gather/cast workspaces",
                            "allocator overhead", "cross-partition copies", "checkpoint host staging"],
                initialization="construct full CPU master model then slice/copy shards; host capacity is separately required")


def placement_budget(config, batch=1, length=8, precision="bf16", mode="inference", *, partitions=2,
                     layer_partitions=None, capacity_bytes=60*2**30):
    with torch.device("meta"):
        model = TideModel(config)
    return _partition_budget(model, batch, length, precision, mode, partitions, layer_partitions, capacity_bytes)


def budget(config, batch, length, precision, mode, *, placement_partitions=0):
    # Allocate only metadata and actual Parameter shapes, never model payloads.
    with torch.device("meta"):
        model = TideModel(config)
    count = sum(p.numel() for p in model.parameters())
    component = parameter_memory_budget(count, mode=mode, precision=precision)
    graph = model.graph
    k_nodes = model.kinds.count("K")
    s_nodes = model.kinds.count("S")
    state = _private_state_budget(model,batch,length,precision)
    kv_bytes = k_nodes*state["K"]
    ssd_bytes = s_nodes*state["S"]
    pending_bytes = len(graph.nodes)*state["pending_retained_storage_upper_per_node"]
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
        memory=component, bounded_state_bytes=kv_bytes+ssd_bytes+pending_bytes,
        state_layout=state,
        layer_placement=(_partition_budget(model,batch,length,precision,mode,placement_partitions,None,60*2**30)
                         if placement_partitions and graph.config.kind != "fc" else None),
        explicit_activation_workspace_allowance_bytes=allowance,
        allowance_method="node outputs x36 autograd/x8 inference, peak incoming x3, largest Parameter x3; estimate, not proof",
        planned_bytes=persistent+component["foreach_scratch_allowance_bytes"]+kv_bytes+ssd_bytes+pending_bytes+allowance,
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
    parser.add_argument("--placement-partitions",type=int,default=0)
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
                row=budget(config,args.batch,args.length,precision,mode,placement_partitions=args.placement_partitions)
                record["observations"].append(dict(name=name,precision=precision,mode=mode,**row))
    atomic(output,record)
    print(json.dumps(dict(output=str(output),rows=len(record["observations"]))))


if __name__ == "__main__":
    main()

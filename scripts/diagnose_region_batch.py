"""Bounded first numeric/support divergence of the current regional batch path.

Intrusive capture, never a performance measurement. The two owners start from
identical CPU parameters/input. Initial inference and optional Timed flush only;
no trajectories, tolerances, support or implementation are modified.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import replace
from decimal import Decimal, localcontext
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess

import torch

from tide import ModelConfig, TideModel
from tide.parameters import NodeLinear, NodeOwners, owner_ids
from tide.precision import precision_context
from benchmarks.workloads import paired_input, parameter_hash, tensor_hash
from scripts.diagnose_bf16 import decimal_quad
from scripts.job import atomic
from scripts.qualify_precision import atomic_tensor, cpu_tree


def difference(a, b):
    if a.shape != b.shape:
        return dict(shape_matches=False, actual=list(a.shape), reference=list(b.shape))
    delta = (a.double() - b.double()).abs()
    return dict(shape_matches=True, dtype_actual=str(a.dtype), dtype_reference=str(b.dtype),
        exact=torch.equal(a, b), different=int((a != b).sum()), elements=a.numel(),
        max_abs=float(delta.max()) if delta.numel() else 0,
        finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()))


class Trace:
    def __init__(self, model):
        self.model, self.phase, self.current = model, "input", ()
        self.entries, self.order, self.calls = {}, [], []
        self.counts = defaultdict(int)
        self.node_region = {node: region for region, nodes in enumerate(model.graph.regions) for node in nodes}

    def put(self, region, stage, tensor, **metadata):
        count = self.counts[self.phase, region, stage]
        self.counts[self.phase, region, stage] += 1
        key = (self.phase, region, stage, count)
        self.order.append(key)
        self.entries[key] = dict(value=tensor.detach().cpu().clone(), metadata=metadata)

    def group(self, stage, tensor):
        offset = 0
        for region in self.current:
            size = len(self.model.graph.regions[region])
            self.put(region, stage, tensor[offset:offset+size])
            offset += size
        if offset != tensor.shape[0]:
            raise ValueError(f"capture split failed: {stage}")

    def node_ids(self, module, ids):
        if module.startswith("memories."):
            kind = module.split(".")[1]
            inverse = {value: node for node, value in self.model.memory_ids[kind].items()}
        elif module.startswith("describe."):
            inverse = {value: node for node, value in self.model.description_ids.items()}
        else:
            return ids
        return [inverse[x] for x in ids]

    def linear_hook(self, name):
        def hook(module, args, output):
            if not self.current:
                return
            x, owners = args
            ids = owner_ids(owners).detach().cpu().tolist()
            nodes = self.node_ids(name, ids)
            call = len(self.calls)
            self.calls.append(dict(module=name, x=x.detach().cpu().clone(), ids=ids,
                possible_ids=owners.possible_ids if isinstance(owners, NodeOwners) else None))
            for region in self.current:
                indices = [i for i, node in enumerate(nodes) if self.node_region[node] == region]
                if indices:
                    self.put(region, "linear."+name, output[indices], call=call, indices=indices,
                             nodes=[nodes[i] for i in indices])
        return hook

    @contextmanager
    def install(self):
        model = self.model
        batch_module = importlib.import_module("tide.region_batch")
        original_batch = batch_module.evaluate_regions
        original_single, original_score = model._evaluate_region, model._score_regions
        handles = []
        def incoming(regions, contents, masses, classes, time):
            self.current = tuple(regions)
            self.group("incoming.contents", contents)
            self.group("incoming.masses", masses)
            self.group("incoming.classes", classes)
            for region in regions:
                self.put(region, "incoming.time", time)
        def single(region, contents, masses, classes, time, *args):
            previous = self.current
            incoming((region,), contents, masses, classes, time)
            try:
                result = original_single(region, contents, masses, classes, time, *args)
                self.put(region, "control", result[1]); self.put(region, "full", result[0])
                return result
            finally:
                self.current = previous
        def batch(owner, regions, contents, masses, classes, time, *args):
            if owner is not model:
                return original_batch(owner, regions, contents, masses, classes, time, *args)
            previous = self.current
            incoming(regions, contents, masses, classes, time)
            try:
                result = original_batch(owner, regions, contents, masses, classes, time, *args)
                for region, value, control in zip(regions, result[0], result[1]):
                    self.put(region, "control", control); self.put(region, "full", value)
                return result
            finally:
                self.current = previous
        def score(regions, z, raw, mass, constant_zero=None):
            for i, region in enumerate(regions):
                self.put(region, "regional.z", z[i], regions=list(regions)); self.put(region, "regional.raw", raw[i])
                self.put(region, "regional.mass", mass[i])
            result = original_score(regions, z, raw, mass, constant_zero)
            for i, region in enumerate(regions):
                self.put(region, "regional.score", result[i])
            return result
        def serial_score_pre(module, args):
            if len(self.current) == 1:
                for name, value in zip(("z", "raw", "mass"), args):
                    self.put(self.current[0], "regional."+name, value)
        def serial_score_post(module, args, output):
            if len(self.current) == 1:
                self.put(self.current[0], "regional.score", output)
        def receive(module, args, output):
            self.group("received", output[0]); self.group("received.u", output[1])
        def describe(module, args, output):
            self.group("describe.z", output[0]); self.group("describe.score", output[1])
        model._evaluate_region, model._score_regions = single, score
        batch_module.evaluate_regions = batch
        handles += [model.receive.register_forward_hook(receive), model.describe.register_forward_hook(describe)]
        for name, module in model.named_modules():
            if isinstance(module, NodeLinear):
                handles.append(module.register_forward_hook(self.linear_hook(name)))
        for module in model.regional:
            handles += [module.register_forward_pre_hook(serial_score_pre), module.register_forward_hook(serial_score_post)]
        try:
            yield
        finally:
            model._evaluate_region, model._score_regions = original_single, original_score
            batch_module.evaluate_regions = original_batch
            for handle in handles:
                handle.remove()


def scalar_dot(left, right):
    with localcontext() as context:
        context.prec = 90
        exact = sum((Decimal.from_float(float(a)) * Decimal.from_float(float(b))
                     for a, b in zip(left, right)), Decimal(0))
        return str(exact), float(exact)


def linear_replay(models, traces, key, precision, device):
    """Same selected-node inputs and exact public dispatch, then scalar oracle."""
    stage_rows, payload = [], []
    for model, trace in zip(models, traces):
        entry = trace.entries[key]; call = trace.calls[entry["metadata"]["call"]]
        module = dict(model.named_modules())[call["module"]]
        ids = torch.tensor(call["ids"], device=device)
        owners = NodeOwners(ids, tuple(call["possible_ids"])) if call["possible_ids"] is not None else ids
        x = call["x"].to(device); flat = x.reshape(len(ids), -1, module.in_features)
        uniform = module._uniform_group(owners)
        weights = module.weight[module.groups[ids]]
        with torch.no_grad(), precision_context(device, precision):
            base = (torch.bmm(flat, weights.transpose(1, 2)) if uniform is None else
                    torch.mm(flat.reshape(-1, module.in_features), module.weight[uniform].T)
                    .reshape(len(ids), -1, module.out_features))
            result = base
            stages = {"base": base}
            if module.shared:
                low = torch.bmm(flat, module.right[ids].transpose(1, 2))
                high = torch.bmm(low, module.left[ids].transpose(1, 2))
                result = result + high
                stages.update(adapter_low=low, adapter_high=high, base_plus_adapter=result)
            if module.bias is not None:
                result = result + module.bias[module.groups[ids], None, :]
            stages["output"] = result.reshape(*x.shape[:-1], module.out_features)
        indices = entry["metadata"]["indices"]
        row = {name: value[indices].detach().cpu() for name, value in stages.items()}
        stage_rows.append(row)
        quantize = lambda t: t.detach().cpu().to(torch.bfloat16).float() if precision == "bf16" else t.detach().cpu()
        payload.append(dict(x=quantize(flat[indices]), weights=quantize(weights[indices]),
                            uniform=uniform, total_nodes=len(ids), stages=row))
    stages = {name: difference(stage_rows[1][name], stage_rows[0][name]) for name in stage_rows[0]}
    same_x = torch.equal(payload[0]["x"], payload[1]["x"])
    same_weight = torch.equal(payload[0]["weights"], payload[1]["weights"])
    record = dict(module=traces[0].calls[traces[0].entries[key]["metadata"]["call"]]["module"],
        quantized_inputs_equal=same_x, effective_weights_equal=same_weight,
        layouts=[dict(uniform=p["uniform"], total_nodes=p["total_nodes"]) for p in payload], stages=stages,
        replay_matches_capture=[difference(row["output"], trace.entries[key]["value"]) for row, trace in zip(stage_rows, traces)])
    a, b = stage_rows[0]["base"], stage_rows[1]["base"]
    mismatch = (a != b).nonzero()
    if same_x and same_weight and mismatch.numel():
        node, row, column = mismatch[0].tolist()
        exact, rounded = scalar_dot(payload[0]["x"][node, row], payload[0]["weights"][node, column])
        dtype = a.dtype
        nearest = float(torch.tensor(rounded, dtype=torch.float64).to(dtype))
        left, right = float(a[node, row, column]), float(b[node, row, column])
        record["independent_base_dot"] = dict(index=[node,row,column], decimal_exact=exact,
            nearest_output_dtype=nearest, dtype=str(dtype), serial=left, batched=right,
            midpoint=(left+right)/2, exact_minus_midpoint=rounded-(left+right)/2,
            serial_equals_nearest=left==nearest, batched_equals_nearest=right==nearest,
            terms=payload[0]["x"].shape[-1])
    return record, payload


def support_witness(model, traces, key):
    left, right = [trace.entries[key]["value"] for trace in traces]
    index = ((left > 0) != (right > 0)).nonzero()[0].tolist()
    node, batch, time = index
    phase, region, _, ordinal = key
    roles = [model.graph.nodes[v].role == "H" for v in model.graph.regions[region]]
    budget = model.config.budget or max(1, model.graph.config.n//32)
    rows = []
    for trace in traces:
        score = trace.entries[phase, region, "regional.score", ordinal]["value"][:,batch,time]
        received = trace.entries[phase, region, "received", ordinal]["value"][:,batch,time]
        actual = trace.entries[key]["value"][:,batch,time]
        oracle = decimal_quad(score, received, roles, budget, model.graph.config.partition)
        rows.append(dict(score=score.tolist(), received=received.tolist(), actual=actual.tolist(),
                         independent_decimal=oracle,
                         support_matches_decimal=[float(x)>0 for x in actual]==[x>0 for x in oracle["control"]]))
    return dict(index=index, roles=roles, budget=budget, rows=rows)


def regional_replay(models, traces, key, precision, device):
    """Replay both submitted shapes and record the first differing operation."""
    from tide.attention import weighted_attention
    from tide.parameters import norm
    from torch.nn.functional import linear, silu
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten
    class ProjectionAudit(TorchDispatchMode):
        def __init__(self):
            self.events=[]
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            output=function(*args,**(kwargs or {}))
            name=str(function)
            if any(token in name for token in ("mm.","add.","add_.","addmm.","baddbmm.")):
                encode=lambda t:dict(shape=list(t.shape),dtype=str(t.dtype),stride=list(t.stride()))
                self.events.append(dict(operator=name,
                    inputs=[encode(t) for t in tree_flatten((args,kwargs))[0] if isinstance(t,torch.Tensor)],
                    outputs=[encode(t) for t in tree_flatten(output)[0] if isinstance(t,torch.Tensor)]))
            return output
    attention_module = importlib.import_module("tide.attention")
    phase, selected, _, ordinal = key
    groups = [[selected], traces[1].entries[phase,selected,"regional.z",ordinal]["metadata"].get("regions",[selected])]
    rows, tensors = [], []
    for index, (model, trace, group) in enumerate(zip(models, traces, groups)):
        networks = [model.regional[model.region_group[r]] for r in group]
        values = {name: torch.stack([trace.entries[phase,r,"regional."+name,ordinal]["value"] for r in group]).to(device)
                  for name in ("z","raw","mass")}
        selected_index = group.index(selected)
        observed = {}; inputs = {}; weights = {}; biases = {}; audits={}
        def remember(name, value):
            observed[name] = value.detach().cpu() if index == 0 else value[selected_index].detach().cpu()
            return value
        def projection(x, name):
            layers = [getattr(net,name) for net in networks]
            if index == 0:
                with ProjectionAudit() as audit:
                    result = linear(x, layers[0].weight, layers[0].bias)
                inputs[name] = x.detach().cpu(); weights[name] = layers[0].weight.detach().cpu()
                biases[name] = None if layers[0].bias is None else layers[0].bias.detach().cpu()
            else:
                matrices = torch.stack([layer.weight for layer in layers])
                flat = x.reshape(len(group), -1, x.shape[-1])
                with ProjectionAudit() as audit:
                    if layers[0].bias is None:
                        result = torch.bmm(flat, matrices.transpose(1,2))
                    else:
                        result = torch.baddbmm(torch.stack([layer.bias for layer in layers])[:,None], flat, matrices.transpose(1,2))
                result = result.reshape(*x.shape[:-1],matrices.shape[1])
                inputs[name] = x[selected_index].detach().cpu(); weights[name] = layers[selected_index].weight.detach().cpu()
                biases[name] = None if layers[0].bias is None else layers[selected_index].bias.detach().cpu()
            audits[name]=audit.events
            return remember(name,result)
        backend = [networks[0].backend]
        native = attention_module._native_attention
        def capture_native(*args, **kwargs):
            backend[0] = args[5]
            return native(*args, **kwargs)
        attention_module._native_attention = capture_native
        try:
            with torch.no_grad(), precision_context(device,precision):
                x = values["z"][0].permute(1,2,0,3) if index == 0 else values["z"].permute(0,2,3,1,4)
                heads = networks[0].heads
                q,k,v = [projection(x,name).reshape(*x.shape[:-1],heads,32).transpose(-3,-2) for name in ("q","k","v")]
                mass = values["mass"][0].permute(1,2,0) if index == 0 else values["mass"].permute(0,2,3,1)
                read = weighted_attention(q,k,v,weights=mass,backend=networks[0].backend)
                remember("attention",read)
                delta = projection(read.transpose(-3,-2).flatten(-2),"o")
                normalized = remember("norm",norm(x+delta))
                hidden = remember("silu",silu(projection(normalized,"first")))
                with torch.autocast(torch.device(device).type,enabled=False):
                    if index == 0:
                        net = networks[0]
                        projected = net.second.weight.T @ net.readout
                        a = delta.to(net.readout.dtype) @ net.readout
                        b = hidden.to(net.readout.dtype) @ projected
                        c = net.second.bias @ net.readout
                        correction = a+b+c
                        score = values["raw"][0] + torch.where(values["mass"][0]>0,correction.permute(2,0,1),0)
                    else:
                        readout = torch.stack([net.readout for net in networks])
                        second = torch.stack([net.second.weight for net in networks])
                        bias = torch.stack([net.second.bias for net in networks])
                        projected = torch.bmm(second.transpose(1,2),readout[...,None])
                        a = torch.bmm(delta.float().reshape(len(group),-1,delta.shape[-1]),readout[...,None])
                        b = torch.bmm(hidden.float().reshape(len(group),-1,hidden.shape[-1]),projected)
                        c = (bias*readout).sum(-1)[:,None,None]
                        correction = (a+b+c).reshape(*x.shape[:-1])
                        score = values["raw"] + torch.where(values["mass"]>0,correction.permute(0,3,1,2),0)
                    remember("score",score)
        finally:
            attention_module._native_attention = native
        if backend[0] == "auto":
            backend[0] = "tiled"
        rows.append(dict(regions=group, effective_attention=backend[0],requested_attention=networks[0].backend,
            zero_priors_per_region=[int((values["mass"][i]==0).sum()) for i in range(len(group))],
            projection_operators=audits,
            replay_matches_capture=difference(observed["score"],trace.entries[key]["value"])))
        tensors.append(dict(stages=observed,inputs=inputs,weights=weights,biases=biases,
                            z=values["z"][selected_index].cpu(), raw=values["raw"][selected_index].cpu(),mass=values["mass"][selected_index].cpu()))
    stages = {name:difference(tensors[1]["stages"][name],tensors[0]["stages"][name]) for name in tensors[0]["stages"]}
    first = next((name for name,comparison in stages.items() if not comparison.get("exact",False)),None)
    record = dict(rows=rows,stages=stages,first_differing_operation=first,
        exact_common_inputs={name:torch.equal(tensors[0][name],tensors[1][name]) for name in ("z","raw","mass")})
    if first in ("q","k","v","o","first"):
        quantize = lambda t: t.to(torch.bfloat16).float() if precision=="bf16" else t
        x0,x1=[quantize(value["inputs"][first]).reshape(-1,value["inputs"][first].shape[-1]) for value in tensors]
        w0,w1=[quantize(value["weights"][first]) for value in tensors]
        a,b=[value["stages"][first].reshape(-1,w0.shape[0]) for value in tensors]
        record["matrix_inputs_equal"]=torch.equal(x0,x1) and torch.equal(w0,w1)
        if record["matrix_inputs_equal"]:
            row,column=(a!=b).nonzero()[0].tolist()
            product,value=scalar_dot(x0[row],w0[column]); exact=product
            bias=tensors[0]["biases"][first]
            bias_value=0.0
            if bias is not None:
                bias_value=float(quantize(bias)[column])
                with localcontext() as context:
                    context.prec=90
                    exact=str(Decimal(exact)+Decimal.from_float(bias_value))
                    value=float(exact)
            left,right=float(a[row,column]),float(b[row,column])
            rounded_product=float(torch.tensor(float(product),dtype=torch.float64).to(a.dtype))
            twice=float(torch.tensor(rounded_product+bias_value,dtype=torch.float64).to(a.dtype))
            record["independent_dot"]=dict(index=[row,column],decimal_exact=exact,
                nearest_output_dtype=float(torch.tensor(value,dtype=torch.float64).to(a.dtype)),
                serial=left,batched=right,midpoint=(left+right)/2,exact_minus_midpoint=value-(left+right)/2,
                terms=x0.shape[-1],output_dtype=str(a.dtype),decimal_product=product,bias=bias_value,
                rounded_product=rounded_product,two_round_output=twice,
                serial_matches_two_round=left==twice,batched_matches_two_round=right==twice)
    if first=="attention":
        q,k,v=[tensors[0]["stages"][name].double().reshape(*tensors[0]["stages"][name].shape[:-1],-1,32).transpose(-3,-2) for name in ("q","k","v")]
        logits=q@k.transpose(-2,-1)/32**.5
        mass=tensors[0]["mass"].double().permute(1,2,0).unsqueeze(-2).unsqueeze(-2)
        numerator=(logits-logits.amax(-1,keepdim=True)).exp()*mass
        denominator=numerator.sum(-1,keepdim=True)
        probability=numerator/denominator.clamp_min(torch.finfo(torch.float64).tiny)
        exact=probability@v
        record["independent_attention_fp64"]=dict(formula="exp shifted dot times mass, normalized quotient, value product",
            against_exact=[difference(value["stages"]["attention"].double(),exact) for value in tensors],
            against_rounded=[difference(value["stages"]["attention"],exact.to(value["stages"]["attention"].dtype)) for value in tensors])
    return record,tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overrides", type=json.loads, default={"region_batch_size":4})
    parser.add_argument("--device", required=True)
    parser.add_argument("--precision", choices=("fp32","bf16"), default="bf16")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--length", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if min(args.batch,args.length,args.threads)<1:
        parser.error("batch, length and threads must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.device.startswith("npu"):
        import torch_npu
        torch.npu.set_device(args.device)
        torch.npu.matmul.allow_hf32 = False; torch.npu.conv.allow_hf32 = False
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    config = ModelConfig.from_dict(json.loads(args.config.read_text()))
    if config.allocation != "quad":
        raise ValueError("this bounded witness supports quad only")
    torch.manual_seed(1107)
    original, candidate = TideModel(config), TideModel(replace(config, **args.overrides))
    candidate.load_state_dict(original.state_dict())
    parameter_identity = parameter_hash(original)
    x = paired_input(args.batch, args.length, config.input_dim or config.d, 1108)
    input_identity = tensor_hash(x)
    models = [original.to(args.device), candidate.to(args.device)]
    traces = []
    for model in models:
        trace = Trace(model)
        with torch.no_grad(), precision_context(args.device, args.precision), trace.install():
            result = model(x.to(args.device))
            if not model.settle:
                trace.phase = "flush"
                model.flush(result.state)
        traces.append(trace)
    common = [key for key in traces[0].order if key in traces[1].entries]
    changed = [key for key in common if not torch.equal(traces[0].entries[key]["value"], traces[1].entries[key]["value"])]
    compatible = [key for key in changed if traces[0].entries[key]["value"].shape == traces[1].entries[key]["value"].shape]
    support = [key for key in common if key[2] == "control" and
               not torch.equal(traces[0].entries[key]["value"]>0, traces[1].entries[key]["value"]>0)]
    record = dict(schema=1, state="completed", source=os.environ.get("TIDE_SOURCE_COMMIT") or
        subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip(),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_dirty=bool(subprocess.check_output(["git","status","--porcelain","--untracked-files=all"],text=True).strip()),
        config=config.to_dict(), overrides=args.overrides, precision=args.precision, device=args.device,
        input_sha256=input_identity, parameter_sha256=parameter_identity,
        physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        cpu_affinity=sorted(os.sched_getaffinity(0)),threads=torch.get_num_threads(),
        common=len(common), changed=len(changed), shape_differences=len(changed)-len(compatible),
        order="serial region traversal, within-region operator order; independent same-layer regions may execute batched",
        qualification="intrusive finite initial-inference/flush witness; no threshold change, no performance result")
    payload = {}
    if compatible:
        key = compatible[0]
        before=common[:common.index(key)]
        record["shape_differences_before_first_numeric"]=[earlier for earlier in before
            if traces[0].entries[earlier]["value"].shape!=traces[1].entries[earlier]["value"].shape]
        record["first_numeric"] = dict(key=key, comparison=difference(traces[1].entries[key]["value"], traces[0].entries[key]["value"]))
        payload["first_numeric"] = [trace.entries[key] for trace in traces]
        if key[2].startswith("linear."):
            record["linear_replay"], payload["linear_replay"] = linear_replay(models, traces, key, args.precision, args.device)
        if key[2]=="regional.score":
            record["regional_replay"],payload["regional_replay"]=regional_replay(models,traces,key,args.precision,args.device)
    if support:
        key = support[0]
        record["first_support"] = dict(key=key, witness=support_witness(original,traces,key))
        payload["first_support"] = [trace.entries[key] for trace in traces]
    record["first_changed_keys"] = compatible[:12]
    atomic_tensor(args.output_dir / "witness.pt", payload)
    atomic(args.output_dir / "result.json", record)
    print(json.dumps({k:record[k] for k in ("state","common","changed","shape_differences","first_numeric","first_support") if k in record}))


if __name__ == "__main__":
    main()

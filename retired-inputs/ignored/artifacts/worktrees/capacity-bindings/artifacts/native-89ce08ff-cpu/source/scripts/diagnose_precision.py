"""Bounded FP32-versus-CPU-FP64 diagnosis using identical quantized parameters.

This uses the same model function at higher precision, not an independent
semantic oracle. Formula tests provide that independent evidence. No thresholds
are relaxed here and no FP16/BF16 or higher-precision execution mode is added.
"""
import argparse
import copy
from dataclasses import fields, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess

import torch
from tide import ModelConfig, TideModel
from scripts.job import atomic


def tensors(value, prefix):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif is_dataclass(value):
        for field in fields(value):
            yield from tensors(getattr(value, field.name), prefix+"."+field.name)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from tensors(item, prefix+"."+str(key))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from tensors(item, prefix+"."+str(index))


def metrics(actual, reference, atol=1e-6, rtol=2e-3):
    a, b = actual.detach().cpu().double(), reference.detach().cpu().double()
    if a.shape != b.shape:
        return dict(shape_matches=False)
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        return dict(shape_matches=True, finite=False)
    delta = (a-b).abs()
    scaled = delta/(atol+rtol*b.abs())
    index = int(scaled.flatten().argmax()) if scaled.numel() else 0
    return dict(shape_matches=True, finite=True,
                max_abs=float(delta.max()) if delta.numel() else 0,
                relative_l2=float(delta.norm()/b.norm().clamp_min(1e-20)),
                max_scaled=float(scaled.max()) if scaled.numel() else 0,
                outside_threshold=int((scaled>1).sum()),
                worst_actual=float(a.flatten()[index]) if a.numel() else 0,
                worst_reference=float(b.flatten()[index]) if b.numel() else 0)


def compare(actual, reference):
    floating, structural = [], []
    if actual.keys() != reference.keys():
        structural.append("tensor names differ")
    for name in sorted(actual.keys() & reference.keys()):
        a, b = actual[name], reference[name]
        if a is None or b is None:
            if (a is None) != (b is None):
                structural.append(name+": disconnected root")
            continue
        if a.shape != b.shape:
            structural.append(name+": shape")
        elif not a.is_floating_point():
            if not torch.equal(a.cpu(), b.cpu()):
                structural.append(name+": integer metadata")
        else:
            value = metrics(a, b)
            floating.append((name, value))
            if name.endswith((".weight", ".strengths")) and not name.startswith("gradient."):
                if not torch.equal(a.detach().cpu()>0, b.detach().cpu()>0):
                    structural.append(name+": positive support")
    return dict(structural_differences=structural,
                nonfinite=[name for name, value in floating if not value["finite"]],
                failed_tensors=[(name, value) for name, value in floating if value.get("outside_threshold", 0)],
                largest_absolute=sorted(floating, key=lambda pair: pair[1].get("max_abs", float("inf")), reverse=True)[:5],
                largest_relative=sorted(floating, key=lambda pair: pair[1].get("relative_l2", float("inf")), reverse=True)[:5])


def evaluate(model, inputs, seed):
    x = inputs.detach().requires_grad_()
    result = model(x)
    names, parameters = zip(*model.named_parameters())
    gradients = torch.autograd.grad((result.readout()*seed).mean(), (x, *parameters), allow_unused=True)
    observed = dict(tensors(result.state, "state"))
    observed.update({"result.values":result.values, "result.strengths":result.strengths,
                     "result.times":result.times, "result.readout":result.readout(),
                     "gradient.input":gradients[0]})
    observed.update(("gradient."+name, gradient) for name, gradient in zip(names, gradients[1:]))
    return {name: None if value is None else value.detach().cpu() for name, value in observed.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--seed", type=int, default=517)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--length", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("choose a new observation path")
    if args.batch<1 or args.length<2:
        parser.error("positive batch and length >=2 required for continuation")
    if args.device.startswith("npu"):
        import torch_npu
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
        torch.npu.set_device(args.device)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    config = ModelConfig.from_dict(json.loads(Path(args.config).read_text()))
    cpu = TideModel(config)
    x = torch.randn(args.batch,args.length,config.input_dim or config.d)
    with torch.no_grad():
        shape = cpu(x).readout().shape
    seed = torch.randn(shape)
    double = copy.deepcopy(cpu).double()
    candidate = copy.deepcopy(cpu).to(args.device)
    observed64 = evaluate(double,x.double(),seed.double())
    observed_cpu = evaluate(cpu,x,seed)
    observed_device = evaluate(candidate,x.to(args.device),seed.to(args.device))
    comparisons = dict(cpu_fp32_vs_fp64=compare(observed_cpu,observed64),
                       device_fp32_vs_fp64=compare(observed_device,observed64),
                       device_fp32_vs_cpu_fp32=compare(observed_device,observed_cpu))
    with torch.no_grad():
        split = args.length//2
        first = double(x[:,:split].double())
        last = double(x[:,split:].double(),first.state)
        continued = dict(tensors(last.state,"state"))
        continued.update({"result.values":torch.cat((first.values,last.values),2),
                          "result.strengths":torch.cat((first.strengths,last.strengths),2),
                          "result.times":torch.cat((first.times,last.times)),
                          "result.readout":torch.cat((first.readout(),last.readout()),1)})
    expected = {name:value for name,value in observed64.items() if not name.startswith("gradient.")}
    comparisons["cpu_fp64_continuation"] = compare(continued,expected)
    source = os.environ.get("TIDE_SOURCE_COMMIT") or subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    record = dict(source=source,harness_source=os.environ.get("TIDE_HARNESS_COMMIT",source),
                  scope=__doc__,arguments=vars(args),config=config.to_dict(),torch=str(torch.__version__),
                  input_sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest(),
                  comparisons=comparisons,threshold=dict(atol=1e-6,rtol=2e-3))
    if args.device.startswith("npu"):
        record.update(torch_npu=str(torch_npu.__version__),physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                      allow_hf32=dict(matmul=torch.npu.matmul.allow_hf32,conv=torch.npu.conv.allow_hf32))
    atomic(output,record)
    for name, comparison in comparisons.items():
        print(json.dumps(dict(comparison=name,structural=comparison["structural_differences"],
                              nonfinite=comparison["nonfinite"],failed_tensors=len(comparison["failed_tensors"]))))
    return int(any(c["structural_differences"] or c["nonfinite"] for c in comparisons.values()))


if __name__ == "__main__":
    raise SystemExit(main())

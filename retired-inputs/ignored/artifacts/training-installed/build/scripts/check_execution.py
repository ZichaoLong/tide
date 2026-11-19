"""Finite same-function checks for execution choices, retaining original tolerances."""
import argparse
from contextlib import nullcontext
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess

import torch

from tide import ModelConfig, TideModel, detach_state
from tide.precision import precision_context
from benchmarks.continuous import checkpoint_resume
from benchmarks.workloads import paired_input, parameter_hash, tensor_hash
from scripts.check_placement import _full, _objective
from scripts.qualify_precision import THRESHOLDS, comparison, cpu_tree
from scripts.job import atomic


def qualify(config, overrides, device, precision, batch, length, steps, checkpoint, progress):
    torch.manual_seed(1107)
    original = TideModel(config)
    candidate = TideModel(replace(config, **overrides))
    candidate.load_state_dict(original.state_dict())
    record = dict(config=config.to_dict(), overrides=overrides,
                  parameter_sha256=parameter_hash(original), checks={}, updates=[],
                  tolerance=THRESHOLDS[precision], state="running")
    models = [original.to(device), candidate.to(device)]
    if precision == "fp64":
        models = [model.double() for model in models]
    dtype = torch.float64 if precision == "fp64" else torch.float32
    x = paired_input(batch, length, config.input_dim or config.d, 1108).to(dtype=dtype)
    record["input_sha256"] = tensor_hash(x)
    x = x.to(device)
    context = lambda: nullcontext() if precision == "fp64" else precision_context(device, precision)
    checks = record["checks"]
    for mode in ("inference", "grad_forward"):
        values, gradients = [], []
        for model in models:
            model.zero_grad(set_to_none=True)
            inp = x.clone().requires_grad_(mode != "inference")
            with torch.set_grad_enabled(mode != "inference"), context():
                result, tail = _full(model, inp, record_work=True)
                loss = _objective(result) + (0 if tail is None else _objective(tail))
            values.append(cpu_tree(dict(result=result, flush=tail)))
            if mode != "inference":
                loss.backward()
                gradients.append(cpu_tree(dict(input=inp.grad, parameters={n:p.grad for n,p in model.named_parameters()})))
            del result, tail, loss, inp
        checks[mode] = comparison(values[1], values[0], THRESHOLDS[precision])
        if gradients:
            checks["gradients"] = comparison(gradients[1], gradients[0], THRESHOLDS[precision])
        progress(record)
    optimizers = [torch.optim.AdamW(model.parameters(), lr=1e-4) for model in models]
    states = [None, None]
    for step in range(steps):
        observations = []
        for index, (model, optimizer) in enumerate(zip(models, optimizers)):
            optimizer.zero_grad(set_to_none=True)
            with context():
                first = model(x[:, :max(1, length//2)], states[index], record_work=True)
                end = first
                loss = _objective(first)
                if length > 1:
                    end = model(x[:, length//2:], first.state, record_work=True)
                    loss = loss + _objective(end)
                if not model.settle:
                    end = model.flush(end.state, record_work=True)
                    loss = loss + _objective(end)
            loss.backward()
            optimizer.step()
            states[index] = detach_state(end.state)
            observations.append(cpu_tree(dict(parameters=model.state_dict(), optimizer=optimizer.state_dict(),
                                              state=states[index], result=end, loss=loss)))
            del first, end, loss
        record["updates"].append(comparison(observations[1], observations[0], THRESHOLDS[precision]))
        progress(record)
    # Existing restore checker retains its original, stricter update thresholds.
    if precision != "fp64":
        record["restore"] = checkpoint_resume(models[1], optimizers[1], states[1], x, precision, checkpoint)
    else:
        record["restore"] = {"state":"not_requested", "reason":"FP64 is a bounded arithmetic diagnostic"}
    rows = list(checks.values()) + record["updates"]
    record["state"] = ("nonfinite" if any(row["nonfinite"] for row in rows) else
                       "structural_failed" if any(row["metadata_failures"] for row in rows) else
                       "finite_deviation" if any(row["threshold_failures"] or row["support_differences"] for row in rows) or
                           record["restore"]["state"] == "finite_deviation" else "within_threshold")
    if record["restore"]["state"] in {"nonfinite", "structural_failed"}:
        record["state"] = record["restore"]["state"]
    progress(record)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--overrides", type=json.loads, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--precision", choices=("fp32", "bf16", "fp64"), default="fp32")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--length", type=int, default=8)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or min(args.batch, args.length, args.steps, args.threads) < 1:
        p.error("positive sizes and a new output path required")
    if args.precision == "fp64" and args.device != "cpu":
        p.error("FP64 diagnostics require CPU")
    if args.device.startswith("npu"):
        import torch_npu
        torch.npu.set_device(args.device)
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
    torch.set_num_threads(args.threads)
    config = ModelConfig.from_dict(json.loads(Path(args.config).read_text()))
    record = dict(schema="tide-execution-qualification-v1", state="running",
                  source=os.environ.get("TIDE_SOURCE_COMMIT") or subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                  arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                  resource_identity=os.environ.get("TIDE_RESOURCE_IDENTITY"),
                  precision=args.precision, torch=str(torch.__version__),
                  qualification="same-function execution candidate; finite deviations retain original thresholds; no performance timing")
    def progress(result):
        record["result"] = result
        atomic(args.output, record)
    try:
        result = qualify(config,args.overrides,args.device,args.precision,args.batch,args.length,args.steps,
                         args.output.with_suffix(".pt"), progress)
        record["state"] = result["state"]
        atomic(args.output,record)
        return 0 if record["state"] in {"within_threshold","finite_deviation"} else 2
    except Exception as error:
        record.update(state="execution_failed", error=repr(error))
        atomic(args.output,record)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

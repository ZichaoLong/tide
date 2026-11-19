"""Finite whole/chunk value, support, state and full-backward witnesses.

The two schedules use identical parameters, inputs and cotangents, retain the
entire graph, and never detach private state between chunks. CPU FP64 is an
explicit diagnostic. Numerical deviations retain the published thresholds.
"""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess

import torch

from tide import ModelConfig, TideModel, TopologyConfig
from tide.precision import precision_context, precision_policy
from scripts.job import atomic
from scripts.qualify_precision import THRESHOLDS, comparison, cpu_tree, fingerprint


def cases():
    return {
        "fc-k-window": (ModelConfig(topology=TopologyConfig(n=1, layers=1), d=64,
                                    memory="K", attention_chunk=32), (65, 64, 2)),
        "fc-s-window": (ModelConfig(topology=TopologyConfig(n=1, layers=1), d=64,
                                    memory="S", attention_chunk=32), (65, 64, 2)),
        "hb-exchange": (ModelConfig(topology=TopologyConfig(kind="exchange", n=2, depth=1, hubs=1),
                                    d=64, memory="X", sharing="layer_shared",
                                    aggregation="incoming_light_64"), (2, 3)),
        "hb-timed": (ModelConfig(topology=TopologyConfig(kind="multiscale", n=2, hubs=1,
                                  profile="compact-5-dd4", timing="timed"), d=64, memory="X",
                                  sharing="region_shared"), (2, 3)),
    }


def schedule_backward(model, value, lengths, precision):
    model.zero_grad(set_to_none=True)
    x = value.detach().clone().requires_grad_()
    context = nullcontext() if precision == "fp64" else precision_context(x.device, precision)
    outputs, state, offset = [], None, 0
    with context:
        for length in lengths:
            result = model(x[:, offset:offset+length], state)
            outputs.append(result)
            state, offset = result.state, offset+length
        if not model.settle:
            result = model.flush(state)
            outputs.append(result)
            state = result.state
        readout = torch.cat([item.readout() for item in outputs], 1)
        # The same deterministic cotangent is independent of execution chunks.
        cotangent = torch.sin(torch.arange(readout.numel(), dtype=torch.float64).reshape(readout.shape))
        cotangent = cotangent.to(device=x.device, dtype=readout.dtype)
        loss = (readout*cotangent).mean()
    gradients = torch.autograd.grad(loss, (x, *model.parameters()), allow_unused=True)
    return cpu_tree(dict(readout=readout,
        strengths=torch.cat([item.strengths for item in outputs], 2),
        times=torch.cat([item.times for item in outputs]), state=state,
        input_gradient=gradients[0],
        parameter_gradients={name: gradient for (name, _), gradient in zip(model.named_parameters(), gradients[1:])}))


def witness(config, lengths, *, device, precision, seed=807):
    torch.manual_seed(seed)
    model = TideModel(config)
    value = torch.randn(2, sum(lengths), config.input_dim or config.d)
    identity = dict(parameters=fingerprint(model.state_dict()), input=fingerprint(value))
    dtype = torch.float64 if precision == "fp64" else torch.float32
    model, value = model.to(device=device, dtype=dtype), value.to(device=device, dtype=dtype)
    whole = schedule_backward(model, value, (sum(lengths),), precision)
    chunked = schedule_backward(model, value, lengths, precision)
    checks = {
        "values_support_state": comparison({k: chunked[k] for k in ("readout", "strengths", "times", "state")},
                                            {k: whole[k] for k in ("readout", "strengths", "times", "state")}, THRESHOLDS[precision]),
        "input_gradient": comparison(chunked["input_gradient"], whole["input_gradient"], THRESHOLDS[precision]),
        "parameter_gradients": comparison(chunked["parameter_gradients"], whole["parameter_gradients"], THRESHOLDS[precision]),
    }
    disconnected = sorted(name for name in whole["parameter_gradients"]
                          if (whole["parameter_gradients"][name] is None) != (chunked["parameter_gradients"][name] is None))
    status = ("nonfinite" if any(c["nonfinite"] for c in checks.values()) else
              "metadata_failed" if disconnected or any(c["metadata_failures"] for c in checks.values()) else
              "finite_deviation" if any(c["threshold_failures"] or c["support_differences"] for c in checks.values()) else
              "within_threshold")
    return dict(config=config.to_dict(), identity=identity, seed=seed, chunk_lengths=lengths,
                input_positions=2*sum(lengths), output_slots=chunked["readout"].shape[0]*chunked["readout"].shape[1],
                final_position=chunked["state"].position, checks=checks, disconnected_differences=disconnected,
                whole_input_gradient_nonzeros=int(torch.count_nonzero(whole["input_gradient"])),
                chunked_input_gradient_nonzeros=int(torch.count_nonzero(chunked["input_gradient"])), state=status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp64"), default="fp32")
    parser.add_argument("--cases", nargs="+", choices=tuple(cases()), default=list(cases()))
    parser.add_argument("--attention-backend", choices=("auto", "math", "tiled"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.threads < 1:
        parser.error("choose a new output identity and positive thread count")
    if args.precision == "fp64" and args.device != "cpu":
        parser.error("FP64 is a finite CPU diagnostic")
    if args.device.startswith("npu"):
        import torch_npu
        torch.npu.set_device(args.device)
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
    torch.set_num_threads(args.threads)
    source = os.environ.get("TIDE_SOURCE_COMMIT") or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    policy = (dict(requested="fp64", device="cpu", purpose="finite diagnostic", arithmetic="float64", autocast=False)
              if args.precision == "fp64" else precision_policy(args.device, args.precision))
    report = dict(source=source, arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                  precision=policy, torch=str(torch.__version__),
                  physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"), cpu_affinity=sorted(os.sched_getaffinity(0)),
                  scope="paired whole/chunk schedule; full BPTT; no detach or optimizer; complete Timed flush in both losses",
                  rows=[])
    atomic(args.output, report)
    for name in args.cases:
        config, lengths = cases()[name]
        data = config.to_dict(); data["attention_backend"] = args.attention_backend
        row = dict(case=name, state="running")
        report["rows"].append(row); atomic(args.output, report)
        try:
            row.update(witness(ModelConfig.from_dict(data), lengths, device=args.device, precision=args.precision))
        except Exception as error:
            import traceback
            traceback.print_exc()
            row.update(state="execution_failed", error=repr(error))
        atomic(args.output, report)
    return int(any(row["state"] in {"nonfinite", "metadata_failed", "execution_failed"} for row in report["rows"]))


if __name__ == "__main__":
    raise SystemExit(main())

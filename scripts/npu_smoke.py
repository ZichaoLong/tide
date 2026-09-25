#!/usr/bin/env python3
"""Run the device-neutral Python graph implementations on one real NPU.

This is a correctness smoke, not a performance qualification.  Native
LibTorch executors are deliberately excluded until a version-matched
libtorch_npu SDK is available.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from foundation_workloads import initialize, inputs_for  # noqa: E402
from tidegraph import Continuation, External  # noqa: E402
from tidegraph.frontier import run as frontier_run  # noqa: E402
from tidegraph.reference import run as streaming_run  # noqa: E402
from tidegraph.runtime import manifest, resolve_device, synchronize  # noqa: E402
from tidegraph.settle import run as settle_run  # noqa: E402
from tidegraph.specialized import run as specialized_run  # noqa: E402
from tidegraph.specialized import settle_layered  # noqa: E402


def _base(case: str) -> dict:
    if case == "pdg-streaming":
        return dict(id="P01", graph="pdg", topology="sparse-ring", width=16, batch=2,
                    sequence=4, body_nodes=4, module="ema", dtype="float32",
                    training_window=None)
    if case == "timed-dag-frontier":
        return dict(id="T01", graph="timed-dag", topology="diamond", width=16, batch=2,
                    sequence=4, body_nodes=4, module="attention", dtype="float32",
                    training_window=None)
    if case == "timed-dag-diamond":
        return dict(id="T01", graph="timed-dag", topology="diamond", width=16, batch=2,
                    sequence=4, body_nodes=4, module="attention", dtype="float32",
                    training_window=None)
    if case == "settle-generic":
        return dict(id="S01", graph="settle", topology="four-layers", width=16, batch=2,
                    sequence=4, body_nodes=8, module="attention-gqa-window128", dtype="float32",
                    training_window=None)
    if case == "settle-layered":
        return dict(id="S01", graph="settle", topology="four-layers", width=16, batch=2,
                    sequence=4, body_nodes=8, module="attention-gqa-window128", dtype="float32",
                    training_window=None)
    if case == "settle-chain":
        return dict(id="S02", graph="settle", topology="chain", width=16, batch=2,
                    sequence=4, body_nodes=4, module="ssm-swiglu", dtype="float32",
                    training_window=None)
    raise ValueError(f"unknown smoke case: {case}")


def _tensor_state(result):
    return {key: value.value.detach().cpu() for key, value in result.continuation.states.items()}


def _compare(cpu_result, npu_result):
    if len(cpu_result.outputs) != len(npu_result.outputs):
        raise AssertionError("output event count differs")
    if len(cpu_result.continuation.pending) != len(npu_result.continuation.pending):
        raise AssertionError("pending message count differs")
    if set(cpu_result.continuation.states) != set(npu_result.continuation.states):
        raise AssertionError("state owner set differs")
    for left, right in zip(cpu_result.outputs, npu_result.outputs):
        if left[:3] != right[:3]:
            raise AssertionError("output coordinates differ")
        torch.testing.assert_close(left[3], right[3].detach().cpu(), atol=4e-4, rtol=4e-3)
    for key, value in _tensor_state(cpu_result).items():
        torch.testing.assert_close(value, _tensor_state(npu_result)[key], atol=4e-4, rtol=4e-3)


def _external(config, graph, values):
    b, sequence, _ = values.shape
    stride = 1 if config["graph"] == "pdg" and config["topology"] != "diamond" else \
        6 if config["topology"] == "large-layered" else len(graph.regions) + 2
    return [External(sample, port, position, stride * position, values[sample, position])
            for sample in range(b) for port in range(len(graph.inputs))
            for position in range(sequence)]


def _execute(case, model, graph, spec, values):
    q = Continuation(graph.identity, values.shape[0])
    if case == "pdg-streaming":
        external = _external(_base(case), graph, values); stride = 1
        return streaming_run(graph, model, q, external, stride * values.shape[1],
                             sealed_until=stride * values.shape[1], mode="hst", trace=True)
    if case == "timed-dag-frontier":
        external = _external(_base(case), graph, values); stride = len(graph.regions) + 2
        return frontier_run(graph, model, q, external, stride * values.shape[1],
                            sealed_until=stride * values.shape[1], mode="hst", trace=True,
                            packed=True)
    if case == "timed-dag-diamond":
        external = _external(_base(case), graph, values); stride = len(graph.regions) + 2
        return specialized_run(graph, model, q, external, stride * values.shape[1],
                               sealed_until=stride * values.shape[1], topology="diamond", mode="hst")
    if case == "settle-generic":
        return settle_run(spec, model, q, values, mode="hard", prefill=True, packed=True)
    if case == "settle-layered":
        return settle_layered(spec, model, q, values, mode="hard")
    if case == "settle-chain":
        from tidegraph.specialized import settle_chain
        return settle_chain(spec, model, q, values, mode="hard")
    raise AssertionError(case)


def run_case(case: str, device: torch.device) -> dict:
    config = _base(case)
    graph, spec, cpu_model = initialize(config)
    cpu_values, _, _, _ = inputs_for(config, graph)
    npu_graph, npu_spec, npu_model = initialize(config, device=device)
    npu_model.load_state_dict({name: value.to(device) for name, value in cpu_model.state_dict().items()})
    npu_values = cpu_values.to(device)
    with torch.no_grad():
        cpu_result = _execute(case, cpu_model, graph, spec, cpu_values)
        npu_result = _execute(case, npu_model, npu_graph, npu_spec, npu_values)
    synchronize(device)
    _compare(cpu_result, npu_result)
    return {"case": case, "status": "passed", "outputs": len(npu_result.outputs),
            "states": len(npu_result.continuation.states), "pending": len(npu_result.continuation.pending),
            "cpu_output_checksum": sum(float(v.detach().double().sum()) for _, _, _, v in cpu_result.outputs),
            "npu_output_checksum": sum(float(v.detach().cpu().double().sum()) for _, _, _, v in npu_result.outputs)}


def _loss(result):
    values = [value.square().mean() for _, _, _, value in result.outputs]
    values += [state.value.square().mean() for state in result.continuation.states.values()]
    return sum(values)


def run_backward(device: torch.device) -> dict:
    config = _base("pdg-streaming")
    graph, spec, cpu_model = initialize(config)
    base, _, _, _ = inputs_for(config, graph)
    npu_graph, npu_spec, npu_model = initialize(config, device=device)
    npu_model.load_state_dict({name: value.to(device) for name, value in cpu_model.state_dict().items()})
    cpu_values = base.clone().requires_grad_(True)
    npu_values = base.to(device).requires_grad_(True)
    cpu_result = _execute("pdg-streaming", cpu_model, graph, spec, cpu_values)
    npu_result = _execute("pdg-streaming", npu_model, npu_graph, npu_spec, npu_values)
    cpu_weight = next(parameter for parameter in cpu_model.parameters() if parameter.requires_grad)
    npu_weight = next(parameter for parameter in npu_model.parameters() if parameter.requires_grad)
    cpu_loss = _loss(cpu_result)
    npu_loss = _loss(npu_result)
    cpu_grads = torch.autograd.grad(cpu_loss, (cpu_values, cpu_weight), allow_unused=True)
    npu_grads = torch.autograd.grad(npu_loss, (npu_values, npu_weight), allow_unused=True)
    synchronize(device)
    for left, right in zip(cpu_grads, npu_grads):
        if left is None or right is None:
            if left is not None or right is not None:
                raise AssertionError("NPU gradient connectivity differs")
        else:
            torch.testing.assert_close(left, right.detach().cpu(), atol=3e-3, rtol=3e-2)
    return {"case": "pdg-backward", "status": "passed", "loss_cpu": float(cpu_loss.detach()),
            "loss_npu": float(npu_loss.detach().cpu()),
            "input_grad_norm": float(npu_grads[0].detach().cpu().norm())}


def run_isolated_vjps(device: torch.device) -> list[dict]:
    from tidegraph.isolated_aggregate import aggregate
    from tidegraph.isolated_linear import linear

    def linear_case(target):
        weight = torch.eye(4, dtype=torch.float32, device=target).requires_grad_()
        rows = [torch.arange(4, dtype=torch.float32, device=target).mul(i + 1).requires_grad_()
                for i in range(2)]
        values = linear(rows, weight)
        loss = sum(value.square().mean() for value in values)
        grads = torch.autograd.grad(loss, (weight, *rows), allow_unused=True)
        return values, grads, loss

    cpu_values, cpu_grads, _ = linear_case(torch.device("cpu"))
    npu_values, npu_grads, linear_loss = linear_case(device)
    for left, right in zip(cpu_values, npu_values):
        torch.testing.assert_close(left, right.detach().cpu(), atol=3e-4, rtol=3e-3)
    for left, right in zip(cpu_grads, npu_grads):
        torch.testing.assert_close(left, right.detach().cpu(), atol=3e-3, rtol=3e-2)

    def aggregate_case(target):
        atoms = [torch.arange(4, dtype=torch.float32, device=target).add(i).requires_grad_() for i in range(4)]
        scales = [torch.tensor(1 + i * .1, dtype=torch.float32, device=target).requires_grad_() for i in range(4)]
        coefficients = torch.tensor([.7, 1.2], dtype=torch.float32, device=target).requires_grad_()
        values = aggregate(atoms, scales, coefficients, 2)
        loss = sum(value.square().mean() for value in values)
        grads = torch.autograd.grad(loss, (coefficients, *atoms, *scales), allow_unused=True)
        return values, grads, loss

    cpu_values, cpu_grads, _ = aggregate_case(torch.device("cpu"))
    npu_values, npu_grads, aggregate_loss = aggregate_case(device)
    for left, right in zip(cpu_values, npu_values):
        torch.testing.assert_close(left, right.detach().cpu(), atol=3e-4, rtol=3e-3)
    for left, right in zip(cpu_grads, npu_grads):
        torch.testing.assert_close(left, right.detach().cpu(), atol=3e-3, rtol=3e-2)
    return [{"case": "isolated-linear-vjp", "status": "passed", "loss": float(linear_loss.detach().cpu())},
            {"case": "isolated-aggregate-vjp", "status": "passed",
             "loss": float(aggregate_loss.detach().cpu())}]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--device-index", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=["pdg-streaming", "timed-dag-frontier",
                                                         "timed-dag-diamond", "settle-generic",
                                                         "settle-layered", "settle-chain"])
    parser.add_argument("--with-backward", action="store_true")
    args = parser.parse_args()
    if not args.device.startswith("npu"):
        parser.error("this script is an explicit NPU smoke; use --device npu[:N]")
    device, reason = resolve_device(args.device, args.device_index)
    if device.type != "npu":
        parser.error("resolved device is not NPU")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    record = {"schema": "tide-npu-python-smoke-v1", "state": "running",
              "runtime": manifest(device, reason, torch.float32), "cases": []}
    (out / "smoke.json").write_text(json.dumps(record, indent=2) + "\n")
    try:
        for case in args.cases:
            row = run_case(case, device)
            record["cases"].append(row)
            (out / "smoke.json").write_text(json.dumps(record, indent=2) + "\n")
        if args.with_backward:
            record["cases"].append(run_backward(device))
            record["cases"].extend(run_isolated_vjps(device))
        record["state"] = "passed"
    except BaseException as error:
        record.update(state="failed", error=f"{type(error).__name__}: {error}")
        (out / "smoke.json").write_text(json.dumps(record, indent=2) + "\n")
        raise
    (out / "smoke.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

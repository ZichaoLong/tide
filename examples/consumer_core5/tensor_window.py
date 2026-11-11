#!/usr/bin/env python3
"""Installed public B×L client: AdamW, raw continuation, fresh-process resume.

Run from any working directory with the installed tidegraph package. The task
and loss are deliberately small, owned here rather than by the foundation API.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import torch
from tidegraph import Continuation, ExecutionPlacement
from tidegraph.runtime import resolve_device
from tidegraph.checkpoint import save, load
from tidegraph.comparison import collect_comparison, equivalent
from tidegraph.weighted import fully_connected, bind, KVConfig, RegionalConfig, eager_runner
from recording import snapshot, json_record, tensor_record


def create(args, *, reference=False):
    topology = fully_connected(2, 2)
    positions = dict(read_frequencies=(.17,), full_frequencies=((.11,), (.23,)),
                     distance_slopes=(.04, .03)) if args.positions else {}
    binding = bind(topology.graph, memories=KVConfig(heads=2, memory_read="qry", window=15, epsilon=.3, **positions),
                   regions=RegionalConfig(k=2), dtype=getattr(torch, args.dtype),
                   device="cpu" if reference else args.device)
    spec = replace(topology, graph=binding.graph)
    optimizer = torch.optim.AdamW(binding.model.parameters(), lr=.0001, foreach=False)
    execute = eager_runner(binding, "python" if reference else args.client,
        ExecutionPlacement("cpu" if reference or args.device == "cpu" else "mixed-c", scoring_dtype=args.dtype),
        native_library=None if reference else args.native_library,
        query_execution="legacy" if reference else "auto", trace=False)
    return binding, spec, optimizer, execute


def step(owner, state, index):
    binding, spec, optimizer, execute = owner
    ref = binding.model.nodes[0].bias
    x = torch.linspace(-.3+.01*index, .7, 24, dtype=ref.dtype).reshape(2, 3, 4).to(ref.device).requires_grad_()
    optimizer.zero_grad(set_to_none=True)
    result = execute.run_positions(spec, x, start_position=3*index, state=state)
    hidden = result.readout(x)
    loss = (hidden*torch.linspace(.2, .8, 4, dtype=ref.dtype, device=ref.device)).square().sum()
    loss.backward()
    gradients = {n: p.grad for n, p in binding.model.named_parameters()}
    if not torch.isfinite(loss) or any(not torch.isfinite(g).all() for g in gradients.values() if g is not None):
        raise FloatingPointError("nonfinite training result")
    state = result.continuation.detach()
    optimizer.step()
    observation = snapshot(dict(outputs=result.outputs, state=state, hidden=hidden, input_gradient=x.grad,
        gradients=gradients, parameters=binding.model.state_dict(), optimizer=optimizer.state_dict()))
    return state, observation, result.execution


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    p.add_argument("--client", choices=("python", "native"), default="python")
    p.add_argument("--native-library")
    p.add_argument("--positions", action="store_true", help="exercise relative rotary/distance KV columns")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--resume", type=Path)
    p.add_argument("--child", action="store_true")
    args = p.parse_args()
    resolve_device(args.device)
    torch.set_num_threads(1 if args.device == "cpu" else 2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    out = args.output_dir.resolve();out.mkdir(parents=True, exist_ok=False)
    identity = dict(interface="tensor-window-v1", device=args.device, dtype=args.dtype, client=args.client,
                    positions=args.positions)
    record = dict(identity=identity, execution="running", strict_numerical="not_checked", recovery="not_checked")
    try:
        owner = create(args)
        b, spec, opt, execute = owner
        state, start = None, 0
        if args.resume:
            meta = json.loads(args.resume.with_suffix(".json").read_text())
            if meta["identity"] != identity:
                raise ValueError("exact resume requires matching execution identity")
            state = load(args.resume, b.graph, b.model, opt)
            start = meta["steps"]
        oracle = None if args.child else create(args, reference=True)
        reference_state, reports = None, []
        for index in range(start, start+args.steps):
            state, actual, execution = step(owner, state, index)
            if execution["schedule"] != "settle-columns-v1":
                raise AssertionError(execution)
            if oracle is not None:
                reference_state, expected, _ = step(oracle, reference_state, index)
                report = collect_comparison(actual, expected, path=f"step{index}", check_device=False)
                if report["discrete"] != "passed":
                    raise AssertionError(report)
                reports.append(report)
        tensor_record(out/"observation.pt", actual)
        save(out/"checkpoint.pt", b.graph, b.model, state, opt)
        json_record(out/"checkpoint.json", dict(identity=identity, steps=start+args.steps))
        record.update(effective_execution=execution, comparisons=reports,
                      strict_numerical="not_checked" if oracle is None else
                      "passed" if all(r["strict_passed"] for r in reports) else "failed")
        if oracle is not None:
            command = [sys.executable, str(Path(__file__).resolve()), "--child", "--device", args.device,
                       "--dtype", args.dtype, "--client", args.client]
            if args.native_library:
                command += ["--native-library", args.native_library]
            if args.positions:
                command += ["--positions"]
            for name, extra in (("whole", ["--steps", "3"]), ("prefix", ["--steps", "2"]),
                    ("resumed", ["--steps", "1", "--resume", str(out/"prefix/checkpoint.pt")])):
                with (out/(name+".log")).open("w") as log:
                    subprocess.run([*command, *extra, "--output-dir", str(out/name)], check=True,
                                   stdout=log, stderr=subprocess.STDOUT, timeout=240)
            equivalent(torch.load(out/"whole/observation.pt", weights_only=False),
                       torch.load(out/"resumed/observation.pt", weights_only=False), atol=0, rtol=0)
            record["recovery"] = "passed"
        record["execution"] = "completed"
    except BaseException as error:
        record.update(execution="failed", error=repr(error));raise
    finally:
        json_record(out/"result.json", record)
    print(json.dumps({k: v for k, v in record.items() if k != "comparisons"}))


if __name__ == "__main__":
    main()

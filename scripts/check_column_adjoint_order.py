#!/usr/bin/env python3
"""Repeat one retained B×L graph's VJP without changing values or optimizer.

This isolates backward accumulation from forward/restart differences. Serial
autograd is an explicit diagnostic only. Fresh-process training continuation is
checked separately by consumer_core5/tensor_window.py or check_column_repeatability.
"""
import argparse
from dataclasses import replace
from pathlib import Path
import os
import subprocess
import _planning_path  # noqa: F401
import torch
from durable_records import write_json
from tidegraph import ExecutionPlacement
from tidegraph.comparison import collect_comparison
from tidegraph.runtime import resolve_device
from tidegraph.weighted import fully_connected, bind, KVConfig, RegionalConfig, eager_runner


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--client", choices=("python", "native"), default="python")
    p.add_argument("--native-library")
    p.add_argument("--placement", choices=("mixed-a", "mixed-b", "mixed-c"), default="mixed-a")
    p.add_argument("--scoring-dtype", choices=("float32", "float64"), default="float32")
    p.add_argument("--autograd-threads", choices=("default", "serial"), default="default")
    p.add_argument("--include-columns", action="store_true")
    p.add_argument("--repetitions", type=int, default=6)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.repetitions < 2:
        p.error("repetitions must be at least two")
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    resolve_device(args.device)
    torch.set_num_threads(1 if args.device == "cpu" else 2)
    torch.set_num_interop_threads(1)
    record = dict(state="running", request={**vars(args), "output_dir": str(out)},
        source=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        visibility=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"), affinity=sorted(os.sched_getaffinity(0)),
        scope="FC4 B2/L3/d4 FP32 KV qry; one fixed forward, first-order VJP repeatability only")
    write_json(out/"result.json", record)
    try:
        topology = fully_connected(2, 2)
        binding = bind(topology.graph, memories=KVConfig(heads=2, memory_read="qry", window=15, epsilon=.3),
                       regions=RegionalConfig(k=2), dtype=torch.float32, device=args.device)
        spec = replace(topology, graph=binding.graph)
        runner = eager_runner(binding, args.client,
            ExecutionPlacement("cpu" if args.device == "cpu" else args.placement, scoring_dtype=args.scoring_dtype),
            native_library=args.native_library, trace=False)
        x = torch.linspace(-.3, .7, 24).reshape(2, 3, 4).to(args.device).requires_grad_()
        result = runner.run_positions(spec, x)
        if result.execution["schedule"] != "settle-columns-v1":
            raise AssertionError(result.execution)
        loss = (result.readout(x)*torch.linspace(.2, .8, 4, device=args.device)).square().sum()
        roots = dict(binding.model.named_parameters()); roots["input"] = x
        if args.include_columns:
            for v, node in result.columns.nodes.items():
                roots.update({f"node{v}.describe.{k}": t for k, t in node.descriptors.items()})
                roots.update({f"node{v}.control": node.controls, f"node{v}.mass": node.window.mass,
                              f"node{v}.content": node.window.content})
                roots.update({f"node{v}.bank.{k}": getattr(node.window.bank, k)
                              for k in ("keys", "values", "weights", "summaries")})
        roots = {k: t for k, t in roots.items() if t.requires_grad}
        first, reports = None, []
        for _ in range(args.repetitions):
            with torch.autograd.set_multithreading_enabled(args.autograd_threads == "default"):
                grads = torch.autograd.grad(loss, tuple(roots.values()), allow_unused=True, retain_graph=True)
            actual = {k: None if g is None else g.detach().cpu() for k, g in zip(roots, grads)}
            if first is None:
                first = actual
            else:
                reports.append(collect_comparison(first, actual, atol=0, rtol=0))
        record.update(effective_execution=result.execution, comparisons=reports,
                      state="passed" if all(r["strict_passed"] for r in reports) else "failed")
    except BaseException as error:
        record.update(state="failed", error=repr(error))
        raise
    finally:
        write_json(out/"result.json", record)
    print(record["state"], out/"result.json")
    if record["state"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

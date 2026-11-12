#!/usr/bin/env python3
"""Repeatable LH direct/public/default operator and autograd diagnosis.

Uses the same material generator and task as benchmark_lh_tensor. This is an
instrumented forward/backward observation, never a throughput measurement or
independent numerical qualification. Native and Python graph counts overlap.
"""
import argparse
from contextlib import nullcontext
import hashlib
import os
from pathlib import Path
import json
import subprocess
import _planning_path  # noqa: F401
from durable_records import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--level", choices=("direct", "public", "default"), default="default")
    p.add_argument("--client", choices=("python", "native"), default="native")
    p.add_argument("--native-library", type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--profile", choices=("cpu", "npu"), default="cpu")
    p.add_argument("--trace-dir", type=Path, help="Required scratch for NPU traces")
    p.add_argument("--census", action="store_true")
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    if a.threads < 1 or a.level == "direct" and a.client != "python":
        p.error("positive threads; direct formulas use client python")
    if a.client == "native" and a.native_library is None:
        p.error("native requires an explicit matching library")
    if a.profile == "npu" and (not a.device.startswith("npu") or a.trace_dir is None):
        p.error("NPU profiling requires an explicit NPU device and scratch trace directory")
    import torch
    from benchmark_lh_tensor import material, ports, output_values, snapshot
    from layer_tensor_baseline import numerics
    from lh_tensor_baseline import StaticKV
    from tidegraph import ExecutionPlacement
    from tidegraph.runtime import resolve_device, manifest
    from tidegraph.planning.materialize import materialize
    from tidegraph.settle import SettleGraph
    from tidegraph.weighted import SettleColumnPlan
    from tidegraph.weighted.execution import eager_runner
    from tidegraph.experiment.work_observation import observe_work
    from tidegraph.experiment.finite import finite_tensors
    from tidegraph.autograd_census import Census, tensor_roots
    torch.set_num_threads(a.threads); torch.set_num_interop_threads(1)
    device, _ = resolve_device(a.device)
    policy = numerics(device, "ieee")
    out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    plan = json.loads(a.plan.read_text())
    binding, _, _ = materialize(plan, max_bytes=512*2**20, dtype=torch.float32, include_inputs=False)
    initial, model_hash, _ = snapshot(binding.model)
    binding.model.to(device)
    spec = SettleGraph(binding.graph, tuple(plan["graph"]["ranks"]), binding="ports", time_stride=plan["stride"])
    placement = ExecutionPlacement("cpu" if device.type == "cpu" else "mixed-c", scoring_dtype="float32")
    runner = (StaticKV(binding, spec) if a.level == "direct" else
              SettleColumnPlan(binding, spec, implementation=a.client, native_library=a.native_library, placement=placement)
              if a.level == "public" else eager_runner(binding, a.client, placement, native_library=a.native_library, trace=False))
    core = None
    if a.client == "native":
        from tidegraph.native_loader import load_native
        core = load_native(a.native_library, backend=device.type)
    x, support = material(plan, torch.float32)
    input_hash = hashlib.sha256(x.numpy().tobytes()+support.numpy().tobytes()).hexdigest()
    record = dict(schema="tide-lh-profile-v1", source=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                  plan=plan, initial_parameters_sha256=model_hash, input_sha256=input_hash,
                  numerical_policy=policy, affinity=sorted(os.sched_getaffinity(0)),
                  runtime=manifest(device, "explicit diagnostic device", torch.float32),
                  visibility=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"), diagnostic_only=True,
                  scope="One reset warmup then instrumented task forward/backward and finite checks; no optimizer, timing or equivalence claim.",
                  execution="running")
    if a.client == "native":
        record["native_build"] = json.loads((a.native_library/"build-manifest.json").read_text())
    write_json(out/"result.json", record)

    def synchronize():
        if device.type != "cpu":
            getattr(torch, device.type).synchronize(device)

    def step(census=None):
        hx, hs = material(plan, torch.float32)
        with torch.profiler.record_function("task::prepare"):
            value = hx.to(device).requires_grad_()
            layout, payload = ports(value, hs)
        with torch.profiler.record_function("task::forward"):
            if a.level == "direct":
                result = runner(value, support=hs.to(device))
                y = result["output_controls"][..., None]*result["output_contents"]
                state = result["states"]
                work = dict(zip(("candidate_events", "selected_events", "visited_edges"), result["counts"].cpu().tolist()))
                work["physical_full_rows"] = result["physical_full_rows"]
            else:
                result = runner.run_ports(layout, payload) if a.level == "public" else runner.run_ports(spec, layout, payload, stop_position=plan["positions"])
                y = output_values(result, value, len(binding.graph.outputs))
                state = result.states if a.level == "public" else result.continuation
                work = dict(result.stats)
            loss = y.square().sum()*1e-6
        if census is not None:
            roots = list(tensor_roots((result, value, y, loss)))
            census.register(roots, label="outer/returned-and-task")
            if core is not None:
                core.register_autograd_roots("outer/returned-and-task", roots)
        with torch.profiler.record_function("task::finite-forward"):
            finite_tensors(dict(y=y, state=state, loss=loss), packing="packed")
        with torch.profiler.record_function("task::backward"):
            loss.backward()
        with torch.profiler.record_function("task::finite-gradients"):
            finite_tensors([p.grad for p in binding.model.parameters()], packing="packed")
        if census is not None:
            record["census"] = dict(python=census.snapshot(), native=None if core is None else dict(core.autograd_census_metrics()),
                                    scope="Overlapping node universes; native explicit saves are not all ATen saved tensors. Never add counts.")
        return work

    try:
        step(); synchronize()
        binding.model.load_state_dict(initial); binding.model.zero_grad(set_to_none=True)
        synchronize()
        if a.profile == "cpu":
            profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True)
        else:
            import torch_npu
            a.trace_dir.mkdir(parents=True, exist_ok=False)
            profiler = torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                record_shapes=True, on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(a.trace_dir.resolve()), analyse_flag=True))
        if core is not None and a.census:
            core.reset_autograd_census(True)
        with profiler, Census() if a.census else nullcontext() as census, observe_work(a.client) as work:
            record["actual_work"] = step(census)
            synchronize()
        record["packing"] = work
        if a.profile == "cpu":
            rows = [dict(name=e.key, count=e.count, self_cpu_us=e.self_cpu_time_total,
                         total_cpu_us=e.cpu_time_total, shapes=e.input_shapes) for e in profiler.key_averages(group_by_input_shape=True)]
            write_json(out/"operators.json", rows)
        record["execution"] = "completed"
    except BaseException as error:
        record.update(execution="failed", error=repr(error)); raise
    finally:
        if core is not None and a.census:
            core.reset_autograd_census(False)
        write_json(out/"result.json", record)


if __name__ == "__main__":
    main()

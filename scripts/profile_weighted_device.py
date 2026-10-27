#!/usr/bin/env python3
"""Finite candidate-only core-5 operator placement and complete training profile.

One warmup and one profiled step, separate from parity and throughput processes.
Question: which device kernels, explicit copies and host scalar operations does
the requested weighted eager placement actually execute? No performance ranking
is inferred from a tiny graph or from profiling overhead.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import torch
from durable_records import write_json
from source_identity import source_state


def inspect_trace(path):
    document = json.loads(path.read_text())
    events = document if isinstance(document, list) else document["traceEvents"]
    hardware = {e["pid"] for e in events if e.get("name") == "process_name"
                and e.get("args", {}).get("name") == "Ascend Hardware"}
    kernels = [e for e in events if e.get("ph") == "X" and
               ("kernel" in str(e.get("cat", "")).lower() or
                e.get("pid") in hardware and str(e.get("args", {}).get("Task Type", "")).startswith("KERNEL"))]
    if not kernels:
        raise RuntimeError("profile has no actual accelerator kernels")
    fallback = [e.get("name") for e in events if any(x in str(e.get("name", "")).lower()
                for x in ("cpu_fallback", "fallback_to_cpu", "fallback to cpu"))]
    if fallback:
        raise RuntimeError("unexpected host fallback events: "+repr(fallback[:10]))
    transfers = Counter(e.get("name") for e in events if any(x in str(e.get("name", "")).lower()
                        for x in ("memcpy", "mem_copy", "_local_scalar_dense")))
    return dict(kernel_events=len(kernels), kernel_names=dict(Counter(e["name"] for e in kernels)),
                kernel_types=dict(Counter(e.get("args", {}).get("Task Type", "cuda") for e in kernels)),
                transfer_and_scalar_events=dict(transfers), observed_host_fallback_events=fallback,
                trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--implementation", choices=("python", "native"), required=True)
    parser.add_argument("--native-library")
    parser.add_argument("--placement", choices=("mixed-a", "mixed-b", "mixed-c"), required=True)
    parser.add_argument("--case", default="heterogeneous-attention")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/"python"))
    from tidegraph import Continuation
    from tidegraph.runtime import resolve_device, manifest, synchronize
    from tidegraph.compare import objective
    from tidegraph.placement import ExecutionPlacement
    from tidegraph.qualification_checks import coverage, plain, finite
    from weighted_device_cases import CASES, fixture, inputs_to
    from weighted_device_checks import runner, check_live
    cases = {c.name: c for c in CASES}
    if args.case not in cases:
        parser.error("unknown case")
    if args.implementation == "native" and not args.native_library:
        parser.error("native requires --native-library")
    if args.native_library:
        args.native_library = str(Path(args.native_library).resolve())
    device, reason = resolve_device(args.device)
    if device.type not in {"cuda", "npu"}:
        parser.error("profile requires an explicit accelerator")
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    placement = ExecutionPlacement(preset=args.placement)
    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=False)
    source, dirty = source_state(root)
    record = dict(state="running", schema="tide-weighted-device-profile-v1", source=source, dirty=dirty,
                  argv=sys.argv[1:], case=asdict(cases[args.case]), implementation=args.implementation,
                  runtime=manifest(device, reason, torch.float32), requested_placement=placement.to_dict(),
                  dispatch_environment={name: os.environ.get(name) for name in
                      ("ASCEND_LAUNCH_BLOCKING", "CUDA_LAUNCH_BLOCKING", "TASK_QUEUE_ENABLE")},
                  resolved_placement={k: str(v) for k, v in placement.resolve(device).items()},
                  warmup_steps=1, profiled_steps=1, tracking="project JSON and raw profiler artifacts",
                  question="actual weighted eager kernel placement, copies and host scalar decisions",
                  scope="input preparation/upload, schedule/packing, forward/loss, backward/finite checks, AdamW and sync",
                  excluded="model construction, fixture generation, optimizer construction, warmup and compilation",
                  caveat="finite small graph; profile is not a throughput result or exhaustive absence-of-fallback proof")
    write_json(out/"result.json", record)
    try:
        binding, cpu_inputs, stop = fixture(cases[args.case])
        torch.save(dict(weights=binding.model.state_dict(), inputs=plain(cpu_inputs)), out/"fixture.pt")
        record["fixture_sha256"] = hashlib.sha256((out/"fixture.pt").read_bytes()).hexdigest()
        binding.model.to(device)
        execute = runner(binding, args.implementation, placement, args.native_library)
        opt = torch.optim.AdamW(binding.model.parameters(), lr=.001)
        def step():
            xs = inputs_to(cpu_inputs, device)
            opt.zero_grad(set_to_none=True)
            r = execute(Continuation(binding.graph.identity, 2), xs, stop, sealed_until=stop)
            loss = objective(r)
            loss.backward()
            checks = [torch.isfinite(p.grad).all() for p in binding.model.parameters() if p.grad is not None]
            if not torch.stack(checks).all():
                raise FloatingPointError("nonfinite weighted gradient")
            opt.step(); synchronize(device)
            return r
        step()
        # Export-only CANN profiling can create cwd scratch files.
        os.chdir(out)
        if device.type == "npu":
            import torch_npu
            profiler = torch_npu.profiler
            profiler.tensorboard_trace_handler(str(out/"raw"), analyse_flag=False)
            activities = [profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU]
        else:
            profiler = torch.profiler
            activities = [profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA]
        with profiler.profile(activities=activities, record_shapes=True) as prof:
            result = step()
        check_live(result, placement, device); finite(result)
        prof.export_chrome_trace(str(out/"trace.json"))
        record.update(inspect_trace(out/"trace.json"), state="passed", workload=coverage(result, binding.graph),
                      candidate_updates=len(result.trace), full_calls=sum(e["active"] for e in result.trace),
                      messages=len(result.messages), actual_stats=result.stats)
        if source_state(root) != (source, dirty):
            raise RuntimeError("source changed during profile")
    except BaseException as error:
        record.update(state="failed", error=repr(error)); raise
    finally:
        write_json(out/"result.json", record)
    print(json.dumps(dict(state=record["state"], kernel_events=record["kernel_events"])))


if __name__ == "__main__":
    main()

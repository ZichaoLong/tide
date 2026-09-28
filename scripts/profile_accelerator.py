#!/usr/bin/env python3
"""Candidate-only eager forward/backward/update trace, separate from CPU parity."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import torch
from durable_records import write_json
from source_identity import source_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--implementation", choices=("python", "native"), required=True)
    parser.add_argument("--native-library")
    parser.add_argument("--case", default="mixed-timed-dag")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "python"))
    from tidegraph import GraphRuntime
    from tidegraph.qualification_inputs import Probe
    from tidegraph.qualification_checks import probe_loss, assert_placement
    from tidegraph.qualification_training import optimizer
    from accelerator_cases import cases, fixture_options
    available = dict(cases(args.implementation))
    if args.case not in available:
        parser.error("unknown case")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    runtime = GraphRuntime(available[args.case], device=args.device, native_library=args.native_library)
    if runtime.device.type not in {"cuda", "npu"}:
        parser.error("profiling requires an explicit accelerator")
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    source, dirty = source_state(root)
    report = dict(schema="tide-device-profile-v1", state="running", source=source, dirty=dirty,
                  case=args.case, runtime=runtime.manifest(), scope="candidate-only warm eager forward/backward/AdamW",
                  caveat="device kernels and host transfers are evidence for this case; not exhaustive absence of fallback")
    write_json(out / "result.json", report)
    try:
        # CANN's export-only mode creates a working-directory scratch tree even
        # with an explicit trace handler. Keep it out of immutable source.
        os.chdir(out)
        if runtime.device.type == "npu":
            import torch_npu
            profiler = torch_npu.profiler
            # The handler fixes the raw output location before profile.start().
            profiler.tensorboard_trace_handler(str(out / "raw"), analyse_flag=False)
            activities = [profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU]
        else:
            profiler = torch.profiler
            activities = [profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA]
        api = getattr(torch, runtime.device.type)
        stream = api.Stream(device=runtime.device)
        stream.wait_stream(api.current_stream(runtime.device))
        session = runtime.session(1)
        probe = Probe(runtime.config, batch_size=1, positions=2,
                      **fixture_options(args.case, runtime.config)).clone(runtime.device)
        opt = optimizer(runtime, "adamw")
        def step(cycle):
            opt.zero_grad(set_to_none=True)
            result = probe.advance(session, cycle=cycle)
            probe_loss(result).backward()
            session.detach()
            opt.step()
            return result
        with api.stream(stream):
            step(0)
            runtime.synchronize()
            with profiler.profile(activities=activities, record_shapes=True) as prof:
                result = step(1)
                runtime.synchronize()
        api.current_stream(runtime.device).wait_stream(stream)
        assert_placement(result, runtime.device)
        assert_placement([p.grad for p in runtime.model.parameters()], runtime.device)
        trace = out / "trace.json"
        prof.export_chrome_trace(str(trace))
        # Vendor export decorators may swallow errors. A real trace is mandatory.
        document = json.loads(trace.read_text())
        events = document if isinstance(document, list) else document["traceEvents"]
        device_pids = {e["pid"] for e in events if e.get("name") == "process_name"
                       and e.get("args", {}).get("name") == "Ascend Hardware"}
        kernels = [e for e in events if e.get("ph") == "X" and
                   ("kernel" in str(e.get("cat", "")).lower() or
                    (e.get("pid") in device_pids and
                     str(e.get("args", {}).get("Task Type", "")).startswith("KERNEL")))]
        if not kernels:
            raise RuntimeError("profile contains no accelerator kernel events")
        fallbacks = [e.get("name") for e in events if any(term in str(e.get("name", "")).lower()
                     for term in ("cpu_fallback", "fallback_to_cpu", "fallback to cpu"))]
        if fallbacks:
            raise RuntimeError("CPU fallback events: " + repr(fallbacks[:10]))
        if source_state(root) != (source, dirty):
            raise RuntimeError("source changed during profiling")
        transfers = Counter(e.get("name") for e in events if any(term in str(e.get("name", "")).lower()
                            for term in ("memcpy", "mem_copy", "_local_scalar_dense")))
        report.update(state="passed", kernel_events=len(kernels),
                      kernel_names=dict(Counter(e["name"] for e in kernels)),
                      kernel_types=dict(Counter(e.get("args", {}).get("Task Type", "cuda") for e in kernels)),
                      transfer_and_scalar_events=dict(transfers), nondefault_stream=True,
                      observed_fallback_events=fallbacks)
    except BaseException as error:
        report.update(state="failed", error=repr(error))
        raise
    finally:
        write_json(out / "result.json", report)
    print(json.dumps(dict(state=report["state"], kernel_events=report["kernel_events"])))


if __name__ == "__main__":
    main()

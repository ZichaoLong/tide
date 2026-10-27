#!/usr/bin/env python3
"""One finite complete-return core-5 measurement or separate profiler process.

Measure whether CPU streaming/prefill, eager placement or resident execution
fits the requested small/medium workload. Every timed window includes input
preparation, packing/scheduling, full CPU result, and (for training) loss, VJP,
finite checks, SGD, continuation detach and synchronization. No reference
trajectory, chosen actions or timing result enters a candidate.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import sys
import time
import torch
from durable_records import write_json
from source_identity import source_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--implementation", choices=("python", "native", "resident"), required=True)
    parser.add_argument("--placement", choices=("native", "mixed-a", "mixed-b", "mixed-c"), default="native")
    parser.add_argument("--schedule", choices=("streaming", "prefill"), default="prefill")
    parser.add_argument("--mode", choices=("inference", "training"), default="training")
    parser.add_argument("--topology", choices=("hb", "dag", "pdg"), default="hb")
    parser.add_argument("--family", choices=("kv", "ssd"), default="kv")
    parser.add_argument("--size", choices=("small", "medium"), default="small")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--native-library")
    parser.add_argument("--resident-library")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-dirty", action="store_true", help="entry-point development only, never controlled evidence")
    args = parser.parse_args()
    if not 1 <= args.steps <= 8:
        parser.error("steps must be 1..8")
    if args.implementation != "python" and not args.native_library:
        parser.error("native/resident requires --native-library")
    if args.implementation == "resident" and (not args.resident_library or args.placement != "native"):
        parser.error("resident requires its explicit library and no eager placement override")
    root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root/"python"))
    from tidegraph.runtime import resolve_device, manifest, synchronize
    from tidegraph.compare import equivalent
    from tidegraph.qualification_checks import finite
    from weighted_benchmark_cases import Workload, fixture, fingerprint
    from weighted_benchmark_runners import Eager, Resident, work
    device, reason = resolve_device(args.device)
    if args.placement != "native" and device.type == "cpu":
        parser.error("mixed placement requires an accelerator")
    if args.implementation == "resident" and device.type == "cpu":
        parser.error("resident requires an accelerator")
    source, dirty = source_state(root)
    if dirty and not args.allow_dirty:
        parser.error("controlled measurement requires clean immutable source")
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    shape = dict(width=4, samples=1, positions=2) if args.size == "small" else dict(width=32, samples=4, positions=4)
    config = Workload(args.topology, args.family, seed=args.seed, **shape)
    binding, inputs, span = fixture(config)
    expected_binding, expected_inputs, _ = fixture(config)
    fixture_hash = fingerprint(binding, inputs, config)
    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=False)
    now = lambda: datetime.now(timezone.utc).isoformat()
    record = dict(schema="tide-weighted-benchmark-v1", state="running", source=source, dirty=dirty,
        started=now(), argv=sys.argv[1:], config=asdict(config), fixture_sha256=fixture_hash,
        runtime=manifest(device, reason, torch.float32), implementation=args.implementation,
        mode=args.mode, schedule=args.schedule, placement=args.placement,
        workload_span=span, warmup_windows=1, measured_windows=1 if args.profile else args.steps,
        kind="profile" if args.profile else "benchmark", tracking="project JSON; execution-flows §1",
        timing="complete CPU return; input prepare/upload, packing, schedule, forward/loss, backward, finite check, SGD, detach, sync",
        excluded="fixture/model/owner/optimizer construction, warmup, validation, final parameter export, profiler parsing",
        load_average_before=list(os.getloadavg()), host_arch=platform.machine(),
        caveat="one process per configuration; evolving continuation; no statistical or universal speed claim",
        dispatch_environment={k: os.environ.get(k) for k in ("ASCEND_LAUNCH_BLOCKING", "CUDA_LAUNCH_BLOCKING", "TASK_QUEUE_ENABLE")})
    write_json(out/"result.json", record)
    candidate = None
    try:
        start = time.perf_counter()
        cls = Resident if args.implementation == "resident" else Eager
        candidate = cls(binding, inputs, span, config, args, device)
        synchronize(device); record["construction_seconds"] = time.perf_counter()-start
        record["effective"] = candidate.manifest()
        actual = []; metrics = []
        start = time.perf_counter(); actual.append(candidate.step(0))
        record["warmup_seconds"] = time.perf_counter()-start
        accelerator = getattr(torch, device.type) if device.type != "cpu" else None
        if accelerator is not None:
            accelerator.reset_peak_memory_stats(device)
        count = record["measured_windows"]
        for step in range(1, count+1):
            if args.profile:
                if device.type == "npu":
                    import torch_npu
                    profiler = torch_npu.profiler
                    activities = [profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU]
                else:
                    profiler = torch.profiler
                    activities = [profiler.ProfilerActivity.CPU]
                    if device.type == "cuda": activities.append(profiler.ProfilerActivity.CUDA)
                os.chdir(out)  # CANN export scratch files belong under artifacts.
                with profiler.profile(activities=activities, record_shapes=True) as prof:
                    actual.append(candidate.step(step))
                prof.export_chrome_trace(str(out/"trace.json"))
                if device.type != "cpu":
                    from profile_weighted_device import inspect_trace
                    record["profile"] = inspect_trace(out/"trace.json")
                else:
                    from source_identity import digest
                    record["profile"] = dict(trace_sha256=digest(out/"trace.json"),
                        operators=[dict(name=x.key, calls=x.count, self_cpu_us=x.self_cpu_time_total)
                                   for x in prof.key_averages()])
            else:
                synchronize(device); start = time.perf_counter()
                actual.append(candidate.step(step))
                elapsed = time.perf_counter()-start
                event = dict(step=step, seconds=elapsed, loss=actual[-1][1], **work(actual[-1][0]))
                metrics.append(event); write_json(out/"metrics.json", metrics)
        parameters = candidate.parameters()
        record["memory"] = dict(process_lifetime_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        if accelerator is not None:
            record["memory"].update(device_peak_allocated_bytes=accelerator.max_memory_allocated(device),
                                    device_peak_reserved_bytes=accelerator.max_memory_reserved(device))
        # Only after the measurement: independent scalar CPU schedule and
        # full-record/optimizer check. It supplies no execution hints.
        reference = Eager(expected_binding, expected_inputs, span, config, args, torch.device("cpu"), reference=True)
        for step, (result, loss) in enumerate(actual):
            expected, expected_loss = reference.step(step)
            finite(result); equivalent(result, expected, check_device=False)
            torch.testing.assert_close(torch.tensor(loss), torch.tensor(expected_loss), rtol=1e-5, atol=1e-6)
        if parameters is not None:
            for name, value in reference.parameters().items():
                torch.testing.assert_close(parameters[name], value, rtol=1e-5, atol=1e-6)
        record.update(state="passed", post_measurement_complete_records=True,
                      parameter_values_checked=parameters is not None,
                      optimizer_values_checked=args.mode == "training" and parameters is not None,
                      metrics=metrics, load_average_after=list(os.getloadavg()))
        if metrics:
            times = [x["seconds"] for x in metrics]
            record["timing_summary"] = dict(total_seconds=sum(times), median_seconds=statistics.median(times),
                min_seconds=min(times), max_seconds=max(times),
                candidate_updates_per_second=sum(x["candidate_updates"] for x in metrics)/sum(times))
        if source_state(root) != (source, dirty):
            raise RuntimeError("source changed during measurement")
    except BaseException as error:
        record.update(state="failed", error=repr(error)); raise
    finally:
        if candidate is not None:
            try:
                candidate.close()
            except BaseException as error:
                record.update(state="failed", cleanup_error=repr(error)); raise
            finally:
                record["finished"] = now(); write_json(out/"result.json", record)
        else:
            record["finished"] = now(); write_json(out/"result.json", record)
    print(json.dumps({k: record[k] for k in ("state", "kind", "fixture_sha256")}))


if __name__ == "__main__":
    main()

"""Finite untimed KV/SSD/pending metadata diagnosis of real inference state.

This is separate from normal performance samples. Every call continues actual
state; no intermediate flush is inserted and no task head is simulated.
"""
import argparse
import json
import os
from pathlib import Path
import resource

import torch

from benchmarks.cache_metadata import CacheMetadataDiagnostic
from benchmarks.inference import (InferenceSchedule, _placement, bounded_parameter_identity,
                                  run_request, source_identity)
from benchmarks.workloads import paired_input, tensor_hash
from scripts.job import atomic
from tide import ModelConfig, TideModel
from tide.precision import precision_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-overrides", type=json.loads, default={})
    device = parser.add_mutually_exclusive_group()
    device.add_argument("--device", default=None)
    device.add_argument("--devices", default=None, help="comma-separated explicitly admitted devices")
    parser.add_argument("--placement", choices=("single", "layer", "region"), default="single")
    parser.add_argument("--transport", choices=("auto", "direct", "host_stage", "synchronized"), default="auto")
    parser.add_argument("--region-concurrency", choices=("serial", "per_device"), default="serial")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--prefill", type=int, default=128)
    parser.add_argument("--prefill-chunk", type=int, default=32)
    parser.add_argument("--decode", type=int, default=256)
    parser.add_argument("--stream-positions", type=int, default=0)
    parser.add_argument("--stream-chunk", type=int, default=128)
    parser.add_argument("--sample-every", type=int, default=64)
    parser.add_argument("--require-kv-eviction", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=751)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not isinstance(args.config_overrides, dict):
        parser.error("config overrides must be a JSON object")
    if min(args.batch, args.threads, args.sample_every) < 1:
        parser.error("batch, threads and sampling interval must be positive")
    if args.output.exists():
        parser.error("choose a new output identity")
    if args.region_concurrency != "serial" and args.placement != "region":
        parser.error("per_device concurrency requires region placement")
    try:
        schedule = InferenceSchedule(args.prefill, args.prefill_chunk, args.decode,
                                     args.stream_positions, args.stream_chunk)
    except ValueError as error:
        parser.error(str(error))
    device_names = (args.devices or args.device or "cpu").split(",")
    if any(name.startswith("npu") for name in device_names):
        import torch_npu
        torch.npu.set_device(device_names[0])
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
    devices = tuple(torch.device(name) for name in device_names)
    if len({d.type for d in devices}) != 1 or devices[0].type not in {"cpu", "cuda", "npu"}:
        parser.error("devices must use one CPU/CUDA/NPU backend")
    if args.placement == "single" and len(devices) != 1:
        parser.error("multiple devices require layer or region placement")
    torch.set_num_threads(args.threads)
    config = ModelConfig.from_dict(json.loads(args.config.read_text()) | args.config_overrides)
    record = dict(schema="tide-inference-cache-metadata-1", source=source_identity(),
                  arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                  config=config.to_dict(), precision=precision_policy(devices[0], args.precision),
                  resource_identity=os.environ.get("TIDE_RESOURCE_IDENTITY", "unspecified"),
                  physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                  cpu_affinity=sorted(os.sched_getaffinity(0)), state="running", stage="construct",
                  timing="all work is intrusive diagnosis; no performance samples or throughput claim",
                  boundary="zero initial state; real full finite schedule; sole final Timed drain; sparse cache snapshots with bounded positive-write ledger")
    atomic(args.output, record)
    try:
        torch.manual_seed(args.seed)
        model = TideModel(config)
        record["parameters"] = bounded_parameter_identity(model)
        if not record["parameters"]["finite"]:
            raise ArithmeticError("initial parameters are nonfinite")
        model, record["placement"] = _placement(model, args, devices)
        model.eval()
        x_cpu = paired_input(args.batch, schedule.positions, config.input_dim or config.d, args.seed+1)
        record["input_sha256"], record["input_shape"] = tensor_hash(x_cpu), list(x_cpu.shape)
        x = x_cpu.to(devices[0]); del x_cpu
        def persist_snapshot(observer):
            record.update(stage="diagnostic", completed_calls=observer.calls,
                          sampled_boundaries=observer.snapshots,
                          write_ledger_failure_count=observer.ledger.failure_count)
            atomic(args.output, record)
        observer = CacheMetadataDiagnostic(model, schedule, args.batch, sample_every=args.sample_every,
                                           on_snapshot=persist_snapshot)
        with observer.ledger:
            state, boundary = run_request(model, x, schedule, precision=args.precision, observer=observer, timing=False)
        diagnosis = observer.report(state, require_kv_eviction=args.require_kv_eviction)
        record.pop("sampled_boundaries", None)
        record["diagnosis"] = diagnosis
        record["actual_calls"] = [{key: call[key] for key in ("phase", "start", "stop", "input_positions", "state_before", "state_after", "output_slots")}
                                  for call in boundary["calls"]]
        record.update(state="passed" if diagnosis["passed"] else "check_failed", stage="complete",
                      process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        atomic(args.output, record)
        print(json.dumps(dict(state=record["state"], output=str(args.output),
            sampled_boundaries=len(diagnosis["snapshots"]), kv_owner_samples_ever_full=diagnosis["kv_owner_samples_ever_full"],
            verified_eviction_owners=diagnosis["kv_owner_samples_with_verified_eviction"])))
        return 0 if diagnosis["passed"] else 1
    except Exception as error:
        record.update(state="execution_failed", error=f"{type(error).__name__}: {error}",
                      process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        atomic(args.output, record)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

"""Supplementary fixed-depth, near-total-parameter cached inference.

This is not a replacement for A-D. Reuse their Transformer, matching rule,
request execution, diagnostics and timing, without constructing a Tide model.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import resource

import torch

from benchmarks.baselines import TransformerBaseline, parameter_count
from benchmarks.inference import (
    InferenceSchedule, _canonical_hash, _check_budget, _clear_model_cache,
    _persist, _placement, add_availability, allocator_stats,
    bounded_parameter_identity, diagnose_request, execution_devices,
    run_request, source_identity, synchronize_all,
)
from benchmarks.matching import match_transformer
from benchmarks.workloads import paired_input, tensor_hash
from tide.precision import precision_policy


def target_reference(path, parameters, schedule, *, batch, seed, precision):
    """An earlier Tide record supplies the target/workload, not new qualification."""
    loaded = json.loads(path.read_text())
    reference = loaded.get("matching_reference", loaded)
    payload = {k: v for k, v in reference.items() if k != "reference_sha256"}
    if reference.get("schema") != "tide-inference-reference-1" or _canonical_hash(payload) != reference.get("reference_sha256"):
        raise ValueError("invalid target reference schema or digest")
    if reference["audit"]["parameters"]["unique_parameters"] != parameters:
        raise ValueError("target parameter count differs from reference")
    expected = dict(schedule=asdict(schedule), batch=batch, seed=seed, precision=precision)
    if any(reference.get(k) != v for k, v in expected.items()):
        raise ValueError("requested workload differs from target reference")
    return dict(path=str(path), file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                reference_sha256=reference["reference_sha256"], source=reference["source"],
                tide_input_width=reference["tide_input_width"],
                tide_input_sha256=reference["tide_input_sha256"],
                scope="reuse total parameter target and exact input schedule; no new Tide timing or numerical qualification")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=int, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--target-reference", type=Path)
    parser.add_argument("--devices", default="cpu")
    parser.add_argument("--transport", choices=("auto", "direct", "host_stage", "synchronized"), default="auto")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--prefill", type=int, default=128)
    parser.add_argument("--prefill-chunk", type=int, default=32)
    parser.add_argument("--decode", type=int, default=16)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--backend", choices=("auto", "math", "tiled", "sdpa", "npu_fusion"), default="auto")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=751)
    parser.add_argument("--allocated-budget-gib", type=float)
    parser.add_argument("--reserved-budget-gib", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.parameters, args.layers, args.batch, args.samples, args.threads, args.window, args.chunk) < 1 or args.warmup < 0:
        parser.error("counts must be positive; warmup may be zero")
    for value in (args.allocated_budget_gib, args.reserved_budget_gib):
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error("memory budgets must be finite and positive")
    if args.output.exists():
        parser.error("choose a new output identity")
    try:
        schedule = InferenceSchedule(args.prefill, args.prefill_chunk, args.decode)
        devices = tuple(torch.device(name) for name in args.devices.split(","))
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    if not devices or len({d.type for d in devices}) != 1 or devices[0].type not in {"cpu", "cuda", "npu"}:
        parser.error("devices must use one CPU/CUDA/NPU backend")
    reference = (target_reference(args.target_reference, args.parameters, schedule,
                 batch=args.batch, seed=args.seed, precision=args.precision) if args.target_reference else None)
    if devices[0].type == "npu":
        import torch_npu
        torch.npu.set_device(devices[0])
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
    torch.set_num_threads(args.threads)
    selected = match_transformer(args.layers, args.parameters, window=args.window)
    record = dict(schema="tide-transformer-depth-results-1", source=source_identity(),
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        schedule=asdict(schedule), precision=precision_policy(devices[0], args.precision),
        comparison=dict(kind="supplementary_fixed_depth_total_parameters", match=selected,
                        target_reference=reference, replaces_joint_classes=False),
        resource_identity=os.environ.get("TIDE_RESOURCE_IDENTITY", "unspecified"),
        physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"), torch=str(torch.__version__),
        cpu_affinity=sorted(os.sched_getaffinity(0)), torch_threads=torch.get_num_threads(),
        host_load_before=os.getloadavg(), state="running", last_stage="construct", observations=[],
        request_boundary="zero state per diagnostic/warmup/sample; real KV continuation within each request",
        timing_scope="same run_request as A-D; warm resident input; excludes construction, hashing, transfer and diagnostics; no cold measurement")
    _persist(args.output, record)
    try:
        torch.manual_seed(args.seed)
        model = TransformerBaseline(**selected["constructor"], chunk=args.chunk, backend=args.backend)
        record["parameters"] = parameter_count(model)
        if record["parameters"] != selected["actual"]["parameters"]:
            raise AssertionError("actual Transformer parameters differ from match")
        record["initial_parameters"] = bounded_parameter_identity(model)
        if not record["initial_parameters"]["finite"]:
            raise ArithmeticError("nonfinite initial parameters")
        model, record["placement"] = _placement(model, args, devices)
        model.eval()
        record["spec"] = model.configuration()
        x_cpu = paired_input(args.batch, schedule.positions, model.d, args.seed+1)
        record["input_sha256"], record["input_shape"] = tensor_hash(x_cpu), list(x_cpu.shape)
        if reference:
            width = reference["tide_input_width"]
            if model.d < width or tensor_hash(x_cpu[..., :width]) != reference["tide_input_sha256"]:
                raise AssertionError("shared input prefix differs from original Tide request")
            record["shared_input_prefix"] = dict(width=width, sha256=reference["tide_input_sha256"])
        x = x_cpu.to(devices[0]); del x_cpu
        active = execution_devices(model, x)
        synchronize_all(active)
        record["last_stage"] = "diagnostic"
        _persist(args.output, record)
        diagnostic, _ = diagnose_request(model, x, schedule, precision=args.precision)
        record["diagnostics"] = diagnostic
        if not diagnostic["finite_outputs"] or not diagnostic["finite_private_state"]:
            raise ArithmeticError("nonfinite inference diagnostic")
        _clear_model_cache(active)
        record["last_stage"] = "warmup"
        _persist(args.output, record)
        for _ in range(args.warmup):
            state, _ = run_request(model, x, schedule, precision=args.precision)
            del state
        for index in range(args.samples):
            record["last_stage"] = f"sample:{index}"
            _persist(args.output, record)
            allocator_stats(active, reset=True)
            state, measured = run_request(model, x, schedule, precision=args.precision)
            measured["allocator"] = allocator_stats(active)
            del state
            add_availability(measured, diagnostic)
            record["observations"].append(measured)
            _persist(args.output, record)
            _check_budget(measured["allocator"], args)
        record.update(state="completed", last_stage="complete", host_load_after=os.getloadavg(),
                      process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        _persist(args.output, record)
        print(json.dumps(dict(state=record["state"], output=str(args.output))))
        return 0
    except Exception as error:
        record.update(state="failed", error=f"{type(error).__name__}: {error}",
                      process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        _persist(args.output, record)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

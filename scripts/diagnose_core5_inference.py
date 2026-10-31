#!/usr/bin/env python3
"""Locate unchanged inference failures using independent schedules and local anchors.

No timing qualification, replayed candidate trajectory, precision intervention,
or tolerance override. Explicit FP64 execution is a different configuration.
"""
import argparse
from dataclasses import asdict
import os
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"python"))
from durable_records import write_json
from source_identity import source_state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workload", type=Path, required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--implementation", choices=("python", "native"), required=True)
    p.add_argument("--native-library")
    p.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    p.add_argument("--scoring-dtype", choices=("float32", "float64"))
    p.add_argument("--placement", choices=("native", "cpu", "mixed-a", "mixed-b", "mixed-c"), default="mixed-c")
    p.add_argument("--schedule", choices=("streaming", "prefill"), default="prefill")
    p.add_argument("--packing", choices=("auto", "scalar", "packed"), default="packed")
    p.add_argument("--windows", type=int, choices=(1, 2), default=1)
    p.add_argument("--max-materialize-bytes", type=int, default=2*1024**3)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--allow-dirty", action="store_true", help="development diagnostic only")
    args = p.parse_args()
    args.scoring_dtype = args.scoring_dtype or args.dtype
    args.mode, args.profile, args.workers, args.peer_devices = "inference", False, 1, None
    if args.implementation == "native" and not args.native_library:
        p.error("native requires an explicit matching library")
    if args.native_library:
        args.native_library = str(Path(args.native_library).resolve())
    from tidegraph.planning import ExecutionSpec
    spec = ExecutionSpec(client="pytorch" if args.implementation == "python" else "python-native",
                         device=args.device, placement=args.placement, dtype=args.dtype,
                         scoring_dtype=args.scoring_dtype, schedule=args.schedule, packing=args.packing).record()
    if not (args.device == "cpu" or args.device.startswith("cuda:")):
        p.error("this bounded diagnostic accepts CPU or one explicit CUDA device")
    from tidegraph.planning.materialize import load, materialize
    from tidegraph.runtime import resolve_device, manifest
    from weighted_benchmark_runners import Eager, work
    from weighted_benchmark_observation import initialize
    from weighted_numeric_differences import first_trace_difference
    from diagnose_weighted_workload import metadata
    from core5_inference_difference import check, first_strict_trace, first_kv_witness
    from cuda_precision import ieee_policy
    source, dirty = source_state(ROOT)
    if dirty and not args.allow_dirty:
        p.error("controlled diagnostic requires clean immutable source")
    plan, config = load(args.workload.resolve())
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    os.chdir(out)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    report = dict(schema="tide-core5-inference-diagnostic-v1", state="running", source=source, dirty=dirty,
                  workload_id=plan["sha256"], model_id=plan["model_id"], config=asdict(config), execution=spec,
                  scope="independent complete inference and post-execution local FP64 anchors; no throughput or checkpoint qualification",
                  windows=[])
    write_json(out/"result.json", report)
    candidate = reference = None
    try:
        device, reason = resolve_device(args.device)
        report["cuda_precision"] = ieee_policy(device)
        initialize(args, device)
        report["runtime"] = manifest(device, reason, getattr(torch, args.dtype))
        create = lambda: materialize(plan, max_bytes=args.max_materialize_bytes, dtype=getattr(torch, args.dtype))
        bound, inputs, span = create()
        candidate = Eager(bound, inputs, span, config, args, device)
        del bound, inputs
        actual = [candidate.step(index)[0] for index in range(args.windows)]
        report["candidate"] = candidate.manifest()
        candidate.close(); candidate = None
        bound, inputs, span = create()
        reference = Eager(bound, inputs, span, config, args, torch.device("cpu"), reference=True)
        for index, result in enumerate(actual):
            expected = reference.step(index)[0]
            with torch.no_grad():
                row = dict(window=index, complete_error=check(result, expected),
                    discrete_error=check(metadata(result), metadata(expected)),
                    first_exact_trace=first_trace_difference(result, expected),
                    first_strict_trace=first_strict_trace(result, expected),
                    kv_write=first_kv_witness(bound, result, expected),
                    candidate_work=work(result), reference_work=work(expected))
            report["windows"].append(row)
            write_json(out/"result.json", report)
        report["state"] = "strict_mismatch" if any(row["complete_error"] for row in report["windows"]) else "observed_match"
    except BaseException as error:
        report.update(state="failed", error=repr(error))
        raise
    finally:
        if candidate is not None:
            candidate.close()
        if reference is not None:
            reference.close()
        write_json(out/"result.json", report)
    return 2 if report["state"] == "strict_mismatch" else 0


if __name__ == "__main__":
    raise SystemExit(main())

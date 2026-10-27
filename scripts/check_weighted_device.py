#!/usr/bin/env python3
"""Bounded core-5 eager target checks; independent CPU oracle, no trajectory prepass.

This checks host scheduling with actual payload/Read/control/selection devices.
It does not qualify resident progression. Use a frozen checkout and a durable
service for a complete run; requested absent backends fail before output creation.
"""
import argparse
from dataclasses import asdict
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
    parser.add_argument("--peer-devices", nargs="+", help="Explicit logical devices; first must match --device")
    parser.add_argument("--implementation", choices=("python", "native"), required=True)
    parser.add_argument("--native-library")
    parser.add_argument("--placement", choices=("cpu", "mixed-a", "mixed-b", "mixed-c"), required=True)
    parser.add_argument("--scoring-dtype", choices=("profile", "float32", "float64"), default="profile")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
    parser.add_argument("--loss-scale", type=float, help="Static FP16 loss scale (default 16); FP32 uses 1")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/"python"))
    from tidegraph.runtime import resolve_device, synchronize, manifest
    from tidegraph.placement import ExecutionPlacement
    from weighted_device_cases import CASES
    from weighted_device_checks import check_case
    from weighted_device_primitives import check as primitives
    from weighted_device_precision import Precision
    try:
        precision = Precision(getattr(torch, args.dtype), args.optimizer, args.loss_scale)
        if args.dtype == "float16" and args.scoring_dtype == "profile":
            raise ValueError("float16 payload requires explicit --scoring-dtype float32 or float64")
    except ValueError as error:
        parser.error(str(error))
    names = {c.name for c in CASES}
    if set(args.case)-names:
        parser.error("unknown case; available: "+", ".join(sorted(names)))
    if args.implementation == "native" and not args.native_library:
        parser.error("native requires --native-library with a matching build manifest")
    device, reason = resolve_device(args.device)
    peers = None
    if args.peer_devices:
        peers = tuple(resolve_device(spec)[0] for spec in args.peer_devices)
        if len(set(peers)) != len(peers) or len(peers) < 2 or peers[0] != device or any(d.type != device.type for d in peers):
            parser.error("peer devices must be distinct, one backend, first equal to --device")
    placement = ExecutionPlacement(preset=args.placement, scoring_dtype=args.scoring_dtype)
    resolved = placement.resolve(device)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=False)
    source, dirty = source_state(root)
    record = dict(state="running", source=source, dirty=dirty, implementation=args.implementation,
                  argv=sys.argv[1:], dispatch_environment={name: os.environ.get(name) for name in
                      ("ASCEND_LAUNCH_BLOCKING", "CUDA_LAUNCH_BLOCKING", "TASK_QUEUE_ENABLE")},
                  runtime=manifest(device, reason, precision.dtype), requested_placement=placement.to_dict(),
                  precision=precision.record(),
                  peer_devices=None if peers is None else [str(d) for d in peers],
                  resolved_placement={k: str(v) for k, v in resolved.items()}, cases=[],
                  scope="eager host schedule; actual device transfers; not resident",
                  tolerances=precision.tolerances, fallback_visibility="requires separate profile/log audit")
    write_json(out/"result.json", record)
    try:
        record["primitives"] = primitives(device, precision.dtype)
        write_json(out/"result.json", record)
        for case in CASES:
            if args.case and case.name not in args.case:
                continue
            entry = dict(configuration=asdict(case), state="running")
            record["cases"].append(entry); write_json(out/"result.json", record)
            path = out/case.name; path.mkdir()
            entry.update(check_case(case, device, args.implementation, placement, path,
                                    args.native_library, training=not args.skip_training, peers=peers,
                                    precision=precision), state="passed")
            for target in peers or (device,):
                synchronize(target)
            write_json(out/"result.json", record)
            print(json.dumps(dict(case=case.name, state="passed")), flush=True)
        if source_state(root) != (source, dirty):
            raise RuntimeError("source changed during device checks")
        record["state"] = "passed"
    except BaseException as error:
        record.update(state="failed", error=repr(error))
        if record["cases"]:
            record["cases"][-1].update(state="failed", error=repr(error))
        raise
    finally:
        write_json(out/"result.json", record)


if __name__ == "__main__":
    main()

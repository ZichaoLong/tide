#!/usr/bin/env python3
"""Fixed-source core-5 CUDA eager cases, independent clients, original assertions.

Every configuration has a separate process and retained result. A failed case
does not suppress unrelated cases; any failure makes this gate fail. This does
not certify resident execution, direct P2P, or performance.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from build_identity import source_hash
from durable_records import write_json
from source_identity import source_state, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--clients", nargs="+", choices=("pytorch", "python-native", "libtorch"), required=True)
    parser.add_argument("--core-build", type=Path)
    parser.add_argument("--placements", nargs="+", choices=("mixed-a", "mixed-b", "mixed-c"), default=["mixed-c"])
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
    parser.add_argument("--packing", choices=("packed", "scalar"), default="packed")
    parser.add_argument("--case", action="append", default=[], help="Python fixture; repeatable, omitted means all")
    parser.add_argument("--cpp-case", action="append", default=[], help="Standalone topology/family; omitted means all")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "python"))
    from weighted_device_cases import CASES
    from tidegraph.runtime import resolve_device, manifest
    from cuda_precision import ieee_policy
    import torch
    source, dirty = source_state(root)
    if dirty:
        parser.error("qualification requires a clean frozen checkout")
    device, reason = resolve_device(args.device)
    if device.type != "cuda":
        parser.error("CUDA device required")
    policy = ieee_policy(device)
    names = args.case or [c.name for c in CASES]
    if set(names) - {c.name for c in CASES}:
        parser.error("unknown Python case")
    cpp_all = [f"{top}/{family}" for top in ("fc", "hb", "nonaligned", "feedback")
               for family in ("kv", "affine", "delta", "gated_delta", "ssd", "heterogeneous")]
    cpp_names = args.cpp_case or cpp_all
    if set(cpp_names) - set(cpp_all):
        parser.error("unknown standalone case")
    core_record = None
    if set(args.clients) - {"pytorch"}:
        if args.core_build is None:
            parser.error("native clients require --core-build")
        args.core_build = args.core_build.resolve()
        core_record = json.loads((args.core_build / "build-manifest.json").read_text())
        if core_record["backend"] != "cuda" or core_record["cpp_source_sha256"] != source_hash(root):
            parser.error("core build backend/source differs")
        for name, expected in core_record["binary_sha256"].items():
            if digest(args.core_build / name) != expected:
                parser.error("core binary differs: " + name)
        if args.packing != "packed" and "libtorch" in args.clients:
            parser.error("standalone training checker declares packed execution")
    probe = torch.arange(4, device=device, dtype=torch.float32, requires_grad=True)
    probe.square().sum().backward()
    torch.testing.assert_close(probe.grad.cpu(), torch.arange(4) * 2, check_dtype=False, atol=0, rtol=0)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               TORCH_DEVICE_BACKEND_AUTOLOAD="0", NVIDIA_TF32_OVERRIDE="0")
    report = dict(schema="tide-core5-cuda-eager-v1", state="running", source=source, dirty=dirty,
                  runtime=manifest(device, reason, getattr(torch, args.dtype)), cuda_precision=policy,
                  core_build=core_record, cases=[], argv=sys.argv[1:],
                  scope="selected single-device core-5 eager cases; independent CPU schedules; no resident, P2P or performance claim",
                  threads=dict(intraop=1, interop=1, workers=1), packing=args.packing,
                  standalone_tf32_override=env["NVIDIA_TF32_OVERRIDE"])
    write_json(out / "result.json", report)
    for client in dict.fromkeys(args.clients):
        for placement in dict.fromkeys(args.placements):
            for case in dict.fromkeys(cpp_names if client == "libtorch" else names):
                label = f"{client}-{placement}-{case.replace('/', '-')}"
                dest = out / label
                if client == "libtorch":
                    command = [str(args.core_build / "tidegraph-weighted-training-check"),
                        "--device", str(device), "--dtype", args.dtype, "--optimizer", args.optimizer,
                        "--placement", placement, "--scoring-dtype", "float32", "--case", case,
                        "--output-dir", str(dest)]
                else:
                    command = [sys.executable, str(root / "scripts/check_weighted_device.py"),
                        "--device", str(device), "--dtype", args.dtype, "--optimizer", args.optimizer,
                        "--placement", placement, "--scoring-dtype", "float32", "--case", case,
                        "--implementation", "native" if client == "python-native" else "python",
                        "--workers", "1", "--packing", args.packing, "--output-dir", str(dest)]
                    if client == "python-native":
                        command += ["--native-library", str(args.core_build)]
                row = dict(client=client, case=case, placement=placement, state="running",
                           command=command, timeout_seconds=180)
                report["cases"].append(row)
                write_json(out / "result.json", report)
                logpath = out / (label + ".log")
                started = time.monotonic()
                with logpath.open("x") as log:
                    try:
                        code = subprocess.run(command, cwd=out, env=env, stdout=log,
                                              stderr=subprocess.STDOUT, timeout=180).returncode
                    except subprocess.TimeoutExpired:
                        code = 124
                row.update(state="passed" if code == 0 else "failed", exit_code=code,
                           elapsed_seconds=time.monotonic() - started, log_sha256=digest(logpath))
                if (dest / "result.json").exists():
                    result = json.loads((dest / "result.json").read_text())
                    row["result_sha256"] = digest(dest / "result.json")
                    if result["state"] != "passed":
                        row["state"] = "failed"
                write_json(out / "result.json", report)
                print(label, row["state"], flush=True)
    if source_state(root) != (source, dirty):
        raise RuntimeError("source changed during checks")
    report["state"] = "passed" if all(c["state"] == "passed" for c in report["cases"]) else "failed"
    write_json(out / "result.json", report)
    return 0 if report["state"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())

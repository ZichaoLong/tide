"""Bounded stable-mass qualification, reusing the existing training/restore checks.

Submit from an immutable commit with scripts/launch.py. This driver does not
allocate devices or change thresholds. Tensor payloads stay available for review;
retirement is a separate action after the reviewed evidence has been committed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from scripts.job import atomic


def configurations():
    from tide import ClockConfig, ModelConfig, TopologyConfig

    fc = ModelConfig(topology=TopologyConfig(n=1, layers=1), d=192, memory="K",
                     kv_width=128, ffn_width=128, clock=ClockConfig(kv_window=8))
    hb = ModelConfig(topology=TopologyConfig(kind="multiscale", n=1, hubs=1,
                     profile="compact-5-dd4"), d=64, memory="X", kv_width=32,
                     ffn_width=128, sharing="region_shared", aggregation="incoming_light_64",
                     reception_qkv_backend="fused", reception_mask_padding=True,
                     allocation="pool", clock=ClockConfig(kv_window=8))
    timed = ModelConfig(topology=TopologyConfig(kind="multiscale", n=1, hubs=1,
                        profile="compact-5-dd4", timing="timed"), d=64, memory="K",
                        kv_width=32, ffn_width=64, sharing="region_shared",
                        clock=ClockConfig(kv_window=8), checkpoint_reception=True)
    return [("fc-kv128-fp32", fc, "fp32"), ("hb-x-pool-fp32", hb, "fp32"),
            ("hb-timed-k-fp32", timed, "fp32"), ("hb-x-pool-bf16", hb, "bf16")]


def retire_payloads(directory):
    """Only completed task-owned .pt outputs; no external source/install paths."""
    paths = sorted(directory.rglob("*.pt"))
    records = []
    for path in paths:
        if path.is_symlink():
            raise ValueError("refuse a symbolic-link tensor output")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        records.append(dict(path=str(path.relative_to(directory)), bytes=path.stat().st_size,
                            sha256=digest.hexdigest()))
    receipt = directory / "payload-retirement.json"
    atomic(receipt, dict(state="reviewed_terminal", files=records,
                        retained="result.json, restore checks/logs, input and package hashes"))
    for path in paths:
        path.unlink()
    atomic(receipt, dict(state="retired", files=records,
                        retained="result.json, restore checks/logs, input and package hashes"))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "npu:0"), default="cpu")
    parser.add_argument("--stage", choices=("primitives", "training", "all"), default="all")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", nargs="+")
    args = parser.parse_args()
    if args.output_dir.exists() or not 1 <= args.threads <= 8:
        parser.error("use a new output directory and 1..8 threads")
    if not os.environ.get("TIDE_SOURCE_COMMIT"):
        parser.error("run with a frozen source identity through scripts/launch.py")
    args.output_dir.mkdir(parents=True)
    record = dict(schema="tide-stable-mass-qualification-v1", state="running",
                  source=os.environ["TIDE_SOURCE_COMMIT"], device=args.device,
                  physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                  resource_identity=os.environ.get("TIDE_RESOURCE_IDENTITY"),
                  cpu_affinity=sorted(os.sched_getaffinity(0)), threads=args.threads,
                  cases=[], purpose="mechanism qualification; no task-quality or performance claim")
    output = args.output_dir / "result.json"
    atomic(output, record)
    try:
        if args.stage in {"primitives", "all"}:
            command = [sys.executable, "-m", "pytest", "-q", "tests/test_stable_reception.py",
                       "tests/test_mass_scaled_attention.py", "--basetemp",
                       str(args.output_dir / "pytest-tmp")]
            env = dict(os.environ, TIDE_STABLE_DEVICE=args.device)
            log = args.output_dir / "primitives.log"
            with log.open("w") as stream:
                completed = subprocess.run(command, env=env, stdout=stream,
                                           stderr=subprocess.STDOUT, timeout=300)
            record["primitives"] = dict(command=command, exit_code=completed.returncode,
                                        device_environment=args.device, log=str(log))
            log_text = log.read_text()
            record["primitives"]["cpu_fallback_observed"] = bool(
                re.search(r"(?:fall.?back|fall back)[^\n]*cpu", log_text.lower()))
            atomic(output, record)
            if completed.returncode:
                raise RuntimeError("primitive qualification failed; see primitives.log")
            if args.device.startswith("npu") and record["primitives"]["cpu_fallback_observed"]:
                raise RuntimeError("CPU fallback observed; this does not qualify the requested NPU arithmetic")
        if args.stage in {"training", "all"}:
            import torch
            from tide import TideModel
            from scripts.check_training_paths import configure_devices, package_identity, run

            torch.set_num_threads(args.threads)
            torch.set_num_interop_threads(1)
            configure_devices([args.device])
            record.update(torch=str(torch.__version__), package_identity=package_identity())
            if args.device.startswith("npu"):
                import torch_npu
                record["torch_npu"] = str(torch_npu.__version__)
            selected = configurations()
            if args.cases is not None:
                if not set(args.cases) <= {name for name, _, _ in selected}:
                    raise ValueError("unknown requested case")
                selected = [row for row in selected if row[0] in args.cases]
            for name, config, precision in selected:
                directory = args.output_dir / name
                directory.mkdir()
                with torch.device("meta"):
                    parameters = sum(p.numel() for p in TideModel(config).parameters())
                if parameters > 1_500_000:
                    raise ValueError("bounded qualification exceeds 1.5M parameters")
                case = dict(name=name, config=config.to_dict(), precision=precision,
                            parameters=parameters, state="running", batch=2, length=8, chunk=4,
                            steps=3, new_process=True)
                record["cases"].append(case)
                atomic(output, record)

                def progress(result):
                    atomic(directory / "result.json", dict(source=record["source"],
                           device=args.device, package_identity=record["package_identity"],
                           name=name, config=config.to_dict(), precision=precision, result=result))

                result = run(config, device=args.device, precision=precision,
                             candidates=["baseline"], batch=2, length=8, steps=3, chunk=4,
                             directory=directory, new_process=True, witness=False, progress=progress)
                case["state"] = result["state"]
                atomic(output, record)
                if result["state"] not in {"within_threshold", "finite_deviation"}:
                    raise RuntimeError("training or exact restore contract failed: " + name)
                case["payload_status"] = "retained pending review, evidence commit and explicit retirement"
                atomic(output, record)
                del result
                gc.collect()
                if args.device.startswith("npu"):
                    torch.npu.empty_cache()
        record["state"] = ("finite_deviation" if any(c["state"] == "finite_deviation"
                           for c in record["cases"]) else "within_threshold")
    except Exception as error:
        record.update(state="failed", error=repr(error))
        atomic(output, record)
        raise
    atomic(output, record)
    print(json.dumps({key: record[key] for key in ("source", "device", "state")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

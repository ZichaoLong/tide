"""Finite parity checks of an actual standalone regional binary.

CPU is the default reference. NPU is only run when explicitly selected by a
caller already admitted by the device queue. A native launcher may activate an
isolated SDK; the Python process must use its own Python runtime environment.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("regional_reference", ROOT / "native/regional/reference.py")
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--native-launcher", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--streams", choices=("same", "pool"), default="same")
    args = parser.parse_args()
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    binary = args.binary.resolve()
    record = dict(schema=1, state="running", device=args.device, streams=args.streams,
        binary=str(binary), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        source={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((ROOT / "native/regional").glob("*")) if p.is_file()},
        python=sys.executable, host=platform.node(), cpu_affinity=sorted(os.sched_getaffinity(0)),
        argv=sys.argv, cases=[])
    reference.atomic_json(output / "status.json", record)
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    tensors = reference.fixture(output / "fixture")
    prefix = [str(binary)] if args.native_launcher is None else [str(args.native_launcher.resolve()), str(binary)]

    def native(name, dtype, mode, workers, repeats, resume=None, fixture=None):
        path = output / name
        command = [*prefix, "--device", args.device, "--dtype", dtype, "--mode", mode,
            "--workers", str(workers), "--inner-threads", "1", "--repeats", str(repeats),
            "--warmup", "1", "--streams", args.streams, "--input-dir", str(fixture or output / "fixture"),
            "--output-dir", str(path)]
        if resume is not None:
            command.extend(["--resume", str(resume)])
        try:
            completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, timeout=120)
        except subprocess.TimeoutExpired as error:
            log = error.stdout or b""
            (output / f"{name}.log").write_bytes(log if isinstance(log, bytes) else log.encode())
            row = dict(name=name, command=command, state="execution_failed", returncode=None, reason="120s timeout")
            record["cases"].append(row)
            reference.atomic_json(output / "status.json", record)
            return None, row
        (output / f"{name}.log").write_text(completed.stdout)
        row = dict(name=name, command=command, returncode=completed.returncode)
        record["cases"].append(row)
        reference.atomic_json(output / "status.json", record)
        if completed.returncode:
            row["state"] = "execution_failed"
            return None, row
        row["native"] = json.loads((path / "result.json").read_text())
        return reference.read_bundle(path / "tensors"), row

    for dtype in ("float32", "bfloat16"):
        for mode in ("inference", "grad_forward", "train"):
            repeats = 3 if mode == "train" else 1
            expected, metadata = reference.run_reference(tensors, device=args.device,
                dtype=dtype, mode=mode, repeats=repeats, warmup=1,
                output=output / f"python-{dtype}-{mode}")
            for workers in (1, 2, 4):
                name = f"native-{dtype}-{mode}-w{workers}"
                actual, row = native(name, dtype, mode, workers, repeats)
                if actual is None:
                    continue
                row["comparison"] = reference.compare(actual, expected, rtol=2e-5, atol=2e-6)
                row["state"] = row["comparison"]["state"]
                if mode == "train":
                    zero = actual["grad.1.w"][0]
                    empty = actual["grad.2.w"]
                    row["zero_prior_right_vjp_nonzero"] = bool(zero.abs().max() > 0)
                    row["empty_prior_vjp_zero"] = bool(torch.equal(empty, torch.zeros_like(empty)))
                    if not (row["zero_prior_right_vjp_nonzero"] and row["empty_prior_vjp_zero"]):
                        row["state"] = "failed"
        expected4, _ = reference.run_reference(tensors, device=args.device, dtype=dtype, mode="train", repeats=4)
        for workers in (1, 2, 4):
            checkpoint = output / f"native-{dtype}-train-w{workers}" / "checkpoint.pt"
            if not checkpoint.is_file():
                continue
            actual, row = native(f"native-{dtype}-restored-w{workers}", dtype, "train", workers, 1, resume=checkpoint)
            if actual is not None:
                row["comparison"] = reference.compare(actual, expected4, rtol=2e-5, atol=2e-6)
                row["state"] = row["comparison"]["state"]
                row["progress_preserved"] = row["native"]["progress_before"] == 3 and row["native"]["progress_after"] == 4
                if not row["progress_preserved"]:
                    row["state"] = "failed"
    # A first-layer bias with no matrix contribution independently identifies
    # the BF16 epilogue; converting after a FP32 bias addition is a different graph.
    rounding = {key: value.clone() for key, value in tensors.items()}
    for key in ("first_weight", "o_weight"):
        rounding[key].zero_()
    rounding["first_bias"].fill_(1.001)
    rounding["w"].fill_(1)
    reference.write_bundle(output / "bias-rounding", rounding)
    expected, _ = reference.run_reference(rounding, device=args.device, dtype="bfloat16", mode="train")
    actual, row = native("native-bias-rounding", "bfloat16", "train", 2, 1, fixture=output / "bias-rounding")
    if actual is not None:
        row["comparison"] = reference.compare(actual, expected, rtol=2e-5, atol=2e-6)
        row["state"] = row["comparison"]["state"]
    corrupt = output / "corrupt-fixture"
    shutil.copytree(output / "fixture", corrupt)
    payload = bytearray((corrupt / "z.f32").read_bytes()); payload[0] ^= 1
    (corrupt / "z.f32").write_bytes(payload)
    negative = {
        "corrupt": ["--input-dir", str(corrupt)],
        "overwrite": ["--output-dir", str(output / "fixture")],
        "bad_workers": ["--workers", "3"],
        "bad_device": ["--device", "cuda"],
    }
    for name, change in negative.items():
        options = {"--device": args.device, "--input-dir": str(output / "fixture"),
                   "--output-dir": str(output / f"reject-{name}"), "--workers": "1"}
        options[change[0]] = change[1]
        command = [*prefix, *(part for pair in options.items() for part in pair)]
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=30)
        (output / f"reject-{name}.log").write_text(completed.stdout)
        record["cases"].append(dict(name=f"reject-{name}", command=command, returncode=completed.returncode,
            state="passed" if completed.returncode != 0 else "failed", expected="nonzero rejection before model execution"))
    record["state"] = "passed" if all(row.get("state") == "passed" for row in record["cases"]) else "failed"
    record["qualification_boundary"] = "regional subpath only; timings are checks, not formal performance results; actual NPU placement requires separate operator trace"
    reference.atomic_json(output / "status.json", record)
    print(json.dumps(dict(state=record["state"], cases=len(record["cases"]), output=str(output))))
    return 0 if record["state"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

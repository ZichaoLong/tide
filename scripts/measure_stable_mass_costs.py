"""Three bounded paired cost repeats using the existing full-path benchmark.

The two source directories and config must already be frozen. Launch inside the
ordinary bounded CPU/NPU service. This observes the cost of a numerical repair;
it does not certify exact equality of two floating-point training trajectories.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.job import atomic


def package_files(source):
    return {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((source / "src/tide").glob("*.py"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--baseline-revision", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "npu:0"), required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    if args.output_dir.exists() or not 1 <= args.threads <= 8:
        parser.error("choose a new output directory and 1..8 threads")
    if not os.environ.get("TIDE_SOURCE_COMMIT"):
        parser.error("a frozen launch identity is required")
    args.output_dir.mkdir(parents=True)
    sources = {"before": (args.baseline_source.resolve(), args.baseline_revision),
               "after": (source, os.environ["TIDE_SOURCE_COMMIT"])}
    identities = {key: dict(revision=revision, path=str(path), files=package_files(path))
                  for key, (path, revision) in sources.items()}
    config_bytes = args.config.read_bytes()
    config = args.output_dir / "config.json"
    config.write_bytes(config_bytes)
    record = dict(schema="tide-stable-mass-costs-v1", state="running", sources=identities,
                  config=json.loads(config_bytes), config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                  device=args.device, physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                  cpu_affinity=sorted(os.sched_getaffinity(0)), threads=args.threads, rows=[],
                  scope="small fixed graph; independent batch2; prompt16/chunk4/decode8; FP32; three paired repeats",
                  boundary="full-path existing benchmark; ordinary inference/full AdamW timing; work/ledger outside timing",
                  limitation="numerical repair changes floating evaluation; check actual work before attributing cost; no 9B extrapolation or task quality")
    output = args.output_dir / "result.json"
    atomic(output, record)
    for repeat in range(3):
        order = ("before", "after") if repeat != 1 else ("after", "before")
        for mode in ("inference", "train"):
            for label in order:
                root, revision = sources[label]
                identity = f"{repeat}-{mode}-{label}"
                result = args.output_dir / (identity + ".json")
                log = args.output_dir / (identity + ".log")
                command = [sys.executable, "-m", "benchmarks.execution_candidates", "--config", str(config.resolve()),
                           "--subject", "tide", "--variant", "baseline", "--mode", mode, "--device", args.device,
                           "--batch", "2", "--length", "16", "--chunk", "4", "--decode", "8", "--warmup", "1",
                           "--samples", "3", "--threads", str(args.threads), "--repeat", str(repeat),
                           "--pair-id", "stable-mass", "--observe-work", "--memory-ledger", "--output", str(result.resolve())]
                env = dict(os.environ)
                env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
                env.update(TIDE_SOURCE_COMMIT=revision, TIDE_HARNESS_COMMIT=revision)
                row = dict(id=identity, command=command, source=revision, source_path=str(root), state="running")
                record["rows"].append(row)
                atomic(output, record)
                try:
                    with log.open("w") as stream:
                        completed = subprocess.run(command, cwd=root, env=env, stdout=stream,
                                                   stderr=subprocess.STDOUT, timeout=180)
                    row["exit_code"] = completed.returncode
                    row["result"] = json.loads(result.read_text()) if result.exists() else None
                    if completed.returncode or row["result"] is None or row["result"]["state"] != "completed_finite":
                        raise RuntimeError("bounded cost child failed: " + identity)
                    if package_files(root) != identities[label]["files"]:
                        raise RuntimeError("frozen package changed during measurement")
                    row["state"] = "completed_finite"
                except Exception as error:
                    row.update(state="failed", error=repr(error))
                    record["state"] = "failed"
                    atomic(output, record)
                    raise
                atomic(output, record)
    record["state"] = "completed_finite"
    atomic(output, record)
    print(json.dumps(dict(state=record["state"], device=args.device, pairs=6)), flush=True)


if __name__ == "__main__":
    main()

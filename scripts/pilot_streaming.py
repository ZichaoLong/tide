#!/usr/bin/env python3
"""Fixed 16-case first streaming comparison; no search or adaptive enlargement."""
import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys
from build_identity import revision
from experiment_record import atomic_json, utc_now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, choices=("cpu",))
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--tracking", choices=("off", "best-effort", "required"), default="best-effort")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]; out = Path(args.output_dir).resolve()
    if args.jobs < 1 or subprocess.check_output(["git", "status", "--porcelain"], cwd=root).strip():
        parser.error("a clean source commit and positive build job count are required")
    out.mkdir(parents=True, exist_ok=False)
    record = {"scope": "streaming-pilot-v1", "source": revision(root), "state": "running", "started": utc_now(),
              "cases": [], "matrix": {"nodes": [32, 4096], "api": ["functional", "cursor"], "workers": [1, 3], "packed": [0, 1]},
              "common": {"device": "cpu", "dtype": args.dtype, "batch": 4, "width": 16, "active_rings": 4,
                         "ticks": 32, "seed": 7, "warmup": 2, "repetitions": 5},
              "comparison_limit": "small shared-weight EMA workload; inspect per-run load/affinity and raw dispersion"}
    def save(): atomic_json(out/"pilot.json", record)
    save(); code = 1
    try:
        subprocess.run([sys.executable, str(root/"scripts/build.py"), "--build-dir", args.build_dir,
                        "--jobs", str(args.jobs)], cwd=root, check=True)
        for nodes, api, workers, packed in itertools.product((32, 4096), ("functional", "cursor"), (1, 3), (0, 1)):
            name = f"n{nodes}-{api}-w{workers}-p{packed}"
            command = [sys.executable, str(root/"scripts/benchmark_streaming.py"), "--device", "cpu", "--dtype", args.dtype,
                       "--nodes", str(nodes), "--api", api, "--workers", str(workers), "--packed", str(packed),
                       "--tracking", args.tracking, "--build-dir", args.build_dir, "--output-dir", str(out/name),
                       "--trackio-dir", str(out.parent/"trackio")]
            result = subprocess.run(command, cwd=root)
            record["cases"].append({"name": name, "command": command, "exit_code": result.returncode}); save()
            if result.returncode:
                raise RuntimeError(f"pilot case failed: {name}")
        if revision(root) != record["source"] or subprocess.check_output(["git", "status", "--porcelain"], cwd=root).strip():
            raise RuntimeError("pilot source changed")
        record.update(state="passed"); code = 0
    except BaseException as error:
        record.update(state="failed", error=f"{type(error).__name__}: {error}")
    finally:
        record.update(finished=utc_now(), exit_code=code); save()
    print(json.dumps({"state": record["state"], "cases": len(record["cases"]), "result": str(out/"pilot.json")}))
    return code


if __name__ == "__main__":
    sys.exit(main())

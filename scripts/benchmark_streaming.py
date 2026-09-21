#!/usr/bin/env python3
"""One controlled native streaming workload, with durable raw metrics and provenance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import statistics
import subprocess
import sys
import uuid
from build_identity import revision, source_hash
from experiment_record import LocalTrackio, atomic_json, read_events, utc_now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, choices=("cpu",))
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=7)
    for name, default in (("nodes", 32), ("active-rings", 4), ("batch", 4), ("width", 16),
                          ("ticks", 32), ("workers", 1), ("warmup", 2), ("repetitions", 5)):
        parser.add_argument("--"+name, type=int, default=default)
    parser.add_argument("--api", choices=("cursor", "functional"), default="cursor")
    parser.add_argument("--packed", type=int, choices=(0, 1), default=0)
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-dirty", action="store_true", help="development smoke only; archives the exact source")
    parser.add_argument("--tracking", choices=("best-effort", "required", "off"), default="best-effort")
    parser.add_argument("--trackio-dir", help="optional local data root; default OUTPUT_PARENT/trackio")
    args = parser.parse_args()
    if (not 8 <= args.nodes <= 1000000 or args.nodes%8 or not 1 <= args.active_rings <= min(128, args.nodes//8)
            or not 1 <= args.batch <= 1024 or not 1 <= args.width <= 4096 or not 1 <= args.ticks <= 100000
            or not 1 <= args.workers <= 64 or not 0 <= args.warmup <= 100 or not 1 <= args.repetitions <= 1000
            or args.seed < 0 or args.seed >= 2**64 or args.ticks*args.active_rings*args.batch > 10000000):
        parser.error("dimensions or seed exceed the bounded benchmark domain")
    root = Path(__file__).resolve().parents[1]
    build, out = Path(args.build_dir).resolve(), Path(args.output_dir).resolve()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    if dirty and not args.allow_dirty:
        parser.error("controlled benchmarks require a clean commit; use --allow-dirty only for development")
    identity = source_hash(root); commit = revision(root)
    wrappers = (Path(__file__).resolve(), root/"scripts/experiment_record.py")
    def python_identity():
        return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in wrappers}
    python_hashes = python_identity()
    compiled = json.loads((build/"build-manifest.json").read_text())
    binary = build/"tidegraph-streaming-bench"
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    if compiled["cpp_source_sha256"] != identity or compiled["binary_sha256"].get(binary.name) != digest:
        parser.error("benchmark build/source mismatch; run scripts/build.py")
    config = {key: getattr(args, key) for key in ("device", "dtype", "seed", "nodes", "active_rings", "batch",
              "width", "ticks", "workers", "warmup", "repetitions", "api", "packed")}
    project, run_id = "tide-graph-execution", out.name+"-"+uuid.uuid4().hex[:10]
    track = LocalTrackio(args.tracking, Path(args.trackio_dir).resolve() if args.trackio_dir else out.parent/"trackio",
                        project, run_id, config)
    command = [str(binary), "--run-id", run_id, "--output-dir", str(out/"native")]
    for key, value in config.items():
        command += ["--"+key.replace("_", "-"), str(value)]
    out.mkdir(parents=True, exist_ok=False)
    now = utc_now()
    record = {"schema_version": 1, "run_id": run_id, "project": project, "name": run_id,
              "status": "running", "created_at": now, "started_at": now, "ended_at": None,
              "source": {"repository": "tide/graph-execution-foundation", "commit": commit, "dirty": bool(dirty),
                         "cpp_source_sha256": identity, "binary_sha256": digest, "python_sha256": python_hashes},
              "command": {"argv": command, "wrapper_argv": sys.argv, "working_directory": str(root)},
              "inputs": {"workload": "parallel-ring-prefix-v1", "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()},
              "runtime": {"host_arch": platform.machine(), "resolved_device": "cpu", "dtype": args.dtype,
                          "seed": args.seed, "python_version": platform.python_version(), "torch_version": compiled["torch"],
                          "cxx11_abi": compiled["cxx11_abi"], "threads": 1, "node_workers": args.workers,
                          "os": platform.platform(), "logical_cpus": os.cpu_count(),
                          "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                          "load_average_before": list(os.getloadavg())},
              "experiment": {"class": "benchmark", "config": config, "global_step_semantics": "one measured repetition after a fresh empty reset",
                             "primary_metric": "perf/advance_seconds", "stop_condition": f"{args.repetitions} measured repetitions",
                             "shared_weights": True, "trace_during_timing": False, "grad_mode": "no_grad"},
              "tracking": track.record,
              "artifacts": {"metrics": "metrics.jsonl", "stdout": "stdout.log", "summary": "summary.json", "native": "native/"}}
    def save(): atomic_json(out/"run.json", record)
    save(); error = None; code = 1; events = []; child_code = None; cancelled = None
    def interrupted(signum, _frame):
        nonlocal cancelled
        cancelled = signum
        raise InterruptedError("benchmark cancelled by signal")
    previous = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        if dirty:
            files = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root)
            subprocess.run(["tar", "--null", "-T", "-", "-czf", str(out/"source.tar.gz")], input=files, cwd=root, check=True)
            record["source"]["archive_sha256"] = hashlib.sha256((out/"source.tar.gz").read_bytes()).hexdigest()
        track.start(); save()
        env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        with (out/"stdout.log").open("w") as log:
            child = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                child_code = child.wait()
            except BaseException:
                child.terminate()
                try: child_code = child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill(); child_code = child.wait()
                raise
        metrics = out/"native/metrics.jsonl"
        if metrics.exists():
            shutil.copyfile(metrics, out/"metrics.jsonl")
            events = read_events(out/"metrics.jsonl", run_id)
        if child_code:
            raise RuntimeError(f"native benchmark exited {child_code}; inspect stdout.log")
        if len(events) != args.repetitions:
            raise ValueError("native benchmark repetition inventory mismatch")
        if (source_hash(root) != identity or revision(root) != commit or python_identity() != python_hashes
                or hashlib.sha256(binary.read_bytes()).hexdigest() != digest):
            raise ValueError("source/binary changed during benchmark")
        track.project_events(events); code = 0
    except BaseException as exception:
        code = 128+cancelled if cancelled else 1
        error = f"{type(exception).__name__}: {exception}"
    finally:
        for signum, handler in previous.items(): signal.signal(signum, handler)
        if (out/"native/metrics.jsonl").exists() and not (out/"metrics.jsonl").exists():
            shutil.copyfile(out/"native/metrics.jsonl", out/"metrics.jsonl")
        try: track.finish()
        except Exception as exception:
            code = 1
            if error is None: error = f"{type(exception).__name__}: {exception}"
        record.update(status="cancelled" if cancelled else "completed" if code == 0 else "failed", ended_at=utc_now())
        summaries = {}
        if events:
            for key in events[0]["metrics"]:
                values = [event["metrics"][key] for event in events]
                summaries[key] = {"median": statistics.median(values), "min": min(values), "max": max(values),
                                  "population_stdev": statistics.pstdev(values)}
        atomic_json(out/"summary.json", {"schema_version": 1, "run_id": run_id, "status": record["status"],
                    "ended_at": record["ended_at"], "exit_code": code, "native_exit_code": child_code,
                    "error": error, "observations": len(events), "metrics": summaries, "tracking": track.record})
        save()
    print(json.dumps({"state": record["status"], "result": str(out/"summary.json"), "tracking": track.record["status"]}))
    return code


if __name__ == "__main__":
    sys.exit(main())

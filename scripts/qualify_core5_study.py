#!/usr/bin/env python3
"""Bounded fixed-source CPU gate for public planning and consumer changes.

Launch through job.py in a detached background.slice service. This directed
gate does not certify the complete accelerator/semantics Cartesian matrix.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from build_resources import add_build_jobs
from durable_records import write_json
from source_identity import source_state, digest

TESTS = (
    "test_build_resources", "test_core5_planning", "test_weighted_workload_plan",
    "test_weighted_inventory", "test_weighted_source_binding", "test_weighted_allocation",
    "test_weighted_entropy_allocation", "test_flow_topology", "test_weighted_benchmark_precision",
    "test_library", "test_greedy_library", "test_weighted_source_execution", "test_weighted_workload_training",
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--build-dir", type=Path, required=True)
    add_build_jobs(p)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    source, dirty = source_state(root)
    if dirty:
        p.error("qualification requires clean fixed source")
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    build = a.build_dir.resolve()
    report = dict(schema="tide-core5-study-qualification-v1", source=source, dirty=dirty,
                  state="running", build_jobs=a.jobs, stages=[],
                  scope="CPU public planning, legacy material compatibility, public runtime and related local semantics")
    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", TIDE_BUILD_DIR=str(build),
               PYTHONPATH=os.pathsep.join((str(root/"python"), str(build), os.environ.get("PYTHONPATH", ""))))

    def run(name, command, seconds):
        start = time.monotonic()
        write_json(out/"result.json", report)
        try:
            with (out/(name+".log")).open("w") as log:
                code = subprocess.run(command, cwd=out, env=env, stdout=log,
                                      stderr=subprocess.STDOUT, timeout=seconds).returncode
        except subprocess.TimeoutExpired:
            code = 124
        report["stages"].append(dict(name=name, command=command, exit_code=code, timeout_seconds=seconds,
                                     seconds=time.monotonic()-start, log_sha256=digest(out/(name+".log"))))
        write_json(out/"result.json", report)
        if code:
            raise RuntimeError(name+" failed; preserved log and terminal code")

    try:
        run("build", [sys.executable, str(root/"scripts/build.py"), "--backend", "cpu", "--build-dir", str(build),
                      "--jobs", str(a.jobs)], 1800)
        run("tests", [sys.executable, "-m", "pytest", "-q", *[str(root/"tests"/(t+".py")) for t in TESTS],
                      "--dtype", "both", "--junitxml="+str(out/"tests.xml")], 900)
        if source_state(root) != (source, dirty):
            raise RuntimeError("source changed during qualification")
        report["state"] = "passed"
    except BaseException as error:
        report.update(state="failed", error=repr(error))
        raise
    finally:
        write_json(out/"result.json", report)


if __name__ == "__main__":
    main()

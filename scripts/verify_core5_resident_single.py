#!/usr/bin/env python3
"""One-device core-5 resident acceptance, with no peer claims or skipped tests.

The independent full target gate retains its three-device hardware requirement.
This gate qualifies its explicit inventory, not the legacy module Cartesian set.
Device residency additionally requires a reviewed target timeline.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

from core5_resident_scope import COMPONENTS, PUBLIC_TESTS
from durable_records import write_json
from python_environment import with_pythonpath
from source_identity import source_state, digest
from verify_resident_target import check_build


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", required=True)
    p.add_argument("--core-build", type=Path, required=True)
    p.add_argument("--python-resident-build", type=Path, required=True)
    p.add_argument("--standalone-resident-build", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    source, dirty = source_state(root)
    if dirty:
        p.error("qualification requires a clean frozen checkout")
    backend = a.device.split(":")[0]
    if backend not in {"cuda", "npu"} or ":" not in a.device:
        p.error("explicit indexed accelerator required")
    core, py, cpp = (x.resolve() for x in (a.core_build, a.python_resident_build, a.standalone_resident_build))
    py_record = check_build(root, py, backend=backend, owner="python")
    cpp_record = check_build(root, cpp, backend=backend, owner="standalone")
    core_record = json.loads((core / "build-manifest.json").read_text())
    if core_record != py_record["core"]:
        p.error("Python core differs from the resident build's recorded core")
    for name, expected in core_record["binary_sha256"].items():
        if digest(core / name) != expected:
            p.error("core binary changed: " + name)
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = with_pythonpath(dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0", OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NVIDIA_TF32_OVERRIDE="0",
        TIDE_BUILD_DIR=str(core), TIDE_RESIDENT_DEVICE=a.device, TIDE_RESIDENT_LIBRARY=str(py)), root / "python", core)
    report = dict(schema="tide-core5-resident-single-v1", state="running", source=source, dirty=dirty,
        device=a.device, owners=1, python_build=py_record, standalone_build=cpp_record,
        component_inventory=list(COMPONENTS), public_inventory=list(PUBLIC_TESTS), stages=[],
        scope="core-5 one-device component, inference, VJP, optimizer, cross-window and checkpoint correctness; no peers, performance or residency timeline claim")
    write_json(out / "result.json", report)

    def run(name, command, seconds):
        row = dict(name=name, command=list(map(str, command)), state="running", timeout_seconds=seconds)
        report["stages"].append(row)
        write_json(out / "result.json", report)
        logpath = out / (name + ".log")
        with logpath.open("x") as log:
            result = subprocess.run(row["command"], cwd=out, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=seconds)
        row.update(exit_code=result.returncode, state="passed" if result.returncode == 0 else "failed",
                   log_sha256=digest(logpath))
        write_json(out / "result.json", report)
        if result.returncode:
            raise RuntimeError(name + " failed; original log retained")

    try:
        run("preflight", [sys.executable, "-c", "import torch;from tidegraph.runtime import resolve_device;"
            f"d,_=resolve_device({a.device!r});x=torch.arange(3,device=d);assert x.cpu().tolist()==[0,1,2]"], 60)
        run("components", [sys.executable, root / "scripts/verify_device_control.py", "--device", a.device,
            "--build-dir", cpp, "--output-dir", out / "components", "--checks", *COMPONENTS], 7200)
        total = 0
        for name in PUBLIC_TESTS:
            stage = name.removesuffix(".py")
            run(stage, [sys.executable, "-m", "pytest", root / "tests" / name, "-q", "--dtype", "both",
                        "--junitxml", out / (stage + ".xml"), "--basetemp", out / (stage + "-tmp")], 1200)
            cases = ET.parse(out / (stage + ".xml")).findall(".//testcase")
            if not cases or any(c.find("skipped") is not None for c in cases):
                raise RuntimeError("one-device inventory requires nonempty tests and zero skips")
            total += len(cases)
        if source_state(root) != (source, dirty):
            raise RuntimeError("source changed during checks")
        report.update(state="passed", public_cases=total, skipped=0)
    except BaseException as error:
        report.update(state="failed", error=repr(error))
        raise
    finally:
        write_json(out / "result.json", report)


if __name__ == "__main__":
    main()

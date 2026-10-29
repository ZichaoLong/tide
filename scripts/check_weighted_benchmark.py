#!/usr/bin/env python3
"""Directed standalone consumer check with actual exported public-model material.

Run from a frozen checkout in a durable job. These tiny cases validate the
consumer facility; their times are not model-scale performance evidence.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from durable_records import write_json
from source_identity import source_state, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--native-library", type=Path, help="matching Python core for CPU static export")
    p.add_argument("--fixture-dir", type=Path, help="previously exported kv.json/ssd.json, no Python core needed")
    p.add_argument("--device", default="cpu", help="cpu or explicitly indexed accelerator")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--allow-dirty", action="store_true")
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    source, dirty = source_state(root)
    if dirty and not a.allow_dirty:
        p.error("fixed-source qualification requires a clean checkout")
    out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    if a.device != "cpu" and not a.device.startswith(("npu:", "cuda:")):
        p.error("device must be cpu or explicitly indexed accelerator")
    if not a.native_library and not a.fixture_dir:
        p.error("provide a Python core for export or --fixture-dir")
    binary, core = a.binary.resolve(), a.native_library.resolve() if a.native_library else None
    fixture_dir = a.fixture_dir.resolve() if a.fixture_dir else None
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               TORCH_DEVICE_BACKEND_AUTOLOAD="0", PYTHONPATH=os.pathsep.join((str(root/"python"), str(core), os.environ.get("PYTHONPATH", ""))))
    report = dict(schema="tide-weighted-consumer-check-v1", source=source, dirty=dirty,
                  binary_sha256=digest(binary), device=a.device, state="running", stages=[])

    def run(name, command):
        command = list(map(str, command))
        with (out/(name+".log")).open("w") as log:
            result = subprocess.run(command, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=120)
        report["stages"].append(dict(name=name, command=command, exit_code=result.returncode,
                                     log_sha256=digest(out/(name+".log"))))
        write_json(out/"result.json", report)
        if result.returncode:
            raise RuntimeError(name+" failed")

    try:
        for family in ("kv", "ssd"):
            material = (fixture_dir or out)/(family+".json")
            if not fixture_dir:
                run("export-"+family, [sys.executable, root/"scripts/export_weighted_benchmark.py", "--family", family,
                    "--native-library", core, "--binary", "--output", material])
            if a.device == "cpu":
                cases = [("float32", "prefill", "packed", 2, "native"),
                         ("float64", "streaming", "scalar", 1, "native"),
                         ("float16", "prefill", "packed", 1, "native")]
            else:
                placements = ("mixed-a", "mixed-b", "mixed-c", "resident") if family == "kv" else ("mixed-c", "resident")
                cases = [("float32", "prefill", "packed", 1, p) for p in placements]
                if family == "kv":
                    cases += [("float16", "prefill", "packed", 1, p) for p in ("mixed-c", "resident")]
            for dtype, schedule, packing, workers, placement in cases:
                name = family+"-"+dtype+"-"+placement
                run(name, [binary, "--fixture", material, "--device", a.device, "--dtype", dtype,
                    "--placement", "native" if placement == "resident" else placement,
                    "--implementation", "resident" if placement == "resident" else "native",
                    "--intraop-threads", "2", "--workers", workers, "--packing", packing,
                    "--mode", "training", "--schedule", schedule, "--steps", "1",
                    "--checkpoint-check", "true", "--output-dir", out/name])
                record = json.loads((out/name/"result.json").read_text())
                if not (record["state"] == "passed" and record["optimizer_values_checked"]
                        and record["final_parameter_gradients_checked"] and record["exact_resume"]["exact"]):
                    raise RuntimeError("incomplete consumer checks: "+name)
        if source_state(root) != (source, dirty):
            raise RuntimeError("consumer source changed during check")
        report["state"] = "passed"
    except BaseException as error:
        report.update(state="failed", error=repr(error)); raise
    finally:
        write_json(out/"result.json", report)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Bounded placement trace of one standalone device component, not throughput."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess

from durable_records import write_json
from build_device_control import component_hash
from source_identity import digest, source_state
from summarize_ascend_profile import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--check", choices=("closure", "numerical", "queue"), required=True)
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    args = parser.parse_args()
    if args.device != "npu" and not args.device.startswith("npu:"):
        parser.error("CANN component profiling requires explicit NPU")
    if args.check != "numerical" and args.dtype != "float32":
        parser.error("closure/queue placement gate currently requires FP32")
    executable = {"closure": "tide-device-closure-check", "numerical": "tide-device-numerical-check",
                  "queue": "tide-packed-queue-check"}[args.check]
    build = args.build_dir.resolve()
    manifest = json.loads((build / "control-build.json").read_text())
    root = Path(__file__).resolve().parents[1]
    if manifest["component_sha256"] != component_hash(root):
        parser.error("component source differs from recorded build")
    binary = build / executable
    if digest(binary) != manifest["binary_sha256"].get(executable):
        parser.error("component binary differs from recorded build")
    msprof = shutil.which("msprof")
    if not msprof:
        parser.error("msprof is unavailable in the active CANN module")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source, dirty = source_state(root)
    command = [msprof, "--output="+str(out / "raw"), "--runtime-api=on", "--task-time=l1",
               "--aicpu=on", "--storage-limit=200MB", str(binary), "--device="+args.device,
               "--dtype="+args.dtype]
    report = dict(schema="tide-device-component-profile-v1", state="running", source=source,
                  dirty=dirty, build=manifest, command=command,
                  scope="component placement; includes construction, inputs and CPU assertions; not throughput")
    write_json(out / "result.json", report)
    try:
        with (out / "profile.log").open("w") as log:
            subprocess.run(command, cwd=out, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
        log = (out / "profile.log").read_text()
        marker = {"closure": "device-closure: passed", "numerical": "device-numerical: passed",
                  "queue": "packed-queue: passed"}[args.check]
        if marker not in log:
            raise RuntimeError("profiled component did not report its acceptance checks")
        if "fall back to run on the CPU" in log or "npu_cpu_fallback" in log:
            raise RuntimeError("profiled component reported host CPU fallback")
        sessions = sorted((out / "raw").glob("PROF_*"))
        if not sessions:
            raise RuntimeError("msprof produced no session")
        for session in sessions:
            if not list(session.rglob("op_summary_*.csv")):
                with (out / "export.log").open("a") as log:
                    subprocess.run([msprof, "--export=on", "--output="+str(session)], cwd=out,
                                   stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
        summaries, closure_rows = [], []
        for directory in sorted({p.parent for p in (out / "raw").rglob("op_summary_*.csv")}):
            summaries.append(dict(path=str(directory.relative_to(out)), **summarize(directory)))
            for path in directory.glob("op_summary_*.csv"):
                with path.open(newline="") as stream:
                    for row in csv.DictReader(stream):
                        if "tide_closure" in " ".join(str(v) for v in row.values()):
                            closure_rows.append(row)
        if not summaries:
            raise RuntimeError("no device operator placement records")
        if args.check == "closure" and not closure_rows:
            raise RuntimeError("no Ascend C closure task in exported trace")
        if source_state(root) != (source, dirty):
            raise RuntimeError("source changed during profiling")
        report.update(state="passed", summaries=summaries, closure_tasks=closure_rows,
                      profile_log_sha256=digest(out / "profile.log"))
    except BaseException as error:
        report.update(state="failed", error=repr(error))
        raise
    finally:
        write_json(out / "result.json", report)
    print(json.dumps(dict(state=report["state"], check=args.check,
                          closure_task_records=len(report["closure_tasks"]))))


if __name__ == "__main__":
    main()

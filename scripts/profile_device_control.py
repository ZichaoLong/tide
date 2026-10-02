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
from device_component_checks import CHECKS, MARKERS, KERNELS
from source_identity import digest, source_state
from summarize_ascend_profile import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--check", choices=tuple(CHECKS), required=True)
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--storage-limit-mb", type=int, default=200,
                        help="Bounded raw collection storage; larger gates may need more than 200 MB")
    parser.add_argument("--application-arg", action="append", default=[], help="Additional recorded component argument")
    args = parser.parse_args()
    if not 1 <= args.storage_limit_mb <= 4096:
        parser.error("storage limit must be between 1 and 4096 MB")
    if args.device != "npu" and not args.device.startswith("npu:"):
        parser.error("CANN component profiling requires explicit NPU")
    executable, dtypes = CHECKS[args.check]
    if args.dtype not in dtypes:
        parser.error("dtype unavailable for this component gate")
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
               "--aicpu=on", f"--storage-limit={args.storage_limit_mb}MB", str(binary), "--device="+args.device,
               "--dtype="+args.dtype, *args.application_arg]
    if args.check in ("precision-control-flow", "peer-control-flow", "peer-shard-control-flow", "peer-state-control-flow"):
        command.append("--control-modes")
    if args.check in ("peer-flow", "peer-control-flow"):
        command.append("--peer-full")
    if args.check in ("peer-shard-flow", "peer-shard-control-flow"):
        command.append("--full-shards=2")
    if args.check in ("peer-state-flow", "peer-state-control-flow", "peer-state-vjp", "peer-state-training"):
        command.extend(("--full-shards=2", "--state-shards"))
    if args.check == "extended-retained":
        command.append("--extended")
    if args.check in ("event-retained", "fiber-retained"):
        command.append("--" + args.check.split("-")[0] + "-cache")
    if args.check == "half-cache-training":
        command.append("--cache")
    if args.check == "peer-resident-accumulation":
        command.extend(("--accumulate", "--resume-devices=0", "--explicit-owners"))
    report = dict(schema="tide-device-component-profile-v1", state="running", source=source,
                  dirty=dirty, build=manifest, command=command,
                  scope="component placement; includes construction, inputs and CPU assertions; not throughput")
    write_json(out / "result.json", report)
    try:
        with (out / "profile.log").open("w") as log:
            subprocess.run(command, cwd=out, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
        log = (out / "profile.log").read_text()
        if MARKERS[args.check] not in log:
            raise RuntimeError("profiled component did not report its acceptance checks")
        # msprof can return zero even when the application segfaults or exits
        # nonzero. A printed marker alone cannot certify orderly completion.
        if "An exception has occurred in process App" in log:
            raise RuntimeError("msprof reported failed application termination")
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
        summaries, closure_rows, transaction_rows, kernel_rows = [], [], [], []
        for directory in sorted({p.parent for p in (out / "raw").rglob("op_summary_*.csv")}):
            summaries.append(dict(path=str(directory.relative_to(out)), **summarize(directory)))
            for path in directory.glob("op_summary_*.csv"):
                with path.open(newline="") as stream:
                    for row in csv.DictReader(stream):
                        if "tide_closure" in " ".join(str(v) for v in row.values()):
                            closure_rows.append(row)
                        if "tide_queue_propose" in " ".join(str(v) for v in row.values()):
                            transaction_rows.append(row)
                        if args.check in KERNELS and KERNELS[args.check] in " ".join(str(v) for v in row.values()):
                            kernel_rows.append(row)
        if not summaries:
            raise RuntimeError("no device operator placement records")
        if args.check == "closure" and not closure_rows:
            raise RuntimeError("no Ascend C closure task in exported trace")
        if args.check == "transaction" and not transaction_rows:
            raise RuntimeError("no Ascend C queue transaction task in exported trace")
        if args.check in KERNELS and not kernel_rows:
            raise RuntimeError("no requested component kernel task in exported trace")
        if source_state(root) != (source, dirty):
            raise RuntimeError("source changed during profiling")
        report.update(state="passed", summaries=summaries, closure_tasks=closure_rows,
                      transaction_tasks=transaction_rows,
                      component_kernel_tasks=kernel_rows,
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

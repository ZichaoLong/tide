#!/usr/bin/env python3
"""Bounded standalone component acceptance, independently of graph qualification."""
import argparse
import json
from pathlib import Path
import subprocess

from build_device_control import component_hash
from build_identity import source_hash
from device_component_checks import CHECKS, MARKERS
from durable_records import write_json
from source_identity import digest, source_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--checks", nargs="+", choices=tuple(CHECKS),
                        default=[name for name in CHECKS if not name.startswith("peer") and name != "resident-sharded-training"],
                        help="Single-device checks by default; peer explicitly requires two visible NPUs")
    parser.add_argument("--full-training-control-check", choices=("strict", "conditioned"), default="strict")
    args = parser.parse_args()
    if args.device != "npu" and not args.device.startswith("npu:"):
        parser.error("component qualification requires explicit NPU")
    root = Path(__file__).resolve().parents[1]
    build = args.build_dir.resolve()
    manifest = json.loads((build / "control-build.json").read_text())
    if manifest["component_sha256"] != component_hash(root) or manifest["core"]["cpp_source_sha256"] != source_hash(root):
        parser.error("component/core source differs from recorded build")
    for check in args.checks:
        name, _ = CHECKS[check]
        if not (build / name).is_file() or digest(build / name) != manifest["binary_sha256"].get(name):
            parser.error("component unavailable or changed: " + name)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source, dirty = source_state(root)
    report = dict(schema="tide-device-components-gate-v1", source=source, dirty=dirty,
                  build=manifest, state="running", device=args.device, cases=[],
                  scope="selected components and declared resident profiles/dtypes; not complete module/dtype/multi-device matrix or throughput qualification")
    write_json(out / "result.json", report)
    try:
        for check in args.checks:
            name, dtypes = CHECKS[check]
            for dtype in dtypes:
                command = [str(build / name), "--device="+args.device, "--dtype="+dtype]
                if check == "full-training":
                    command.append("--control-check=" + args.full_training_control_check)
                if check in ("precision-control-flow", "peer-control-flow", "peer-shard-control-flow", "peer-state-control-flow"):
                    command.append("--control-modes")
                if check in ("peer-flow", "peer-control-flow"):
                    command.append("--peer-full")
                if check in ("peer-shard-flow", "peer-shard-control-flow"):
                    command.append("--full-shards=2")
                if check in ("peer-state-flow", "peer-state-control-flow", "peer-state-vjp", "peer-state-training"):
                    command.extend(("--full-shards=2", "--state-shards"))
                if check == "extended-retained":
                    command.append("--extended")
                if check in ("event-retained", "fiber-retained"):
                    command.append("--" + check.split("-")[0] + "-cache")
                if check == "half-cache-training":
                    command.append("--cache")
                if check == "peer-resident-accumulation":
                    command.extend(("--accumulate", "--resume-devices=0", "--explicit-owners"))
                if check == "peer-resident-compact-contexts":
                    command.extend(("--compact-contexts", "--explicit-owners"))
                if check == "peer-resident-contexts":
                    command.extend(("--contexts", "--explicit-owners"))
                if check == "peer-resident-compact-journals":
                    command.extend(("--compact-journals", "--resume-devices=0", "--explicit-owners"))
                if check == "peer-resident-compact-projections":
                    command.extend(("--emission", "--compact-journals", "--explicit-owners"))
                log_path = out / (check+"-"+dtype+".log")
                with log_path.open("w") as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                            timeout=360 if check in ("peer-resident-compact-projections", "peer-resident-compact-journals", "peer-resident-compact-contexts", "peer-resident-contexts", "peer-resident-accumulation", "resident-sharded-training", "peer-sharded-vjp", "peer-sharded-training", "peer-state-flow", "peer-state-control-flow", "peer-state-vjp", "peer-state-training") else 120)
                text = log_path.read_text()
                passed = result.returncode == 0 and MARKERS[check] in text
                if "fall back to run on the CPU" in text or "npu_cpu_fallback" in text:
                    passed = False
                report["cases"].append(dict(check=check, dtype=dtype, command=command,
                    state="passed" if passed else "failed", exit_code=result.returncode,
                    log=log_path.name, log_sha256=digest(log_path)))
                write_json(out / "result.json", report)
                print(check, dtype, report["cases"][-1]["state"], flush=True)
                if not passed:
                    raise RuntimeError("device gate failed: " + str(log_path))
        if source_state(root) != (source, dirty):
            raise RuntimeError("source changed during device gate")
        report["state"] = "passed"
    except BaseException as error:
        report.update(state="failed", error=repr(error))
        raise
    finally:
        write_json(out / "result.json", report)


if __name__ == "__main__":
    main()

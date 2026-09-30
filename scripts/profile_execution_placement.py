#!/usr/bin/env python3
"""Bounded CANN placement trace of the public mixed executor, not throughput."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from build_identity import source_hash
from durable_records import write_json
from source_identity import digest, source_state
from summarize_ascend_profile import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    if args.device != "npu" and not args.device.startswith("npu:"):
        parser.error("CANN placement profiling requires explicit NPU")
    root = Path(__file__).resolve().parents[1]
    build = args.build_dir.resolve()
    binary = build / "tidegraph-placement-check"
    normal = build / "build-manifest.json"
    if normal.exists():
        manifest = json.loads(normal.read_text())
        if manifest["cpp_source_sha256"] != source_hash(root):
            parser.error("profile source differs from the native build")
        expected = manifest["binary_sha256"][binary.name]
    else:
        # Development-only isolated relinks retain exact core source and binary
        # identities. Formal qualification uses the clean full build above.
        manifest = json.loads((build / "checker-build.json").read_text())
        if manifest["state"] != "passed":
            parser.error("development checker build did not pass")
        for name, identity in manifest["core_production_files"].items():
            if digest(root / name) != identity:
                parser.error("profile production source differs: " + name)
        sources = [Path(x) for command in manifest["commands"] for x in command if x.endswith("/cpp/test/placement.cpp")]
        if len(sources) != 1 or digest(sources[0]) != digest(root / "cpp/test/placement.cpp"):
            parser.error("profile checker source differs")
        expected = manifest["checker_sha256"]
    if digest(binary) != expected:
        parser.error("profile executable differs from recorded build")
    msprof = shutil.which("msprof")
    if not msprof:
        parser.error("msprof is unavailable in the active CANN module")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source, dirty = source_state(root)
    command = [msprof, "--output="+str(out / "raw"), "--runtime-api=on", "--task-time=l1",
               "--aicpu=on", "--storage-limit=200MB", str(binary), "--device="+args.device,
               "--dtype=float32", "--profile-smoke"]
    report = dict(schema="tide-execution-placement-profile-v1", state="running", source=source,
                  dirty=dirty, build=manifest, command=command,
                  scope="mixed-C ranking/control plus CPU-FP64-Read anchor; host event dispatch; includes assertions; not throughput")
    write_json(out / "result.json", report)
    try:
        with (out / "profile.log").open("w") as log:
            subprocess.run(command, cwd=out, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
        log = (out / "profile.log").read_text()
        if "standalone-placement: passed" not in log or "scope=profile-smoke" not in log:
            raise RuntimeError("profiled checker did not pass its bounded assertions")
        if "An exception has occurred in process App" in log:
            raise RuntimeError("msprof reported failed application termination")
        if "fall back to run on the CPU" in log or "npu_cpu_fallback" in log:
            raise RuntimeError("unexpected host CPU fallback")
        for session in sorted((out / "raw").glob("PROF_*")):
            if not list(session.rglob("op_summary_*.csv")):
                with (out / "export.log").open("a") as stream:
                    subprocess.run([msprof,"--export=on","--output="+str(session)],cwd=out,
                                   stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=180)
        summaries = [dict(path=str(p.relative_to(out)), **summarize(p))
                     for p in sorted({f.parent for f in (out / "raw").rglob("op_summary_*.csv")})]
        if not summaries:
            raise RuntimeError("no device operator placement records")
        if source_state(root) != (source, dirty):
            raise RuntimeError("source changed during profiling")
        report.update(state="passed",summaries=summaries,profile_log_sha256=digest(out / "profile.log"),
                      aicpu_sort_diagnostic="running on AiCpu" in log,host_fallback_diagnostic=False)
    except BaseException as error:
        report.update(state="failed", error=repr(error))
        raise
    finally:
        write_json(out / "result.json", report)
    print(json.dumps(dict(state=report["state"],aicpu_sort_diagnostic=report["aicpu_sort_diagnostic"])))


if __name__ == "__main__":
    main()

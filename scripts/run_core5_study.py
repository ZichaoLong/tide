#!/usr/bin/env python3
"""Run one explicitly selected study case/execution; preserve all terminal results.

Launch in a fixed-source detached background.slice job. This entry has a finite
child timeout, never retries and never auto-expands a performance sweep.
"""
import argparse
from datetime import datetime, timezone
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import _planning_path  # noqa: F401
from tidegraph.planning.study import expand, case, command
from durable_records import write_json
from source_identity import source_state, digest
from core5_study_records import standalone_identity, accept_result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, default=Path(__file__).resolve().parents[1]/"benchmarks/core5-study-v1.json")
    p.add_argument("--list", action="store_true")
    p.add_argument("--case")
    p.add_argument("--execution")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--native-library", type=Path)
    p.add_argument("--export-library", type=Path, help="Python-owned CPU core used only to export standalone material")
    p.add_argument("--resident-library", type=Path)
    p.add_argument("--standalone-binary", type=Path)
    p.add_argument("--fixture", type=Path, help="already exported exact matching workload; otherwise export it")
    p.add_argument("--resident-limits", type=Path)
    p.add_argument("--checkpoint-max-bytes", type=int, help="explicit native graph file budget; LibTorch checkpoint check only")
    p.add_argument("--max-materialize-bytes", type=int, default=2*1024**3)
    a = p.parse_args()
    study = json.loads(a.study.read_text())
    if a.list:
        print(json.dumps(dict(cases=expand(study), executions=study["executions"]), indent=2)); return
    if not a.case or not a.execution or not a.output_dir:
        p.error("select exactly one --case, --execution and new --output-dir")
    root = Path(__file__).resolve().parents[1]
    source, dirty = source_state(root)
    if dirty:
        p.error("controlled study requires clean fixed source")
    r = case(study, a.case, a.execution)
    out = a.output_dir.resolve()
    for key in ("native_library", "export_library", "resident_library", "standalone_binary", "fixture", "resident_limits"):
        if getattr(a, key) is not None:
            setattr(a, key, getattr(a, key).resolve())
    standalone = r["execution"]["requested"]["client"] == "libtorch"
    binary = standalone_identity(a.standalone_binary, root) if standalone and a.standalone_binary else None
    fixture = a.fixture or out/"fixture.json"
    if standalone and a.fixture:
        data = json.loads(fixture.read_text())
        if data.get("workload", {}).get("sha256") != r["plan"]["sha256"]:
            p.error("external fixture does not match requested workload identity")
    if standalone and not a.fixture and not a.export_library:
        p.error("standalone material export requires --export-library (Python-owned CPU core)")
    argv = command(r, python=sys.executable, root=root, output=out/"measurement", workload=out/"workload.json",
                   native_library=a.native_library, resident_library=a.resident_library,
                   standalone_binary=a.standalone_binary, fixture=fixture,
                   max_materialize_bytes=a.max_materialize_bytes, resident_limits=a.resident_limits,
                   checkpoint_max_bytes=a.checkpoint_max_bytes)
    out.mkdir(parents=True, exist_ok=False)
    now = lambda: datetime.now(timezone.utc).isoformat()
    record = {k: v for k, v in r.items() if k not in {"plan", "inventory"}}
    from tidegraph.planning.facts import describe
    record["configuration_facts"] = describe(r["plan"])
    record.update(schema="tide-core5-run-v1", source=source, dirty=dirty, state="running", started=now(),
        model_id=r["plan"]["model_id"], workload_id=r["plan"]["sha256"],
        parameters=r["inventory"]["totals"], recipe=r["plan"]["recipe"],
        graph_counts=r["plan"]["graph"]["counts"], command=argv, stages=[],
        standalone_build=binary, study_sha256=digest(a.study),
        resident_limits_sha256=digest(a.resident_limits) if a.resident_limits else None,
        materialization_ceiling_bytes=a.max_materialize_bytes,
        checkpoint_max_bytes=a.checkpoint_max_bytes if a.checkpoint_max_bytes is not None else (2**32 if standalone else None),
        tracking="project JSON; execution-flows §1", load_average_before=list(os.getloadavg()))
    write_json(out/"workload.json", r["plan"]); write_json(out/"inventory.json", r["inventory"])
    write_json(out/"run.json", record)
    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS=str(r["execution"]["requested"]["intraop_threads"]),
               MKL_NUM_THREADS=str(r["execution"]["requested"]["intraop_threads"]), OPENBLAS_NUM_THREADS="1",
               PYTHONPATH=os.pathsep.join((str(root/"python"), os.environ.get("PYTHONPATH", ""))))

    def run(name, args):
        with (out/(name+".log")).open("w") as log:
            with subprocess.Popen(list(map(str, args)), cwd=out, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True) as child:
                try:
                    code = child.wait(timeout=r["measurement"]["timeout_seconds"])
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                    code = 124
        record["stages"].append(dict(name=name, exit_code=code, log_sha256=digest(out/(name+".log"))))
        write_json(out/"run.json", record)
        if code:
            raise RuntimeError(name+" failed with exit code "+str(code))

    try:
        if standalone and not a.fixture:
            run("export", [sys.executable, root/"scripts/export_weighted_benchmark.py", "--workload", out/"workload.json",
                "--native-library", a.export_library, "--binary", "--max-materialize-bytes", a.max_materialize_bytes,
                "--output", fixture])
        run("measurement", argv)
        result = json.loads((out/"measurement/result.json").read_text())
        accept_result(r, result)
        if source_state(root) != (source, dirty):
            raise RuntimeError("source changed during study")
        record.update(state="passed", result="measurement/result.json", result_sha256=digest(out/"measurement/result.json"))
    except BaseException as error:
        record.update(state="failed", error=repr(error)); raise
    finally:
        record.update(finished=now(), load_average_after=list(os.getloadavg()))
        if (out/"measurement/result.json").exists():
            record.update(result="measurement/result.json", result_sha256=digest(out/"measurement/result.json"))
        write_json(out/"run.json", record)


if __name__ == "__main__":
    main()

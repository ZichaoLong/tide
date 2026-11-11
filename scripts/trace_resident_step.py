"""Locate a resident failure at a public operation boundary, using the real step.

One diagnostic step, no reference trajectory and no performance qualification.
Run in a frozen checkout and an allocated device environment for long workloads.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

from durable_records import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--site")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    from tidegraph.experiment.config import resolve
    resolved = resolve(args.config, site=args.site)
    effective = resolved["execution"]["effective"]
    if effective["placement"] != "resident" or effective["client"] != "python-native":
        parser.error("requires the python-native resident owner; standalone uses its own executable")
    from tidegraph.experiment.site import launch_environment
    environment = launch_environment(resolved["site"], resolved["execution"])
    if any(os.environ.get(k) != v for k, v in environment.items()):
        os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
                  dict(os.environ, **environment))
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"resolved.json", resolved)
    record = dict(schema="tide-resident-step-diagnostic-v1", state="running", events=[],
                  config_id=resolved["config_id"], pid=os.getpid(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  affinity=sorted(os.sched_getaffinity(0)),
                  visibility=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                  scope="real public step with boundary records and one extra metrics read after advance; not throughput or correctness qualification")

    def save():
        write_json(out/"result.json", record)

    def observe(name, operation):
        def wrapped(*values, **keywords):
            event = dict(operation=name, state="running")
            record["events"].append(event)
            save()
            start = time.monotonic()
            try:
                value = operation(*values, **keywords)
                event["state"] = "completed"
                return value
            except Exception as error:
                event.update(state="failed", error=dict(type=type(error).__name__, message=str(error)))
                raise
            finally:
                event["host_seconds"] = time.monotonic()-start
                save()
        return wrapped

    owner = None
    save()
    try:
        from tidegraph.experiment.engine import _configure, create
        from tidegraph.experiment.admission import preflight
        from tidegraph.experiment.identity import capture
        record["runtime"] = _configure(resolved)
        record["admission"] = preflight(resolved, out, start_window=0)
        if record["admission"]["refused"]:
            raise ValueError("capacity preflight: "+"; ".join(record["admission"]["refused"]))
        write_json(out/"identity.json", capture(resolved["site"]["paths"]))
        owner = observe("construct", create)(resolved)
        session = owner.owner
        for name in ("advance", "cotangents", "backward", "step"):
            if hasattr(session, name):
                setattr(session, name, observe(name, getattr(session, name)))
        advance = session.advance

        def with_metrics(*values, **keywords):
            result = advance(*values, **keywords)
            record["forward_metrics"] = dict(session.metrics())
            save()
            return result

        session.advance = with_metrics
        observe("public_step", owner.step)(0)
        record["backward_statistics"] = getattr(owner, "last_backward_statistics", None)
        record["state"] = "completed"
    except Exception as error:
        record.update(state="failed", error=dict(type=type(error).__name__, message=str(error)),
                      traceback=traceback.format_exc())
    finally:
        if owner is not None:
            try:
                observe("close", owner.close)()
            except Exception as error:
                record.update(state="failed", cleanup_error=repr(error))
        save()
    print(json.dumps(dict(state=record["state"], result=str(out/"result.json"))))
    return int(record["state"] != "completed")


if __name__ == "__main__":
    sys.exit(main())

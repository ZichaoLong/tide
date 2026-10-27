#!/usr/bin/env python3
"""Observe process launch -> runtime-ready separately from complete window timing.

The marker follows DSO verification and a synchronized first backend operation,
before fixture construction. The observed interval includes exec, imports/loader,
argument/source checks and pipe delivery; this is process-cold, not OS-cache-cold.
"""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time
from durable_records import write_json
from source_identity import source_state

READY = b"TIDE_WEIGHTED_RUNTIME_READY_V1"


def observe(command, log, timeout):
    """Bounded child observation; a ready marker alone never implies success."""
    start = time.perf_counter()
    ready = []
    timed_out = False
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             start_new_session=True)
    pending = b""
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(child.stdout, selectors.EVENT_READ)
            while selector.get_map():
                left = timeout-(time.perf_counter()-start)
                if left <= 0:
                    timed_out = True
                    os.killpg(child.pid, signal.SIGKILL)
                    break
                for key, _ in selector.select(min(left, 1)):
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    observed = time.perf_counter()-start
                    log.write(chunk)
                    pending += chunk
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        if line == READY:
                            ready.append(observed)
            try:
                code = child.wait(timeout=max(.01, timeout-(time.perf_counter()-start)))
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(child.pid, signal.SIGKILL)
                code = child.wait(timeout=5)
        log.flush()
        os.fsync(log.fileno())
        return dict(state="passed" if code == 0 and len(ready) == 1 and not timed_out else "failed",
                    exit_code=code, timed_out=timed_out, ready_markers=len(ready),
                    launch_to_runtime_ready_seconds=ready[0] if len(ready) == 1 else None,
                    process_seconds=time.perf_counter()-start)
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)
        child.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not 0 < args.timeout_seconds <= 3600:
        parser.error("command and finite timeout in (0,3600] required")
    root = Path(__file__).resolve().parents[1]
    source, dirty = source_state(root)
    if dirty:
        parser.error("process observation requires clean immutable source")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    now = lambda: datetime.now(timezone.utc).isoformat()
    record = dict(schema="tide-weighted-process-observation-v1", state="running", source=source,
                  dirty=dirty, command=command, cwd=str(Path.cwd()), started=now(),
                  timeout_seconds=args.timeout_seconds, load_average_before=list(os.getloadavg()),
                  boundary=__doc__, tracking="project JSON; execution-flows §1")
    write_json(out/"result.json", record)
    try:
        with (out/"process.log").open("xb") as log:
            record.update(observe(command, log, args.timeout_seconds))
        if source_state(root) != (source, dirty):
            raise RuntimeError("source changed during process observation")
    except BaseException as error:
        record.update(state="failed", error=repr(error))
        raise
    finally:
        record.update(finished=now(), load_average_after=list(os.getloadavg()))
        write_json(out/"result.json", record)
    return 0 if record["state"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

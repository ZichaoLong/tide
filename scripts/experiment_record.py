"""Small project-owned benchmark records; optional local Trackio projection.

Adapted from the experiment skill's portable recording contract. Fresh runs only;
raw C++ metric events remain authoritative and are never rewritten for Trackio.
"""
import datetime
import json
import math
import os
from pathlib import Path
import tempfile
import warnings


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_json(path, value):
    text = json.dumps(value, indent=2, allow_nan=False)+"\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text); handle.flush(); os.fsync(handle.fileno())
        temporary.replace(path)
        if path.read_text() != text:
            raise OSError("record read-back mismatch")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_events(path, run_id):
    events = []
    for line in path.read_text().splitlines():
        event = json.loads(line)
        metrics = event.get("metrics")
        if (event.get("schema_version") != 1 or event.get("run_id") != run_id
                or type(event.get("sequence")) is not int or event["sequence"] != len(events)
                or type(event.get("step")) is not int or event["step"] != len(events)
                or not isinstance(metrics, dict) or not metrics
                or any(not isinstance(k, str) or not k or type(v) not in (int, float) or not math.isfinite(v)
                       for k, v in metrics.items())):
            raise ValueError("invalid benchmark metric event")
        stamp = datetime.datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
        elapsed = event.get("elapsed_seconds")
        if (stamp.tzinfo is None or type(elapsed) not in (int, float) or not math.isfinite(elapsed)
                or elapsed < 0 or (events and elapsed < events[-1]["elapsed_seconds"])):
            raise ValueError("invalid benchmark metric time")
        events.append(event)
    return events


class LocalTrackio:
    def __init__(self, mode, directory, project, name, config):
        self.mode, self.client = mode, None
        self.record = {"mode": mode, "backend": "trackio", "project": project, "run_name": name,
                       "destination_kind": "local", "local_data_root": str(directory.resolve()),
                       "storage_mode": "auto", "status": "disabled" if mode == "off" else "not-started",
                       "delivery_acknowledged": False}
        self.project, self.name, self.config = project, name, config

    def attempt(self, stage, function):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always"); function()
            if caught:
                raise RuntimeError("Trackio emitted warnings; delivery is uncertain")
        except Exception as error:
            self.record.update(status="degraded", failure_stage=stage, failure_type=type(error).__name__)
            if self.mode == "required":
                raise RuntimeError(f"required Trackio failed during {stage}: {type(error).__name__}") from error

    def start(self):
        if self.mode == "off":
            return
        def initialize():
            if os.environ.get("TRACKIO_SERVER_URL") or os.environ.get("TRACKIO_SPACE_ID"):
                raise ValueError("this benchmark adapter supports a local Trackio projection only")
            os.environ["TRACKIO_DIR"] = self.record["local_data_root"]
            os.environ["TRACKIO_STORAGE_MODE"] = self.record["storage_mode"]
            import trackio
            self.client = trackio
            run = trackio.init(project=self.project, name=self.name, config=self.config,
                               resume="never", embed=False, auto_log_cpu=False, auto_log_gpu=False)
            self.record.update(status="healthy", run_id=getattr(run, "id", None),
                               version=getattr(trackio, "__version__", "unknown"))
        self.attempt("init", initialize)

    def project_events(self, events):
        if self.client is None or self.record["status"] != "healthy":
            return
        for event in events:
            self.attempt("log", lambda: self.client.log(event["metrics"], step=event["step"]))
            if self.record["status"] != "healthy":
                break

    def finish(self):
        if self.client is not None:
            self.attempt("finish", self.client.finish)

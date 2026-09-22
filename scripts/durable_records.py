"""Small dependency-free persistence boundary for handoffs and job records."""
import json
import os
from pathlib import Path
import tempfile


def replace_text(path, text):
    """Publish a whole record; preserve the previous value before replacement.

    A directory-fsync failure is reported after publication; the new complete
    value may already exist. Concurrent writers need external coordination.
    """
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path, record):
    replace_text(path, json.dumps(record, indent=2, allow_nan=False) + "\n")


def read_jobs(directory):
    """Retain healthy records even if a different historical file is damaged."""
    records, errors = [], []
    for path in Path(directory).glob("*/status.json"):
        try:
            data = json.loads(path.read_text())
            if not isinstance(data, dict) or data.get("state") not in {"starting", "running", "passed", "failed", "cancelled"}:
                raise ValueError("unknown job record/state")
            if not isinstance(_record_time(data), str) or not isinstance(data.get("source"), str):
                raise ValueError("invalid job source/start metadata")
            if data["state"] in {"passed", "failed", "cancelled"}:
                code = data.get("exit_code")
                if type(code) is not int or (code == 0) != (data["state"] == "passed"):
                    raise ValueError("job terminal state/exit code disagree")
            records.append((path, data))
        except (OSError, ValueError, TypeError) as error:
            errors.append((path, str(error)))
    records.sort(key=lambda item: _record_time(item[1]), reverse=True)
    return records, errors


def _record_time(data):
    # A documented launcher failure may have no process start time. Keep its
    # observed terminal time distinct; never turn it into an invented start.
    if (data.get("record_kind") == "postmortem from observed terminal systemd status"
            and data.get("state") in {"passed", "failed", "cancelled"} and "started" not in data):
        return data.get("observed")
    return data.get("started")

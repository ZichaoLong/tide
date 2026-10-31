"""Dependency-free atomic records shared by installed tools and launchers."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile


def now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4*1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def replace_text(path, text):
    """Fsync a same-directory staging file before replacing a complete record."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix="."+path.name+".", suffix=".tmp", dir=path.parent)
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


def write_json(path, value, *, exclusive=False):
    path = Path(path)
    if exclusive and path.exists():
        raise FileExistsError(f"refusing to replace {path}; choose a new output path")
    replace_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}:{error.lineno}:{error.colno}: invalid JSON: {error.msg}") from error

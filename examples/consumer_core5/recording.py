"""Application-owned records; never write into the dependency checkout."""
from dataclasses import fields, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def publish(path, writer):
    """Fsync then exclusively publish a whole file; refuse overwrites/races."""
    path = Path(path)
    fd, staging = tempfile.mkstemp(dir=path.parent, prefix='.'+path.name, suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(staging, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(staging).unlink(missing_ok=True)


def json_record(path, value):
    data = (json.dumps(value, indent=2, allow_nan=False)+'\n').encode()
    publish(path, lambda stream: stream.write(data))


def tensor_record(path, value):
    import torch
    publish(path, lambda stream: torch.save(value, stream))


def snapshot(value):
    """Actual values only; preserve None, int64, routes and all state fields."""
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if is_dataclass(value):
        return {f.name: snapshot(getattr(value, f.name)) for f in fields(value) if f.name != 'stats'}
    if isinstance(value, dict):
        return {k: snapshot(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(snapshot(v) for v in value)
    return value


def application_identity():
    root = Path(__file__).resolve().parent
    names = ('train.py', 'task.py', 'checkpoint.py', 'dependency.py', 'recording.py')
    return fingerprint({name: sha256(root/name) for name in names})

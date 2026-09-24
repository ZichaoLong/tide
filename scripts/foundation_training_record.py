"""Small v2 training audit artifacts, separate from resumable checkpoints."""
import hashlib
import os
from pathlib import Path
import tempfile
import torch


def publish(path, model):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    owners, aliases = {}, {}
    for name, p in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(p), []).append(name)
        owners.setdefault(id(p), (name, p))
    record = dict(schema="tide-training-observation-v1",
                  values={name: p.detach().clone() for name, p in owners.values()},
                  gradients={name: None if p.grad is None else p.grad.detach().clone() for name, p in owners.values()},
                  aliases=list(aliases.values()), scope="final-window owners/gradients, not a resume checkpoint")
    fd, temporary = tempfile.mkstemp(prefix=".training-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(record, stream); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic, no overwrite, same directory/filesystem.
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())

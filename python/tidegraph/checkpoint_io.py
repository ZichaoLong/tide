"""Durable exclusive publication: a checkpoint path never exposes partial bytes."""
import errno
import os
from pathlib import Path
import tempfile
import torch


def publish(path, record):
    target = Path(path)
    if os.path.lexists(target):
        raise FileExistsError(errno.EEXIST, "checkpoint already exists", str(target))
    descriptor, staging = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as output:
            torch.save(record, output)
            output.flush()
            os.fsync(output.fileno())
        # Same-directory hard link provides atomic, no-overwrite publication,
        # including when another writer wins after the initial existence check.
        os.link(staging, target)
    finally:
        os.unlink(staging)
    directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)

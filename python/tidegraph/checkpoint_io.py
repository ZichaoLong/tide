"""Durable exclusive publication: a checkpoint path never exposes partial bytes."""
import errno
import os
from pathlib import Path
import tempfile
import torch
from .checkpoint_metrics import phase


def publish(path, record):
    target = Path(path)
    if os.path.lexists(target):
        raise FileExistsError(errno.EEXIST, "checkpoint already exists", str(target))
    descriptor, staging = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as output:
            with phase("io.serialize"):
                torch.save(record, output)
            with phase("io.flush_fsync_file"):
                output.flush()
                os.fsync(output.fileno())
        # Same-directory hard link provides atomic, no-overwrite publication,
        # including when another writer wins after the initial existence check.
        with phase("io.publish_link"):
            os.link(staging, target)
    finally:
        os.unlink(staging)
    directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with phase("io.fsync_directory"):
            os.fsync(directory)
    finally:
        os.close(directory)

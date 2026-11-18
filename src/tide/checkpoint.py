"""Ordinary trusted training checkpoints for the new model API."""
import os
from pathlib import Path
import tempfile

import torch

from .config import ModelConfig
from .model import TideModel, detach_state


def save_checkpoint(path, model, *, state=None, optimizer=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(format="tide-instances-1", config=model.config.to_dict(),
                   parameters=model.state_dict(), state=detach_state(state),
                   optimizer=None if optimizer is None else optimizer.state_dict())
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix="."+path.name+".")
    try:
        with os.fdopen(fd,"wb") as stream:
            torch.save(payload,stream)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary,path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path, *, device="cpu"):
    """Load a file created by save_checkpoint. Only load trusted pickle files."""
    payload = torch.load(path,map_location=device,weights_only=False)
    if payload["format"]!="tide-instances-1":
        raise ValueError("unknown checkpoint format")
    dtype = next(v.dtype for v in payload["parameters"].values() if v.is_floating_point())
    model = TideModel(ModelConfig.from_dict(payload["config"])).to(device=device,dtype=dtype)
    model.load_state_dict(payload["parameters"])
    return model,payload["state"],payload["optimizer"]

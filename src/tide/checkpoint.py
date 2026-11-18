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
    placement = getattr(model,"_placement_spec",None)
    if placement is not None and optimizer is not None and not hasattr(optimizer,"_tide_placement"):
        raise ValueError("placed training checkpoints require placement_adamw")
    payload = dict(format="tide-instances-2" if placement is not None else "tide-instances-1",
                   config=model.config.to_dict(), placement=placement,
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


def load_checkpoint(path, *, device="cpu", placement_devices=None):
    """Load a trusted checkpoint, optionally remapping its layer shard devices.

    Sharded checkpoints keep the saved layer/parameter/optimizer layout. Supply
    ``placement_devices`` for each saved partition, then construct the returned
    model's optimizer with ``placement_adamw`` before loading its slots. Without
    an explicit mapping every partition is placed on ``device`` (CPU by default).
    Dense format-1 checkpoints retain their original model API.
    """
    # Decode on CPU first: a multi-device checkpoint must never transiently put
    # all parameters or optimizer slots on one capacity-limited accelerator.
    payload = torch.load(path,map_location="cpu",weights_only=False)
    if payload["format"] not in {"tide-instances-1","tide-instances-2"}:
        raise ValueError("unknown checkpoint format")
    dtype = next(v.dtype for v in payload["parameters"].values() if v.is_floating_point())
    model = TideModel(ModelConfig.from_dict(payload["config"])).to(dtype=dtype)
    placement = payload.get("placement")
    if placement is not None:
        from .placement import place_model, place_state
        if placement["kind"] != "hb-layer-shards-1":
            raise ValueError("unknown checkpoint placement layout")
        devices = ([device]*len(placement["devices"]) if placement_devices is None else placement_devices)
        if len(devices) != len(placement["devices"]):
            raise ValueError("restored placement must preserve the saved number of partitions")
        place_model(model,devices,layer_partitions=placement["layer_partitions"],
                    transport=placement.get("requested_transport",placement.get("transport","direct")))
        model.load_state_dict(payload["parameters"])
        state = place_state(model,payload["state"])
    else:
        if placement_devices is not None:
            raise ValueError("dense checkpoints do not contain a sharded optimizer layout")
        from .placement import move_tensors
        model.to(device=device)
        model.load_state_dict(payload["parameters"])
        state = move_tensors(payload["state"],device)
    return model,state,payload["optimizer"]

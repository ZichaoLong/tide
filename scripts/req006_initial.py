"""Optional immutable post-prime material for paired REQ-006 diagnostics.

This is a diagnostic snapshot, not the downstream application's checkpoint or
the foundation continuation codec. Actions/modes and physical device IDs may
differ; configuration, inputs, prime origin, runtime and model state may not.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from req006_material import tensor_digest

SCHEMA = "tide-req006-initial-state-v1"


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _plain_json(value):
    # TorchVersion is a str subclass: serializing it directly requires a custom
    # pickle global, which weights_only=True intentionally does not allow.
    return json.loads(json.dumps(value, allow_nan=False))


def initial_identity(config, material, *, progress, input_seed, prime_updates, runtime, downstream):
    """Freeze semantic/numeric material, allowing measure/profile/census pairing."""
    runtime_keys = ("backend", "dtype", "host_architecture", "torch", "torch_git", "python",
                    "cxx11_abi", "torch_npu", "cann_version", "device_name", "matmul_allow_hf32",
                    "cuda_runtime", "cudnn", "capability", "compiled_architectures",
                    "matmul_allow_tf32", "cudnn_allow_tf32", "threads", "float32_matmul_precision",
                    "deterministic_algorithms")
    return _plain_json(dict(configuration_sha256=canonical_digest(config),
                input_sha256=tensor_digest(material), input_shape=list(material["input_ids"].shape),
                progress_tokens=progress, input_seed=input_seed, model_seed=config["seed"],
                dtype=config["dtype"], prime_updates=prime_updates,
                downstream_source=downstream["commit"], downstream_content_sha256=downstream["content_sha256"],
                runtime={k: runtime[k] for k in runtime_keys if k in runtime},
                private_state="independent document blocks; no cross-update private state"))


def _file_digest(stream):
    stream.seek(0)
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(2**20), b""):
        digest.update(block)
    return digest.hexdigest()


def _tensor_bytes(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.numel()*value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(v) for v in value)
    return 0


def _metadata(path, document, digest):
    return dict(schema=SCHEMA, path_hint=str(path.resolve()), bytes=path.stat().st_size,
                sha256=digest, identity_sha256=document["identity_sha256"],
                initial_state_sha256=document["initial_state_sha256"],
                weights_sha256=document["weights_sha256"], producer=document["producer"])


def save_initial(path, state, identity, *, producer):
    """Fsync staging, then publish without overwriting even a concurrent writer.

    A directory-fsync failure is reported after publication; the complete new
    file may then exist. Existing files are never replaced. No raw model writes
    occur in the measured interval.
    """
    import torch
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError("initial-state output already exists: "+str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(path.parent).free < _tensor_bytes(state)+2**20:
        raise OSError("insufficient filesystem space for initial-state staging")
    identity, producer = _plain_json(identity), _plain_json(producer)
    document = dict(schema=SCHEMA, identity=identity, identity_sha256=canonical_digest(identity),
                    state=state, initial_state_sha256=tensor_digest(state),
                    weights_sha256=tensor_digest(state["model"]), producer=dict(producer))
    fd, name = tempfile.mkstemp(prefix="."+path.name+".", suffix=".tmp", dir=path.parent)
    staging = Path(name)
    try:
        with os.fdopen(fd, "w+b") as stream:
            torch.save(document, stream)
            stream.flush()
            os.fsync(stream.fileno())
            digest = _file_digest(stream)
        os.link(staging, path)  # Atomic no-replace publication on the same filesystem.
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return _metadata(path, document, digest)
    finally:
        staging.unlink(missing_ok=True)


def load_initial(path, identity, *, max_bytes=8*2**30):
    """Validate identity and all state bytes before the caller restores owners."""
    import torch
    path = Path(path)
    if path.stat().st_size > max_bytes:
        raise ValueError("initial-state artifact exceeds byte bound")
    with path.open("rb") as stream:
        digest = _file_digest(stream)
        stream.seek(0)
        document = torch.load(stream, map_location="cpu", weights_only=True)
        if _file_digest(stream) != digest:
            raise ValueError("initial-state file changed while reading")
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ValueError("unknown initial-state schema")
    actual_identity = document.get("identity")
    if actual_identity != identity:
        keys = sorted(k for k in identity if not isinstance(actual_identity, dict)
                      or actual_identity.get(k) != identity[k])
        raise ValueError("initial-state identity mismatch: "+", ".join(keys))
    if document.get("identity_sha256") != canonical_digest(identity):
        raise ValueError("initial-state identity digest mismatch")
    state = document.get("state")
    if not isinstance(state, dict) or set(state) != {"model", "optimizer", "cpu_rng", "device_rng"}:
        raise ValueError("initial-state payload has unexpected fields")
    if (tensor_digest(state) != document.get("initial_state_sha256")
            or tensor_digest(state["model"]) != document.get("weights_sha256")):
        raise ValueError("initial-state payload digest mismatch")
    return state, _metadata(path, document, digest)

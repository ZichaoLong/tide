"""Atomic experiment checkpoint generations around public value checkpoints."""
import os
from pathlib import Path
import shutil
import tempfile
from .records import digest, read_json, write_json
from ..checkpoint_metrics import phase

SCHEMA = "tide-experiment-checkpoint-v1"
KEYS = ("model_id", "workload_id", "task_id", "execution_id")


def locate(path):
    path = Path(path).resolve()
    if (path/"latest.json").is_file():
        name = read_json(path/"latest.json")["directory"]
        if Path(name).name != name:
            raise ValueError("checkpoint pointer must name a local generation")
        path = path/name
    elif (path/"checkpoints"/"latest.json").is_file():
        return locate(path/"checkpoints")
    record = read_json(path/"metadata.json")
    if record.get("schema") != SCHEMA:
        raise ValueError("resume: unsupported experiment checkpoint schema")
    for name, expected in record["files"].items():
        if Path(name).name != name or digest(path/name) != expected:
            raise ValueError("resume: checkpoint file identity differs: "+name)
    return path, record


def validate(path, resolved, identity):
    path, record = locate(path)
    for name in KEYS:
        if record[name] != resolved[name]:
            raise ValueError(f"resume: {name} differs; exact continuation requires the same model/task/execution")
    if record["mode"] != resolved["measurement"]["mode"]:
        raise ValueError("resume: inference/training mode differs")
    if record["package_sha256"] != identity["package"]["installed_files_sha256"]:
        raise ValueError("resume: installed code differs; use the fixed dependency lock for exact continuation")
    if record["builds"] != {k: v["binary_sha256"] for k, v in identity["builds"].items()}:
        raise ValueError("resume: binary identity differs; portable handoff is a separate contract")
    return path, record


def save(owner, output, resolved, identity, next_window):
    root = Path(output)/"checkpoints"
    root.mkdir(exist_ok=True)
    name = f"window-{next_window:06d}"
    final = root/name
    if final.exists():
        raise FileExistsError(final)
    temporary = Path(tempfile.mkdtemp(prefix=".staging-", dir=root))
    old = read_json(root/"latest.json") if (root/"latest.json").exists() else None
    try:
        with phase("generation.owner_save"):
            owner.save(temporary)
        with phase("generation.hash_files"):
            files = {p.name: digest(p) for p in sorted(temporary.iterdir()) if p.is_file()}
        size = sum(p.stat().st_size for p in temporary.iterdir() if p.is_file())
        if size > resolved["resources"]["checkpoint_max_bytes"]:
            raise ValueError(f"resources.checkpoint_max_bytes: actual checkpoint {size} exceeds the declared budget")
        record = dict(schema=SCHEMA, **{k: resolved[k] for k in KEYS}, mode=resolved["measurement"]["mode"],
                      package_sha256=identity["package"]["installed_files_sha256"],
                      builds={k: v["binary_sha256"] for k, v in identity["builds"].items()},
                      next_window=next_window, span=resolved["plan"]["span"], files=files, bytes=size)
        with phase("generation.publish_metadata_pointer"):
            write_json(temporary/"metadata.json", record)
            os.rename(temporary, final)
            write_json(root/"latest.json", dict(directory=name, next_window=next_window))
        # Only our previous, published generation is retired, after the new
        # fsynced pointer is durable. A stopped run always has its last boundary.
        if old is not None and old["directory"] != name:
            old_name = old["directory"]
            if Path(old_name).name != old_name or not old_name.startswith("window-"):
                raise ValueError("invalid previous checkpoint generation")
            with phase("generation.retire_previous"):
                shutil.rmtree(root/old_name)
        return final, record
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def exact_next(owner, create, path, index):
    """Serial new owner; raw decode and next full window, outside timers."""
    from ..compare import equivalent
    with phase("resume.snapshot_saved_values"):
        saved = owner.decoded_state()
    with phase("resume.original_next_window"):
        actual, _ = owner.step(index, return_mode="complete")
    with phase("resume.snapshot_next_parameters"):
        params, training = owner.parameters(), owner.training_state()
    with phase("resume.close_original"):
        owner.close()
    with phase("resume.construct_and_load"):
        restored = create(checkpoint=path)
    try:
        with phase("resume.compare_decoded_values"):
            equivalent(saved, restored.decoded_state(), atol=0, rtol=0)
        with phase("resume.restored_next_window"):
            expected, _ = restored.step(index, return_mode="complete")
        with phase("resume.compare_next_window"):
            equivalent(actual, expected, atol=0, rtol=0)
        with phase("resume.compare_next_parameters_optimizer"):
            equivalent(params, restored.parameters(), atol=0, rtol=0)
            equivalent(training, restored.training_state(), atol=0, rtol=0)
    finally:
        restored.close()
    return dict(state="passed", exact=True, next_window=index,
                scope="serial new owner: full decoded values and next complete window/parameters/gradients/optimizer/task")

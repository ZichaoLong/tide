"""Bounded, resumable HTTPS asset reads with publisher hashes and atomic completion.

The caller selects a proxy through its launch environment. Proxy credentials and
signed redirect URLs are never included in persistent records or exceptions.
"""
import fcntl
import hashlib
import os
from pathlib import Path
from .asset_transport import receive
from .catalog import local_path, model, safe_relative
from .resources import disk_admission
from ..experiment.records import fingerprint, now, read_json, write_json


def _hasher(spec):
    if spec.get("sha256"):
        return hashlib.sha256(), spec["sha256"]
    if not spec.get("git_blob"):
        raise ValueError("asset has neither SHA256 nor Git blob identity")
    value = hashlib.sha1()
    value.update(f"blob {spec['bytes']}\0".encode())
    return value, spec["git_blob"]


def verify_file(path, spec):
    if not path.is_file() or path.is_symlink() or path.stat().st_size != spec["bytes"]:
        return False
    digest, expected = _hasher(spec)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024**2), b""):
            digest.update(block)
    return digest.hexdigest() == expected


def download_file(url, path, spec, progress, *, opener=None, check_budget=lambda _: None,
                  chunk_bytes=32*1024**2, retries=6, event=lambda _: None, **transport):
    """One bounded attempt. Publish only after exact size and publisher hash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if verify_file(path, spec):
            return "verified_existing"
        raise RuntimeError(f"existing asset fails publisher hash: {path.name}")
    partial = path.with_name(path.name+".part")
    if partial.is_symlink():
        raise ValueError("partial asset must not be a symbolic link")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > spec["bytes"]:
        raise RuntimeError("partial asset is larger than its published size")
    check_budget(spec["bytes"]-offset)
    digest, expected = _hasher(spec)
    if offset:
        with partial.open("rb") as stream:
            for block in iter(lambda: stream.read(4*1024**2), b""):
                digest.update(block)
    offset = receive(url, partial, spec['bytes'], digest, progress, check_budget,
                     opener=opener, chunk_bytes=chunk_bytes, retries=retries, event=event, **transport)
    if digest.hexdigest() != expected:
        raise RuntimeError(f"publisher hash mismatch: {path.name}; partial retained")
    os.replace(partial, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    progress(offset)
    return "downloaded_verified"


def fetch(key, root, project_root, *, project_gib=80, account_gib=300, chunk_mib=32, retries=6, timeout_seconds=60):
    spec = model(key)
    destination = local_path(root, key)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination/".fetch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = destination/'asset-state.json'
        if previous.exists():
            old = read_json(previous)
            (destination/'attempts').mkdir(exist_ok=True)
            write_json(destination/'attempts'/(fingerprint(old)+'.json'), old)
        record = dict(schema="tide-lm-asset-state-v1", model=key, revision=spec["revision"],
                      manifest_id=fingerprint(spec), started=now(), state="running", files={},
                      transport=dict(chunk_mib=chunk_mib, retries_per_file=retries,timeout_seconds=timeout_seconds), events=[])
        def save():
            write_json(destination/"asset-state.json", record)
        def budget(additional):
            return disk_admission(project_root, additional, project_gib=project_gib, account_gib=account_gib)
        save()
        try:
            for entry in spec["files"]:
                relative = safe_relative(entry["path"])
                path = destination/relative
                if path.resolve().parent != destination.resolve() and destination.resolve() not in path.resolve().parents:
                    raise ValueError("asset path escaped destination")
                record.update(current_file=entry["path"], current_bytes=0, current_total=entry["bytes"])
                def progress(count):
                    record.update(current_bytes=count, updated=now())
                    save()
                def event(value):
                    record['events'].append(dict(file=entry['path'], time=now(), **value)); save()
                url = f"https://huggingface.co/{spec['id']}/resolve/{spec['revision']}/{entry['path']}"
                state = download_file(url, path, entry, progress, check_budget=budget,
                                      chunk_bytes=chunk_mib*1024**2, retries=retries, timeout_seconds=timeout_seconds,event=event)
                record["files"][entry["path"]] = dict(state=state, bytes=entry["bytes"], mtime_ns=path.stat().st_mtime_ns)
                print(f"{key}: {entry['path']} {state}", flush=True)
            record.update(state="complete", finished=now())
            save()
        except BaseException as error:
            record.update(state="failed", finished=now(), error=str(error)[:1000])
            save()
            raise
    return record


def ready(key, root):
    path = local_path(root, key)/"asset-state.json"
    if not path.exists():
        return False
    record = read_json(path)
    if record.get("state") != "complete" or record.get("manifest_id") != fingerprint(model(key)):
        return False
    for name, entry in record.get("files", {}).items():
        asset = path.parent/safe_relative(name)
        if not asset.is_file() or asset.stat().st_size != entry["bytes"] or asset.stat().st_mtime_ns != entry.get("mtime_ns"):
            return False
    return len(record.get("files", {})) == len(model(key)["files"])

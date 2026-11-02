"""Bounded, resumable HTTPS asset reads with publisher hashes and atomic completion.

The caller selects a proxy through its launch environment. Proxy credentials and
signed redirect URLs are never included in persistent records or exceptions.
"""
import fcntl
import hashlib
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
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


def download_file(url, path, spec, progress, *, opener=None, check_budget=lambda _: None):
    """One attempt. Keep an incomplete file for an explicitly resumed command."""
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
    if offset < spec["bytes"]:
        headers = {"User-Agent": "tidegraph-lm-assets/1", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        opener = opener or urllib.request.build_opener()
        try:
            response = opener.open(urllib.request.Request(url, headers=headers), timeout=60)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"asset HTTP {error.code}: {path.name}") from None
        except Exception as error:
            raise RuntimeError(f"asset connection {type(error).__name__}: {path.name}") from None
        with response, partial.open("ab") as stream:
            if offset and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                raise RuntimeError("server did not honor the requested resume offset")
            length = response.headers.get("Content-Length")
            if length is not None and int(length) != spec["bytes"]-offset:
                raise RuntimeError("asset Content-Length differs from pinned manifest")
            last = 0.
            while offset < spec["bytes"]:
                try:
                    block = response.read(min(1024**2, spec["bytes"]-offset))
                except Exception as error:
                    stream.flush()
                    os.fsync(stream.fileno())
                    raise RuntimeError(f"asset body {type(error).__name__}: {path.name}") from None
                if not block:
                    raise RuntimeError("asset ended before its published size")
                stream.write(block)
                digest.update(block)
                offset += len(block)
                if time.monotonic()-last >= 30:
                    stream.flush()
                    check_budget(0)
                    progress(offset)
                    last = time.monotonic()
            stream.flush()
            os.fsync(stream.fileno())
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


def fetch(key, root, project_root, *, project_gib=80, account_gib=300):
    spec = model(key)
    destination = local_path(root, key)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination/".fetch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = dict(schema="tide-lm-asset-state-v1", model=key, revision=spec["revision"],
                      manifest_id=fingerprint(spec), started=now(), state="running", files={})
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
                url = f"https://huggingface.co/{spec['id']}/resolve/{spec['revision']}/{entry['path']}"
                state = download_file(url, path, entry, progress, check_budget=budget)
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

"""Offline admission of an original checkpoint from a content-verified East OBS catalog.

An archive is enabled by a SHA-bound, itself uploaded and verified index receipt.
Outside this scope the existing local-file checks apply unchanged.
"""
import contextlib
import contextvars
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path


_current = contextvars.ContextVar("tide_checkpoint_archive", default=None)


def _checked(condition, message):
    if not condition:
        raise ValueError(message)


def _timestamp(value):
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() > 0
    except (AttributeError, TypeError, ValueError):
        return False


def _read_catalog(ref):
    proof_bytes = Path(ref["path"]).read_bytes()
    _checked(hashlib.sha256(proof_bytes).hexdigest() == ref["sha256"], "archive index receipt SHA differs")
    proof = json.loads(proof_bytes)
    raw = Path(proof["source"]).read_bytes()
    _checked(proof.get("schema") == "tide-r3-verified-index/v1" and proof.get("etag")
             and _timestamp(proof.get("verified")) and len(raw) == proof["size"]
             and hashlib.sha256(raw).hexdigest() == proof["sha256"], "verified archive index changed")
    data = json.loads(gzip.decompress(raw))
    prefix = "obs://" + data["east_bucket"] + "/" + data["east_suffix_prefix"].strip("/") + "/"
    _checked(data.get("schema") == "tide-r3-verified-inputs/v1" and data.get("east_region") == "cn-east-4"
             and data.get("published_selection", {}).get("passed") is True
             and proof["paths"] == len(data["paths"]) and proof["objects"] == len(data["objects"])
             and proof["uri"] == prefix + proof["key"], "archive index identities or coverage differ")
    return data, prefix


@contextlib.contextmanager
def checkpoint_reads(refs, *, authoritative=False):
    """Missing files can be verified by archive. Explicit remote review can select that copy directly."""
    if not refs:
        yield
        return
    old = _current.get()
    if old and old[0] == refs:
        yield
        return
    state = refs, [_read_catalog(ref) for ref in refs], authoritative
    token = _current.set(state)
    try:
        yield
    finally:
        _current.reset(token)


def archive_record(path, sha, size=None):
    state = _current.get()
    path = Path(path)
    if not state or path.suffix != ".pt" or not path.is_absolute() or not state[2] and path.exists():
        return None
    for data, prefix in state[1]:
        item = data["paths"].get(str(path))
        if item is None:
            continue
        obj = data["objects"].get(sha, {})
        _checked(item["sha256"] == sha == obj.get("sha256") and size in (None, obj.get("size"))
                 and type(obj.get("size")) is int and obj["size"] > 0 and item["size"] == obj["size"]
                 and item["key"] == obj["key"] and obj["key"].rsplit("/", 1)[-1] == sha
                 and obj["uri"] == prefix + obj["key"] and obj.get("etag")
                 and _timestamp(obj.get("content_read_at")) and _timestamp(obj.get("verified")),
                 "archive path, content identity, size or pinned OBS readback differs: " + str(path))
        return obj
    return None


def checkpoint_matches(checkpoint):
    if archive_record(checkpoint["path"], checkpoint["sha256"], checkpoint.get("size_bytes")):
        return True
    from .io import sha256
    return sha256(checkpoint["path"]) == checkpoint["sha256"]

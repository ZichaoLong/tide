"""Dependency-free installed package identity shared by public entry points."""
import hashlib
import json
from pathlib import Path


CONTENT_SCOPE = "recursive package files and relative names; __pycache__ and .pyc/.pyo excluded"


def content_identity(root):
    """Hash actual installed content; no Git, wheel or runtime provenance inferred."""
    root = Path(root)
    files = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4*1024**2), b""):
                digest.update(chunk)
        files[path.relative_to(root).as_posix()] = digest.hexdigest()
    if not files:
        raise ValueError("installed package contains no readable files")
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return dict(installed_files_sha256=hashlib.sha256(encoded).hexdigest(),
                files_sha256=files, content_scope=CONTENT_SCOPE)

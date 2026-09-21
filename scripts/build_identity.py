"""Small shared build identity helpers; no dependency on Torch or skill files."""
import hashlib
from pathlib import Path
import subprocess


def source_hash(root):
    digest = hashlib.sha256()
    files = [root / "CMakeLists.txt"] + sorted((root / "cpp").rglob("*"))
    for path in files:
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def revision(root):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()

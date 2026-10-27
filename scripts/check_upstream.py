#!/usr/bin/env python3
"""Verify the adopted textbook snapshot without Torch, network or old artifacts."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys


def check_entry(root, lock, upstream):
    """Verify one immutable adoption, including nested local-definition files."""
    folder = root / lock["snapshot_directory"]
    if (len(set(lock["sources"])) != len(lock["sources"])
            or set(lock["sources"]) != set(lock["sha256"])):
        raise ValueError("upstream source and hash inventories differ")
    paths = list(folder.rglob("*"))
    if any(p.is_symlink() for p in paths):
        raise ValueError("upstream snapshot cannot contain symlinks")
    if {p.relative_to(folder).as_posix() for p in paths if p.is_file()} != set(lock["sources"]):
        raise ValueError("upstream snapshot inventory differs from the lock")
    for name in lock["sources"]:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != name:
            raise ValueError("upstream filenames must be canonical relative paths")
        data = (folder / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != lock["sha256"][name]:
            raise ValueError(f"upstream snapshot checksum mismatch: {name}")
        if upstream is not None:
            original = subprocess.check_output([
                "git", "-C", str(upstream), "show",
                f'{lock["revision"]}:{lock["directory"]}/{name}',
            ])
            if data != original:
                raise ValueError(f"snapshot differs from pinned upstream: {name}")
    return {"state": "passed", "semantic_version": lock["semantic_version"],
            "revision": lock["revision"], "files": len(lock["sources"]),
            "compared_with_upstream_git": upstream is not None}


def check(root, upstream=None):
    lock = json.loads((root / "docs/upstream.json").read_text())
    result = check_entry(root, lock, upstream)
    result["legacy"] = [check_entry(root, entry, upstream)
                        for entry in lock.get("legacy", [])]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-repo", type=Path,
                        help="also compare with the pinned Git objects in this checkout")
    args = parser.parse_args()
    try:
        print(json.dumps(check(Path(__file__).resolve().parents[1], args.upstream_repo)))
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"upstream verification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

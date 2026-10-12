#!/usr/bin/env python3
"""Verify the adopted textbook snapshot without Torch, network or old artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def check(root, upstream=None):
    lock = json.loads((root / "docs/upstream.json").read_text())
    folder = root / lock["snapshot_directory"]
    if set(lock["sources"]) != set(lock["sha256"]):
        raise ValueError("upstream source and hash inventories differ")
    if {p.name for p in folder.iterdir()} != set(lock["sources"]):
        raise ValueError("upstream snapshot inventory differs from the lock")
    for name in lock["sources"]:
        if Path(name).name != name:
            raise ValueError("upstream filenames must be local basenames")
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

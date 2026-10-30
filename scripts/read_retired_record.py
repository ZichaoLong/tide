#!/usr/bin/env python3
"""Inspect one retired record; never execute commands or restore absolute paths."""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import sys


def records(archive):
    with gzip.open(archive, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--path", help="exact original path label, including any :suffix")
    group.add_argument("--list-prefix", help="list at most 50 original path labels")
    parser.add_argument("--output", type=Path, help="write this record to a new file")
    parser.add_argument("--archive", type=Path, help="explicit archive downloaded from the frozen old history")
    args = parser.parse_args()
    if args.output and not args.path:
        parser.error("--output requires --path")
    if args.output and args.output.exists():
        parser.error("output already exists")
    root = Path(__file__).resolve().parents[1]
    receipt = json.loads((root / "docs/evidence/artifact-retirement-20261012.json").read_text())
    archive = args.archive or root / receipt["archive"]["path"]
    if not archive.is_file():
        parser.error("archive is retained on graph-execution-foundation-old only; "
                     "see docs/history-migration.md and supply --archive PATH (no automatic download)")
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != receipt["archive"]["sha256"]:
        raise ValueError("retired archive checksum mismatch")
    if args.list_prefix is not None:
        count = 0
        for record in records(archive):
            if record["path"].startswith(args.list_prefix):
                if count < 50:
                    print(json.dumps({k: record[k] for k in ("path", "kind", "bytes")}, ensure_ascii=False))
                count += 1
        print(f"{count} matches; at most 50 shown")
        return 0
    target = next((r for r in records(archive) if r["path"] == args.path), None)
    if target is None:
        parser.error("record not found; use --list-prefix")
    print(json.dumps({k: target[k] for k in ("path", "kind", "bytes", "sha256")}, ensure_ascii=False))
    if args.output:
        blob = next(r for r in records(archive)
                    if r["sha256"] == target["sha256"] and "data_b64" in r)
        data = base64.b64decode(blob["data_b64"], validate=True)
        if len(data) != target["bytes"] or hashlib.sha256(data).hexdigest() != target["sha256"]:
            raise ValueError("record checksum mismatch")
        with args.output.open("xb") as stream:
            stream.write(data)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, StopIteration) as error:
        print(f"retired record read failed: {error}", file=sys.stderr)
        sys.exit(1)

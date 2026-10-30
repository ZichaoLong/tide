#!/usr/bin/env python3
"""Independently compare every mapped Git tree, parent and commit metadata.

Requires separate old/new object stores. Does not invoke filter-repo, change
refs, execute historical source, or reconstruct missing results.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess
from durable_records import write_json
from history_map import MANIFEST, ROOT, git, load_map


def commit_parts(repo, oid):
    raw = git(repo, "cat-file", "commit", oid)
    header, message = raw.split(b"\n\n", 1)
    parents, metadata = [], []
    for line in header.split(b"\n"):
        if line.startswith(b"parent "):
            parents.append(line[7:].decode("ascii"))
        elif not line.startswith(b"tree "):
            metadata.append(line)
    return parents, b"\n".join(metadata) + b"\n\n" + message


def tree(repo, oid, removed):
    entries = git(repo, "ls-tree", "-rz", "--full-tree", oid).split(b"\0")
    kept, absent = [], []
    for entry in entries:
        if not entry:
            continue
        _, path = entry.split(b"\t", 1)
        (absent if path == removed else kept).append(entry)
    return b"\0".join(kept), absent


def verify(old_repo, new_repo, manifest=MANIFEST):
    meta, mapping = load_map(manifest)
    old_set = set(git(old_repo, "rev-list", meta["old_head"]).decode().splitlines())
    new_set = set(git(new_repo, "rev-list", meta["filtered_head"]).decode().splitlines())
    if old_set != set(mapping) or new_set != set(mapping.values()):
        raise ValueError("map does not cover exactly both reachable histories")
    removed = meta["removed"]["path"].encode()
    digest = hashlib.sha256()
    removed_commits = 0
    for old, new in sorted(mapping.items()):
        old_parents, old_metadata = commit_parts(old_repo, old)
        new_parents, new_metadata = commit_parts(new_repo, new)
        if [mapping[p] for p in old_parents] != new_parents:
            raise ValueError(f"parent mismatch at {old}")
        if old_metadata != new_metadata:
            raise ValueError(f"commit metadata/message mismatch at {old}")
        old_tree, old_removed = tree(old_repo, old, removed)
        new_tree, new_removed = tree(new_repo, new, removed)
        if new_removed or old_tree != new_tree:
            raise ValueError(f"tree mismatch beyond removed archive at {old}")
        if old_removed:
            removed_commits += 1
            if len(old_removed) != 1 or old_removed[0].split(b"\t")[0] != (
                    "100644 blob " + meta["removed"]["blob"]).encode():
                raise ValueError(f"unexpected archive identity at {old}")
        digest.update(old.encode() + new.encode())
        digest.update(hashlib.sha256(old_tree).digest())
        digest.update(hashlib.sha256(old_metadata).digest())
    reachable = {line.split()[0] for line in git(
        new_repo, "rev-list", "--objects", meta["filtered_head"]).decode().splitlines()}
    if meta["removed"]["blob"] in reachable:
        raise ValueError("removed blob remains reachable")
    blob = git(old_repo, "cat-file", "blob", meta["removed"]["blob"])
    if (len(blob) != meta["removed"]["bytes"] or
            hashlib.sha256(blob).hexdigest() != meta["removed"]["sha256"]):
        raise ValueError("original archive bytes do not match manifest")
    return {"state": "passed", "old_head": meta["old_head"],
            "filtered_head": meta["filtered_head"], "commits": len(mapping),
            "rewritten": sum(a != b for a, b in mapping.items()),
            "archive_present_old_commits": removed_commits,
            "archive_reachable_in_new_history": False,
            "all_other_paths_modes_blobs_equal": True,
            "parents_metadata_messages_equal": True,
            "comparison_sha256": digest.hexdigest(),
            "map_sha256": meta["map"]["sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-repo", type=Path, required=True)
    parser.add_argument("--new-repo", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    try:
        result = verify(args.old_repo, args.new_repo, args.manifest)
        write_json(args.output, result)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"history verification failed: {error}\n")
    print(f"{result['commits']} commits independently verified; {result['rewritten']} rewritten")


if __name__ == "__main__":
    main()

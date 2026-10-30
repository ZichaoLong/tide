"""Foundation commit identities across the archive-only history migration.

Contract: docs/history-migration.md. This resolves source, never test results,
dependency locks, upstream textbook identities or historical binary hashes.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/history/migration-20261030.json"
OID = re.compile(r"[0-9a-f]{40}\Z")


def git(repo, *args):
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], stderr=subprocess.PIPE,
        env={**os.environ, "GIT_NO_REPLACE_OBJECTS": "1"})


def load_map(manifest=MANIFEST):
    manifest = Path(manifest)
    meta = json.loads(manifest.read_text())
    if meta.get("schema_version") != 1:
        raise ValueError("unsupported history map schema")
    data = (manifest.parent / meta["map"]["file"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != meta["map"]["sha256"]:
        raise ValueError("history map checksum mismatch")
    rows = data.decode("ascii").splitlines()
    if not rows or rows[0].split() != ["old", "new"]:
        raise ValueError("invalid history map header")
    mapping = {}
    targets = set()
    for row in rows[1:]:
        fields = row.split()
        if len(fields) != 2 or any(not OID.fullmatch(x) or x == "0" * 40 for x in fields):
            raise ValueError("invalid or dropped history map entry")
        old, new = fields
        if old in mapping or new in targets:
            raise ValueError("history map is not one-to-one")
        mapping[old] = new
        targets.add(new)
    changed = sum(old != new for old, new in mapping.items())
    if len(mapping) != meta["map"]["commits"] or changed != meta["map"]["rewritten"]:
        raise ValueError("history map count mismatch")
    if mapping.get(meta["old_head"]) != meta["filtered_head"]:
        raise ValueError("history head mapping mismatch")
    return meta, mapping


def resolve(value, repo=ROOT, manifest=MANIFEST):
    if not re.fullmatch(r"[0-9a-fA-F]{7,40}", value):
        raise ValueError("commit must be 7–40 hexadecimal digits")
    value = value.lower()
    _, mapping = load_map(manifest)
    old_matches = {old for old in mapping if old.startswith(value)}
    candidates = {mapping[old] for old in old_matches}
    candidates.update(new for new in mapping.values() if new.startswith(value))
    # Post-migration commits are accepted too. Old objects in a full clone
    # normalize through the map, so their presence does not create ambiguity.
    for oid in git(repo, "rev-parse", f"--disambiguate={value}").decode().splitlines():
        if git(repo, "cat-file", "-t", oid).strip() == b"commit":
            candidates.add(mapping.get(oid, oid))
    if not candidates:
        raise ValueError("unknown foundation commit; upstream repositories have separate identities")
    if len(candidates) != 1:
        raise ValueError("ambiguous commit prefix; provide the full identity")
    target = candidates.pop()
    try:
        git(repo, "cat-file", "-e", target + "^{commit}")
    except subprocess.CalledProcessError as error:
        raise ValueError("mapped commit missing locally; use the complete new branch, not a shallow clone") from error
    return {"input": value, "commit": target, "original_matches": sorted(old_matches),
            "rewritten": any(mapping[old] != old for old in old_matches)}

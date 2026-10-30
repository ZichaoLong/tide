"""Independent synthetic Git DAGs, including rejection of non-archive edits."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from history_map import load_map, resolve
from verify_history_mapping import verify


def git(repo, *args, data=None):
    return subprocess.check_output(["git", "-C", str(repo), *args], input=data).strip().decode()


def blob(repo, content):
    return git(repo, "hash-object", "-w", "--stdin", data=content)


def tree(repo, entries):
    data = b"".join(f"{mode} blob {blob(repo, content)}\t{name}".encode() + b"\0"
                    for name, mode, content in entries)
    return git(repo, "mktree", "-z", data=data)


def commit(repo, tree_id, parents, message):
    text = f"tree {tree_id}\n" + "".join(f"parent {p}\n" for p in parents)
    text += "author Test <test@example.invalid> 1000000000 +0800\n"
    text += "committer Test <test@example.invalid> 1000000001 +0000\n\n" + message + "\n"
    return git(repo, "hash-object", "-t", "commit", "-w", "--stdin", data=text.encode())


def write_manifest(path, mapping, old_head, new_head, archive):
    data = "old new\n" + "".join(f"{a} {b}\n" for a, b in mapping.items())
    (path.parent / "map.tsv").write_text(data)
    path.write_text(json.dumps({"schema_version": 1, "old_head": old_head,
        "filtered_head": new_head, "map": {"file": "map.tsv",
        "sha256": hashlib.sha256(data.encode()).hexdigest(), "commits": len(mapping),
        "rewritten": sum(a != b for a, b in mapping.items())}, "removed": archive}))


def histories(tmp_path, mutation=None):
    old_repo, new_repo = tmp_path / "old", tmp_path / "new"
    for repo in (old_repo, new_repo):
        repo.mkdir()
        git(repo, "init", "-q")
    source = [("source.txt", "100644", b"actual source\n"),
              ("tab\tname", "100644", b"filename with tab\n")]
    archive = ("archive.gz", "100644", b"original archive bytes")
    pairs = []
    # A root, independent left/right commits, then a merge. Even commits whose
    # sole original change was the archive must survive with their metadata.
    for index, parent_indices in enumerate(((), (0,), (0,), (1, 2))):
        pair = []
        for side, repo in enumerate((old_repo, new_repo)):
            entries = list(source)
            parents = [pairs[i][side] for i in parent_indices]
            message = f"commit {index}"
            if side == 0 and index in (1, 3):
                entries.append(archive)
            if side == 1 and index == 3:
                if mutation == "source": entries[0] = ("source.txt", "100644", b"wrong source")
                if mutation == "mode": entries[0] = ("source.txt", "100755", source[0][2])
                if mutation == "message": message += " modified"
                if mutation == "parents": parents.reverse()
                if mutation == "archive": entries.append(archive)
            pair.append(commit(repo, tree(repo, entries), parents, message))
        pairs.append(pair)
    for side, repo in enumerate((old_repo, new_repo)):
        git(repo, "update-ref", "refs/heads/test", pairs[-1][side])
    manifest = tmp_path / "manifest.json"
    archive_meta = {"path": "archive.gz", "blob": blob(old_repo, archive[2]),
                    "bytes": len(archive[2]), "sha256": hashlib.sha256(archive[2]).hexdigest()}
    write_manifest(manifest, dict(pairs), *pairs[-1], archive_meta)
    return old_repo, new_repo, manifest, pairs


def test_complete_history_and_resolver(tmp_path):
    old, new, manifest, pairs = histories(tmp_path)
    report = verify(old, new, manifest)
    assert report["commits"] == 4 and report["rewritten"] == 2
    assert report["archive_present_old_commits"] == 2
    for a, b in pairs:
        for value in (a, a[:7], b, b[:7]):
            assert resolve(value, new, manifest)["commit"] == b
    post = commit(new, tree(new, [("new", "100644", b"later")]), [pairs[-1][1]], "later work")
    assert resolve(post[:7], new, manifest)["commit"] == post
    # Having both histories present must not make an old ID ambiguous.
    git(new, "fetch", "-q", str(old), "refs/heads/test:refs/heads/old")
    assert resolve(pairs[-1][0][:7], new, manifest)["commit"] == pairs[-1][1]
    result = subprocess.run([sys.executable, str(ROOT / "scripts/resolve_history_commit.py"),
        pairs[-1][0][:7], "--repo", str(new), "--manifest", str(manifest), "--json"],
        capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["rewritten"] is True


@pytest.mark.parametrize("mutation,match", [("source", "tree mismatch"), ("mode", "tree mismatch"),
    ("archive", "tree mismatch"), ("parents", "parent mismatch"), ("message", "metadata/message mismatch")])
def test_reject_non_archive_history_change(tmp_path, mutation, match):
    old, new, manifest, _ = histories(tmp_path, mutation)
    with pytest.raises(ValueError, match=match): verify(old, new, manifest)


def test_bad_or_incomplete_mapping_and_missing_target(tmp_path):
    old, new, manifest, pairs = histories(tmp_path)
    original = (tmp_path / "map.tsv").read_text()
    (tmp_path / "map.tsv").write_text(original + "\n")
    with pytest.raises(ValueError, match="checksum"): load_map(manifest)
    meta = json.loads(manifest.read_text())
    write_manifest(manifest, dict(pairs[1:]), *pairs[-1], meta["removed"])
    with pytest.raises(ValueError, match="cover exactly"): verify(old, new, manifest)
    fake = "f" * 40
    mapping = dict(pairs); mapping[pairs[-1][0]] = fake
    write_manifest(manifest, mapping, pairs[-1][0], fake, meta["removed"])
    with pytest.raises(ValueError, match="missing locally"): resolve(pairs[-1][0], new, manifest)


def test_prefix_ambiguity_and_unknown_input(tmp_path):
    _, new, manifest, pairs = histories(tmp_path)
    meta = json.loads(manifest.read_text())
    mapping = {"abcdef0" + "1" * 33: pairs[0][1], "abcdef0" + "2" * 33: pairs[-1][1]}
    write_manifest(manifest, mapping, list(mapping)[-1], pairs[-1][1], meta["removed"])
    with pytest.raises(ValueError, match="ambiguous"): resolve("abcdef0", new, manifest)
    for value, message in [("HEAD", "hexadecimal"), ("--all", "hexadecimal"), ("f" * 40, "unknown")]:
        with pytest.raises(ValueError, match=message): resolve(value, new, manifest)


def test_duplicate_and_dropped_entries_rejected(tmp_path):
    _, _, manifest, pairs = histories(tmp_path)
    meta = json.loads(manifest.read_text())
    mapping = dict(pairs); mapping[pairs[-1][0]] = pairs[0][1]
    write_manifest(manifest, mapping, pairs[-1][0], pairs[0][1], meta["removed"])
    with pytest.raises(ValueError, match="one-to-one"): load_map(manifest)
    mapping[pairs[-1][0]] = "0" * 40
    write_manifest(manifest, mapping, pairs[-1][0], "0" * 40, meta["removed"])
    with pytest.raises(ValueError, match="dropped"): load_map(manifest)

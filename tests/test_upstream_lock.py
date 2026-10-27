"""Immutable definition provenance, independent of Torch/model behavior."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("check_upstream", Path(__file__).resolve().parents[1]/"scripts/check_upstream.py")
checker = importlib.util.module_from_spec(spec); spec.loader.exec_module(checker)


def test_real_adoption_matches_all_snapshot_bytes():
    result = checker.check(Path(__file__).resolve().parents[1])
    assert result["revision"] == "0770593838a37c790a568226930567ed60f99b83"
    assert result["semantic_version"] == "tide-core-5" and result["files"] == 19
    assert result["legacy"][0]["semantic_version"] == "tide-core-3"


@pytest.mark.parametrize("fault", ["changed", "missing", "extra", "duplicate", "symlink"])
def test_nested_snapshot_fails_closed(tmp_path, fault):
    folder=tmp_path/"docs/snapshot/local"; folder.mkdir(parents=True)
    p=folder/"definition.md"; p.write_bytes(b"fixed definition")
    lock=dict(snapshot_directory="docs/snapshot", sources=["local/definition.md"],
              sha256={"local/definition.md": hashlib.sha256(p.read_bytes()).hexdigest()},
              semantic_version="test", revision="fixed")
    if fault == "changed": p.write_bytes(b"different")
    if fault == "missing": p.unlink()
    if fault == "extra": (folder/"draft.md").write_text("draft")
    if fault == "duplicate": lock["sources"] *= 2
    if fault == "symlink":
        p.unlink(); p.symlink_to(tmp_path/"target")
    (tmp_path/"docs/upstream.json").write_text(json.dumps(lock))
    with pytest.raises(ValueError): checker.check(tmp_path)

"""Archived SHA receipts are valid only for the same delivered destination revision."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import verify_tide_regional_inputs as audit
import tide_cloud_submit as submit


@pytest.mark.parametrize(("previous_version", "final_etag", "reads", "fails"), [
    ("v2", "e1", 0, False),
    ("v1", "e1", 1, False),
    ("v1", "different", 1, True),
])
def test_reuse_and_reject_changed_guiyang_revision(tmp_path, monkeypatch, previous_version, final_etag, reads, fails):
    h = "a" * 64
    target = dict(key=audit.object_key(h), operation="existing", oms_task="12", size=5, sha256=h)
    old = dict(**target, etag="e1", version=previous_version, method="previous full read", verified_at="earlier")
    node = dict(id="1", name="existing", platform=dict(id="1", name="existing", phase="Running"))
    heads = iter(dict(sha256=h, size=5, etag=e, version="v2") for e in ("e1", final_etag))

    class Region:
        side = "guiyang"

        def head(self, key):
            assert key == target["key"]
            return next(heads)

    calls = []
    monkeypatch.setattr(audit.Store, "named", lambda name: Region())
    monkeypatch.setattr(audit, "pinned_ssh", lambda candidate: candidate == node)
    monkeypatch.setattr(audit, "incoming", lambda storage: {h: target})
    monkeypatch.setattr(audit, "prior_receipts", lambda *args: {h: old})

    def read_remote(allocation, store, key, sha256, size, *, background):
        assert allocation == node and background and (key, sha256, size) == (target["key"], h, 5)
        calls.append(sha256)
        return {"sha256": sha256, "size": size}

    monkeypatch.setattr(audit, "verify_object", read_remote)
    file = tmp_path / "receipts/guiyang-node" / (h + ".json")
    if fails:
        with pytest.raises(ValueError, match="changed during regional readback"):
            audit.verify(tmp_path, node, 1)
        assert not file.exists()
    else:
        result = audit.verify(tmp_path, node, 1)
        assert result["by_method"] == {("node" if reads else "reused"): 1}
        assert audit.read(file)["version"] == "v2"
    assert len(calls) == reads


def test_only_background_read_reduces_its_process_priority(monkeypatch):
    changes = []
    monkeypatch.setattr(submit, "verified_queue", lambda *args: (None,))
    monkeypatch.setattr(submit.os, "nice", changes.append)
    monkeypatch.setattr(submit.blobs, "read_signed", lambda url, sha256, size: size)
    request = dict(action="verify", sha256="a"*64, size=3, url="example")
    for background in (False, True):
        assert submit.read_on_node({}, {**request, "background":background}, None, None, None)["size"] == 3
    assert changes == [15]

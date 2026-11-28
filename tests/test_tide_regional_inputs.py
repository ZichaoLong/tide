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

    def read_remote(allocation, store, pending):
        assert allocation == node and pending == {h: target}
        calls.append(h)
        return {h: 5}

    monkeypatch.setattr(audit, "verify_objects", read_remote)
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


def test_batch_uses_one_background_connection_for_multiple_object_hashes(monkeypatch):
    objects = {h * 64: {"key":"objects/" + h * 64, "size":size} for h, size in (("a", 1), ("b", 2))}
    connections, priority = [], []
    monkeypatch.setattr(submit.blobs, "signed_get", lambda store, key: "private-" + key)
    monkeypatch.setattr(submit, "verified_queue", lambda *args: (None,))
    monkeypatch.setattr(submit.os, "nice", priority.append)

    def read(url, h, size):
        assert url == "private-" + objects[h]["key"] and size == objects[h]["size"]
        return size

    def remote_call(node, claims, *, files):
        connections.append(node)
        return submit.read_on_node(node, files, None, None, None)

    monkeypatch.setattr(submit.blobs, "read_signed", read)
    monkeypatch.setattr(submit, "remote_call", remote_call)
    assert submit.verify_objects("pinned-node", None, objects) == {h: row["size"] for h, row in objects.items()}
    assert connections == ["pinned-node"] and priority == [15]


def test_interrupted_connection_preserves_other_batches_and_redacts_errors(tmp_path, monkeypatch):
    hashes = [f"{i:064x}" for i in range(13)]
    objects = {h: {"sha256":h, "size":4, "key":audit.object_key(h)} for h in hashes}
    node = {"id":"1", "name":"old", "platform":{"id":"1", "name":"old", "phase":"Running"}}

    class Region:
        side = "guiyang"

        def head(self, key):
            return {"size":4, "etag":"fixed", "version":None, "sha256":key.rsplit("/", 1)[1]}

    failed = []
    monkeypatch.setattr(audit.Store, "named", lambda name: Region())
    monkeypatch.setattr(audit, "pinned_ssh", lambda found: found == node)
    monkeypatch.setattr(audit, "incoming", lambda storage: objects)
    monkeypatch.setattr(audit, "prior_receipts", lambda *args: {})
    monkeypatch.setattr(audit.time, "sleep", lambda seconds: None)

    def interrupt_first_batch(allocation, region, pending):
        if hashes[0] in pending:
            failed.append(True)
            raise OSError("sensitive simulated query details")
        return {h: row["size"] for h, row in pending.items()}

    monkeypatch.setattr(audit, "verify_objects", interrupt_first_batch)
    with pytest.raises(ValueError, match="unverified regional batches") as error:
        audit.verify(tmp_path, node, 2)
    assert "sensitive" not in str(error.value)
    assert len(failed) == 4
    assert {p.stem for p in (tmp_path / "receipts/guiyang-node").glob("*.json")} == set(hashes[6:])

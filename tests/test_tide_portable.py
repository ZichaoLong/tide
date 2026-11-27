"""Copied task/gate bytes, moved worker outputs and contradictory attempt evidence."""

from pathlib import Path
import sys

import pytest

from pilotlab.gate3 import read_evidence
from pilotlab.io import sha256

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from tide_batch import aggregate, export
from tide_portable import Frozen, module_binding, pack, read, write


def test_export_binds_sha256_and_single_part_obs_etag_to_the_same_bytes(tmp_path):
    source = tmp_path / "input"
    source.write_bytes(b"abc")
    target = tmp_path / "obs"
    receipt = export(source, target, "products/output")
    assert receipt == {
        "path": "products/output", "size": 3, "sha256": sha256(source),
        "content_md5": "900150983cd24fb0d6963f7d28e17f72",
    }
    assert export(source, target, "products/output") == receipt
    source.write_bytes(b"xyz")
    with pytest.raises(ValueError, match="published artifact differs"):
        export(source, target, "products/output")


def project_fixture(tmp):
    data, corpus = tmp / "data", tmp / "corpus"
    data.mkdir()
    corpus.mkdir()
    write(corpus / "manifest.json", {})
    write(
        data / "manifest.json",
        {
            "schema_version": "tide-pilot-data-v3",
            "source_directory": str(corpus),
            "source_manifest_sha256": sha256(corpus / "manifest.json"),
            "files": {},
            "source_files": {},
        },
    )
    src = tmp / "original" / "pilotlab" / "worker3.py"
    src.parent.mkdir(parents=True)
    src.write_text("# witness\n")
    identity = {
        "schema": "pilot-r3-source-v1",
        "data_manifest_sha256": sha256(data / "manifest.json"),
        "source_files": {str(src): sha256(src)},
    }
    auth = tmp / "authorization.json"
    write(auth, {"selection": "frozen"})
    gate = tmp / "stage-b-gate.json"
    taskfile = tmp / "task.json"
    task = {"id": "r", "source_identity": identity, "phase": "b", "stage_b_gate": str(gate)}
    write(taskfile, task)
    write(
        gate,
        {
            "schema": "pilot-stage-b-gate-v1",
            "status": "passed",
            "source_identity": identity,
            "allowed_task_ids": ["r"],
            "task_sha256": {"r": sha256(taskfile)},
            "report_path": str(auth),
            "report_sha256": sha256(auth),
            "manifest_path": str(auth),
            "manifest_sha256": sha256(auth),
        },
    )
    dst = tmp / "export"
    pack(dst, [taskfile], data)
    return dst, taskfile, gate, auth


def test_frozen_gate_and_scientific_inputs_relocate_without_rewriting(tmp_path):
    from pilotlab import worker3

    dst, taskfile, gate, auth = project_fixture(tmp_path)
    target = tmp_path / "unrelated-location"
    dst.rename(target)
    frozen = Frozen(target)
    task, dest = frozen.request("r", sha256(taskfile))
    assert taskfile.read_bytes() == dest.read_bytes() and gate.read_bytes() == frozen.path(gate).read_bytes()
    gate.unlink()
    taskfile.unlink()
    auth.unlink()
    with (
        module_binding(worker3, "Path", frozen.path),
        module_binding(worker3, "sha256", lambda p: sha256(frozen.path(p))),
    ):
        assert worker3.validate_gate(task, sha256(dest))["status"] == "passed"
        frozen.path(auth).write_text("{}")
        with pytest.raises(ValueError, match="hash mismatch"):
            worker3.validate_gate(task, sha256(dest))
    with pytest.raises(ValueError, match="changed input"):
        Frozen(target)


def test_attempt_aggregation_detects_conflicts_missing_evidence_and_allows_gate_readback(tmp_path):
    frozen, taskfile, _, _ = project_fixture(tmp_path)
    task = read(taskfile)
    fingerprint = sha256(taskfile)

    def attempt(root, name="one", *, started=1):
        taskdir = root / "products" / "r"
        taskdir.mkdir(parents=True)
        file = taskdir / "checkpoint"
        file.write_bytes(b"content")
        cp = {"path": "/old/cloud/not-mounted.pt", "sha256": sha256(file)}
        result = {
            "id": "r",
            "status": "completed",
            "task_sha256": fingerprint,
            "source_identity": task["source_identity"],
            "checkpoint": cp,
            "health": {"eligible": True},
            "recovery": {"checkpoint_load": {"status": "exact"}},
        }
        write(taskdir / "raw.json", result)
        write(taskdir / "verdict.json", {"status": "completed"})
        artifacts = {
            k: {"path": p.relative_to(root).as_posix(), "sha256": sha256(p), "size": p.stat().st_size}
            for k, p in (
                ("result", taskdir / "raw.json"),
                ("verdict", taskdir / "verdict.json"),
                ("checkpoint", file),
            )
        }
        artifacts["checkpoint"]["step"] = 4
        row = {
            "schema": "maflow-task-result/v1",
            "project": "demo",
            "batch": "batch",
            "plan_sha256": "0" * 64,
            "task": "r",
            "attempt": name,
            "fingerprint": fingerprint,
            "status": "completed",
            "artifacts": artifacts,
            "started": started,
        }
        write(root / "products/tasks/r.json", row)
        return row

    one = tmp_path / "first"
    attempt(one)
    assert aggregate(frozen, [])["missing"] == ["r"]
    out = aggregate(frozen, [one])
    assert out["complete"] and aggregate(frozen, [one, one]) == out
    entry = out["tasks"]["r"]["evidence"]
    _, original, checked = read_evidence(entry, checkpoint=True)
    assert (
        original["checkpoint"]["path"] == "/old/cloud/not-mounted.pt"
        and Path(checked["checkpoint_path"]).exists()
    )
    entry["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="relocated"):
        read_evidence(entry, checkpoint=True)
    two = tmp_path / "second"
    attempt(two, started=2)
    broken = aggregate(frozen, [one, two])
    assert not broken["complete"] and "conflicting" in broken["problems"][0]["error"]
    attempt(tmp_path / "third", "two")
    assert not aggregate(frozen, [one, tmp_path / "third"])["complete"]
    (one / "products/r/checkpoint").unlink()
    invalid = aggregate(frozen, [one])
    assert not invalid["complete"] and invalid["problems"] and invalid["missing"]

"""CPU-only scientific frontier, archive and physical/logic resource contracts."""
from copy import deepcopy
from pathlib import Path
import time

import pytest

from pilotlab import cloud_campaign3 as cloud, gate3, round3, worker3
from pilotlab.scheduler3 import freeze_json, read
from test_pilot_round3 import simulation
from test_pilot_gate3 import campaign, formal_inputs, write_json
from test_runtime_migration3 import inputs, write


def test_remote_checkpoint_can_be_reopened_after_cache_eviction(tmp_path):
    _, _, checksum, _, result = inputs(tmp_path)
    absent = str(tmp_path / ("a" * 64))
    result["checkpoint"]["path"] = "/cloud/actual/end.pt"
    raw = write(tmp_path / "cloud", result)
    fp = result["task_sha256"]
    verdict = {"task": result["id"], "fingerprint": fp, "status": "completed",
               "execution_runtime": result["execution_runtime"]}
    verdict_hash = write(tmp_path / "verdict", verdict)
    receipt = dict(schema="tide-cloud-result/v1", task=result["id"], fingerprint=fp, status="completed",
                   attempt="attempt", job="allocated-job", artifacts={"result.json": {"sha256": raw},
                   "project-verdict.json": {"sha256": verdict_hash}, "checkpoint.pt": {"sha256": "a"*64, "size": 20}})
    receipt_hash = write(tmp_path / "receipt", receipt)
    evidence = dict(schema="tide-cloud-checkpoint/v1", path=absent, size=20, sha256="a"*64, verified_sha256="a"*64,
                    verified_unix=1, uri="obs://bucket/objects/"+"a"*64, producer={"job": receipt["job"],
                    "attempt": "attempt", "fingerprint": fp, "receipt_sha256": receipt_hash})
    def ref(name, checksum):
        return {"path": str(tmp_path / name), "sha256": checksum}
    entry = dict(task_path=str(tmp_path / "task"), result_path=str(tmp_path / "cloud"),
                 runtime_plan=ref("plan", checksum), cloud_verdict=ref("verdict", verdict_hash),
                 cloud_receipt=ref("receipt", receipt_hash), checkpoint_path=absent, checkpoint_sha256="a"*64,
                 checkpoint_receipt=ref("object", write(tmp_path / "object", evidence)))
    assert not Path(absent).exists()
    _, checked, audited = gate3.read_evidence(entry, checkpoint=True)
    assert checked.relocated_checkpoint == absent and audited["checkpoint_state"].startswith("remote_payload_verified")
    assert gate3.read_evidence(audited, checkpoint=True)[2] == audited
    entry["checkpoint_receipt"] = ref("object", write(tmp_path / "object", {**evidence, "verified_sha256": "b"*64}))
    with pytest.raises(ValueError, match="remote checkpoint"):
        gate3.read_evidence(entry, checkpoint=True)


def test_distinct_formal_source_grants_preserve_frozen_worker_checks(campaign, tmp_path, monkeypatch):
    manifest_path, report_path, paths = formal_inputs(campaign, tmp_path)
    manifest = read(manifest_path)
    selected = read(report_path)["selected"]
    bound = gate3.stage_source_contract(manifest, selected)
    task = read(paths[0])
    alternate = {**task["source_identity"], "foundation": "alternate"}
    bound["models"][task["model_id"]]["b_source"] = alternate
    gate = tmp_path / "gate.json"
    variant = gate3.gate_path_for_source(gate, alternate, manifest["source_identity"])
    task.update(source_identity=alternate, gate_file=str(variant))
    write_json(paths[0], task)
    monkeypatch.setattr(gate3, "stage_source_contract", lambda *args: bound)
    gate3.create_gate(gate, manifest_path=manifest_path, report_path=report_path, task_paths=paths)
    assert len(read(gate)["allowed_task_ids"]) == len(paths)
    assert read(variant)["allowed_task_ids"] == [task["id"]]
    assert worker3.validate_gate(task, gate3.sha256(paths[0]))["status"] == "passed"
    with pytest.raises(ValueError, match="source identity"):
        worker3.validate_gate({**task, "gate_file": str(gate)}, gate3.sha256(paths[0]))


def test_measured_concurrency_distinguishes_boards_chips_and_overlapping_workers():
    now = time.time()
    node = {"workers": [{"state": "serving", "at": now, "cards": {"0": {"chips": {"0": {}, "1": {}}}},
                         "capacity": {"threads": 12}}]}
    rows = [(0, 10, 8, (0, chip)) for chip in (0, 1) for _ in range(3)]
    widths, observed = cloud.measured_parallelism({"job": node}, {("job", "npu"): rows}, 4)
    assert observed["job"]["boards"] == 1 and observed["job"]["logical_chips"] == 2
    assert widths["npu"] == pytest.approx([4.8, 4.8]) and widths["cpu"] == [3, 3]
    assert observed["job"]["backends"]["npu"]["wall_seconds"] == 10


def test_portable_frontier_reaches_all_native_phases(simulation, monkeypatch):
    native = simulation()
    c = cloud.CloudCampaign.__new__(cloud.CloudCampaign)
    c.__dict__.update(vars(native))
    c.subjects = [dict(m, effective_key=m["model_key"]) for m in c.subjects]
    c.fresh, c.review, c.records, c.original_tasks = set(), {"subjects": {}}, {}, {}
    c.initial, c.accepted, c.peaks, c.entries = {}, {}, {}, {}
    monkeypatch.setattr(c, "manifest", lambda: round3.Campaign.manifest(c))
    monkeypatch.setattr(c, "forecast", lambda resources: {"fits": True, "reasons": []})
    monkeypatch.setattr(cloud, "audit_stage_a", round3.audit_stage_a)
    def grant(path, **kwargs):
        report = read(kwargs["report_path"])
        assert report["status"] == "passed"
        return freeze_json(path, {"manifest_sha256": gate3.sha256(kwargs["manifest_path"]),
               "report_sha256": gate3.sha256(kwargs["report_path"]),
               "task_sha256": {read(f)["id"]: gate3.sha256(f) for f in kwargs["task_paths"]}})
    monkeypatch.setattr(cloud, "create_gate", grant)
    phases = []
    for _ in range(32):
        c.models, c.pending, c.blocked = list(c.subjects), {}, []
        out = c.stages({})
        assert not out["blocked"], out
        phases.append(out["phase"])
        if out["finished"]:
            break
        assert out["tasks"]
        wave = [read(row["task_path"]) for row in out["tasks"]]
        answers = c.supervisor.run(out["phase"], wave)
        for task in wave:
            mid = task["id"]
            result = c.supervisor.state["tasks"][mid]
            if task["phase"] == "b_confirmation":
                assert answers[mid]["training"]["segment_tokens"] == 0
                answers[mid]["confirmation_final"] = dict(status="completed", split="confirmation", nll=3., tokens=131072)
                write_json(result["result_path"], answers[mid])
            c.records[mid] = {k: result[k] for k in ("task_path","result_path")}
    assert set(phases) == {"stage-a-subjects", "stage-a-control-capacity", "stage-a-controls",
                           "stage-b-formal", "stage-b-repeat", "stage-b-confirmation", "completed"}
    final = read(c.output / "report.json")
    assert final["formal"] == 9 and final["repeats"] == 18 and final["confirmed"] == 27

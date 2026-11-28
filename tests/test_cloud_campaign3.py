"""CPU-only scientific frontier, archive and physical/logic resource contracts."""
from pathlib import Path
import time

import pytest

from pilotlab import cloud_campaign3 as cloud, gate3, round3, worker3
from pilotlab.scheduler3 import freeze_json, read
from test_pilot_round3 import simulation
from test_pilot_chain_receipt3 import chain
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
    c.initial, c.accepted, c.peaks, c.entries, c.hbm_evidence = {}, {}, {}, {}, {}
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


def test_unseen_formal_shape_profiles_the_whole_idle_chip_before_formal(tmp_path):
    c = cloud.CloudCampaign.__new__(cloud.CloudCampaign)
    c.output, c.hbm_evidence, c.pending, c.blocked = tmp_path, {}, {}, []
    spec = dict(id="model-b", model_id="model", device="npu:0", source_identity={"source":"v2"},
                model={"family":"tide"}, batch_size=2, sequence_length=128, stream_chunks=4, phase="b",
                cpu_threads=4, seed=round3.FORMAL_SEED, target_tokens=12*gate3.MI, eval_tokens=16384)
    def pending(probe):
        c.pending[probe["id"]] = {"scientific_task": probe}
    c.resolution = pending
    now = time.time()
    chip = {"health": "OK", "total": 64*2**30, "used": 2**30, "processes": []}
    nodes = {"one": {"phase":"Running", "workers":[{"state":"serving","at":now,
        "cards":{"0":{"process_table_observed":True, "chips":{"1":chip}}}}]}}
    assert not c.formal_hbm([spec], {"observed_unix":now, "node_capacity":nodes}, "formal")
    row, = c.pending.values()
    task = row["scientific_task"]
    assert task["phase"] == "preflight" and task["seed"] != round3.FORMAL_SEED and not task.get("gate_file")
    assert task["target_tokens"] == 2*128*4*2 and row["reserve_chip"] is True and row["hbm_mib"] == 63*1024
    assert not (tmp_path / "stage-b-formal-hbm.json").exists()
    shape = cloud.hbm_shape(spec)
    c.hbm_evidence[shape] = {"reservation_bytes": 14*2**30, "observed_bytes": 9*2**30, "shape": shape,
                             "basis": "same_source_and_shape_capacity_probe"}
    assert c.formal_hbm([spec], {}, "formal") and spec["reserved_hbm_bytes"] == 14*2**30


def test_bound_archive_reconstructs_an_evicted_original_chain_without_weakening_other_paths(chain, tmp_path):
    import gzip
    import json
    from pilotlab.checkpoint_archive3 import checkpoint_reads
    from test_pilot_chain_receipt3 import verify
    old = read(chain["reference"]["path"])
    row = old["retained_checkpoints"][0]
    cp = Path(row["path"])
    checksum = gate3.sha256(cp)
    key = "prod/objects/sha256/" + checksum
    size = cp.stat().st_size
    stamp = "2026-11-28T11:00:00+00:00"
    item = {"key":key,"size":size,"sha256":checksum}
    index = {"schema":"tide-r3-verified-inputs/v1","east_region":"cn-east-4","east_bucket":"archive",
             "east_suffix_prefix":"a/", "published_selection":{"passed":True},
             "paths":{str(cp):item},
             "objects":{checksum:{**item,"etag":"remote-generation","uri":"obs://archive/a/"+key,
                                  "content_read_at":stamp,"verified":stamp}}}
    packed = tmp_path / "archive.gz"
    packed.write_bytes(gzip.compress(json.dumps(index).encode()))
    proof = {"schema":"tide-r3-verified-index/v1","source":str(packed), "verified":stamp,"etag":"index-generation",
             "paths":1,"objects":1,"key":"index.gz","uri":"obs://archive/a/index.gz",
             "size":packed.stat().st_size, "sha256":gate3.sha256(packed)}
    proof_path = tmp_path / "proof.json"
    entries = [{"path": str(proof_path), "sha256": write(proof_path, proof)}]
    cp.unlink()
    with pytest.raises(ValueError, match="missing"):
        verify(chain)
    with checkpoint_reads(entries):
        assert verify(chain)["receipt"] == chain["reference"]
        assert gate3.read_evidence(chain["args"]["choice"]["confirmation"][0], checkpoint=True)[2]["checkpoint_state"] == "original_verified_in_archive"
    packed.write_bytes(b"replacement")
    with pytest.raises(ValueError, match="verified archive index changed"):
        with checkpoint_reads(entries):
            verify(chain)


def test_forecast_uses_matched_shape_concurrency_and_charges_physical_boards(tmp_path):
    now = time.time()
    shape = dict(source_identity={"dataset":"frozen"}, model={"family":"tide"}, model_id="same",
                 device="npu:0", batch_size=2, sequence_length=32, stream_chunks=4, cpu_threads=2)
    local = tmp_path / "source-task"
    write(local, shape)
    historical = tmp_path / "seed"
    write(historical, {"training":{"tokens_per_second": 100.}})
    selected = {"same":{"confirmation":[{"task_path":str(local),"result_path":str(historical)}]}}
    accepted = {}
    for i in range(4):
        tid, task, raw, receipt = str(i), tmp_path/f"task{i}", tmp_path/f"result{i}", tmp_path/f"receipt{i}"
        fp = write(task, {**shape,"id":tid,"phase":"a3"})
        sha = write(raw, {"task_sha256":fp,"id":tid,"training":{"seconds":10,"segment_tokens":500}})
        write(receipt, {"job":"job","status":"completed","fingerprint":fp,"task":tid,"started":100,"finished":110,
                        "board":0, "chip":i%2, "artifacts":{"result.json":{"sha256":sha}}})
        accepted[fp] = dict(task_path=str(task), result_path=str(raw), receipt_path=str(receipt))
    capacity = {"job":{"phase":"Running","workers":[{"state":"serving","at":now, "capacity":{"threads":16},
                                                     "cards":{"0":{"chips":{"0":{},"1":{}}}}}]}}
    resources = dict(observed_unix=now, deadline_unix=now+1e9, remaining_device_seconds=1e9,
                     allocated_physical_boards=1, active_nodes=1, node_capacity=capacity)
    estimate = cloud.cloud_forecast(selected, resources, accepted=accepted)
    assert estimate["fits"], estimate["reasons"]
    assert estimate["rates"]["same"]["slow_rate"] == 50
    assert estimate["parallelism_by_backend"]["npu"] == [4, 4] and estimate["logical_chips"] == 2
    assert estimate["physical_boards_charged"] == 1 and estimate["accepted_attempts"] == 4
    assert estimate["wall_seconds_range"] == estimate["allocated_board_seconds_range"]
    resources["remaining_device_seconds"] = 10
    assert not cloud.cloud_forecast(selected, resources, accepted=accepted)["fits"]


def test_complete_archive_replaces_the_staged_indices_for_candidate_and_dispatch(tmp_path):
    from pilotlab.checkpoint_archive3 import archive_proofs

    old = tmp_path / "selected-hot-proof.json"
    frontier = tmp_path / "selected-frontier-proof.json"
    whole = tmp_path / "remaining-checkpoints-proof.json"
    write(old, {"index": "hot"})
    write(frontier, {"index": "frontier"})
    assert [Path(ref["path"]).name for ref in archive_proofs(tmp_path)] == [old.name, frontier.name]
    expected = write(whole, {"index": "complete"})
    assert archive_proofs(tmp_path) == [{"path": str(whole), "sha256": expected}]


def test_control_matching_waits_for_real_subject_activity_before_catalog_generation(tmp_path):
    from pilotlab.scheduler3 import Blocked

    c = cloud.CloudCampaign.__new__(cloud.CloudCampaign)
    c.output, c.subjects, c.selected = tmp_path, [{"id":"ready"}, {"id":"pending"}], {"ready": {}}
    with pytest.raises(Blocked, match="1 missing"):
        c.controls()
    assert not (tmp_path / "control-catalog.json").exists()

"""The selected A3 activity must survive an exact zero-update continuation."""
from pathlib import Path

import pytest

from pilotlab import chain_receipt3 as chains, gate3 as gate, round3 as rnd
from pilotlab.catalog3 import model_key
from pilotlab.prepare_resume3 import reference
from pilotlab.scheduler3 import Blocked
from pilotlab.source_migration3 import ADAPTER, RECEIPT, digest
from test_pilot_gate3 import model_row, write_json
from test_pilot_transfer3 import single_old_selection


def frozen_identity(root, changed=False):
    package = root / "pilotlab"
    package.mkdir(parents=True)
    paths = {}
    for name in ("worker3", "models", "models3", "resume_adapter3"):
        path = package / (name+".py")
        path.write_text(name+" changed" if changed and name=="models3" else name)
        paths[str(path)] = gate.sha256(path)
    return {"source_files":paths, "data_manifest_sha256":"fixed", "runtime_packages":{}, "python_version":"same"}


@pytest.mark.parametrize("damage", [None, "seed", "hash", "step", "source", "segment"])
def test_zero_step_activity_recomputed_by_selection_receipt_and_admission(tmp_path, damage):
    current = frozen_identity(tmp_path/"new")
    old = frozen_identity(tmp_path/"old", changed=damage=="source")
    model = model_row(0)
    model["model"]["config"]["allocation"] = "quad"
    model["model_key"] = model["effective_key"] = model_key(model["model"])
    selected = single_old_selection(tmp_path/"chain", current, model)
    choice = selected[model["id"]]
    direct_path = Path(choice["confirmation"][1]["result_path"])
    direct = gate.load(direct_path)
    direct["work"][-1]["work"]["activity"].update(version="unique-external-token-parameters-v1",
        whole_model_parameters=1500, vocabulary_parameters=500, mean_active_whole_model_parameters=1000,
        min_active_backbone_parameters=300, max_active_backbone_parameters=600, semantic_full_owner_tick_rows=20)
    write_json(direct_path,direct)
    entry = choice["confirmation"][0]
    task, result, _ = gate.read_evidence(entry)
    target, end_step = task["target_tokens"], 8192
    task.update(evaluation_start_tokens=gate.MI, eval_tokens=128)
    original = {**task, "source_identity":old, "seed":202 if damage=="seed" else task["seed"]}
    old_task = reference(write_json(tmp_path/"previous-task.json", original))
    data = tmp_path/"previous-checkpoint.pt"
    data.write_bytes(b"old state")
    cp = {**reference(data), "steps":end_step, "tokens":target, "state_sha256":"model-optimizer-stream-rng"}
    initial = {"status":"completed", "split":"selection", "tokens":128, "requested_tokens":128, "nll":4.0}
    sample = {"qualified":True, "version":"unique-external-token-parameters-v1", "whole_model_parameters":1500,
              "backbone_total_parameters":1000, "mean_active_backbone_parameters":350.0, "measured_external_tokens":256,
              "vocabulary_parameters":500, "mean_active_whole_model_parameters":850,
              "min_active_backbone_parameters":320, "max_active_backbone_parameters":400, "semantic_full_owner_tick_rows":10}
    base = {"schema":"pilot-r3-worker-result-v1", "id":task["id"], "phase":"a3", "model_id":task["model_id"], "parameters":1500}
    prior = {**result, **base, "source_identity":old, "task_sha256":old_task["sha256"], "checkpoint":cp,
             "status":"budget_exhausted", "stop_reason":"signal", "validation_initial":initial,
             "training":{"from_tokens":gate.MI, "tokens":target, "steps":end_step, "segment_steps":4096},
             "work":[{"step":end_step-int(damage=="step"), "tokens":target, "work":{"activity":sample}}]}
    old_result = reference(write_json(tmp_path/"previous-result.json", prior))
    receipt = {"schema":RECEIPT, "mode":"signal", "task_id":task["id"], "new_source_digest":digest(current),
               "original_task":old_task, "result":old_result, "checkpoint":cp,
               "initial_evaluation":{"mode":"inherit", "exposure":gate.MI, "field":"validation_initial",
                                     "result":old_result, "observation":initial}}
    task["stage_a_migration"] = reference(write_json(tmp_path/"migration.json", receipt))
    new_task = reference(write_json(Path(entry["task_path"]), task))
    package = tmp_path/"new"/"pilotlab"
    result.update(base, checkpoint={**result["checkpoint"], "steps":end_step, "tokens":target,"state_sha256":cp["state_sha256"]},
                  work=[], task_sha256=new_task["sha256"],
                  training={"from_steps":end_step,"steps":end_step,"segment_steps":int(damage=="segment"),
                            "from_tokens":target,"tokens":target,"segment_tokens":0},
                  resume={**cp,"from_steps":end_step,"from_tokens":target,"state_reused":True,"source_task_id":task["id"],
                          "checkpoint_load":{"status":"exact"},"fallback_used":False},
                  execution_adapter={"schema":ADAPTER,"receipt":task["stage_a_migration"],"original_task":old_task,
                                     "original_result":old_result,"new_task":new_task, "checkpoint":cp,
                                     "adapter":reference(package/"resume_adapter3.py"),"worker":reference(package/"worker3.py")})
    write_json(Path(entry["result_path"]),result)
    if damage=="hash":
        Path(old_result["path"]).write_text("{}")
    pairs = [gate.read_evidence(e)[:2] for e in choice["confirmation"]]
    tasks, results = zip(*pairs)
    manifest = dict(subjects=[model], models=[model], source_identity=current, tuning_seeds=[101,202],
                    diagnostic_seeds=[101,202,303], formal_seed=404)
    receipt_args = dict(model=model,choice=choice,source_identity=current,tuning_seeds=[101,202],diagnostic_seeds=[101,202,303])
    if damage:
        with pytest.raises(Blocked):
            rnd.mean_activity(results,tasks)
        with pytest.raises(ValueError):
            chains.write_model_chain_receipt(tmp_path/"chain.json", **receipt_args)
        report = gate.audit_stage_a(manifest,selected,[],remaining_resources_ok=True)
        assert any("confirmation_evidence_error" in str(r) for r in report["blocking_reasons"])
        return
    choice["activity"] = rnd.mean_activity(results,tasks)
    link = chains.write_model_chain_receipt(tmp_path/"chain.json", **receipt_args)
    measured = gate.audit_stage_a(manifest,selected,[],remaining_resources_ok=True)["measured_activity"][model["id"]]
    assert measured == gate.load(link["path"])["activity"]
    assert all(choice["activity"][k]==v for k,v in measured.items())
    assert measured["measured_external_tokens"]==384 and measured["mean_active_backbone_parameters"]==400
    assert measured["inherited_samples"][0]["result"]==old_result
    assert choice["activity"]["mean_active_whole_model_parameters"]==900
    assert choice["activity"]["min_active_backbone_parameters"]==300 and choice["activity"]["max_active_backbone_parameters"]==600
    assert choice["activity"]["semantic_full_owner_tick_rows"]==30

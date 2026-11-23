"""The stage gate follows real segments, without laundering missing evidence."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from pilotlab import gate3 as g
from pilotlab.continuation3 import continuation_origin
from pilotlab.io import sha256
from test_pilot_gate3 import make_campaign, write_json


def reference(path):
    return {"path": str(path), "sha256": sha256(path)}


@pytest.mark.parametrize("migration", [False, True], ids=["same-source", "source-migration"])
@pytest.mark.parametrize("steps,tokens,flag,damage,passes", [
    (0, 0, False, None, True), (1, 128, True, None, True),
    (0, 0, None, None, False), (0, 0, 0, None, False), (1, 128, False, None, False),
    (1, 0, False, None, False), (0, 128, False, None, False),
    (0, 0, False, "fallback", False), (0, 0, False, "load", False),
    (0, 0, False, "digest", False), (0, 0, False, "gap", False),
])
def test_verified_private_state_at_origin(tmp_path, migration, steps, tokens, flag, damage, passes):
    from pilotlab.resume_adapter3 import initial_origin
    from pilotlab.source_migration3 import ADAPTER, RECEIPT, digest

    def bind(name, value):
        return reference(write_json(tmp_path/name,value))

    worker, adapter, cp_file = (tmp_path/name for name in ("worker.py","adapter.py","checkpoint.pt"))
    for path in (worker,adapter,cp_file):
        path.write_bytes(path.name.encode())
    identity = {"source_files":{str(p):sha256(p) for p in (worker,adapter)}, "data_manifest_sha256":"data",
                "python_version":"3", "runtime_packages":{}}
    task = dict(id="t", phase="a1", model_id="m", model={"family":"transformer"}, eval_tokens=16,
                target_tokens=256, source_identity=identity)
    task_ref = bind("original.json",task)
    cp = {**reference(cp_file), "steps":steps, "tokens":tokens, "state_sha256":"original-state"}
    previous = {k:task[k] for k in ("id","phase","model_id","source_identity")}
    previous.update(schema="pilot-r3-worker-result-v1", task_sha256=task_ref["sha256"], status="budget_exhausted",
        stop_reason="signal", checkpoint=cp, training=dict(from_tokens=0,tokens=tokens,steps=steps),
        validation_initial=dict(status="completed",split="selection",tokens=16,requested_tokens=16,nll=1.))
    prev_ref = bind("previous.json",previous)
    receipt = dict(schema="pilot-r3-task-resume-receipt-v2", task_id="t", original_task=task_ref,
                   source_identity=identity, result=prev_ref, checkpoint=cp)
    if migration:
        receipt.update(schema=RECEIPT, mode="signal", new_source_digest=digest(identity),
                       initial_evaluation=initial_origin(prev_ref,task,task_ref["sha256"]))
    receipt_ref = bind("receipt.json",receipt)
    mark = dict(schema="pilot-r3-execution-adapter-v1", receipt=receipt_ref, original_task=task_ref,
                original_result=prev_ref, adapter=reference(adapter), worker=reference(worker), checkpoint=cp)
    if migration:
        task = {**task, "stage_a_migration":receipt_ref}
        task_ref = bind("new.json",task)
        mark.update(schema=ADAPTER,new_task=task_ref)
    resume = dict(path=cp["path"],sha256=cp["sha256"],source_task_id="t",state_sha256=cp["state_sha256"],
        from_tokens=tokens,from_steps=steps,fallback_used=damage=="fallback",
        checkpoint_load={"status":"not_tested" if damage=="load" else "exact"})
    if flag is not None:
        resume["state_reused"] = flag
    if damage == "digest":
        resume.pop("state_sha256")
    if damage == "gap":
        resume["from_tokens"] += 128
    result = {**previous, "task_sha256":task_ref["sha256"], "execution_adapter":mark, "resume":resume,
              "training":dict(from_steps=steps,from_tokens=tokens,steps=2,tokens=256,segment_steps=2-steps,segment_tokens=256-tokens)}
    if passes:
        assert continuation_origin(task,result) == previous
    else:
        with pytest.raises(ValueError):
            continuation_origin(task,result)


@pytest.fixture(params=["a2", "a3"])
def chain(tmp_path, request):
    worker, adapter = tmp_path/"worker.py", tmp_path/"adapter.py"
    worker.write_text("# immutable worker")
    adapter.write_text("# immutable adapter")
    identity = {"source_files": {str(worker): sha256(worker)}}
    manifest, selected, _ = make_campaign(tmp_path/"round", subject_count=1, identity=identity)
    choice = selected[manifest["subjects"][0]["id"]]
    phase = request.param
    entry = choice["selection_a2" if phase == "a2" else "confirmation"][0]
    parent_entry = choice["selection_a1" if phase == "a2" else "selection_a2"][0]
    parent = g.load(parent_entry["task_path"]), g.load(parent_entry["result_path"])
    task, result = g.load(entry["task_path"]), g.load(entry["result_path"])
    task_ref = reference(entry["task_path"])
    result.update(id=task["id"], model_id=task["model_id"], phase=phase)
    first_start = result["training"]["from_tokens"]
    result["training"].update(from_steps=0, steps=0)
    result["resume"].update(from_steps=0)
    prior_refs = []
    for i in range(2):
        partial = deepcopy(result)
        start = partial["training"]["from_tokens"]
        partial.update(status="budget_exhausted", stop_reason="signal",
            health={"eligible": False, "blocking_reasons": ["declared_training_exposure_not_reached"]})
        partial["training"].update(tokens=start+128, steps=i+1, segment_tokens=128, segment_steps=1)
        cp = tmp_path/f"interrupted-{i}.pt"
        cp.write_bytes(f"complete state {i}".encode())
        partial["checkpoint"] = {**reference(cp), "tokens":start+128, "steps":i+1, "state_sha256":str(i)}
        p = write_json(tmp_path/f"partial-{i}.json", partial)
        prior_refs.append(reference(p))
        receipt = {"schema":"pilot-r3-task-resume-receipt-v2", "task_id":task["id"],
            "original_task":task_ref, "source_identity":identity, "result":reference(p),
            "prior_results":list(prior_refs), "checkpoint":partial["checkpoint"]}
        receipt_ref = reference(write_json(tmp_path/f"receipt-{i}.json", receipt))
        result = deepcopy(result)
        result["training"].update(from_tokens=start+128, from_steps=i+1,
            tokens=task["target_tokens"], steps=i+2,
            segment_tokens=task["target_tokens"]-start-128, segment_steps=1)
        result["resume"] = {**partial["checkpoint"], "from_tokens":start+128, "from_steps":i+1,
            "state_reused":True, "fallback_used":False, "runtime_override":True,
            "checkpoint_load":{"status":"exact"}}
        result["execution_adapter"] = {"schema":"pilot-r3-execution-adapter-v1", "adapter":reference(adapter),
            "worker":reference(worker), "receipt":receipt_ref, "original_task":task_ref,
            "original_result":reference(p), "checkpoint":partial["checkpoint"]}
    return task, result, parent, first_start


def test_two_interruptions_pass_both_stage_and_parent_gates_without_rewriting_counters(chain):
    task, result, parent, start = chain
    before = deepcopy(result)
    assert g.eligible(task, result, require_resume=True) == []
    assert g.continuation_matches(parent, task, result)
    assert continuation_origin(task,result)["training"]["from_tokens"] == start
    assert result["training"]["from_tokens"] == start+256
    assert result == before


@pytest.mark.parametrize("damage", ["task", "result_hash", "checkpoint_hash", "adapter_hash",
    "load", "state", "gap", "steps", "segment", "missing_adapter", "unrelated_parent", "ancestor_science"])
def test_changed_chain_never_qualifies_from_end_checkpoint_readback_alone(chain, damage):
    task, result, parent, _ = chain
    a = result["execution_adapter"]
    if damage == "task":
        task["seed"] += 1
    elif damage in {"result_hash", "adapter_hash"}:
        a["original_result" if damage=="result_hash" else "adapter"]["sha256"] = "0"*64
    elif damage == "checkpoint_hash":
        Path(a["checkpoint"]["path"]).write_bytes(b"damaged")
    elif damage == "load":
        result["resume"]["checkpoint_load"]["status"] = "not_tested"
    elif damage == "state":
        result["resume"]["state_sha256"] = "different"
    elif damage == "gap":
        result["training"]["from_tokens"] += 128
    elif damage == "steps":
        result["resume"]["from_steps"] += 1
    elif damage == "segment":
        result["training"]["segment_tokens"] -= 1
    elif damage == "missing_adapter":
        result.pop("execution_adapter")
    elif damage == "unrelated_parent":
        parent[1]["checkpoint"]["sha256"] = "different"
    else:
        path = Path(a["original_result"]["path"])
        previous = g.load(path)
        previous["health"]["blocking_reasons"].append("nonfinite_gradient")
        write_json(path,previous)
        a["original_result"] = reference(path)
        receipt = g.load(a["receipt"]["path"])
        receipt["result"] = reference(path)
        write_json(a["receipt"]["path"],receipt)
        a["receipt"] = reference(a["receipt"]["path"])
    if damage != "unrelated_parent":
        assert g.eligible(task,result,require_resume=True)
    assert not g.continuation_matches(parent,task,result)

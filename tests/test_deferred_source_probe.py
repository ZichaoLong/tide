from pathlib import Path

from pilotlab.data3 import DocumentStore3
from pilotlab.prepare_resume3 import reference
from scripts.deferred_source_probe import compare, one_step
from test_pilot_resume_adapter3 import interrupted


def test_paired_update_restores_the_original_checkpoint_on_both_arms(interrupted):
    task = interrupted["task"]
    checkpoint = interrupted["result"]["checkpoint"]
    case = {"task":reference(interrupted["task_path"]), "model_id":task["model_id"], "key":task["id"],
            "start":"checkpoint_successor", "evidence":checkpoint, "checkpoint_task_id":task["id"], "sample":True}
    store = DocumentStore3(Path(interrupted["data"]))
    args = task["source_identity"], {}, store, "cpu"
    old = one_step(case, "historical", *args)
    new = one_step(case, "reference", *args)
    assert old["outcome"] == "updated" and old["from_step"] == checkpoint["steps"]
    assert old["checkpoint_state_sha256"] == checkpoint["state_sha256"]
    assert compare(old, new)["outcome"] == "sample_exact"
    assert compare(old, dict(new, input_sha256="other"))["outcome"] == "different_start"

"""Replay an identified CPU failure from its verified Stage A checkpoint."""
import argparse
import json
import os
from pathlib import Path
import sys

import torch

from pilotlab import worker3 as w
from pilotlab.data3 import DocumentStore3
from pilotlab.io import atomic_json, sha256
from pilotlab.models3 import build_model
from pilotlab.resume_adapter3 import checked
from pilotlab.source_migration3 import digest, read_bound


def owners(task, data, device):
    model = build_model(task["model"], data.vocab_size, task["init_recipe"]).to(device)
    return model, w.optimizer_for(model, task)


def advance(model, optimizer, batch, state, device, task, tokens, sample=False):
    x, y, reset = batch
    result = w.train_update(model, optimizer, x, y, None if reset else state, device, task, tokens, sample=sample)
    return result["state"], {k: result[k] for k in ("loss", "objective", "lr", "grad_norm", "backward_divisor")}


def run(args):
    plan = json.loads(args.plan.read_text())
    old, previous = read_bound(plan["original_task"]), read_bound(plan["old_result"])
    task = {**old, "source_identity": read_bound(plan["source_identity"])}
    data = DocumentStore3(plan["data"])
    w.validate_source_identity(task, Path(plan["data"]))
    device = w.initialize("cpu", task)
    cp = previous["checkpoint"]
    output = args.plan.parent
    if args.restore:
        direct = json.loads((output/"direct.json").read_text())
        record = direct["saved"]
        payload, origin = w.load_checkpoint(record["path"], task, device, record["sha256"])
        assert not origin["fallback_used"] and payload["state_sha256"] == record["state_sha256"]
        model, opt = owners(task, data, device)
        state, stream = w.restore_owners(payload, model, opt, data, task, device)
        x, y, reset = stream.next()
        state, row = advance(model, opt, (x, y, reset), state, device, task, payload["tokens"])
        end = w.checkpoint_payload(model, opt, state, stream, task, device, payload["tokens"]+y.numel(),
                                   payload["steps"]+1, payload["initialization"])
        assert end["state_sha256"] == direct["successor_state_sha256"] and row == direct["successor"]
        atomic_json(output/"qualification.json", {**direct, "status":"passed", "restore_pid":os.getpid(),
                    "restored_step":end["steps"], "successor_state_exact":True})
        return
    assert previous["task_sha256"] == plan["original_task"]["sha256"] and previous["id"] == task["id"]
    failure = torch.load(checked(previous["failure_capture"], "failed batch"), weights_only=False, map_location="cpu")
    assert failure["task"] == old and failure["resumable"] is False and failure["update_in_progress"]
    journal = {r["step"]:r for r in map(json.loads, (Path(plan["old_result"]["path"]).parent/"steps.jsonl").read_text().splitlines())}
    payload, origin = w.load_checkpoint(cp["path"], old, device, cp["sha256"])
    assert not origin["fallback_used"] and payload["state_sha256"] == cp["state_sha256"]
    model, opt = owners(task, data, device)
    state, stream = w.restore_owners(payload, model, opt, data, old, device)
    step, tokens, init = payload["steps"], payload["tokens"], payload["initialization"]
    del payload
    while step < failure["completed_steps"]:
        batch = stream.next()
        state, row = advance(model, opt, batch, state, device, task, tokens, journal[step+1]["sampled_diagnostics"])
        tokens += batch[1].numel()
        step += 1
        assert row["loss"] == journal[step]["loss"] and row["lr"] == journal[step]["lr"]
    x, y, reset = stream.next()
    assert torch.equal(x, failure["inputs"]) and torch.equal(y, failure["targets"])
    # The failed worker assigned the scheduled LR before attempting backward.
    for group in opt.param_groups:
        group["lr"] = w.learning_rate(task, tokens+y.numel())
    for name, value in (("model", model.state_dict()), ("optimizer", opt.state_dict()),
                        ("private_state", None if reset else state)):
        assert w.digest_tree(value) == w.digest_tree(failure[name]), name
    del failure
    before = w.digest_tree(model.state_dict())
    state, fault = advance(model, opt, (x,y,reset), state, device, task, tokens)
    tokens += y.numel()
    step += 1
    assert fault["backward_divisor"] in (8,16) and before != w.digest_tree(model.state_dict())
    saved = w.save_checkpoint(output, w.checkpoint_payload(model, opt, state, stream, task, device, tokens, step, init))
    assert w.verify_saved_checkpoint(saved, task, data, device)["status"] == "exact"
    batch = stream.next()
    state, successor = advance(model, opt, batch, state, device, task, tokens)
    end = w.checkpoint_payload(model, opt, state, stream, task, device, tokens+batch[1].numel(), step+1, init)
    atomic_json(output/"direct.json", {"schema":"pilot-r3-backward-range-qualification-v1",
        "device":str(device), "threads":torch.get_num_threads(), "pid":os.getpid(),
        "original_task":plan["original_task"], "old_result":plan["old_result"],
        "official_checkpoint":cp, "failure_input":previous["failure_capture"],
        "new_source_digest":digest(task["source_identity"]), "fault_step":step, "fault_batch_exact":True,
        "fault_owners_exact":True, "fault_divisor":fault["backward_divisor"], "fault":fault, "saved":saved,
        "successor":successor, "successor_state_sha256":end["state_sha256"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    run(args)
    if not args.restore:
        # Drop the direct process and its allocation arena before reloading.
        os.execv(sys.executable, [sys.executable, __file__, str(args.plan), "--restore"])

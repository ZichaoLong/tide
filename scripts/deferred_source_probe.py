"""Compare bounded Stage A updates from two frozen workers on the original backend.

Run each source in a separate interpreter. Publish scalar observations and state
digests, never overwrite a trajectory, and never turn a probe into a selection.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def checksum(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bound(ref):
    path = Path(ref["path"])
    if checksum(path) != ref["sha256"]:
        raise ValueError(f"changed input {path}")
    return json.loads(path.read_bytes())


def publish(path, value):
    from pilotlab.io import atomic_json
    atomic_json(path, value)


def one_step(case, arm, identity, model_policy, data, device_tag):
    import torch
    from pilotlab import worker3 as w
    from pilotlab.models3 import build_model
    from pilotlab.worker import StreamBatch, digest_tree, rng_state, synchronize, tree_map

    original = bound(case["task"])
    if arm == "historical" and original["source_identity"] != identity:
        raise ValueError("historical task differs from the executed worker identity")
    task = dict(original, source_identity=identity)
    if arm != "historical" and model_policy:
        config = {**original["model"]["config"], **model_policy}
        task["model"] = {**original["model"], "config": config}
    if original["device"] != device_tag or task["model_id"] != case["model_id"]:
        raise ValueError("probe backend or historical model changed")
    w.validate_task(task)
    device = w.initialize(device_tag, task)
    model = build_model(task["model"], data.vocab_size, init_recipe=task["init_recipe"]).to(device)
    optimizer = w.optimizer_for(model, task)
    state, tokens, steps = None, 0, 0
    witness = case["start"]
    checkpoint_digest = None
    reset = None
    if witness == "failure_capture":
        ref = case["evidence"]
        if checksum(ref["path"]) != ref["sha256"]:
            raise ValueError("changed saved failure")
        saved = torch.load(ref["path"], weights_only=False, map_location="cpu")
        if (saved["task"] != original or saved.get("resumable") is not False
                or saved.get("update_in_progress") is not True):
            raise ValueError("saved witness is not the bound failed update")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        state = tree_map(saved["private_state"], lambda t: t.to(device))
        x, y = saved["inputs"], saved["targets"]
        tokens, steps = saved["completed_tokens"], saved["completed_steps"]
        # This saved witness has no stream or RNG. Check the owners it did capture.
        for old, actual in ((saved["model"], model.state_dict()), (saved["optimizer"], optimizer.state_dict()),
                            (saved["private_state"], state)):
            if digest_tree(old) != digest_tree(actual):
                raise ValueError("saved failure owners changed on load")
        del saved
    else:
        if witness != "seed_start":
            ref = case["evidence"]
            payload = w._read_checkpoint(Path(ref["path"]), ref["sha256"])
            w.validate_checkpoint(payload, original, device)
            if (payload["task_id"] != case["checkpoint_task_id"] or payload["tokens"] != ref["tokens"]
                    or payload["steps"] != ref["steps"]):
                raise ValueError("checkpoint owner or exposure changed")
            state, stream = w.restore_owners(payload, model, optimizer, data, original, device)
            tokens, steps, checkpoint_digest = payload["tokens"], payload["steps"], payload["state_sha256"]
            del payload
        else:
            if original.get("resume_checkpoint") or original.get("evaluation_start_tokens", 0):
                raise ValueError("seed origin cannot replace a parent checkpoint")
            stream = StreamBatch(data, original)
        x, y, reset = stream.next()
        if reset:
            state = None
    if y.numel() != original["batch_size"] * original["sequence_length"]:
        raise ValueError("saved batch violates the trajectory shape")
    before = digest_tree([model.state_dict(), optimizer.state_dict(), state])
    row = dict(task=original["id"], case=case["key"], start=witness, from_step=steps, from_tokens=tokens,
               batch_tokens=y.numel(), input_sha256=digest_tree((x,y)), owner_sha256=before,
               checkpoint_state_sha256=checkpoint_digest, reset=reset, rng_sha256=digest_tree(rng_state(device)))
    try:
        update = w.train_update(model, optimizer, x, y, state, device, task, tokens, sample=case["sample"])
        synchronize(device)
    except (RuntimeError, ArithmeticError, ValueError) as error:
        return dict(row, outcome="update_error", error_type=type(error).__name__, error=str(error),
                    capacity_error=w.is_capacity_error(error, device), memory=w.device_memory(device))
    metrics = {k: float(update[k]) for k in ("loss", "objective", "grad_norm", "lr")}
    if not all(math.isfinite(x) for x in metrics.values()):
        raise ArithmeticError("nonfinite returned metrics")
    work = update.get("work") or {}
    keys = ("active_ratio", "min_active_backbone_parameters", "max_active_backbone_parameters",
            "semantic_full_owner_tick_rows", "measured_external_tokens")
    work = {key: work[key] for key in keys if key in work}
    return dict(row, outcome="updated", metrics=metrics, activity=work,
                clipped_gradient_sha256=digest_tree({n:p.grad for n,p in model.named_parameters()}),
                successor_sha256=digest_tree([model.state_dict(), optimizer.state_dict(), update["state"]]),
                memory=w.device_memory(device))


def run_arm(plan, arm, out, wall_end):
    from pilotlab import worker3 as w
    from pilotlab.data3 import DocumentStore3
    import torch
    config = plan["arms"][arm]
    identity = bound(config["identity"])
    w.validate_source_identity({"source_identity": identity}, Path(plan["data"]))
    store = DocumentStore3(plan["data"])
    results = {}
    for case in plan["cases"]:
        if time.time() >= wall_end:
            raise TimeoutError("source probe exhausted its time window")
        try:
            row = one_step(case, arm, identity, config["model_policy"], store, os.environ["RUNQ_DEVICE"])
        except Exception as e:
            row = {"case":case["key"], "outcome":"input_or_runtime_error", "error_type":type(e).__name__, "error":str(e)}
        results[case["key"]] = row
        publish(out/(arm+".json"), results)
        gc.collect()
        if os.environ["RUNQ_DEVICE"].startswith("npu") and hasattr(torch, "npu") and torch.npu.is_initialized():
            torch.npu.empty_cache()


def compare(a, b):
    if "input_or_runtime_error" in (a["outcome"], b["outcome"]):
        return {"outcome": "input_or_runtime_error"}
    if any(a[k] != b[k] for k in ("from_step", "from_tokens", "input_sha256", "owner_sha256")):
        return {"outcome": "different_start"}
    if a["start"] != "failure_capture" and a["rng_sha256"] != b["rng_sha256"]:
        return {"outcome": "different_start_rng"}
    if a["outcome"] != "updated" or b["outcome"] != "updated":
        return {"outcome": a["outcome"] + "_to_" + b["outcome"]}
    fields = ("metrics", "activity", "clipped_gradient_sha256", "successor_sha256")
    return {"outcome": "sample_exact" if all(a[k] == b[k] for k in fields) else "sample_changed",
            "equal": {k:a[k] == b[k] for k in fields},
            "delta": {k:b["metrics"][k]-v for k,v in a["metrics"].items()}}


def execute(args):
    if checksum(args.plan) != args.sha256:
        raise ValueError("changed probe plan")
    plan = json.loads(args.plan.read_bytes())
    if checksum(__file__) != plan["probe_sha256"] or os.environ["RUNQ_DEVICE"] != plan["device"]:
        raise ValueError("changed probe runtime or backend")
    out = args.plan.parent/"observations"
    if args.arm:
        run_arm(plan, args.arm, out, args.wall_end)
        return
    from pilotlab.preflight3 import finish_token_accounting
    from pilotlab.prepare_campaign3 import budget_remaining, storage_observation
    budget_remaining(plan["ledger"], time.time())
    storage_observation(out, reserve=0, snapshot_bytes=0)
    out.mkdir(exist_ok=False)
    wall_end = min(plan["deadline"], time.time()+plan["max_run_seconds"])
    packaged = {}
    try:
        for arm, spec in plan["arms"].items():
            command = [sys.executable, str(Path(__file__).resolve()), str(args.plan), args.sha256,
                       "--arm", arm, "--wall-end", str(wall_end)]
            subprocess.run(command, cwd=spec["source"], env={**os.environ, "PYTHONPATH":os.pathsep.join(
                [spec["source"], str(Path(spec["foundation"])/"src")])}, check=True, timeout=max(1,wall_end-time.time()))
            packaged[arm] = json.loads((out/(arm+".json")).read_bytes())
        history = packaged["historical"]
        comparisons = {arm:{key:compare(history[key], value) for key,value in rows.items()}
                       for arm,rows in packaged.items() if arm != "historical"}
        invalid = any(row["outcome"] == "input_or_runtime_error" for rows in comparisons.values() for row in rows.values())
        status = "review_incomplete" if invalid else "observations_recorded"
        publish(out/"comparison.json", dict(status=status, plan_sha256=args.sha256, comparisons=comparisons,
                model=plan["model"], device=plan["device"], physical_card=os.environ.get("RUNQ_PHYSICAL_DEVICE"),
                qualification="sampled updates only", recorded_unix=time.time()))
        if invalid:
            raise RuntimeError("one or more historical inputs or runtime checks failed")
    finally:
        charges = {}
        for arm in plan["arms"]:
            path = out/(arm+".json")
            if path.exists():
                observed = json.loads(path.read_bytes())
                charges[arm] = {"training":{"segment_tokens":sum(x["batch_tokens"] for x in observed.values() if x["outcome"]=="updated")}}
        finish_token_accounting(plan["ledger"], str(args.plan.parent), charges,
                                "observed" if sys.exc_info()[0] is None else "incomplete")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("plan", type=Path)
    p.add_argument("sha256")
    p.add_argument("--arm")
    p.add_argument("--wall-end", type=float)
    execute(p.parse_args())

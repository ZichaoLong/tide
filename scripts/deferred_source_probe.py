"""Compare Stage A updates from frozen workers on the original backend.

Pass a frozen source plan or an attempt referencing that plan, its original
probe, and any previous observation files. Each attempt has its own directory.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time


def checksum(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bound(ref):
    path = Path(ref["path"])
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != ref["sha256"]:
        raise ValueError(f"changed input {path}")
    return json.loads(data)


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


def run_arm(plan, arm, out, previous, step):
    from pilotlab import worker3 as w
    from pilotlab.data3 import DocumentStore3
    import torch
    config = plan["arms"][arm]
    identity = bound(config["identity"])
    w.validate_source_identity({"source_identity": identity}, Path(plan["data"]))
    store = DocumentStore3(plan["data"])
    results = dict(previous)
    for case in plan["cases"]:
        if case["key"] in previous:
            continue
        if time.time() >= plan["deadline"]:
            raise TimeoutError("campaign deadline reached")
        try:
            row = step(case, arm, identity, config["model_policy"], store, os.environ["RUNQ_DEVICE"])
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


def prior_observations(request, plan, source_ref):
    cases = {case["key"]: case for case in plan["cases"]}
    if len(cases) != len(plan["cases"]):
        raise ValueError("duplicate comparison point")
    previous = {arm: {} for arm in plan["arms"]}
    predecessor = request.get("previous_attempt", source_ref)
    if predecessor != source_ref and bound(predecessor).get("source_plan") != source_ref:
        raise ValueError("previous attempt used a different source plan")
    directory = Path(predecessor["path"]).parent / "observations"
    for arm, ref in request.get("prior_observations", {}).items():
        if arm not in previous or Path(ref["path"]) != directory / (arm + ".json"):
            raise ValueError("unbound previous observation file")
        for key, row in bound(ref).items():
            if key not in cases or row["case"] != key:
                raise ValueError("previous observation has a different comparison point")
            if row["outcome"] == "input_or_runtime_error":
                continue
            case = cases[key]
            if (row["outcome"] not in {"updated", "update_error"} or row["start"] != case["start"]
                    or row["batch_tokens"] != case["tokens_per_arm"] or row["task"] != bound(case["task"])["id"]):
                raise ValueError("previous observation differs from its frozen case")
            previous[arm][key] = row
    return previous


def run_with_budget(command, spec, plan):
    from pilotlab.prepare_campaign3 import budget_remaining

    def remaining():
        now = time.time()
        budget = budget_remaining(plan["ledger"], now)
        if budget["deadline_unix"] != plan["deadline"]:
            raise ValueError("probe and campaign deadlines differ")
        seconds = budget["remaining_wall_seconds"]
        if plan["device"].startswith("npu"):
            seconds = min(seconds, budget["remaining_preparation_device_seconds"], budget["remaining_device_seconds"])
        return seconds

    # A polling interval checks the shared budget; it never expires a healthy arm.
    end = time.monotonic() + remaining()
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([spec["source"], str(Path(spec["foundation"]) / "src")])}
    with subprocess.Popen(command, cwd=spec["source"], env=env) as proc:
        try:
            while True:
                seconds = min(remaining(), end - time.monotonic())
                if seconds <= 0:
                    raise TimeoutError("campaign deadline or device-time budget reached")
                try:
                    code = proc.wait(timeout=min(15, seconds))
                    break
                except subprocess.TimeoutExpired:
                    continue
        finally:
            if proc.poll() is None:
                proc.kill()
        if code:
            raise subprocess.CalledProcessError(code, command)


def execute(args):
    request_ref = {"path": str(args.plan.resolve()), "sha256": args.sha256}
    request = bound(request_ref)
    source_ref = request.get("source_plan", request_ref)
    plan = bound(source_ref) if source_ref != request_ref else request
    if checksum(__file__) != request["probe_sha256"] or os.environ["RUNQ_DEVICE"] != plan["device"]:
        raise ValueError("changed probe runtime or backend")
    step, compare_rows = one_step, compare
    if "source_plan" in request:
        # Use the original measurement and comparison functions, including for a resumed arm.
        source = request["source_probe"]
        if checksum(source["path"]) != source["sha256"] or source["sha256"] != plan["probe_sha256"]:
            raise ValueError("changed original probe")
        original = runpy.run_path(source["path"])
        step, compare_rows = original["one_step"], original["compare"]
    previous = prior_observations(request, plan, source_ref)
    out = args.plan.parent / "observations"
    if args.arm:
        run_arm(plan, args.arm, out, previous[args.arm], step)
        return

    from pilotlab.preflight3 import finish_token_accounting
    from pilotlab.prepare_campaign3 import budget_remaining, storage_observation
    budget = budget_remaining(plan["ledger"], time.time())
    if plan["deadline"] != budget["deadline_unix"]:
        raise ValueError("probe and campaign deadlines differ")
    pending_tokens = sum(case["tokens_per_arm"] for rows in previous.values()
                         for case in plan["cases"] if case["key"] not in rows)
    claim_ref = source_ref if request.get("use_unstarted_reservation") else request_ref
    claim_run = str(Path(claim_ref["path"]).parent)
    if pending_tokens:
        claim = json.loads(Path(plan["ledger"]).read_bytes())["preparation_token_reservations"][claim_run]
        if (claim["plan_sha256"] != claim_ref["sha256"] or claim["status"] != "reserved"
                or claim["reserved_tokens"] != pending_tokens):
            raise ValueError("missing or mismatched reservation for outstanding observations")
    storage_observation(out, reserve=0, snapshot_bytes=0)
    out.mkdir(exist_ok=False)
    packaged = {}
    try:
        for arm, rows in previous.items():
            if rows:
                publish(out / (arm + ".json"), rows)
        for arm, spec in plan["arms"].items():
            path = out / (arm + ".json")
            if len(previous[arm]) < len(plan["cases"]):
                command = [sys.executable, str(Path(__file__).resolve()), str(args.plan.resolve()), args.sha256,
                           "--arm", arm]
                run_with_budget(command, spec, plan)
            packaged[arm] = json.loads(path.read_bytes())
            if packaged[arm].keys() != {case["key"] for case in plan["cases"]}:
                raise ValueError(f"incomplete observation coverage for {arm}")
        history = packaged["historical"]
        comparisons = {arm:{key:compare_rows(history[key], value) for key,value in rows.items()}
                       for arm,rows in packaged.items() if arm != "historical"}
        invalid = any(row["outcome"] == "input_or_runtime_error" for rows in comparisons.values() for row in rows.values())
        status = "review_incomplete" if invalid else "observations_recorded"
        publish(out/"comparison.json", dict(status=status, plan_sha256=source_ref["sha256"],
                attempt=request_ref, prior_observations=request.get("prior_observations", {}), comparisons=comparisons,
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
                charges[arm] = {"training": {"segment_tokens": sum(row["batch_tokens"] for key, row in observed.items()
                                      if key not in previous[arm] and row["outcome"] == "updated")}}
        if pending_tokens:
            finish_token_accounting(plan["ledger"], claim_run, charges,
                                    "observed" if sys.exc_info()[0] is None else "incomplete")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("plan", type=Path)
    p.add_argument("sha256")
    p.add_argument("--arm")
    execute(p.parse_args())

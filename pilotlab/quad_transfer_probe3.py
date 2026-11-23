"""Finite CPU proof of the selected quad recipes across two frozen sources.

The new frozen copy of this file drives one fresh process per source/model.
Children import the designated source's real worker/model/data modules and
execute exactly three ordinary updates. Observation hooks never change a
gradient, optimizer argument, input, private state, or RNG. This is a finite
CPU equality witness, not an NPU trajectory or learning-quality claim.
"""
from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


STEPS = 3
SEQUENCE_LENGTH = 128
MAX_MODELS = 24
PATHS = ("forward", "backward", "optimizer", "state")
RECIPE_FIELDS = ("model_id", "model", "seed", "recipe_id", "init_recipe", "batch_size", "sequence_length",
                 "stream_chunks", "cpu_threads", "precision", "deterministic", "lr", "schedule_tokens",
                 "warmup_tokens", "lr_floor_ratio", "weight_decay", "grad_clip", "auxiliary_loss_weight", "device")


def load(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def reference(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path)}


def checked(entry):
    if reference(entry["path"]) != entry:
        raise ValueError("probe reference changed: " + str(entry.get("path")))
    return load(entry["path"])


def process_identity():
    text = Path("/proc/self/stat").read_text()
    fields = text[text.rfind(")") + 2:].split()
    return {"pid": os.getpid(), "start_ticks": int(fields[19]), "ppid": os.getppid(),
            "pgrp": int(fields[2]), "state": fields[0]}


def observe_descendants(result, current_key=None):
    """Follow recorded start identities and subreaper-adopted descendants."""
    from pilotlab.scheduler import family_members, process_table, same_live

    table = process_table(strict=True)
    seen = result.setdefault("all_seen_processes", {})
    seeds = {os.getpid()} | {row["pid"] for row in seen.values() if same_live(row, table)}
    identities = {pid: table[pid] for pid in family_members(seeds, table) - {os.getpid()} if pid in table}
    for pid, identity in identities.items():
        key = f"{pid}:{identity['start_ticks']}"
        seen.setdefault(key, identity)
        if current_key:
            result["children"][current_key].setdefault("all_seen_processes", {}).setdefault(key, identity)
    return {pid: row for pid, row in identities.items() if row["state"] != "Z"}


def reap_orphans(result, *, direct_pid=None):
    from pilotlab.scheduler import proc_identity

    for row in result.get("all_seen_processes", {}).values():
        if row["pid"] == direct_pid:
            continue
        current = proc_identity(row["pid"], strict=True)
        if current and current["start_ticks"] == row["start_ticks"] and current["ppid"] == os.getpid():
            try:
                os.waitpid(row["pid"], os.WNOHANG)
            except ChildProcessError:
                pass


def cleanup_descendants(result, process=None, current_key=None):
    from pilotlab.scheduler import proc_identity

    deadline = time.monotonic() + 10
    while True:
        members = observe_descendants(result, current_key)
        for row in members.values():
            current = proc_identity(row["pid"], strict=True)
            if current and current["start_ticks"] == row["start_ticks"]:
                try:
                    os.kill(row["pid"], signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if process is not None:
            process.poll()
        reap_orphans(result, direct_pid=process.pid if process else None)
        members = observe_descendants(result, current_key)
        if not members or time.monotonic() >= deadline:
            result["remaining_descendants"] = list(members.values())
            if current_key:
                result["children"][current_key].update(remaining_descendants=list(members.values()),
                    return_code=process.returncode if process else result["children"][current_key].get("return_code"),
                    terminal_observed_unix=time.time())
            return not members
        time.sleep(.05)


def check_deadline(deadline):
    if time.time() >= deadline:
        raise TimeoutError("finite quad CPU probe deadline reached")


def build_plan(old_prepared, new_prepared, selected, *, timeout_seconds=1800):
    """Read and bind the requested frozen recipes; this function runs no model."""
    from pilotlab.transfer3 import _source_comparison

    if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool) or not 0 < timeout_seconds <= 3600:
        raise ValueError("probe wall-clock limit must be in (0, 3600] seconds")
    references = {"old_prepared": reference(old_prepared), "new_prepared": reference(new_prepared),
                  "selected": reference(selected)}
    old, new, choices = (checked(references[key]) for key in ("old_prepared", "new_prepared", "selected"))
    if old.get("schema") != "pilot-r3-prepared-inputs-v1" or new.get("schema") != "pilot-r3-prepared-inputs-v1":
        raise ValueError("two frozen round-three prepared inputs are required")
    comparison = _source_comparison(old, new)
    if Path(old["python"]).absolute() != Path(new["python"]).absolute():
        raise ValueError("both CPU sides must use the same declared interpreter")
    if Path(old["budget_ledger"]).resolve() != Path(new["budget_ledger"]).resolve():
        raise ValueError("both sides must retain the original campaign budget")
    model_ids = sorted(choices)
    if not 1 <= len(model_ids) <= MAX_MODELS or any(not isinstance(mid, str) or not mid for mid in model_ids):
        raise ValueError("one finite selection of 1..24 quad models is required")
    recipes, evidence = {}, {}
    for model_id in model_ids:
        choice = choices[model_id]
        confirmations = choice.get("confirmation", [])
        if len(confirmations) != 2:
            raise ValueError("each chosen recipe requires its two recorded confirmation tasks")
        candidates = []
        evidence[model_id] = []
        for item in confirmations:
            bound = {key: reference(item[key + "_path"]) for key in ("task", "result")}
            task, result = checked(bound["task"]), checked(bound["result"])
            if (task.get("model_id") != model_id or task.get("phase") != "a3"
                    or task.get("source_identity") != old["source_identity"]
                    or task.get("recipe_id") != choice.get("recipe_id")
                    or task.get("init_recipe") != choice.get("init_recipe")
                    or task.get("model", {}).get("family") != "tide"
                    or task.get("model", {}).get("config", {}).get("allocation") != "quad"
                    or result.get("status") != "completed"
                    or result.get("source_identity") != task["source_identity"]
                    or result.get("task_sha256") != bound["task"]["sha256"]):
                raise ValueError("selected quad task/result/recipe/source binding mismatch: " + model_id)
            if (type(task.get("batch_size")) is not int or task["batch_size"] < 1
                    or task.get("sequence_length") != SEQUENCE_LENGTH
                    or type(task.get("seed")) is not int or task["seed"] < 0
                    or type(task.get("cpu_threads")) is not int or not 1 <= task["cpu_threads"] <= len(os.sched_getaffinity(0))):
                raise ValueError("probe needs the selected batch/128-token/seed/CPU-thread contract")
            candidates.append({key: copy.deepcopy(task[key]) for key in RECIPE_FIELDS if key in task})
            evidence[model_id].append(bound)
        candidates.sort(key=lambda task: task["seed"])
        if candidates[0]["seed"] == candidates[1]["seed"] or any(
                candidates[0].get(key) != candidates[1].get(key) for key in RECIPE_FIELDS if key != "seed"):
            raise ValueError("confirmation seeds must be distinct and use one identical recipe")
        task = candidates[0]
        task.update(id="quad-transfer-" + model_id, phase="preflight", eval_tokens=0,
                    target_tokens=STEPS * task["batch_size"] * SEQUENCE_LENGTH)
        recipes[model_id] = task
    budget_path = Path(new["budget_ledger"]).resolve()
    budget = load(budget_path)
    if (budget.get("schema") != "pilot-r3-budget-v1" or budget.get("max_device_seconds") != 7200 * 3600
            or budget.get("preparation_max_processed_tokens") != 64 * 2**20
            or budget.get("deadline_unix") != budget.get("started_unix", 0) + 30 * 86400):
        raise ValueError("the original finite campaign budget is required")
    started = time.time()
    deadline = min(started + timeout_seconds, budget["deadline_unix"])
    check_deadline(deadline)
    return {"schema": "pilot-r3-quad-transfer-probe-plan-v1", **references,
            "old_source_identity": old["source_identity"], "new_source_identity": new["source_identity"],
            "source_comparison": comparison, "model_ids": model_ids, "recipes": recipes,
            "selected_evidence": evidence, "steps_per_model": STEPS, "tested_paths": list(PATHS),
            "budget_ledger": str(budget_path), "reserved_tokens": 2 * sum(task["target_tokens"] for task in recipes.values()),
            "reservation_formula": "2 sources * 3 updates * 128 loss tokens * sum(selected batch_size per model)",
            "max_parallel": 1, "backend": "cpu", "device_seconds": 0,
            "probe": reference(Path(new["source"]) / "pilotlab/quad_transfer_probe3.py"),
            "started_unix": started, "deadline_unix": deadline,
            "input_policy": "independent same-seed real DocumentStore3/StreamBatch; three saved batches must match exactly",
            "scope": "fresh initialization using selected recipe; no old checkpoint loading, evaluation, or formal learning"}


def verify_references(plan):
    for key in ("old_prepared", "new_prepared", "selected"):
        checked(plan[key])
    if reference(plan["probe"]["path"]) != plan["probe"]:
        raise ValueError("frozen probe code changed")
    for entries in plan["selected_evidence"].values():
        for entry in entries:
            checked(entry["task"])
            checked(entry["result"])


def child_environment(prepared, task):
    return dict(os.environ, PYTHONPATH=os.pathsep.join((prepared["source"], str(Path(prepared["foundation"]) / "src"))),
        PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", PYTHONHASHSEED="0",
        TORCH_DEVICE_BACKEND_AUTOLOAD="0", ASCEND_RT_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS=str(task["cpu_threads"]), MKL_NUM_THREADS=str(task["cpu_threads"]), OPENBLAS_NUM_THREADS="1")


def run_child(plan_path, side, model_id, output, *, parent_pid=None):
    """Called only in a fresh process, with the side's source import roots."""
    if parent_pid is not None:
        if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "PR_SET_PDEATHSIG failed")
        if os.getppid() != parent_pid:
            raise RuntimeError("probe parent exited before child ownership was established")
    if os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") != "0":
        raise ValueError("CPU probe requires TORCH_DEVICE_BACKEND_AUTOLOAD=0")
    from pilotlab.io import atomic_json
    from pilotlab import worker3 as worker
    from pilotlab.data3 import DocumentStore3
    from pilotlab.models3 import build_model
    import torch

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan = load(plan_path)
    prepared = checked(plan[side + "_prepared"])
    identity = prepared["source_identity"]
    task = copy.deepcopy(plan["recipes"][model_id])
    task["source_identity"] = identity
    expected_worker = Path(prepared["source"]) / "pilotlab/worker3.py"
    result = {"schema": "pilot-r3-quad-transfer-side-v1", "status": "running", "side": side,
        "model_id": model_id, "source_identity": identity, "plan": reference(plan_path),
        "probe": plan["probe"], "process": process_identity(), "cgroup": Path("/proc/self/cgroup").read_text().strip(),
        "device": "cpu", "started_unix": time.time(),
        "selected_device": task.get("device"), "runtime_override": "CPU equality diagnostic; selected backend/recipe evidence remains original",
        "state_digest_fields": ["model", "optimizer", "private_state", "stream", "rng", "tokens", "steps"],
        "state_digest_exclusions": "source identity, task ID, initialization description and boundary text; contract compared separately; private state/clocks retained in full",
        "training": {"steps": 0, "segment_tokens": 0}, "digests": {}, "observations": [], "inputs": []}
    paths = {key: [] for key in PATHS}
    hooks = []
    try:
        if side not in {"old", "new"} or model_id not in plan["model_ids"] or plan["steps_per_model"] != STEPS:
            raise ValueError("unexpected independent probe side/model/update contract")
        if Path(worker.__file__).resolve() != expected_worker.resolve() or reference(__file__) != plan["probe"]:
            raise ValueError("child imported an unrequested frozen worker or probe")
        check_deadline(plan["deadline_unix"])
        worker.validate_task(task)
        worker.validate_source_identity(task, Path(prepared["data"]))
        device = worker.initialize("cpu", task)
        store = DocumentStore3(prepared["data"])
        model = build_model(task["model"], store.vocab_size, task.get("init_recipe", "default")).to(device)
        optimizer = worker.optimizer_for(model, task)
        stream = worker.StreamBatch(store, task)
        state, tokens = None, 0
        current = {}
        names = dict(model.named_parameters())
        result["initial_parameter_sha256"] = worker.digest_tree(model.state_dict())
        result["contract"] = worker.contract(task, "cpu")

        def forward_observation(module, args, value):
            if "forward" in current:
                raise RuntimeError("probe expected exactly one root forward per ordinary update")
            current["forward"] = worker.digest_tree({"logits": value.logits, "private_state": value.state,
                "auxiliary_loss": getattr(value, "auxiliary_loss", None), "work": value.work})

        def gradient_observation(name):
            def observe(parameter):
                current["gradients_before_clip"][name] = worker.digest_tree(parameter.grad)
            return observe

        def before_optimizer(module, args, kwargs):
            current["optimizer_before"] = worker.digest_tree(optimizer.state_dict())
            current["parameters_before"] = worker.digest_tree(model.state_dict())
            current["gradients_after_clip"] = worker.digest_tree({name: parameter.grad for name, parameter in names.items()})

        def after_optimizer(module, args, kwargs):
            current["optimizer_after"] = worker.digest_tree(optimizer.state_dict())
            current["parameters_after"] = worker.digest_tree(model.state_dict())

        hooks.append(model.register_forward_hook(forward_observation))
        hooks.extend(parameter.register_post_accumulate_grad_hook(gradient_observation(name))
                     for name, parameter in names.items() if parameter.requires_grad)
        hooks.extend((optimizer.register_step_pre_hook(before_optimizer), optimizer.register_step_post_hook(after_optimizer)))
        for step in range(STEPS):
            check_deadline(plan["deadline_unix"])
            current.clear()
            current["gradients_before_clip"] = {name: None for name in names}
            inputs, targets, reset = stream.next()
            if reset:
                state = None
            input_path = output / f"input-{step + 1:03d}.json"
            atomic_json(input_path, {"step": step + 1, "inputs": inputs.tolist(), "targets": targets.tolist(), "state_reset": reset})
            result["inputs"].append(reference(input_path))
            before_rng = worker.digest_tree(worker.rng_state(device))
            update = worker.train_update(model, optimizer, inputs, targets, state, device, task, tokens, sample=True)
            state = update["state"]
            tokens += targets.numel()
            if set(current) != {"forward", "gradients_before_clip", "gradients_after_clip", "parameters_before",
                                "parameters_after", "optimizer_before", "optimizer_after"}:
                raise RuntimeError("an ordinary forward/backward/optimizer observation is missing")
            payload = worker.checkpoint_payload(model, optimizer, state, stream, task, device, tokens, step + 1, {})
            paths["forward"].append({"tensors": current["forward"], "loss": update["loss"], "objective": update["objective"]})
            paths["backward"].append({"before_clip": current["gradients_before_clip"],
                                      "after_clip": current["gradients_after_clip"], "norm_before_clip": update["grad_norm"]})
            paths["optimizer"].append({key: current[key] for key in
                ("parameters_before", "parameters_after", "optimizer_before", "optimizer_after")})
            paths["state"].append(payload["state_sha256"])
            row = {"step": step + 1, "tokens": tokens, "input_sha256": worker.digest_tree((inputs, targets)),
                   "state_reset": reset, "rng_before_sha256": before_rng,
                   "rng_after_sha256": worker.digest_tree(worker.rng_state(device)), "lr": update["lr"],
                   "loss": update["loss"], "objective": update["objective"],
                   **{key + "_sha256": worker.digest_tree(paths[key][-1]) for key in PATHS}}
            result["observations"].append(row)
            result["training"].update(steps=step + 1, segment_tokens=tokens)
            atomic_json(output / "progress.json", result)
        if tokens != task["target_tokens"]:
            raise RuntimeError("CPU probe exceeded or failed to consume its exact finite reservation")
        result["digests"] = {key: worker.digest_tree(paths[key]) for key in PATHS}
        worker.validate_source_identity(task, Path(prepared["data"]))
        loaded = {}
        for name, module in tuple(sys.modules.items()):
            if name == "torch_npu" or name.startswith("torch_npu."):
                raise RuntimeError("CPU probe unexpectedly imported the NPU backend")
            if (name == "tide" or name.startswith("tide.") or name == "pilotlab" or name.startswith("pilotlab.")) and getattr(module, "__file__", None):
                path = str(Path(module.__file__).resolve())
                if path not in identity["source_files"] or sha256(path) != identity["source_files"][path]:
                    raise ValueError("executed module is outside the frozen side identity: " + name)
                loaded[name] = reference(path)
        result.update(status="completed", executed_modules=loaded, vendor_imported=False)
    except BaseException as error:
        result.update(status="failed", reason=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        for hook in hooks:
            hook.remove()
        result["finished_unix"] = time.time()
        atomic_json(output / "result.json", result)
    return 0 if result["status"] == "completed" else 1


def compare_sides(plan, results):
    observations, reasons = {}, []
    for model_id in plan["model_ids"]:
        old, new = (results.get(side + ":" + model_id, {}) for side in ("old", "new"))
        row = {}
        expected = plan["recipes"][model_id]["target_tokens"]
        for side, value in (("old", old), ("new", new)):
            if (value.get("status") != "completed" or value.get("source_identity") != plan[side + "_source_identity"]
                    or value.get("model_id") != model_id or value.get("side") != side or value.get("device") != "cpu"
                    or value.get("training", {}).get("steps") != STEPS
                    or value.get("training", {}).get("segment_tokens") != expected
                    or len(value.get("observations", [])) != STEPS or len(value.get("inputs", [])) != STEPS):
                reasons.append(f"{model_id}:{side}: incomplete bound CPU evidence")
        identities = [tuple(value.get("process", {}).get(key) for key in ("pid", "start_ticks")) for value in (old, new)]
        if any(any(type(item) is not int or item <= 0 for item in identity) for identity in identities) or identities[0] == identities[1]:
            reasons.append(model_id + ": independent process identities missing or reused")
        for field in ("contract", "initial_parameter_sha256"):
            if old.get(field) != new.get(field) or old.get(field) is None:
                reasons.append(model_id + ": unequal " + field)
        if ([item.get("sha256") for item in old.get("inputs", [])] !=
                [item.get("sha256") for item in new.get("inputs", [])]):
            reasons.append(model_id + ": diagnostic input mismatch")
        for key in PATHS:
            left, right = old.get("digests", {}).get(key), new.get("digests", {}).get(key)
            row[key] = {"old_sha256": left, "new_sha256": right}
            if (not isinstance(left, str) or len(left) != 64 or any(c not in "0123456789abcdef" for c in left)
                    or left != right):
                reasons.append(f"{model_id}:{key}: not exact")
        observations[model_id] = row
    return observations, reasons


def run(old_prepared, new_prepared, selected, output, *, timeout_seconds=1800):
    from pilotlab.io import atomic_json
    from pilotlab.preflight3 import claim_token_budget, finish_token_accounting
    from pilotlab.prepare3 import verify_data
    from pilotlab.transfer3 import TEST_SCHEMA, _source_comparison
    from pilotlab.scheduler import child_subreaper, proc_identity

    plan = build_plan(old_prepared, new_prepared, selected, timeout_seconds=timeout_seconds)
    if reference(__file__) != plan["probe"] or plan["new_source_identity"]["source_files"].get(str(Path(__file__).resolve())) != plan["probe"]["sha256"]:
        raise ValueError("execute the probe from its declared new frozen source")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("inspect the existing probe instead of repeating its finite reservation")
    prepared = {side: checked(plan[side + "_prepared"]) for side in ("old", "new")}
    for data in {item["data"] for item in prepared.values()}:
        verify_data(Path(data))
    output.mkdir(parents=True, exist_ok=False)
    plan_path = output / "plan.json"
    atomic_json(plan_path, plan)
    claim_token_budget(Path(plan["budget_ledger"]), str(output), plan["reserved_tokens"], sha256(plan_path))
    result = {"schema": TEST_SCHEMA, "status": "running", "exact": False,
        "old_source_identity": plan["old_source_identity"], "new_source_identity": plan["new_source_identity"],
        "model_ids": plan["model_ids"], "tested_paths": list(PATHS), "steps_per_model": STEPS,
        "plan": reference(plan_path), "controller": process_identity(), "controller_cgroup": Path("/proc/self/cgroup").read_text().strip(),
        "reserved_tokens": plan["reserved_tokens"],
        "reservation_formula": plan["reservation_formula"], "observations": {}, "children": {},
        "started_unix": time.time(), "deadline_unix": plan["deadline_unix"], "device_seconds": 0,
        "scope": "exact CPU witness for these saved batches, seeds and recipes; no NPU trajectory or learning equivalence claimed"}
    results, process, key = {}, None, None
    previous_subreaper = child_subreaper(True)

    def interrupted(signum, frame):
        raise InterruptedError("quad CPU probe stop requested: " + str(signum))

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    for sig in previous:
        signal.signal(sig, interrupted)
    try:
        for index, model_id in enumerate(plan["model_ids"]):
            for side in ("old", "new"):
                check_deadline(plan["deadline_unix"])
                verify_references(plan)
                child_output = output / "children" / f"model-{index:02d}-{side}"
                child_output.parent.mkdir(parents=True, exist_ok=True)
                command = [prepared[side]["python"], "-u", plan["probe"]["path"], "--child-plan", str(plan_path),
                           "--side", side, "--model-id", model_id, "--output", str(child_output), "--parent-pid", str(os.getpid())]
                log_path = child_output.with_suffix(".log")
                key = side + ":" + model_id
                with log_path.open("x") as log:
                    process = subprocess.Popen(command, cwd=prepared[side]["source"], env=child_environment(prepared[side], plan["recipes"][model_id]),
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    result["children"][key] = {"argv": command, "pid": process.pid, "output": str(child_output), "log": str(log_path)}
                    identity = proc_identity(process.pid, strict=True)
                    if identity is None:
                        raise RuntimeError("independent child start identity unavailable")
                    result["children"][key]["process"] = identity
                    result.setdefault("all_seen_processes", {})[f"{identity['pid']}:{identity['start_ticks']}"] = identity
                    atomic_json(output / "report.json", result)
                    child_deadline = min(time.time() + 300, plan["deadline_unix"])
                    last_recorded = 0
                    while True:
                        check_deadline(child_deadline)
                        observe_descendants(result, key)
                        if time.monotonic() - last_recorded >= 1:
                            atomic_json(output / "report.json", result)
                            last_recorded = time.monotonic()
                        try:
                            code = process.wait(timeout=.25)
                            break
                        except subprocess.TimeoutExpired:
                            pass
                observe_descendants(result, key)
                reap_orphans(result, direct_pid=process.pid)
                remaining = observe_descendants(result, key)
                result["children"][key].update(return_code=code, remaining_descendants=list(remaining.values()),
                                              terminal_observed_unix=time.time())
                if remaining:
                    raise RuntimeError("CPU child left live descendants after exit")
                process = None
                result["children"][key].update(return_code=code, result=reference(child_output / "result.json"))
                value = checked(result["children"][key]["result"])
                results[key] = value
                if (any(value.get("process", {}).get(field) != result["children"][key]["process"].get(field)
                        for field in ("pid", "start_ticks"))
                        or value.get("plan") != result["plan"] or value.get("probe") != plan["probe"]):
                    raise ValueError("independent child result does not bind its process/plan/probe")
                if code != 0 or value.get("status") != "completed":
                    raise RuntimeError(f"independent CPU probe failed: {key}: {value.get('reason')}")
                for entry in value["inputs"]:
                    checked(entry)
                atomic_json(output / "report.json", result)
        verify_references(plan)
        _source_comparison(prepared["old"], prepared["new"])
        observations, reasons = compare_sides(plan, results)
        result.update(observations=observations, blocking_reasons=reasons,
                      exact=not reasons, status="passed" if not reasons else "failed")
    except BaseException as error:
        result.update(status="failed", exact=False, reason=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        try:
            if not cleanup_descendants(result, process, key):
                result.update(status="failed", exact=False, reason="probe_descendant_cleanup_incomplete")
        except Exception as error:
            result.update(status="failed", exact=False, cleanup_observation_error=f"{type(error).__name__}: {error}")
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        child_subreaper(previous_subreaper)
        finish_token_accounting(Path(plan["budget_ledger"]), str(output), results, result["status"])
        result.update(finished_unix=time.time(), actual_completed_loss_tokens=sum(v["training"]["segment_tokens"] for v in results.values()))
        atomic_json(output / "report.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-prepared", type=Path)
    parser.add_argument("--new-prepared", type=Path)
    parser.add_argument("--selected", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--child-plan", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--side", choices=("old", "new"), help=argparse.SUPPRESS)
    parser.add_argument("--model-id", help=argparse.SUPPRESS)
    parser.add_argument("--parent-pid", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child_plan:
        return run_child(args.child_plan, args.side, args.model_id, args.output, parent_pid=args.parent_pid)
    if not all((args.old_prepared, args.new_prepared, args.selected)):
        parser.error("--old-prepared, --new-prepared and --selected are required")
    result = run(args.old_prepared, args.new_prepared, args.selected, args.output, timeout_seconds=args.timeout_seconds)
    print(json.dumps({"status": result["status"], "exact": result["exact"], "report": str(args.output.resolve() / "report.json")}, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

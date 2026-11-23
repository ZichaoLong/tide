"""Finite numerical risk screen; diagnostic exposure never becomes stage A evidence."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from .io import atomic_json, sha256
from .scheduler3 import Blocked, read


def reference(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path)}


def checked(ref):
    if sha256(ref["path"]) != ref["sha256"]:
        raise ValueError("bound input changed: " + ref["path"])
    return read(ref["path"])


def deadline_check(deadline):
    if time.time() >= deadline - 30:
        raise TimeoutError("risk screen closing reserve reached")


def budget_tokens(task):
    """Real updates, one replay, FP64 updates, two evaluations and causal probe."""
    quantum = task["batch_size"] * task["sequence_length"]
    refs = len(task.get("fp64_steps", []))
    train = (task["screen_steps"] + 1 + refs) * quantum
    return train, train + 2 * task["eval_tokens"] + 32


def snapshot(model, optimizer, state):
    from .worker import cpu_snapshot
    return {"parameters": cpu_snapshot(model.state_dict()),
            "optimizer": cpu_snapshot(optimizer.state_dict()), "private_state": cpu_snapshot(state),
            "gradients": {name: None if p.grad is None else p.grad.detach().cpu().clone()
                          for name, p in model.named_parameters()}}


def errors(actual, expected):
    import torch
    from .worker import compare_tree, tree_map, leaves
    a, b = [tree_map(value, lambda t: t.double() if t.is_floating_point() else t)
            for value in (actual, expected)]
    gate = compare_tree(a, b, atol=1e-5, rtol=1e-4)
    fields = {}
    for key in actual:
        aa, bb = dict(leaves(a[key])), dict(leaves(b[key]))
        bad, delta2, norm2, largest = 0, 0., 0., 0.
        for name in aa.keys() & bb.keys():
            x, y = aa[name], bb[name]
            if not isinstance(x, torch.Tensor) or not x.is_floating_point() or not x.numel():
                continue
            diff = x-y
            bad += int((diff.abs() > 1e-5 + 1e-4*y.abs()).sum())
            delta2 += float(diff.square().sum())
            norm2 += float(y.square().sum())
            largest = max(largest, float(diff.abs().max()))
        fields[key] = {"failed_elements": bad, "max_abs_error": largest,
                       "relative_l2_error": (delta2/norm2)**.5 if norm2 else None}
    return {"original_threshold_gate": gate, "fields": fields,
            "scope": "same implementation promoted to CPU FP64; not an independent model oracle"}


def fp64_update(owner, inputs, targets, recipe, vocab_size):
    import torch
    from .models3 import build_model
    from .worker3 import optimizer_for, learning_rate
    from .worker import tree_map, restore_rng, rng_state, digest_tree
    saved_rng = rng_state(torch.device("cpu"))
    try:
        model = build_model(recipe["model"], vocab_size, recipe["init_recipe"]).double()
        model.load_state_dict(owner["model"])
        if recipe.get("forced_reception_tiled"):
            model.core.receive.backend = "tiled"
        optimizer = optimizer_for(model, recipe)
        optimizer.load_state_dict(owner["optimizer"])
        state = tree_map(owner["private_state"], lambda t: t.double() if t.is_floating_point() else t)
        restore_rng(owner["rng"], torch.device("cpu"))
        reference_rng = rng_state(torch.device("cpu"))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(recipe, owner["tokens"] + targets.numel())
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        output = model(inputs, state)
        loss = torch.nn.functional.cross_entropy(output.logits.flatten(0, 1), targets.flatten())
        aux = getattr(output, "auxiliary_loss", None)
        objective = loss if aux is None else loss + recipe.get("auxiliary_loss_weight", 1.) * aux
        objective.backward()
        raw = {name: None if p.grad is None else p.grad.detach().clone() for name, p in model.named_parameters()}
        torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["grad_clip"], error_if_nonfinite=True, foreach=False)
        optimizer.step()
        if owner["rng"].get("npu") is not None and digest_tree(reference_rng) != digest_tree(rng_state(torch.device("cpu"))):
            raise ValueError("cross-device FP64 diagnostic requires a non-stochastic update")
        result = snapshot(model, optimizer, output.state)
        result["gradients"] = raw
        result["parameter_delta"] = {name: p.detach()-before[name] for name, p in model.named_parameters()}
        return result
    finally:
        restore_rng(saved_rng, torch.device("cpu"))


def restore_child(args, task, device, store):
    from . import worker3 as w
    from .models3 import build_model
    from .worker import atomic_torch_save, digest_tree
    owner = w._read_checkpoint(args.restore_prefix, args.restore_sha256)
    w.validate_checkpoint(owner, task, device)
    model = build_model(task["model"], store.vocab_size, task["init_recipe"]).to(device)
    optimizer = w.optimizer_for(model, task)
    state, stream = w.restore_owners(owner, model, optimizer, store, task, device)
    if task.get("forced_reception_tiled"):
        model.core.receive.backend = "tiled"
    inputs, targets, reset = stream.next()
    update = w.train_update(model, optimizer, inputs, targets, None if reset else state,
                            device, task, owner["tokens"])
    final = snapshot(model, optimizer, update["state"])
    final["stream_sha256"] = digest_tree(stream.state_dict())
    final["rng_sha256"] = digest_tree(w.rng_state(device))
    atomic_torch_save(args.output / "final.pt", final)
    atomic_json(args.output / "restore.json", {"load": "exact", "input_sha256": digest_tree((inputs, targets)),
        "completed_loss_tokens": targets.numel(), "final": reference(args.output / "final.pt")})
    return 0


def worker(args):
    task = read(args.task)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    from . import worker3 as w
    from .data3 import DocumentStore3
    from .models3 import build_model
    from .worker import atomic_torch_save, cpu_snapshot, digest_tree, compare_tree, causality_check, synchronize
    import torch
    result = {"schema": "pilot-r3-risk-screen-result-v1", "id": task["id"], "model_id": task["model_id"],
        "phase": "preflight", "status": "running", "task_sha256": sha256(args.task),
        "source_identity": task["source_identity"], "started_unix": time.time(), "device": args.device,
        "physical_visibility": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "training": {"tokens": 0, "segment_tokens": 0}, "evaluation": {"tokens": 0},
        "scope": "bounded risk screen; not formal A/B or source-equivalence evidence", "observations": []}
    atomic_json(output / "task.json", task)
    device, paths, originals, hooks = None, Counter(), {}, []
    try:
        w.validate_source_identity(task, args.data)
        recipe = copy.deepcopy(task)
        # Diagnostic exposure is separate from cumulative trajectory exposure.
        recipe["target_tokens"] = task["schedule_tokens"]
        w.validate_task(recipe)
        device = w.initialize(args.device, task)
        store = DocumentStore3(args.data)
        if args.restore_prefix:
            return restore_child(args, recipe, device, store)
        if task.get("qualification_tests"):
            for ref in task["qualification_tests"]:
                if sha256(ref["path"]) != ref["sha256"]:
                    raise ValueError("frozen qualification test changed")
            command = [sys.executable, "-m", "pytest", "-q", "-s", "-p", "no:cacheprovider",
                       *[ref["path"] for ref in task["qualification_tests"]]]
            qualification = subprocess.run(command, capture_output=True, text=True,
                env={**os.environ, "TIDE_STABLE_DEVICE": str(device)}, timeout=300)
            (output / "qualification.log").write_text(qualification.stdout + qualification.stderr)
            result["qualification"] = {"argv": command, "returncode": qualification.returncode,
                                       "log": reference(output / "qualification.log")}
            if qualification.returncode:
                raise ArithmeticError("independent singleton/support qualification failed")
        model = build_model(task["model"], store.vocab_size, task["init_recipe"]).to(device)
        optimizer = w.optimizer_for(model, task)
        tokens = steps = 0
        state, stream = None, w.StreamBatch(store, task)
        if task.get("origin_task"):
            origin = checked(task["origin_task"])
            owner = w._read_checkpoint(Path(task["origin_checkpoint"]["path"]), task["origin_checkpoint"]["sha256"])
            w.validate_checkpoint(owner, origin, device)
            if w.contract(origin, str(device)) != w.contract(recipe, str(device)):
                raise ValueError("screen must preserve the trained trajectory recipe")
            state, stream = w.restore_owners(owner, model, optimizer, store, recipe, device)
            tokens, steps = owner["tokens"], owner["steps"]
            result["source_transition"] = {"origin_task": task["origin_task"],
                "origin_checkpoint": task["origin_checkpoint"], "load": "exact",
                "origin_source_identity": owner["source_identity"], "from_tokens": tokens, "from_steps": steps,
                "scope": "diagnostic continuation on repaired source; original trajectory is retained"}
        if task.get("forced_reception_tiled"):
            model.core.receive.backend = "tiled"
        result["initialization"] = {"recipe": getattr(model, "initialization_recipe", task["init_recipe"]),
            "seed": task["seed"], "parameters_sha256": digest_tree(model.state_dict()),
            "trained_checkpoint": bool(task.get("origin_task"))}
        # Count actual selected implementations without altering their outputs.
        import tide.attention as attention
        for name in ("_native_attention", "_math_attention"):
            if hasattr(attention, name):
                original = getattr(attention, name)
                originals[name] = original
                def observed(*values, _name=name, _call=original, **kw):
                    paths[_name] += 1
                    return _call(*values, **kw)
                setattr(attention, name, observed)
        for name in ("_TiledAttention", "_MassScaledAttention", "_SingleKeyAttention"):
            cls = getattr(attention, name)
            original = cls.forward
            originals[name] = original
            def forward(ctx, *values, _name=name, _call=original):
                paths[_name] += 1
                return _call(ctx, *values)
            cls.forward = staticmethod(forward)
        saved_rng = w.rng_state(device)
        result["causality"] = causality_check(model, store, task, device)
        w.restore_rng(saved_rng, device)
        if not result["causality"]["within_tolerance"]:
            raise ArithmeticError("causality gate failed")
        result["validation_initial"] = w.evaluate(model, store, task, device, args.deadline_unix - 30)
        result["evaluation"]["tokens"] += result["validation_initial"]["tokens"]
        start_tokens, start_steps = tokens, steps
        result["fp64"] = []
        quantum = task["batch_size"] * task["sequence_length"]
        raw = {}
        def capture(name):
            def observe(p):
                if capture.enabled:
                    raw[name] = p.grad.detach().cpu().clone()
            return observe
        capture.enabled = False
        hooks = [p.register_post_accumulate_grad_hook(capture(name)) for name, p in model.named_parameters() if p.requires_grad]
        train_start = time.monotonic()
        for index in range(task["screen_steps"]):
            deadline_check(args.deadline_unix)
            if index == task["screen_steps"] - 1:
                prefix = w.checkpoint_payload(model, optimizer, state, stream, recipe, device, tokens, steps, result["initialization"])
                result["checkpoint"] = w.save_checkpoint(output, prefix)
            inputs, targets, reset = stream.next()
            if reset:
                state = None
            capture.enabled = index in task.get("fp64_steps", [])
            if capture.enabled:
                owner = {"model": cpu_snapshot(model.state_dict()), "optimizer": cpu_snapshot(optimizer.state_dict()),
                         "private_state": cpu_snapshot(state), "rng": w.rng_state(device), "tokens": tokens}
                raw.clear()
            sampled = index in {0, task["screen_steps"]-1} or index % 32 == 0
            update = w.train_update(model, optimizer, inputs, targets, state, device, recipe, tokens, sample=sampled)
            state, tokens, steps = update["state"], tokens + quantum, steps + 1
            result["training"]["segment_tokens"] += quantum
            if capture.enabled:
                if device.type != "cpu" and digest_tree(owner["rng"]) != digest_tree(w.rng_state(device)):
                    raise ValueError("cross-device FP64 diagnostic requires an unchanged CPU/device RNG")
                actual = snapshot(model, optimizer, state)
                actual["gradients"] = {name: raw.get(name) for name, _ in model.named_parameters()}
                actual["parameter_delta"] = {name: p.detach().cpu()-owner["model"][name] for name, p in model.named_parameters()}
                expected = fp64_update(owner, inputs, targets, recipe, store.vocab_size)
                comparison = errors(actual, expected)
                comparison["execution_device"] = str(device)
                comparison["rng_policy"] = "same saved CPU RNG; non-CPU updates must consume no CPU/device randomness"
                result["training"]["segment_tokens"] += quantum
                atomic_torch_save(output / f"precision-{index}.pt", {"actual": actual, "fp64": expected})
                result["fp64"].append({"step": steps, **comparison, "witness": reference(output / f"precision-{index}.pt")})
                gate = comparison["original_threshold_gate"]
                if not gate["finite"] or not gate["structure_equal"] or not gate["integer_equal"]:
                    raise ArithmeticError("FP64 diagnostic nonfinite or structural disagreement")
            row = {"step": steps, "tokens": tokens, "loss": update["loss"], "grad_norm": update["grad_norm"],
                   "lr": update["lr"], "reset": reset, "input_sha256": digest_tree((inputs, targets))}
            if sampled:
                row.update(health=update["health"], work=update["work"])
            result["observations"].append(row)
            atomic_json(output / "progress.json", result)
        synchronize(device)
        result["trajectory"] = {"from_tokens": start_tokens, "to_tokens": tokens,
            "from_steps": start_steps, "to_steps": steps, "seconds_including_precision_checks": time.monotonic()-train_start}
        final = snapshot(model, optimizer, state)
        final["stream_sha256"] = digest_tree(stream.state_dict())
        final["rng_sha256"] = digest_tree(w.rng_state(device))
        atomic_torch_save(output / "final.pt", final)
        child_output = output / "restored"
        command = [sys.executable, "-u", "-m", "pilotlab.risk_screen3", "--task", str(args.task),
            "--output", str(child_output), "--data", str(args.data), "--device", args.device,
            "--deadline-unix", str(args.deadline_unix), "--restore-prefix", result["checkpoint"]["path"],
            "--restore-sha256", result["checkpoint"]["sha256"]]
        child = subprocess.run(command, capture_output=True, text=True, timeout=min(300, max(1, args.deadline_unix-time.time()-30)))
        result["fresh_process"] = {"argv": command, "returncode": child.returncode, "stderr": child.stderr[-6000:]}
        if child.returncode:
            raise RuntimeError("fresh-process continuation failed: " + child.stderr[-2000:])
        restored = read(child_output / "restore.json")
        final_restored = torch.load(restored["final"]["path"], map_location="cpu", weights_only=False)
        recovery = compare_tree(final, final_restored, atol=1e-5, rtol=1e-4)
        result["recovery"] = {"load": restored["load"], "trajectory": recovery, "result": reference(child_output / "restore.json")}
        if (restored["input_sha256"] != result["observations"][-1]["input_sha256"] or not recovery["finite"]
                or not recovery["structure_equal"] or not recovery["integer_equal"]):
            raise ArithmeticError("fresh-process state/stream/discrete continuation mismatch")
        result["training"]["segment_tokens"] += restored["completed_loss_tokens"]
        result["validation_final"] = w.evaluate(model, store, task, device, args.deadline_unix-30)
        result["evaluation"]["tokens"] += result["validation_final"]["tokens"]
        if any(result[key]["status"] != "completed" for key in ("validation_initial", "validation_final")):
            raise RuntimeError("screen evaluation did not complete")
        result["training"]["tokens"] = result["training"]["segment_tokens"]
        if result["training"]["tokens"] != task["target_tokens"]:
            raise RuntimeError("processed diagnostic token accounting mismatch")
        w.validate_source_identity(task, args.data)
        result.update(status="completed", health={"eligible": True, "blocking_reasons": []})
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
                      health={"eligible": False, "blocking_reasons": [str(error)]})
    finally:
        for hook in hooks:
            hook.remove()
        if originals:
            for name, original in originals.items():
                if name in ("_TiledAttention", "_MassScaledAttention", "_SingleKeyAttention"):
                    getattr(attention, name).forward = staticmethod(original)
                else:
                    setattr(attention, name, original)
        if device is not None:
            result["device_memory"] = w.device_memory(device)
        result.update(attention_paths=dict(paths), finished_unix=time.time())
        if not args.restore_prefix or result["status"] == "failed":
            atomic_json(output / "result.json", result)
    return 0 if result["status"] == "completed" else 1


def controller(args):
    from .preflight3 import claim_token_budget, finish_token_accounting
    from .repair_probe3 import RepairSupervisor
    from .prepare_campaign3 import budget_remaining
    plan = read(args.plan)
    for filename, expected in plan["source_identity"]["source_files"].items():
        if sha256(filename) != expected:
            raise ValueError("frozen risk screen source changed")
    budget_remaining(Path(plan["config"]["budget_ledger"]), time.time())
    if any(budget_tokens(task) != (task["target_tokens"], task["budget_tokens"]) for task in plan["tasks"]):
        raise ValueError("diagnostic processing ceiling differs from declared operations")
    claim_token_budget(Path(plan["config"]["budget_ledger"]), str(args.output.resolve()),
                       plan["reserved_tokens"], sha256(args.plan))
    supervisor = RepairSupervisor(plan["config"], args.output)
    signal.signal(signal.SIGTERM, supervisor.request_stop)
    signal.signal(signal.SIGINT, supervisor.request_stop)
    report = {"schema": "pilot-r3-risk-screen-v1", "status": "running", "stage_b_gate": "closed",
              "plan": reference(args.plan), "started_unix": time.time()}
    results = {}
    try:
        if supervisor.state["tasks"]:
            raise Blocked("inspect existing screen; do not repeat finite exposure")
        results = supervisor.run("numerical-risk-screen", plan["tasks"])
        if any(result.get("health", {}).get("eligible") is not True for result in results.values()):
            raise Blocked("screen mechanism failure")
        report.update(status="passed_mechanism_screen", finite_precision_deviations=[key for key, result in results.items()
            if any(not row["original_threshold_gate"]["within_tolerance"] for row in result["fp64"])
            or not result["recovery"]["trajectory"]["within_tolerance"]])
        supervisor.finish("completed")
    except BaseException as error:
        supervisor.stop_reason = supervisor.stop_reason or f"{type(error).__name__}: {error}"
        supervisor.stop_all()
        report.update(status="blocked", reason=supervisor.stop_reason)
    finally:
        results = {key: read(row["result_path"], {}) for key, row in supervisor.state["tasks"].items() if row.get("result_path")}
        report.update(result_files={key: reference(row["result_path"]) for key, row in supervisor.state["tasks"].items()
                                    if row.get("result_path") and Path(row["result_path"]).is_file()},
                      counts=supervisor.state["counts"], finished_unix=time.time(),
                      remaining_descendants=list(supervisor.owned_members().values()), device_seconds=supervisor.device_seconds())
        finish_token_accounting(Path(plan["config"]["budget_ledger"]), str(args.output.resolve()), results, report["status"])
        atomic_json(args.output / "risk-report.json", report)
        supervisor.close()
    return 0 if report["status"] == "passed_mechanism_screen" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", type=Path)
    action.add_argument("--task", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--deadline-unix", type=float)
    parser.add_argument("--restore-prefix", type=Path)
    parser.add_argument("--restore-sha256")
    args = parser.parse_args(argv)
    return controller(args) if args.plan else worker(args)


if __name__ == "__main__":
    raise SystemExit(main())

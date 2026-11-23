"""One finite, queued Shannon repair qualification; never formal learning.

The controller uses the ordinary Supervisor and original preparation ledger.
One physical lease covers scalar checks, three saved-input updates and one
fresh-process restored update. The exact reservation is four saved batches;
uncertain failures retain that full reservation. Finite trajectory differences
are reported separately from exact loading and numerical mechanism admission.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from .io import atomic_json, sha256
from .scheduler3 import Blocked, Supervisor, read


def cases():
    rows = []
    for budget in (1, 2, 4, 8):
        for gap in (16.647329807281494, 40.):
            rows.append({"id": f"n16-k{budget}-gap{gap:g}",
                "values": [[6.] * budget + [6. - gap] * (16 - budget), [1.] * 16,
                           [1.] * 16, [0.] * 16, [1.] * 16, float(budget)],
                "cotangent": [1.] + [0.] * 15})
        upper = [.15, .25, .45, .65] + [1.] * 12
        rows.append({"id": f"n16-k{budget}-mixed-bounds",
            "values": [[8., 5., 3., 2., .8, .2, -.4, -1., -2., -3., -4., -5., -6., -7., -8., -9.],
                       [x / 2 for x in upper], upper, [.02, 0., .03, 0.] + [0.] * 12, upper, float(budget)],
            "cotangent": [1., -.5, .25, -.75] * 4})
    rows.append({"id": "actual-failure-row", "values": [[6.040277004241943, -10.60705280303955],
        [1., 1.], [1., 1.], [0., 0.], [1., 1.], 1.], "cotangent": [1., 0.]})
    rows.append({"id": "near-exhausted-floors", "values": [[6., 6., 6.], [1.] * 3,
        [1.] * 3, [.1, .9, 0.], [1.] * 3, 1.], "cotangent": [0., 0., 1.]})
    return rows


def independent_reference(values, cotangent):
    """CPU Decimal KKT dual solve plus analytic VJP; no production solve."""
    with localcontext() as context:
        context.prec = 80
        scores, weights, mu, lower, upper = [[Decimal.from_float(float(x)) for x in row]
                                            for row in values[:5]]
        budget = Decimal.from_float(float(values[5]))
        high = min(budget, sum(upper))
        low = max(sum(lower), min(Decimal(1), sum(weights), high))

        def response(multiplier):
            return [max(l, min(u, (score - 1 - multiplier).exp()))
                    for score, l, u in zip(scores, lower, upper)]

        initial = response(Decimal(0))
        target = max(low, min(high, sum(initial)))
        result = initial
        if target != sum(initial):
            left, right = Decimal(-4096), Decimal(4096)
            for _ in range(256):
                middle = (left + right) / 2
                if sum(response(middle)) > target:
                    left = middle
                else:
                    right = middle
            result = response((left + right) / 2)
        slopes = [v if l < v < u else Decimal(0) for v, l, u in zip(result, lower, upper)]
        roots = [Decimal.from_float(float(v)) for v in cotangent]
        average = (sum(v * g for v, g in zip(slopes, roots)) / sum(slopes)
                   if target != sum(initial) and sum(slopes) else Decimal(0))
        gradients = [v * (g - average) for v, g in zip(slopes, roots)]
        return {"values": [float(v) for v in result], "score_vjp": [float(v) for v in gradients],
                "method": "80-digit CPU Decimal constrained entropy dual and analytic active-face VJP"}


def check_deadline(deadline, reserve=0):
    if time.time() + reserve >= deadline:
        raise TimeoutError("repair diagnostic wall-clock deadline")


def processing_bound(failure):
    """Exactly three observed batches and one fresh-process repeated batch."""
    targets = failure["targets"]
    if targets.ndim != 2 or targets.numel() <= 0:
        raise ValueError("a nonempty saved target batch is required")
    total = 4 * targets.numel()
    if total > 2**18:
        raise ValueError("repair probe exceeds its finite processed-token ceiling")
    return total


def numerical_checks(device, deadline):
    import torch
    from tide.allocation import allocate
    records = []
    for case in cases():
        check_deadline(deadline, 180)
        args = [torch.tensor(v, dtype=torch.float32, device=device, requires_grad=True) for v in case["values"]]
        cotangent = torch.tensor(case["cotangent"], dtype=torch.float32, device=device)
        expected = independent_reference([v.detach().cpu().tolist() for v in args], case["cotangent"])
        actual = allocate(*args, kind="shannon")
        gradient, = torch.autograd.grad(actual, (args[0],), cotangent)
        got, dg = actual.detach().cpu().double(), gradient.detach().cpu().double()
        want, dw = (torch.tensor(expected[key], dtype=torch.float64) for key in ("values", "score_vjp"))
        support = bool((got[want > 0] > 0).all())
        finite = bool(got.isfinite().all() and dg.isfinite().all())
        gradient_support = bool((dg[dw.float() != 0] != 0).all())
        bounds = bool(((actual >= args[3]) & (actual <= args[4])).all())
        value_ok = bool(torch.allclose(got, want, rtol=3e-5, atol=2.**-149))
        vjp_ok = bool(torch.allclose(dg, dw, rtol=3e-5, atol=2.**-149))
        records.append({"id": case["id"], "positive_support": support, "gradient_support": gradient_support,
            "finite": finite, "bounds": bounds, "value_within_threshold": value_ok, "vjp_within_threshold": vjp_ok,
            "rtol": 3e-5, "atol": 2.**-149, "values": got.tolist(), "score_vjp": dg.tolist(),
            "reference": expected, "passed": support and finite and gradient_support and bounds and value_ok and vjp_ok})
    for budget in (1, 2, 4, 8):
        check_deadline(deadline, 180)
        scores = torch.tensor([6.] * budget + [-120.] * (16 - budget), device=device)
        try:
            allocate(scores, torch.ones_like(scores), torch.ones_like(scores), 0., torch.ones_like(scores),
                     float(budget), "shannon")
            error = None
        except ArithmeticError as caught:
            error = str(caught)
        records.append({"id": f"n16-k{budget}-unrepresentable-positive-guard", "error": error,
                        "passed": error == "Shannon allocation underflow would erase a positive sender"})
    return {"schema": "pilot-r3-shannon-npu-local-v1", "cases": records,
            "status": "passed" if all(v["passed"] for v in records) else "failed",
            "device": str(device), "precision": "fp32", "processed_loss_tokens": 0}


def verified_input(task):
    path = Path(task["failure_result"])
    if sha256(path) != task["failure_result_sha256"]:
        raise ValueError("frozen diagnostic failure-result identity changed")
    record = read(path)
    capture = record["failure_capture"]
    if sha256(capture["path"]) != capture["sha256"]:
        raise ValueError("frozen failure payload changed")
    for filename, expected in task["source_identity"]["source_files"].items():
        if sha256(filename) != expected:
            raise ValueError(f"frozen repair source changed: {filename}")
    return capture


def updates(task, output, device, deadline, *, prefix=None):
    import torch
    from .models3 import build_model
    from .worker3 import optimizer_for, train_update
    from .worker import (atomic_torch_save, compare_tree, cpu_snapshot, digest_tree, restore_rng,
                         rng_state, synchronize, tree_map)
    capture = verified_input(task)
    failure = torch.load(capture["path"], map_location="cpu", weights_only=False)
    if task["target_tokens"] != processing_bound(failure) or task["budget_tokens"] != task["target_tokens"]:
        raise ValueError("repair request must reserve exactly four saved batches")
    recipe = failure["task"]
    owner = torch.load(prefix, map_location="cpu", weights_only=False) if prefix else failure
    if prefix and owner.get("schema") != "pilot-r3-repair-diagnostic-prefix-v1":
        raise ValueError("only the matching repair diagnostic prefix may be restored")
    if prefix and owner.get("source_identity") != task["source_identity"]:
        raise ValueError("diagnostic prefix source identity mismatch")
    model = build_model(recipe["model"], failure["model"]["embedding.weight"].shape[0], recipe["init_recipe"]).to(device)
    model.load_state_dict(owner["model"])
    optimizer = optimizer_for(model, recipe)
    optimizer.load_state_dict(owner["optimizer"])
    state = tree_map(owner["private_state"], lambda value: value.to(device))
    if prefix:
        restore_rng(owner["rng"], device)
    expected_load = {name: owner[name] for name in ("model", "optimizer", "private_state")}
    actual_load = {"model": cpu_snapshot(model.state_dict()), "optimizer": cpu_snapshot(optimizer.state_dict()),
                   "private_state": cpu_snapshot(state)}
    loading = compare_tree(expected_load, actual_load, atol=0, rtol=0)
    if prefix:
        loading["rng_exact"] = digest_tree(owner["rng"]) == digest_tree(rng_state(device))
    else:
        loading["rng_exact"] = None
    if not loading["exact"] or prefix and loading["rng_exact"] is not True:
        raise ArithmeticError("diagnostic checkpoint state or RNG did not load exactly")
    tokens = owner.get("completed_tokens", failure["completed_tokens"])
    completed, rows, prefix_record = 0, [], None
    total_steps = 1 if prefix else 3
    for index in range(total_steps):
        check_deadline(deadline, 15)
        if not prefix and index == total_steps - 1:
            saved = output / "checkpoints/diagnostic-prefix.pt"
            saved.parent.mkdir(parents=True, exist_ok=True)
            payload = {"schema": "pilot-r3-repair-diagnostic-prefix-v1", "source_identity": task["source_identity"],
                "model": cpu_snapshot(model.state_dict()), "optimizer": cpu_snapshot(optimizer.state_dict()),
                "private_state": cpu_snapshot(state), "rng": rng_state(device), "completed_tokens": tokens,
                "scope": "fixed-input diagnostic only; no campaign stream or resumable learning contract"}
            atomic_torch_save(saved, payload)
            prefix_record = {"path": str(saved), "sha256": sha256(saved), "size_bytes": saved.stat().st_size,
                             "kind": "diagnostic_prefix_not_campaign_checkpoint", "tokens": tokens, "steps": 2}
            atomic_json(output / "checkpoint.json", {"schema": "pilot-r3-checkpoint-generations-v1",
                                                       "current": prefix_record, "previous": None})
        started = time.monotonic()
        update = train_update(model, optimizer, failure["inputs"], failure["targets"], state,
                              device, recipe, tokens, sample=False)
        synchronize(device)
        state, tokens = update["state"], tokens + failure["targets"].numel()
        completed += failure["targets"].numel()
        rows.append({"update": index + 1, "loss": update["loss"], "gradient_norm": update["grad_norm"],
                     "seconds": time.monotonic() - started})
        atomic_json(output / "replay-progress.json", {"completed_loss_tokens": completed, "updates": rows,
                                                       "initial_state_load": loading, "checkpoint": prefix_record})
    final = {"model": cpu_snapshot(model.state_dict()), "optimizer": cpu_snapshot(optimizer.state_dict()),
             "private_state": cpu_snapshot(state), "gradients": {name: None if p.grad is None else p.grad.detach().cpu().clone()
                                                                 for name, p in model.named_parameters()}}
    path = output / "final-state.pt"
    atomic_torch_save(path, final)
    return {"status": "completed", "completed_loss_tokens": completed, "updates": rows,
            "initial_state_load": loading, "checkpoint": prefix_record,
            "final_state": {"path": str(path), "sha256": sha256(path), "state_sha256": digest_tree(final)}}


def worker(args):
    task = read(args.task)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = {"schema": "pilot-r3-repair-probe-result-v1", "id": task["id"], "model_id": task["model_id"],
        "phase": "preflight", "purpose": "numerical_repair_diagnostic_not_learning", "status": "running",
        "source_identity": task["source_identity"], "task_sha256": sha256(args.task), "started_unix": time.time(),
        "physical_visibility": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"), "device": args.device,
        "training": {"tokens": 0, "segment_tokens": 0}, "evaluation": {"tokens": 0}}
    device = None
    try:
        verified_input(task)
        check_deadline(args.deadline_unix, 180)
        from .worker3 import initialize, device_memory
        from .worker import compare_tree
        import torch
        device = initialize(args.device, task)
        if args.restore_prefix:
            restored = updates(task, output, device, args.deadline_unix, prefix=args.restore_prefix)
            atomic_json(output / "restore-result.json", restored)
            return 0
        numerical = numerical_checks(device, args.deadline_unix)
        numeric_path = output / "numerical.json"
        atomic_json(numeric_path, numerical)
        result["numerics"] = {"path": str(numeric_path), "sha256": sha256(numeric_path), "status": numerical["status"]}
        if numerical["status"] != "passed":
            raise ArithmeticError("independent NPU Shannon values/support/VJP qualification failed")
        replay = updates(task, output, device, args.deadline_unix)
        result["replay"] = replay
        result["checkpoint"] = replay["checkpoint"]
        restored_output = output / "restored"
        command = [sys.executable, "-u", "-m", "pilotlab.repair_probe3", "--task", str(args.task.resolve()),
            "--output", str(restored_output), "--data", str(args.data.resolve()), "--device", args.device,
            "--deadline-unix", str(args.deadline_unix), "--restore-prefix", replay["checkpoint"]["path"]]
        child = subprocess.run(command, capture_output=True, text=True, timeout=min(120, max(1, args.deadline_unix - time.time())))
        result["fresh_process"] = {"argv": command, "returncode": child.returncode,
                                   "stdout": child.stdout[-4000:], "stderr": child.stderr[-4000:]}
        if child.returncode:
            raise RuntimeError("fresh-process diagnostic update failed")
        restored = read(restored_output / "restore-result.json")
        expected = torch.load(replay["final_state"]["path"], map_location="cpu", weights_only=False)
        actual = torch.load(restored["final_state"]["path"], map_location="cpu", weights_only=False)
        trajectory = compare_tree(expected, actual, atol=1e-5, rtol=1e-4)
        trajectory["status"] = "within_threshold" if trajectory["within_tolerance"] else "outside_threshold"
        result["recovery"] = {"checkpoint_load": restored["initial_state_load"], "numeric_trajectory": trajectory,
            "restored_result_path": str(restored_output / "restore-result.json"),
            "restored_result_sha256": sha256(restored_output / "restore-result.json")}
        if not trajectory["finite"] or not trajectory["structure_equal"] or not trajectory["integer_equal"]:
            raise ArithmeticError("restored diagnostic trajectory has nonfinite or structural mismatch")
        completed = replay["completed_loss_tokens"] + restored["completed_loss_tokens"]
        if completed != task["target_tokens"]:
            raise ArithmeticError("diagnostic processed-token reservation mismatch")
        result["training"] = {"tokens": completed, "segment_tokens": completed, "steps": 4,
            "scope": "three saved-input diagnostic updates plus one restored update; not a formal learning trajectory"}
        result["health"] = {"eligible": True, "blocking_reasons": [],
                            "strict_trajectory_eligible": trajectory["within_tolerance"]}
        result["status"], result["stop_reason"] = "completed", "finite_diagnostic_completed"
    except BaseException as error:
        result.update(status="failed", stop_reason=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        result["health"] = {"eligible": False, "blocking_reasons": [str(error)]}
        progress = read(output / "replay-progress.json", {})
        restored_progress = read(output / "restored/replay-progress.json", {})
        tokens = progress.get("completed_loss_tokens", 0) + restored_progress.get("completed_loss_tokens", 0)
        result["training"] = {"tokens": tokens, "segment_tokens": tokens,
                              "scope": "completed diagnostic loss tokens; uncertain attempts retain full budget reservation"}
        if progress.get("checkpoint"):
            result["checkpoint"] = progress["checkpoint"]
    finally:
        if device is not None:
            from .worker3 import device_memory
            result["device_memory"] = device_memory(device)
        result["finished_unix"] = time.time()
        if args.restore_prefix and result["status"] == "failed":
            atomic_json(output / "restore-result.json", result)
        elif not args.restore_prefix:
            atomic_json(output / "result.json", result)
    return 0 if result["status"] == "completed" else 1


class RepairSupervisor(Supervisor):
    def resource_reason(self):
        reason = super().resource_reason()
        if reason:
            return reason
        if time.time() >= self.config["deadline_unix"]:
            return "repair_walltime_limit"
        if self.device_seconds() >= self.config["segment_max_device_seconds"]:
            return "repair_device_hour_limit"
        return None


def controller(plan_path, output):
    from .preflight3 import claim_token_budget, finish_token_accounting
    plan = read(plan_path)
    for filename, expected in plan["source_identity"]["source_files"].items():
        if sha256(filename) != expected:
            raise ValueError("repair controller source changed")
    ledger = read(plan["config"]["budget_ledger"])
    if (ledger["max_device_seconds"] != 7200 * 3600 or ledger["preparation_max_device_seconds"] != 48 * 3600
            or ledger["preparation_max_processed_tokens"] != 64 * 2**20
            or ledger["deadline_unix"] != plan["campaign_deadline_unix"]
            or ledger["deadline_unix"] != ledger["started_unix"] + 30 * 86400):
        raise ValueError("original absolute campaign budget must be retained")
    if len(plan["tasks"]) != 1 or plan["reserved_tokens"] != plan["tasks"][0]["budget_tokens"]:
        raise ValueError("repair probe must retain its single finite reservation")
    claim_token_budget(Path(plan["config"]["budget_ledger"]), str(output.resolve()), plan["reserved_tokens"], sha256(plan_path))
    supervisor = RepairSupervisor(plan["config"], output)
    signal.signal(signal.SIGTERM, supervisor.request_stop)
    signal.signal(signal.SIGINT, supervisor.request_stop)
    status, results, report = "blocked", {}, {"schema": "pilot-r3-repair-preflight-v1", "status": "running",
        "plan_path": str(plan_path.resolve()), "plan_sha256": sha256(plan_path), "source_identity": plan["source_identity"],
        "started_unix": time.time(), "learning": "not_measured", "stage_b_gate": "closed"}
    try:
        if supervisor.state["tasks"]:
            raise Blocked("inspect_existing_repair_probe_instead_of_restarting")
        results = supervisor.run("shannon-repair-preflight", plan["tasks"])
        value = next(iter(results.values()))
        if value.get("health", {}).get("eligible") is not True:
            raise Blocked("repair_numeric_or_state_load_not_qualified")
        strict = value["recovery"]["numeric_trajectory"]["within_tolerance"]
        status = "passed" if strict else "completed_with_finite_trajectory_difference"
        report.update(status=status, numerics=value["numerics"], recovery=value["recovery"],
                      checkpoint=value["checkpoint"], actual_processed_loss_tokens=value["training"]["tokens"])
        supervisor.finish("completed")
    except BaseException as error:
        supervisor.stop_reason = supervisor.stop_reason or f"{type(error).__name__}: {error}"
        supervisor.stop_all()
        report.update(status="blocked", reason=supervisor.stop_reason)
    finally:
        for task_id, row in supervisor.state["tasks"].items():
            if row.get("result_path"):
                results[task_id] = read(row["result_path"], {})
        report.update(finished_unix=time.time(), result_files={key: {"path": value.get("result_path"),
            "sha256": value.get("result_sha256")} for key, value in supervisor.state["tasks"].items()},
            remaining_descendants=list(supervisor.owned_members().values()),
            device_seconds=supervisor.device_seconds())
        finish_token_accounting(Path(plan["config"]["budget_ledger"]), str(output.resolve()), results, status)
        atomic_json(output / "repair-report.json", report)
        supervisor.close()
    return 0 if status != "blocked" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    request = parser.add_mutually_exclusive_group(required=True)
    request.add_argument("--task", type=Path)
    request.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--deadline-unix", type=float)
    parser.add_argument("--restore-prefix", type=Path)
    args = parser.parse_args(argv)
    return controller(args.plan, args.output) if args.plan else worker(args)


if __name__ == "__main__":
    raise SystemExit(main())

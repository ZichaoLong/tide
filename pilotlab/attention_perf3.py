"""Paired finite NPU timing, with fixed states and retained numerical discrepancies."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time
import traceback

from .io import atomic_json, sha256
from .scheduler3 import Blocked, read


def summary(values):
    return {"samples_seconds": values, "median_seconds": statistics.median(values),
            "min_seconds": min(values), "max_seconds": max(values)}


def bound(ref):
    if sha256(ref["path"]) != ref["sha256"]:
        raise ValueError("benchmark input identity changed: " + ref["path"])
    return read(ref["path"])


@contextmanager
def trace_attention():
    import tide.attention as a
    counts, original = Counter(), {}
    def describe(name, q, k):
        counts[(name, tuple(q.shape), tuple(k.shape))] += 1
    original["native"] = a._native_attention
    def native(q, k, *args, **kwargs):
        describe("native", q, k)
        return original["native"](q, k, *args, **kwargs)
    a._native_attention = native
    for name in ("_TiledAttention", "_MassScaledAttention", "_SingleKeyAttention"):
        original[name] = getattr(a, name).forward
        def forward(ctx, q, k, *args, _name=name):
            describe(_name, q, k)
            return original[_name](ctx, q, k, *args)
        getattr(a, name).forward = staticmethod(forward)
    try:
        yield counts
    finally:
        a._native_attention = original["native"]
        for name in ("_TiledAttention", "_MassScaledAttention", "_SingleKeyAttention"):
            getattr(a, name).forward = staticmethod(original[name])


def micros(task, device, output, deadline):
    import torch
    from tide.attention import weighted_attention
    from .worker import compare_tree, cpu_snapshot
    results = []
    for case in task["micro_cases"]:
        if time.time() >= deadline-60:
            raise TimeoutError("benchmark closing reserve")
        b, h, qlen, klen, d = case["shape"]
        generator = torch.Generator().manual_seed(918)
        q = torch.randn(b, h, qlen, d, generator=generator) * .2
        k = torch.randn(b, h, klen, d, generator=generator) * .2
        v = torch.randn(b, h, klen, d, generator=generator)
        w = torch.rand(b, h, 1, klen, generator=generator) + .5 if case["trainable_prior"] else None
        values = [x.to(device).requires_grad_() for x in (q, k, v)]
        if w is not None:
            w = w.to(device).requires_grad_()
        roots = (*values, w) if w is not None else tuple(values)
        mask = None
        if case["mask"] == "singleton":
            mask = torch.zeros(1, 1, qlen, klen, dtype=torch.bool, device=device)
            mask[..., -1] = True
        elif case["mask"] == "causal":
            mask = torch.ones(qlen, klen, dtype=torch.bool, device=device).tril()
        cotangent = torch.randn(b, h, qlen, d, generator=generator).to(device) * .01
        def call(backend, backward):
            if backward:
                out = weighted_attention(*values, w, mask, backend)
                grads = torch.autograd.grad(out, roots, cotangent)
                return (out, grads)
            with torch.no_grad():
                return weighted_attention(*values, w, mask, backend)
        checked = {backend: cpu_snapshot(call(backend, True)) for backend in ("auto", "tiled")}
        comparison = compare_tree(checked["auto"], checked["tiled"], atol=1e-5, rtol=1e-4)
        if not comparison["finite"] or not comparison["structure_equal"]:
            raise ArithmeticError("nonfinite or malformed primitive comparison")
        with trace_attention() as paths:
            call("auto", True)
        row = {"case": case, "comparison": comparison, "auto_paths": [
            {"implementation": key[0], "q_shape": key[1], "k_shape": key[2], "calls": n}
            for key, n in paths.items()], "timing": {}}
        for mode in ("forward", "forward_backward"):
            backward = mode == "forward_backward"
            for backend in ("auto", "tiled"):
                for _ in range(task["micro_warmup"]):
                    call(backend, backward)
            samples = {"auto": [], "tiled": []}
            for repeat in range(task["micro_repeats"]):
                for backend in (("auto", "tiled") if repeat % 2 == 0 else ("tiled", "auto")):
                    torch.npu.synchronize()
                    start = time.perf_counter()
                    for _ in range(task["micro_iterations"]):
                        call(backend, backward)
                    torch.npu.synchronize()
                    samples[backend].append((time.perf_counter()-start)/task["micro_iterations"])
            row["timing"][mode] = {key: summary(value) for key, value in samples.items()}
            row["timing"][mode]["tiled_over_auto_time"] = statistics.median(samples["tiled"])/statistics.median(samples["auto"])
        results.append(row)
        atomic_json(output / "micro-progress.json", results)
    return results


def models(task, device, data, output, deadline, result):
    import torch
    from . import worker3 as w
    from .models3 import build_model
    from .data3 import DocumentStore3
    from .worker import cpu_snapshot, compare_tree, digest_tree, tree_map
    store = DocumentStore3(data)
    rows = []
    for case in task["model_cases"]:
        if time.time() >= deadline-120:
            raise TimeoutError("benchmark closing reserve")
        recipe = bound(case["task"])
        cp = case["checkpoint"]
        owner = w._read_checkpoint(Path(cp["path"]), cp["sha256"])
        w.validate_checkpoint(owner, recipe, recipe["device"])
        stream = w.StreamBatch(store, recipe, owner["stream"])
        inputs, targets, reset = stream.next()
        quantum = targets.numel()
        rng = w.rng_state(device)
        runners, diagnostic = {}, {}
        for backend in ("auto", "tiled"):
            spec = copy.deepcopy(recipe["model"])
            spec["config"]["attention_backend"] = backend
            spec["config"]["local_attention_backend"] = None
            model = build_model(spec, store.vocab_size, recipe["init_recipe"]).to(device)
            optimizer = w.optimizer_for(model, recipe)
            runners[backend] = model, optimizer
        def run(backend, *, sample=False):
            model, optimizer = runners[backend]
            # All restorations and input state copies are outside timing.
            model.load_state_dict(owner["model"])
            optimizer.load_state_dict(copy.deepcopy(owner["optimizer"]))
            state = None if reset else tree_map(owner["private_state"], lambda t: t.to(device).clone())
            w.restore_rng(rng, device)
            torch.npu.synchronize()
            start = time.perf_counter()
            update = w.train_update(model, optimizer, inputs, targets, state, device, recipe, owner["tokens"], sample=sample)
            torch.npu.synchronize()
            seconds = time.perf_counter()-start
            result["training"]["segment_tokens"] += quantum
            return seconds, update
        row = {"label": case["label"], "model_id": recipe["model_id"], "batch_size": recipe["batch_size"],
               "sequence_length": recipe["sequence_length"], "origin_device": recipe["device"],
               "original_checkpoint": cp, "input_sha256": digest_tree((inputs, targets)), "reset": reset,
               "cpu_threads": task["cpu_threads"], "timing_scope": "synchronized warm single training update, includes host input transfer/loss/backward/clipping/Adam; state restoration excluded",
               "restore_scope": "identical saved model/optimizer/private state for each update, benchmark NPU RNG held fixed; not cross-backend exact continuation"}
        for backend in ("auto", "tiled"):
            with trace_attention() as paths:
                _, update = run(backend, sample=True)
            model, optimizer = runners[backend]
            diagnostic[backend] = {"model": cpu_snapshot(model.state_dict()), "optimizer": cpu_snapshot(optimizer.state_dict()),
                "state": cpu_snapshot(update["state"]), "gradients": {n: None if p.grad is None else p.grad.detach().cpu().clone()
                    for n, p in model.named_parameters()}}
            row[backend + "_diagnostic"] = {"loss": update["loss"], "grad_norm": update["grad_norm"],
                "health": update["health"], "work": update["work"], "paths": [
                    {"implementation": key[0], "q_shape": key[1], "k_shape": key[2], "calls": count}
                    for key, count in paths.items()]}
            for _ in range(task["model_warmup"]):
                run(backend)
        row["numerical_comparison"] = compare_tree(diagnostic["auto"], diagnostic["tiled"], atol=1e-5, rtol=1e-4)
        row["activity_comparison"] = compare_tree(row["auto_diagnostic"]["work"], row["tiled_diagnostic"]["work"], atol=0, rtol=0)
        samples = {"auto": [], "tiled": []}
        for repeat in range(task["model_repeats"]):
            for backend in (("auto", "tiled") if repeat % 2 == 0 else ("tiled", "auto")):
                seconds, _ = run(backend)
                samples[backend].append(seconds)
        row["timing"] = {backend: {**summary(samples[backend]),
            "tokens_per_second": quantum/statistics.median(samples[backend])} for backend in samples}
        row["tiled_over_auto_time"] = statistics.median(samples["tiled"])/statistics.median(samples["auto"])
        rows.append(row)
        atomic_json(output / "model-progress.json", rows)
        atomic_json(output / "progress.json", result)
        del runners, diagnostic
    return rows


def worker(args):
    task = read(args.task)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    from . import worker3 as w
    result = {"schema": "pilot-r3-attention-performance-v1", "status": "running", "id": task["id"],
              "model_id": task["model_id"], "phase": "preflight", "task_sha256": sha256(args.task),
              "source_identity": task["source_identity"], "started_unix": time.time(),
              "training": {"tokens": 0, "segment_tokens": 0, "scope": "discarded fixed-state timing updates; no learning trajectory"},
              "evaluation": {"tokens": 0}, "stage_b_gate": "closed"}
    atomic_json(output / "task.json", task)
    def stop(signum, frame):
        raise InterruptedError("performance probe signal " + str(signum))
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        w.validate_source_identity(task, args.data)
        device = w.initialize(args.device, task)
        result.update(device=str(device), physical_visibility=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"))
        hardware = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=15)
        (output / "hardware-before.txt").write_text(hardware.stdout + hardware.stderr)
        result["micro"] = micros(task, device, output, args.deadline_unix)
        result["models"] = models(task, device, args.data, output, args.deadline_unix, result)
        result["training"]["tokens"] = result["training"]["segment_tokens"]
        if result["training"]["tokens"] != task["target_tokens"]:
            raise ArithmeticError("performance token ceiling differs from completed fixed-state samples")
        w.validate_source_identity(task, args.data)
        hardware = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=15)
        (output / "hardware-after.txt").write_text(hardware.stdout + hardware.stderr)
        result.update(status="completed", device_memory=w.device_memory(device))
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        result["finished_unix"] = time.time()
        atomic_json(output / "result.json", result)
    return 0 if result["status"] == "completed" else 1


def controller(args):
    from .preflight3 import claim_token_budget, finish_token_accounting
    from .repair_probe3 import RepairSupervisor
    from .prepare_campaign3 import budget_remaining
    plan = read(args.plan)
    budget_remaining(Path(plan["config"]["budget_ledger"]), time.time())
    if len(plan["tasks"]) != 1 or plan["reserved_tokens"] != plan["tasks"][0]["target_tokens"]:
        raise ValueError("finite single-card benchmark reservation mismatch")
    claim_token_budget(Path(plan["config"]["budget_ledger"]), str(args.output.resolve()), plan["reserved_tokens"], sha256(args.plan))
    supervisor = RepairSupervisor(plan["config"], args.output)
    signal.signal(signal.SIGTERM, supervisor.request_stop)
    signal.signal(signal.SIGINT, supervisor.request_stop)
    report = {"schema": "pilot-r3-attention-performance-controller-v1", "status": "running",
        "plan_path": str(args.plan.resolve()), "plan_sha256": sha256(args.plan), "stage_b_gate": "closed"}
    results = {}
    try:
        if supervisor.state["tasks"]:
            raise Blocked("inspect existing benchmark instead of repeating it")
        results = supervisor.run("paired-attention-performance", plan["tasks"])
        supervisor.finish("completed")
        report["status"] = "completed_timing_only"
    except BaseException as error:
        supervisor.stop_reason = supervisor.stop_reason or f"{type(error).__name__}: {error}"
        supervisor.stop_all()
        report.update(status="blocked", reason=supervisor.stop_reason)
    finally:
        for key, row in supervisor.state["tasks"].items():
            if row.get("result_path"):
                results[key] = read(row["result_path"], {})
        finish_token_accounting(Path(plan["config"]["budget_ledger"]), str(args.output.resolve()), results, report["status"])
        report.update(finished_unix=time.time(), device_seconds=supervisor.device_seconds(),
            remaining_descendants=list(supervisor.owned_members().values()),
            result_files={key: {"path": row.get("result_path"), "sha256": row.get("result_sha256")}
                          for key, row in supervisor.state["tasks"].items()})
        atomic_json(args.output / "performance-report.json", report)
        supervisor.close()
    return 0 if report["status"] == "completed_timing_only" else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", type=Path)
    action.add_argument("--task", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data", type=Path)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--deadline-unix", type=float)
    args = p.parse_args(argv)
    return controller(args) if args.plan else worker(args)


if __name__ == "__main__":
    raise SystemExit(main())

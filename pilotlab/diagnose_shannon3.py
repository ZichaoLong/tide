"""Bounded replay of an R3 failure, separate from the campaign and its ledger.

The capture is diagnostic, never a resumable campaign checkpoint. All replay
updates use its one saved input/target pair; they are not learning experiments.
Select explicit frozen project/foundation paths. Device admission and durable
submission belong to the caller; this module never acquires or submits work.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def entropy_reference(scores, lower, upper, low_total, high_total):
    """80-digit scalar KKT reference with no production solve/reconstruction."""
    with localcontext() as context:
        context.prec = 80
        scores, lower, upper = [[Decimal.from_float(float(x)) for x in row]
                                 for row in (scores, lower, upper)]
        low, high = Decimal.from_float(float(low_total)), Decimal.from_float(float(high_total))

        def response(multiplier):
            return [max(l, min(u, (score - 1 - multiplier).exp()))
                    for score, l, u in zip(scores, lower, upper)]

        initial = response(Decimal(0))
        target = max(low, min(high, sum(initial)))
        result = initial
        if sum(initial) != target:
            left, right = Decimal(-4096), Decimal(4096)
            if not sum(response(left)) >= target >= sum(response(right)):
                raise ValueError("diagnostic scalar reference bracket does not cover this input")
            for _ in range(256):
                middle = (left + right) / 2
                if sum(response(middle)) > target:
                    left = middle
                else:
                    right = middle
            result = response((left + right) / 2)
        return {"values": [float(x) for x in result], "target": float(target),
                "positive_support": [x > 0 for x in result],
                "method": "independent 80-digit Decimal KKT dual solve"}


def configure_imports(source, foundation):
    """The -m entry package must not override the explicitly selected source."""
    source, foundation = Path(source).resolve(), Path(foundation).resolve()
    if not (source / "pilotlab/models3.py").is_file() or not (foundation / "src/tide/allocation.py").is_file():
        raise ValueError("explicit existing project and foundation sources are required")
    sys.path[:0] = [str(source), str(foundation / "src")]
    if "pilotlab" in sys.modules:
        sys.modules["pilotlab"].__path__ = [str(source / "pilotlab")]
    return source, foundation


def replay(args):
    result_path, output = args.result.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("diagnostic output must be new; original evidence is never replaced")
    result = json.loads(result_path.read_text())
    capture_path = Path(result["failure_capture"]["path"])
    if file_hash(capture_path) != result["failure_capture"]["sha256"]:
        raise ValueError("original failure capture hash mismatch")
    source, foundation = configure_imports(args.source, args.foundation)
    import torch
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401
    from pilotlab.models3 import build_model
    from pilotlab.worker3 import initialize, optimizer_for, train_update
    from pilotlab.worker import (atomic_torch_save, compare_tree, cpu_snapshot, digest_tree,
                                 restore_rng, rng_state, synchronize, tree_map)
    from pilotlab.io import atomic_json
    allocation = importlib.import_module("tide.allocation")
    capture = torch.load(capture_path, map_location="cpu", weights_only=False)
    task = capture["task"]
    device = initialize(args.device, dict(task, cpu_threads=args.threads))
    report = {"schema": "pilot-r3-shannon-failure-diagnostic-v1", "status": "running",
        "started_unix": time.time(), "original_result": str(result_path),
        "original_result_sha256": file_hash(result_path), "failure_capture": str(capture_path),
        "failure_capture_sha256": file_hash(capture_path), "source": str(source), "foundation": str(foundation),
        "allocation_path": str(Path(allocation.__file__).resolve()), "allocation_sha256": file_hash(allocation.__file__),
        "source_files": {str(path): file_hash(path) for path in sorted((foundation / "src/tide").rglob("*.py"))},
        "device": str(device), "precision": "fp32", "cpu_threads": args.threads,
        "original_completed_steps": capture["completed_steps"],
        "original_completed_tokens": capture["completed_tokens"],
        "original_error": capture["error"], "original_capture_resumable": capture["resumable"],
        "scope": "diagnostic replay of saved input/targets; no campaign task, learning, or recovery qualification",
        "atol": 1e-5, "rtol": 1e-4, "updates": []}
    output.mkdir(parents=True, exist_ok=True)
    original_save = allocation._save_solution
    calls = 0

    def observe(ctx, c, scores, mu, lower, upper, low_active, high_active, kind, high_total, originals,
                *extra, **kwargs):
        nonlocal calls
        calls += 1
        bad = (upper > 0) & (c == 0) & (high_total > lower.sum(-1, keepdim=True))
        if kind == "shannon" and bool(bad.any()):
            rows = bad.any(-1)
            tensors = {name: value.detach().cpu() for name, value in
                       zip(("c", "scores", "mu", "lower", "upper", "low_total", "high_total"),
                           (c, scores, mu, lower, upper, originals[4], high_total))}
            saved = output / "allocation-failure.pt"
            atomic_torch_save(saved, tensors)
            indexes = rows.nonzero().cpu().tolist()
            details = []
            for index in indexes[:8]:
                position = tuple(index)
                row = {name: tensors[name][position].tolist() for name in ("scores", "lower", "upper", "c")}
                low = float(torch.broadcast_to(tensors["low_total"], (*c.shape[:-1], 1))[position].item())
                high = float(torch.broadcast_to(tensors["high_total"], (*c.shape[:-1], 1))[position].item())
                row["reference"] = entropy_reference(row["scores"], row["lower"], row["upper"], low, high)
                details.append(dict(index=index, **row))
            report["allocation_failure"] = {"call": calls, "path": str(saved), "sha256": file_hash(saved),
                                             "bad_sender_count": int(bad.sum()), "rows": details}
        return original_save(ctx, c, scores, mu, lower, upper, low_active, high_active, kind,
                             high_total, originals, *extra, **kwargs)

    allocation._save_solution = observe
    try:
        owner = capture
        if args.resume_checkpoint:
            owner = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
            if owner.get("schema") != "pilot-r3-diagnostic-checkpoint-v1":
                raise ValueError("only a diagnostic checkpoint may be replayed here")
        model = build_model(task["model"], capture["model"]["embedding.weight"].shape[0], task["init_recipe"]).to(device)
        model.load_state_dict(owner["model"])
        optimizer = optimizer_for(model, task)
        optimizer.load_state_dict(owner["optimizer"])
        state = tree_map(owner["private_state"], lambda tensor: tensor.to(device))
        if "rng" in owner:
            restore_rng(owner["rng"], device)
        tokens = owner.get("completed_tokens", capture["completed_tokens"])
        prefix = None
        final_payload = None
        for step in range(args.steps):
            if args.restore_check and step == args.steps - 1:
                prefix = output / "prefix.pt"
                atomic_torch_save(prefix, {"schema": "pilot-r3-diagnostic-checkpoint-v1",
                    "model": cpu_snapshot(model.state_dict()), "optimizer": cpu_snapshot(optimizer.state_dict()),
                    "private_state": cpu_snapshot(state), "completed_tokens": tokens, "rng": rng_state(device)})
            started = time.monotonic()
            update = train_update(model, optimizer, capture["inputs"], capture["targets"], state,
                                  device, task, tokens, sample=False)
            synchronize(device)
            state, tokens = update["state"], tokens + capture["targets"].numel()
            final_payload = {"model": cpu_snapshot(model.state_dict()), "optimizer": cpu_snapshot(optimizer.state_dict()),
                "private_state": cpu_snapshot(state),
                "gradients": {name: None if p.grad is None else p.grad.detach().cpu().clone()
                              for name, p in model.named_parameters()}}
            report["updates"].append({"index": step + 1, "loss": update["loss"], "grad_norm": update["grad_norm"],
                "seconds": time.monotonic() - started, "state_digest": digest_tree(final_payload),
                "parameter_sha256": digest_tree(final_payload["model"]),
                "gradient_sha256": digest_tree(final_payload["gradients"])})
        final_path = output / "final-state.pt"
        atomic_torch_save(final_path, final_payload)
        report["final_state"] = {"path": str(final_path), "sha256": file_hash(final_path)}
        if args.restore_check:
            child_output = output / "restored"
            child_argv = [sys.executable, str(Path(__file__).resolve()), "--result", str(result_path),
                "--source", str(source), "--foundation", str(foundation), "--output", str(child_output),
                "--device", args.device, "--steps", "1", "--threads", str(args.threads),
                "--resume-checkpoint", str(prefix)]
            child = subprocess.run(child_argv, env={**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"},
                                   capture_output=True, text=True, timeout=120)
            report["fresh_process"] = {"argv": child_argv, "returncode": child.returncode,
                                       "stdout": child.stdout[-4000:], "stderr": child.stderr[-4000:]}
            if child.returncode:
                raise RuntimeError("diagnostic fresh-process replay failed")
            restored = torch.load(child_output / "final-state.pt", map_location="cpu", weights_only=False)
            comparison = compare_tree(final_payload, restored, atol=1e-5, rtol=1e-4)
            report["fresh_process"]["comparison"] = comparison
            if not comparison["within_tolerance"]:
                raise ArithmeticError("diagnostic fresh-process update exceeds the unchanged resume thresholds")
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=str(error), exception=type(error).__name__, traceback=traceback.format_exc())
    finally:
        allocation._save_solution = original_save
        report.update(allocation_calls=calls, finished_unix=time.time())
        atomic_json(output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("result", "source", "foundation", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, choices=range(1, 5), default=3)
    parser.add_argument("--threads", type=int, choices=(1, 4, 8), default=1)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--restore-check", action="store_true")
    args = parser.parse_args(argv)
    report = replay(args)
    print(json.dumps({key: report.get(key) for key in ("status", "error", "allocation_calls", "updates", "fresh_process")},
                     ensure_ascii=False, indent=2))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

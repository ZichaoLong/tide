"""Replay the r09 failure and its real successor batches with a pinned Tide source."""
import argparse
import copy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / ".local/pilot-r3/campaign-r08/frozen"
WITNESS = ROOT / "docs/evidence/pilot-r3-r09-shannon-underflow.json"
CHECKPOINT_SHA = "35124c711e01558a5765c0e98d77b4b1de88a78d490374c06501d82db65d9fa2"
POLICY = "positive-floor-v1"


def read(path):
    return json.loads(path.read_text())


def bound_task(original, src, expected, data):
    from pilotlab import worker3 as w

    task = copy.deepcopy(original)
    task["id"] += "-tl-req-007-diagnostic"
    task["model"]["config"]["shannon_support_policy"] = POLICY
    files = task["source_identity"]["source_files"]
    worker_files = {p: digest for p, digest in files.items() if Path(p).is_relative_to(FROZEN / "source")}
    base = {p.relative_to(src / "tide").as_posix(): w.sha256(p) for p in (src / "tide").rglob("*.py")}
    if not base or w.canonical_digest(base) != expected:
        raise ValueError("foundation source does not match the requested tree SHA256")
    task["source_identity"]["source_files"] = worker_files | {str(src / "tide" / p): h for p, h in base.items()}
    w.validate_source_identity(task, data)
    return task


def original_rows(rows, device):
    import torch
    from tide import observe_shannon_support
    from tide.allocation import allocate

    s = torch.tensor([r["scores"] for r in rows], device=device, requires_grad=True)
    cap = torch.ones_like(s)
    budget = torch.ones(len(rows), device=device, requires_grad=True)
    with observe_shannon_support() as counts:
        c = allocate(s, cap, cap, torch.zeros_like(cap), cap, budget, kind="shannon", support_policy=POLICY)
    weak = s.argmin(-1, keepdim=True)
    root = torch.zeros_like(s).scatter_(1, weak, 1e30)
    ds, dk = torch.autograd.grad(c, (s, budget), root)
    # Independent binary logistic derivative, evaluated in host binary64.
    odds = [math.exp(-abs(a-b)) for a, b in s.tolist()]
    p = [z/(1+z) for z in odds]
    g = root.max().item()
    expected = torch.tensor([g*x*(1-x) for x in p], device=device)
    total = torch.tensor([g*x for x in p], device=device)
    observed = (ds.gather(1, weak).flatten(), -ds.gather(1, 1-weak).flatten(), dk)
    maxima = []
    for actual, wanted in zip(observed, (expected, expected, total)):
        torch.testing.assert_close(actual, wanted, rtol=3e-5, atol=0)
        maxima.append(float(((actual-wanted)/wanted).abs().max()))
    assert weak.flatten().tolist() == [r["index"][-1] for r in rows]
    assert bool((c.gather(1, weak) == 2.**-149).all()) and bool((c.gather(1, 1-weak) == 1).all())
    assert counts.repaired_coordinates == len(rows)
    return {"counts": asdict(counts), "max_relative_error_weak_strong_budget": maxima,
            "first_scores": s[0].tolist(), "first_stored": c[0].tolist(), "first_score_gradient": ds[0].tolist()}


def owners(task, store, device):
    from pilotlab.models3 import build_model
    from pilotlab.worker3 import optimizer_for

    model = build_model(task["model"], store.vocab_size, task["init_recipe"]).to(device)
    return model, optimizer_for(model, task)


def update(model, optimizer, state, batch, task, device, tokens):
    import torch
    from pilotlab import worker3 as w
    from tide import observe_shannon_support

    x, y, reset = batch
    with observe_shannon_support() as counts:
        result = w.train_update(model, optimizer, x, y, None if reset else state, device, task, tokens)
        w.synchronize(device)
    gradients = {name: p.grad for name, p in model.named_parameters()}
    trees = w.cpu_snapshot({"model": model.state_dict(), "adamw": optimizer.state_dict(),
                            "private_state": result["state"], "gradients_after_clip": gradients})
    for field, tree in trees.items():
        for path, t in w.leaves(tree):
            if isinstance(t, torch.Tensor) and t.is_floating_point() and not bool(t.isfinite().all()):
                raise ArithmeticError(f"non-finite {field}{path}")
    stats = {key: result[key] for key in ("loss", "objective", "grad_norm", "lr")}
    return result["state"], trees, dict(stats, counts=asdict(counts), reset=reset,
        input_sha256=w.digest_tree((x, y)), gradient_tensors=sum(g is not None for g in gradients.values()),
        state_sha256={key: w.digest_tree(value) for key, value in trees.items()})


def direct(args, record, task, original, store, device):
    import torch
    from pilotlab import worker3 as w
    from tide.precision import precision_context

    capture_ref = record["failure"]["capture"]
    path = ROOT / capture_ref["path"]
    assert w.sha256(path) == capture_ref["sha256"]
    capture = torch.load(path, map_location="cpu", weights_only=False)
    assert capture["task"] == original and capture["resumable"] is False
    index = read(path.parent / "checkpoint.json")["current"]
    assert index["sha256"] == CHECKPOINT_SHA
    official, origin = w.load_checkpoint(index["path"], original, original["device"], CHECKPOINT_SHA)
    assert origin["sha256"] == CHECKPOINT_SHA and official["steps"] == 1203
    report = {"pid": os.getpid(), "source_sha256": args.sha256, "physical_device": os.getenv("ASCEND_RT_VISIBLE_DEVICES"),
              "capture_sha256": capture_ref["sha256"], "official_checkpoint": origin,
              "rows": original_rows(record["numeric_witness"]["all_failing_coordinates"][device.type], device)}
    if device.type == "npu":
        # Restore all original owners on their actual backend, then explicitly
        # apply the new recipe to one real update from the official checkpoint.
        model, opt = owners(task, store, device)
        state, early_stream = w.restore_owners(official, model, opt, store, original, device)
        _, _, report["official_migration_step_1204"] = update(
            model, opt, state, early_stream.next(), task, device, official["tokens"])
        del model, opt, state

    stream = w.StreamBatch(store, original, copy.deepcopy(official["stream"]))
    log = {r["step"]: r for r in map(json.loads, (path.parent / "steps.jsonl").read_text().splitlines())}
    for step in range(official["steps"] + 1, capture["completed_steps"] + 1):
        x, y, reset = stream.next()
        assert reset == log[step]["state_reset"]
        if "input_sha256" in log[step]:
            assert w.digest_tree((x, y)) == log[step]["input_sha256"]
    batch = stream.next()
    assert torch.equal(batch[0], capture["inputs"]) and torch.equal(batch[1], capture["targets"])
    assert capture["completed_steps"] == 1255 and capture["completed_tokens"] == 321280
    report["stream_alignment"] = {"from_step": official["steps"], "to_step": 1256, "failure_batch_exact": True}
    state = w.tree_map(capture["private_state"], lambda t: t.to(device))
    model, _ = owners(original, store, device)
    model.load_state_dict(capture["model"])
    try:
        with precision_context(device, original["precision"]):
            model(batch[0].to(device), state)
    except ArithmeticError as error:
        if str(error) != record["failure"]["error"]:
            raise
        report["strict_error"] = str(error)
    else:
        raise AssertionError("strict no longer reproduces the failure")

    # The failure capture contains no RNG/stream checkpoint. Bind the verified
    # stream and a freshly initialized diagnostic RNG; never publish it to A.
    w.initialize(args.device, task)
    model, opt = owners(task, store, device)
    model.load_state_dict(capture["model"])
    opt.load_state_dict(capture["optimizer"])
    origin_hashes = {key: w.digest_tree(capture[key]) for key in ("model", "optimizer", "private_state")}
    assert all(w.digest_tree(actual) == origin_hashes[key] for key, actual in
               (("model", model.state_dict()), ("optimizer", opt.state_dict()), ("private_state", state)))
    report["capture_owners_exact"] = True
    report["updates"] = []
    tokens = capture["completed_tokens"]
    for step in (1256, 1257, 1258):
        if step != 1256:
            batch = stream.next()
        state, trees, info = update(model, opt, state, batch, task, device, tokens)
        tokens += batch[1].numel()
        report["updates"].append(dict(info, step=step, tokens=tokens))
        if step == 1256:
            assert info["counts"]["repaired_coordinates"] == 23
            assert info["state_sha256"]["model"] != origin_hashes["model"]
            assert info["state_sha256"]["adamw"] != origin_hashes["optimizer"]
        if step == 1257:
            payload = w.checkpoint_payload(model, opt, state, stream, task, device, tokens, step,
                {"origin": "explicit strict-to-floor failure-capture diagnostic", "capture_sha256": capture_ref["sha256"]})
            report["new_checkpoint"] = w.save_checkpoint(args.output, payload)
    w.atomic_torch_save(args.output / "expected.pt", trees)
    report["stream_sha256"], report["rng_sha256"] = w.digest_tree(stream.state_dict()), w.digest_tree(w.rng_state(device))
    return report


def restored(args, task, store, device):
    import torch
    from pilotlab import worker3 as w
    from pilotlab.worker import compare_tree

    baseline = read(args.output / "direct.json")
    ref = baseline["new_checkpoint"]
    saved, source = w.load_checkpoint(ref["path"], task, device, ref["sha256"])
    assert source["sha256"] == ref["sha256"] and not source["fallback_used"] and saved["steps"] == 1257
    model, opt = owners(task, store, device)
    state, stream = w.restore_owners(saved, model, opt, store, task, device)
    _, actual, info = update(model, opt, state, stream.next(), task, device, saved["tokens"])
    expected = torch.load(args.output / "expected.pt", map_location="cpu", weights_only=False)
    comparisons = {k: compare_tree(v, actual[k]) for k, v in expected.items()}
    prior = baseline["updates"][-1]
    discrete = (all(info[k] == prior[k] for k in ("input_sha256", "counts", "reset", "gradient_tensors", "lr"))
                and baseline["stream_sha256"] == w.digest_tree(stream.state_dict())
                and baseline["rng_sha256"] == w.digest_tree(w.rng_state(device)))
    scalars = {k: abs(prior[k]-info[k]) for k in ("loss", "objective", "grad_norm")}
    passed = discrete and all(v["within_tolerance"] for v in comparisons.values()) and all(
        math.isclose(info[k], prior[k], abs_tol=1e-5, rel_tol=1e-4) for k in scalars)
    exact = (discrete and all(value == 0 for value in scalars.values())
             and info["state_sha256"] == prior["state_sha256"])
    return {"pid": os.getpid(), "physical_device": os.getenv("ASCEND_RT_VISIBLE_DEVICES"),
            "loaded_state_exact": True, "checkpoint_sha256": source["sha256"],
            "step": 1258, "discrete_exact": discrete, "scalar_abs_diffs": scalars,
            "comparison": comparisons, "bitwise_replay": exact, "passed": passed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundation-src", required=True, type=Path)
    parser.add_argument("--sha256", required=True, help="canonical digest of relative tide/*.py file hashes")
    parser.add_argument("--device", required=True, choices=("cpu", "npu:0"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--phase", choices=("direct", "restored"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.phase is None:
        args.output.mkdir(parents=True, exist_ok=False)
        for phase in ("direct", "restored"):
            subprocess.run([sys.executable, __file__, *sys.argv[1:], "--phase", phase], check=True)
        return
    src = args.foundation_src.resolve()
    sys.path[:0] = [str(FROZEN / "source"), str(src)]
    from pilotlab import worker3 as w
    from pilotlab.data3 import DocumentStore3

    record = read(WITNESS)
    task_ref = record["failure"]["task"]
    assert w.sha256(ROOT / task_ref["path"]) == task_ref["sha256"]
    original = read(ROOT / task_ref["path"])
    data = Path(read(FROZEN / "prepared.json")["data"])
    task = bound_task(original, src, args.sha256, data)
    device = w.initialize(args.device, task)
    store = DocumentStore3(data)
    report = (direct(args, record, task, original, store, device) if args.phase == "direct"
              else restored(args, task, store, device))
    w.validate_source_identity(task, data)
    w.atomic_json(args.output / (args.phase + ".json"), report)
    if not report.get("passed", True):
        raise ArithmeticError("fresh-process comparison exceeded the worker thresholds")
    print(args.device, args.phase, "passed", flush=True)


if __name__ == "__main__":
    main()

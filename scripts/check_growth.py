#!/usr/bin/env python3
"""Bounded admission checks for the selected pretrained residual task model.

This checks real cached weights. It is not a language benchmark, convergence
test, architecture-value claim, or a replacement for core5 independent checks.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def check_numerics(report, name, actual, expected, *, explore):
    """Preserve strict failures; exploration only continues finite differences."""
    import torch
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    record = dict(finite=finite, atol=1e-5, rtol=1e-4)
    report["numerical_checks"][name] = record
    if not finite:
        record["strict"] = "failed_nonfinite"
        report["strict_numeric"] = "failed"
        raise FloatingPointError(f"nonfinite values in {name}")
    error = (actual-expected).abs()
    allowed = 1e-5 + 1e-4 * expected.abs()
    close = bool(torch.allclose(actual, expected, atol=1e-5, rtol=1e-4))
    record.update(strict="passed" if close else "failed", max_abs=float(error.max().cpu()),
                  max_tolerance_ratio=float((error/allowed).max().cpu()),
                  values_exceeding_tolerance=int((error > allowed).sum().cpu()))
    if not close:
        report["strict_numeric"] = "failed"
        report["strict_numeric_failures"].append(name)
        if not explore:
            raise AssertionError(f"{name} failed strict numerical tolerance")
    elif report["strict_numeric"] == "not_checked":
        report["strict_numeric"] = "passed"
    return close


def save_fixture(path, report, tensors):
    """Small CPU tensors for a later same-model CPU/NPU comparison."""
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    cpu_tensors = {key: value.detach().cpu().clone() for key, value in tensors.items()}
    torch.save(dict(schema="tide-growth-calibration-fixture-v1", device=report["device"],
        seed=report["seed"], model=report["model"], tensors=cpu_tensors), temporary)
    os.replace(temporary, path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    report["fixture"] = dict(path=str(path), sha256=digest,
                             shapes={name: list(value.shape) for name, value in cpu_tensors.items()},
                             scope="untrained logits, same inputs; no gradient or optimizer fixture")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--branch", choices=("tide", "mlp"), default="tide")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--explore-numerics", action="store_true",
        help="continue finite cache/prefix differences, preserving every strict failure")
    parser.add_argument("--fixture-output", type=Path,
        help="optional small untrained-logit fixture for same-model CPU/NPU comparison")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.fixture_output is not None and args.fixture_output.exists():
        raise FileExistsError(args.fixture_output)
    if args.fixture_output is not None and args.fixture_output.resolve() == args.output.resolve():
        raise ValueError("fixture output and JSON report must have different paths")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import torch
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401; site environment and allocation are launcher-owned
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    from tide_lab.growth import GrowthClassifier
    from tide_lab.tasks import RetrievalTask
    started = time.monotonic()
    report = dict(schema="tide-growth-admission-v1", state="running", device=args.device,
        execution="running", strict_numeric="not_checked", strict_numeric_failures=[],
        numerical_checks={}, seed=args.seed,
        numerical_policy=dict(atol=1e-5, rtol=1e-4,
                              mode="explore_finite" if args.explore_numerics else "strict"), checks={},
        scope="selected checkpoint, tiny synthetic episodes, eager FP32; two SGD ingress steps",
        application_resume="not checked here; runner-owned full application test",
        core5_correctness="separate independent core5 admission required")
    write_json(args.output, report)
    try:
        task = RetrievalTask()
        model = GrowthClassifier(args.checkpoint_path, task.vocab_size, task.value_vocab_size,
                                 branch=args.branch, seed=args.seed,
                                 freeze_backbone=args.freeze_backbone).to(args.device)
        model.eval()
        report["model"] = model.facts()
        tokens, target = task.batch("train", 2, 0, args.seed, args.device)
        checks = report["checks"]
        with torch.no_grad():
            initial = model(tokens)
            unbranched = model(tokens, branch_enabled=False)
            checks["task_initial_finite"] = bool(torch.isfinite(initial).all() and torch.isfinite(unbranched).all())
            if not checks["task_initial_finite"]:
                raise FloatingPointError("nonfinite initial task output")
            checks["task_initial_exact"] = bool(torch.equal(initial, unbranched))
            if not checks["task_initial_exact"]:
                raise AssertionError("zero branch changed initial task logits")
            prompt = "The capital of France is"
            ordinary_ids = model.tokenizer(prompt, return_tensors="pt")["input_ids"].to(args.device)
            original = model.lm(ordinary_ids, use_cache=True, return_dict=True, logits_to_keep=1)
            neutral = model.language_last_logits(ordinary_ids)
            checks["original_lm_finite"] = bool(torch.isfinite(original.logits).all() and torch.isfinite(neutral).all())
            if not checks["original_lm_finite"]:
                raise FloatingPointError("nonfinite original or neutral LM output")
            checks["language_initial_exact"] = bool(torch.equal(original.logits[:, -1], neutral))
            if not checks["language_initial_exact"]:
                raise AssertionError("zero branch changed initial language logits")
            prefix = model.lm(ordinary_ids[:, :-1], use_cache=True, return_dict=True, logits_to_keep=1)
            resumed = model.lm(ordinary_ids[:, -1:], use_cache=True,
                               past_key_values=prefix.past_key_values, return_dict=True, logits_to_keep=1)
            checks["cached_lm_finite"] = bool(torch.isfinite(resumed.logits).all())
            if not checks["cached_lm_finite"]:
                raise FloatingPointError("nonfinite cached LM output")
            error = (original.logits[:, -1] - resumed.logits[:, -1]).abs()
            checks["original_lm_cache_max_abs"] = float(error.max().cpu())
            if args.fixture_output is not None:
                save_fixture(args.fixture_output, report, dict(
                    task_tokens=tokens, task_targets=target, ordinary_token_ids=ordinary_ids,
                    initial_task_logits=initial, original_lm_last_logits=original.logits[:, -1],
                    cached_lm_last_logits=resumed.logits[:, -1]))
            checks["original_lm_cache_close"] = check_numerics(report, "original_lm_cache",
                original.logits[:, -1], resumed.logits[:, -1], explore=args.explore_numerics)
            checks["original_lm_next_token"] = model.tokenizer.decode(
                original.logits[:, -1].argmax(-1).tolist())
            if not args.skip_generation:
                generated = model.lm.generate(ordinary_ids, max_new_tokens=8, do_sample=False,
                    use_cache=True, pad_token_id=model.tokenizer.eos_token_id)
                checks["original_lm_greedy_continuation"] = model.tokenizer.decode(
                    generated[0, ordinary_ids.shape[1]:], skip_special_tokens=True)
            state = None
            prefix_checks = []
            checks["prefix_continuation"] = prefix_checks
            for position in range(tokens.shape[1]):
                streamed, state = model.infer_tokens(tokens[:, position:position+1], state)
                whole = model(tokens[:, :position+1])
                close = check_numerics(report, f"task_prefix_{position+1}", streamed, whole,
                                       explore=args.explore_numerics)
                prefix_checks.append(dict(length=position+1, strict_close=close))
            checks["all_prefix_continuation_close"] = all(item["strict_close"] for item in prefix_checks)
            checks["initial_task_loss"] = float(torch.nn.functional.cross_entropy(initial, target).cpu())
        del original, neutral, prefix, resumed, state
        # Unique public parameters include tied embedding/LM-head storage once.
        params = [p for p in model.parameters() if p.requires_grad]
        if len(params) != len({id(p) for p in params}):
            raise AssertionError("optimizer contains duplicate shared parameters")
        checks["optimizer_unique_parameters"] = True
        optimizer = torch.optim.SGD(params, lr=1e-3)
        frozen_versions = {name:p._version for name,p in model.lm.named_parameters()}
        if args.freeze_backbone:
            if any(p.requires_grad or id(p) in {id(q) for q in params} for p in model.lm.parameters()):
                raise AssertionError('frozen backbone entered optimizer')
            checks['backbone_excluded_from_optimizer'] = True
        records = []
        checks["gradient_ingress"] = records
        model.train()
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens)
            loss = torch.nn.functional.cross_entropy(logits, target)
            loss.backward()
            if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all()
                                             for p in params):
                raise FloatingPointError("nonfinite loss or gradient")
            inner = [(name, p.grad) for name, p in model.named_parameters()
                     if name.startswith("branch_core.") or name.startswith("branch_input.")]
            maximum = max((float(g.detach().abs().max().cpu()) for _, g in inner if g is not None), default=0.)
            output = float(model.branch_output.weight.grad.detach().abs().max().cpu())
            records.append(dict(step=step, loss=float(loss.detach().cpu()),
                                inner_gradient_max_abs=maximum, output_gradient_max_abs=output))
            if step == 0 and (maximum != 0 or output <= 0):
                raise AssertionError("zero-output-projection gradient boundary differs from expected")
            if step == 1 and maximum <= 0:
                raise AssertionError("task gradient did not enter branch after output projection update")
            optimizer.step()
            if args.freeze_backbone:
                if any(p.grad is not None or p.requires_grad or p._version!=frozen_versions[name]
                       for name,p in model.lm.named_parameters()):
                    raise AssertionError('frozen backbone acquired a gradient or in-place parameter mutation')
                checks['backbone_no_grad_or_parameter_mutation'] = True
        model.eval()
        with torch.no_grad():
            on, off = model(tokens), model(tokens, branch_enabled=False)
            if not torch.isfinite(on).all() or not torch.isfinite(off).all():
                raise FloatingPointError("nonfinite post-update task output")
            checks["post_update_ablation_max_abs"] = float((on-off).abs().max().cpu())
            if checks["post_update_ablation_max_abs"] <= 0:
                raise AssertionError("trained branch has no measurable task-logit contribution")
            _, state = model.infer_tokens(tokens[:, :-1])
            streamed, state = model.infer_tokens(tokens[:, -1:], state)
            checks["post_update_continuation_max_abs"] = float((on-streamed).abs().max().cpu())
            checks["post_update_continuation_close"] = check_numerics(
                report, "post_update_continuation", on, streamed, explore=args.explore_numerics)
        report["execution"] = "completed"
        report["state"] = ("completed_with_strict_failures" if report["strict_numeric"] == "failed"
                           else "passed")
    except BaseException as error:
        report.update(state="failed", execution="aborted", error=repr(error))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(args.output, report)
    print(json.dumps(dict(state=report["state"], output=str(args.output),
                          execution=report["execution"], strict_numeric=report["strict_numeric"],
                          elapsed_seconds=report["elapsed_seconds"])))


if __name__ == "__main__":
    main()

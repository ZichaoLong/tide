#!/usr/bin/env python3
"""Bounded admission checks for the selected pretrained residual task model.

This checks real cached weights. It is not a language benchmark, convergence
test, architecture-value claim, or a replacement for core5 independent checks.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--branch", choices=("tide", "mlp"), default="tide")
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
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
        seed=args.seed, numerical_policy=dict(atol=1e-5, rtol=1e-4), checks={},
        scope="selected checkpoint, tiny synthetic episodes, eager FP32; two SGD ingress steps",
        application_resume="not checked here; runner-owned full application test",
        core5_correctness="separate independent core5 admission required")
    write_json(args.output, report)
    try:
        task = RetrievalTask()
        model = GrowthClassifier(args.checkpoint_path, task.vocab_size, task.value_vocab_size,
                                 branch=args.branch, seed=args.seed).to(args.device)
        model.eval()
        report["model"] = model.facts()
        tokens, target = task.batch("train", 2, 0, args.seed, args.device)
        checks = report["checks"]
        with torch.no_grad():
            initial = model(tokens)
            unbranched = model(tokens, branch_enabled=False)
            checks["task_initial_exact"] = bool(torch.equal(initial, unbranched))
            if not checks["task_initial_exact"]:
                raise AssertionError("zero branch changed initial task logits")
            prompt = "The capital of France is"
            ordinary_ids = model.tokenizer(prompt, return_tensors="pt")["input_ids"].to(args.device)
            original = model.lm(ordinary_ids, use_cache=True, return_dict=True, logits_to_keep=1)
            neutral = model.language_last_logits(ordinary_ids)
            checks["language_initial_exact"] = bool(torch.equal(original.logits[:, -1], neutral))
            if not checks["language_initial_exact"]:
                raise AssertionError("zero branch changed initial language logits")
            prefix = model.lm(ordinary_ids[:, :-1], use_cache=True, return_dict=True, logits_to_keep=1)
            resumed = model.lm(ordinary_ids[:, -1:], use_cache=True,
                               past_key_values=prefix.past_key_values, return_dict=True, logits_to_keep=1)
            error = (original.logits[:, -1] - resumed.logits[:, -1]).abs()
            checks["original_lm_cache_max_abs"] = float(error.max().cpu())
            checks["original_lm_cache_close"] = bool(torch.allclose(
                original.logits[:, -1], resumed.logits[:, -1], atol=1e-5, rtol=1e-4))
            if not checks["original_lm_cache_close"]:
                raise AssertionError("original LM cache numerical check failed strict tolerance")
            checks["original_lm_finite"] = bool(torch.isfinite(original.logits).all())
            checks["original_lm_next_token"] = model.tokenizer.decode(
                original.logits[:, -1].argmax(-1).tolist())
            if not checks["original_lm_finite"]:
                raise FloatingPointError("nonfinite original LM output")
            if not args.skip_generation:
                generated = model.lm.generate(ordinary_ids, max_new_tokens=8, do_sample=False,
                    use_cache=True, pad_token_id=model.tokenizer.eos_token_id)
                checks["original_lm_greedy_continuation"] = model.tokenizer.decode(
                    generated[0, ordinary_ids.shape[1]:], skip_special_tokens=True)
            state = None
            for position in range(tokens.shape[1]):
                streamed, state = model.infer_tokens(tokens[:, position:position+1], state)
                whole = model(tokens[:, :position+1])
                if not torch.allclose(streamed, whole, atol=1e-5, rtol=1e-4):
                    raise AssertionError("cached backbone/replayed branch prefix mismatch")
            checks["all_prefix_continuation_close"] = True
            checks["initial_task_loss"] = float(torch.nn.functional.cross_entropy(initial, target).cpu())
        del original, neutral, prefix, resumed, state
        # Unique public parameters include tied embedding/LM-head storage once.
        params = [p for p in model.parameters() if p.requires_grad]
        if len(params) != len({id(p) for p in params}):
            raise AssertionError("optimizer contains duplicate shared parameters")
        checks["optimizer_unique_parameters"] = True
        optimizer = torch.optim.SGD(params, lr=1e-3)
        records = []
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
        checks["gradient_ingress"] = records
        model.eval()
        with torch.no_grad():
            on, off = model(tokens), model(tokens, branch_enabled=False)
            checks["post_update_ablation_max_abs"] = float((on-off).abs().max().cpu())
            if checks["post_update_ablation_max_abs"] <= 0:
                raise AssertionError("trained branch has no measurable task-logit contribution")
            _, state = model.infer_tokens(tokens[:, :-1])
            streamed, state = model.infer_tokens(tokens[:, -1:], state)
            checks["post_update_continuation_max_abs"] = float((on-streamed).abs().max().cpu())
            checks["post_update_continuation_close"] = bool(torch.allclose(
                on, streamed, atol=1e-5, rtol=1e-4))
            if not checks["post_update_continuation_close"]:
                raise AssertionError("trained branch continuation failed strict tolerance")
        report["state"] = "passed"
    except BaseException as error:
        report.update(state="failed", error=repr(error))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(args.output, report)
    print(json.dumps(dict(state=report["state"], output=str(args.output),
                          elapsed_seconds=report["elapsed_seconds"])))


if __name__ == "__main__":
    main()

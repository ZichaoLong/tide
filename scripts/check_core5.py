#!/usr/bin/env python3
"""Independent downstream acceptance for a tiny installed core-5 consumer.

No training qualification is inferred from these finite checks. Exact
application resume is exercised by the project runner, not this script.
"""
import argparse
from dataclasses import fields, is_dataclass
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback

import torch

from tide_lab.core5 import Core5Classifier
from tide_lab.tasks import RetrievalTask


def snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if is_dataclass(value):
        return {f.name: snapshot(getattr(value, f.name)) for f in fields(value) if f.name != "stats"}
    if isinstance(value, dict):
        return {k: snapshot(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(snapshot(v) for v in value)
    return value


def canonical_result(result):
    value = snapshot(result)
    value["trace"].sort(key=lambda e: (e["time"], e["batch"], e["node"]))
    value["outputs"].sort(key=lambda e: e[:3])
    value["messages"].sort(key=lambda a: (a["batch"], a["node"], a["time"], a["source"], a["position"]))
    return value


def equivalent(a, b, atol, rtol, path="root"):
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor):
            raise AssertionError(path + ": tensor type mismatch")
        torch.testing.assert_close(a, b, atol=atol if a.is_floating_point() else 0,
                                   rtol=rtol if a.is_floating_point() else 0,
                                   equal_nan=False, msg=lambda m: path + ": " + m)
    elif isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise AssertionError(path + ": dictionary key mismatch")
        for k in a:
            equivalent(a[k], b[k], atol, rtol, f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            raise AssertionError(path + ": sequence shape/type mismatch")
        for i, (x, y) in enumerate(zip(a, b)):
            equivalent(x, y, atol, rtol, f"{path}[{i}]")
    elif a != b:
        raise AssertionError(f"{path}: {a!r} != {b!r}")


def atomic_json(path, value):
    path = Path(path)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".result-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    p.add_argument("--send", choices=("dense", "sparse"), default="dense")
    p.add_argument("--topology", choices=("fc", "hb"), default="fc")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--adam-eps", type=float, default=1e-8,
                   help="explicit optimizer recipe; never changes numerical tolerances")
    p.add_argument("--training-steps", type=int, choices=(1, 2), default=2)
    a = p.parse_args()
    if a.device != "cpu" and not a.device.startswith("npu:"):
        raise ValueError("first batch accepts only explicit CPU/NPU devices")
    if a.device.startswith("npu:"):
        if a.dtype != "float32":
            raise ValueError("NPU acceptance uses explicit float32")
        import torch_npu  # site admission must precede this script
        torch.npu.set_device(a.device)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    a.output_dir.mkdir(parents=True, exist_ok=False)
    dtype = getattr(torch, a.dtype)
    atol, rtol = (1e-10, 1e-8) if dtype == torch.float64 else (1e-6, 1e-5)
    task = RetrievalTask(dict(n_keys=4, n_values=4, pairs=2, delay=0))
    options = dict(width=8, topology=a.topology, depth=2 if a.topology == "fc" else 1,
                   region_width=4, send=a.send, temperature=a.temperature, seed=7, dtype=dtype)
    model = Core5Classifier(task.vocab_size, task.value_vocab_size, **options).to(a.device)
    oracle = Core5Classifier(task.vocab_size, task.value_vocab_size, **options)
    report = dict(schema="tide-lab-core5-acceptance-v1", state="running", device=a.device,
                  dtype=a.dtype, strict_tolerance=dict(atol=atol, rtol=rtol),
                  model=model.facts(), task=task.to_dict(), checks={},
                  optimizer=dict(kind="AdamW", lr=1e-3, eps=a.adam_eps, weight_decay=.01),
                  independent_training_steps=[],
                  scope="tiny task integration only; not model learning, scale, native or resident qualification")

    def record(name, fn):
        try:
            detail = fn()
            report["checks"][name] = dict(state="passed", detail=detail)
        except Exception as error:
            report["checks"][name] = dict(state="failed", error=str(error), traceback=traceback.format_exc())
        atomic_json(a.output_dir / "result.json", report)

    def gradients(m):
        return {name: snapshot(param.grad) for name, param in m.named_parameters() if param.requires_grad}

    def train_step(m, opt, x, target, implementation):
        opt.zero_grad(set_to_none=True)
        features = m.embedding(x)
        features.retain_grad()
        result = m.encoder.run_features(features, implementation=implementation, trace=True)
        logits = m.head(m.encoder.project(result, features))
        loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in m.parameters()):
            raise FloatingPointError("nonfinite task loss/gradient")
        observation = dict(result=canonical_result(result), logits=snapshot(logits), loss=snapshot(loss),
                           input_gradient=snapshot(features.grad), gradients=gradients(m))
        opt.step()
        observation.update(parameters=snapshot(m.state_dict()), optimizer=snapshot(opt.state_dict()))
        return observation

    def independent_training():
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3,
                               eps=a.adam_eps, weight_decay=.01)
        ref_opt = torch.optim.AdamW([p for p in oracle.parameters() if p.requires_grad], lr=1e-3,
                                   eps=a.adam_eps, weight_decay=.01)
        expected_ids = {id(p) for p in model.parameters() if p.requires_grad}
        actual = [p for group in opt.param_groups for p in group["params"]]
        assert len(actual) == len({id(p) for p in actual}) and {id(p) for p in actual} == expected_ids
        failures = []
        for step in range(a.training_steps):
            x, y = task.batch("train", 2, step, 43)
            observed = train_step(model, opt, x.to(a.device), y.to(a.device), "eager")
            expected = train_step(oracle, ref_opt, x, y, "reference")
            subchecks = {"step": step}
            for component in observed:
                try:
                    equivalent(observed[component], expected[component], atol, rtol, component)
                    subchecks[component] = dict(state="passed")
                except AssertionError as error:
                    subchecks[component] = dict(state="failed", error=str(error))
                    failures.append(f"step {step} {component}: {error}")
            report["independent_training_steps"].append(subchecks)
        if failures:
            raise AssertionError("\n".join(failures))
        nonzero = [n for n, p in model.named_parameters() if p.grad is not None and bool(p.grad.abs().max() > 0)]
        assert any(n.startswith("encoder.graph_model.") for n in nonzero)
        assert "embedding.weight" in nonzero and "head.weight" in nonzero
        return dict(steps=a.training_steps, nonzero_gradient_parameters=len(nonzero), optimizer="AdamW",
                    complete_state=True, output_routes=True, input_roots=True)

    def shared_gradient_sum():
        model.zero_grad(set_to_none=True)
        aliases = {}
        for name, param in model.named_parameters(remove_duplicate=False):
            aliases.setdefault(id(param), []).append(name)
        shared = [names for names in aliases.values() if len(names) > 1]
        assert shared, "test must contain actual aliased graph parameters"
        x, y = task.batch("train", 2, 3, 43, device=a.device)
        torch.nn.functional.cross_entropy(model(x), y, reduction="sum").backward()
        batch_grad = gradients(model)
        model.zero_grad(set_to_none=True)
        for i in range(2):
            torch.nn.functional.cross_entropy(model(x[i:i+1]), y[i:i+1], reduction="sum").backward()
        equivalent(batch_grad, gradients(model), atol, rtol)
        return dict(shared_parameter_groups=len(shared), samples=2,
                    comparison="one batched loss versus accumulated independent sample losses")

    def inference_schedule_and_causality():
        x, _ = task.batch("val", 2, 0, 71, device=a.device)
        with torch.no_grad():
            features = model.embedding(x)
            eager = model.encoder.run_features(features, trace=True)
            streaming = model.encoder.run_features(features, implementation="streaming", trace=True)
            chunked = model.encoder.run_features(features, chunk_size=1, trace=True)
            equivalent(canonical_result(eager), canonical_result(streaming), atol, rtol)
            equivalent(canonical_result(eager), canonical_result(chunked), atol, rtol)
            prefix = model.encoder.run_features(features[:, :1], trace=True)
            equivalent(snapshot(model.encoder.project(eager, features, position=0)),
                       snapshot(model.encoder.project(prefix, features[:, :1])), atol, rtol)
            last = model.encoder.project(eager, features)
            assert torch.isfinite(last).all()
        return dict(prefill_streaming=True, chunked_continuation=True,
                    causality="position zero invariant to future inputs", inference_shape=list(last.shape))

    def sending_observation():
        x, _ = task.batch("val", 2, 0, 71, device=a.device)
        model.encoder.observe = True
        with torch.no_grad():
            model(x)
        if a.send == "dense":
            assert model.last_metrics["activity/full_fraction"] == 1.0
        return dict(model.last_metrics, scope="observed support for this fixture; sparse is not required to sparsify every input")

    record("independent_complete_training_adamw", independent_training)
    record("shared_parameter_and_sample_gradient_sum", shared_gradient_sum)
    record("inference_causality_prefill_streaming", inference_schedule_and_causality)
    record("sending_support_observation", sending_observation)
    if a.device.startswith("npu:"):
        torch.npu.synchronize()
    report["state"] = "passed" if all(r["state"] == "passed" for r in report["checks"].values()) else "failed"
    atomic_json(a.output_dir / "result.json", report)
    print(json.dumps(dict(state=report["state"], checks={k: v["state"] for k, v in report["checks"].items()})))
    return 0 if report["state"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())

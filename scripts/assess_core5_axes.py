#!/usr/bin/env python3
"""Bounded core-5 assessment with separate numerical and mechanism axes.

The training source is read-only. A finite strict numerical disagreement is
retained and does not skip discrete checks. Only such disagreements may return
zero for an explicitly authorized learning exploration. This is neither an
exact-resume check nor evidence that the model learned the task.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

ATOL, RTOL = 1e-6, 1e-5


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def alias_groups(model):
    groups = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        groups.setdefault(id(param), []).append(name)
    return sorted(sorted(names) for names in groups.values())


class Audit:
    """Visit all comparable leaves; never short-circuit on a float mismatch."""
    def __init__(self):
        self.discrete = dict(leaves=0, mismatches=0, first_errors=[])
        self.numeric = dict(tensor_leaves=0, elements=0, failed_leaves=0, failed_elements=0,
                            max_abs_error=0., reference_max_abs=0., candidate_max_abs=0.,
                            reference_sum_squares=0., error_sum_squares=0., first_failures=[])
        self.nonfinite = []

    def discrete_error(self, path, reason):
        self.discrete["mismatches"] += 1
        if len(self.discrete["first_errors"]) < 8:
            self.discrete["first_errors"].append(dict(path=path, reason=reason))

    def visit(self, candidate, reference, path="root"):
        import torch
        if isinstance(candidate, torch.Tensor) or isinstance(reference, torch.Tensor):
            self.discrete["leaves"] += 1
            if not isinstance(candidate, torch.Tensor) or not isinstance(reference, torch.Tensor):
                self.discrete_error(path, "tensor/type mismatch")
                return
            if candidate.shape != reference.shape or candidate.dtype != reference.dtype:
                self.discrete_error(path, f"shape/dtype mismatch: {candidate.shape}/{candidate.dtype} vs {reference.shape}/{reference.dtype}")
                return
            if not candidate.is_floating_point():
                if not torch.equal(candidate, reference):
                    self.discrete_error(path, "non-floating tensor differs")
                return
            a, b = candidate.double(), reference.double()
            if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
                self.nonfinite.append(path)
                return
            if not a.numel():
                return
            error = (a - b).abs()
            bad = error > ATOL + RTOL * b.abs()
            n = self.numeric
            n["tensor_leaves"] += 1
            n["elements"] += a.numel()
            n["failed_leaves"] += int(bool(bad.any()))
            n["failed_elements"] += int(bad.sum())
            n["max_abs_error"] = max(n["max_abs_error"], float(error.max()))
            n["reference_max_abs"] = max(n["reference_max_abs"], float(b.abs().max()))
            n["candidate_max_abs"] = max(n["candidate_max_abs"], float(a.abs().max()))
            n["reference_sum_squares"] += float(b.square().sum())
            n["error_sum_squares"] += float(error.square().sum())
            if bool(bad.any()) and len(n["first_failures"]) < 8:
                coordinate = tuple(int(i) for i in bad.nonzero()[0])
                n["first_failures"].append(dict(path=path, index=list(coordinate),
                    candidate=float(a[coordinate]), reference=float(b[coordinate]),
                    abs_error=float(error[coordinate]), strict_limit=float(ATOL + RTOL*b[coordinate].abs()),
                    leaf_reference_max_abs=float(b.abs().max()), leaf_max_abs_error=float(error.max())))
            return
        if isinstance(candidate, dict) or isinstance(reference, dict):
            self.discrete["leaves"] += 1
            if not isinstance(candidate, dict) or not isinstance(reference, dict):
                self.discrete_error(path, "mapping/type mismatch")
                return
            if candidate.keys() != reference.keys():
                self.discrete_error(path, "mapping keys differ")
            for key in candidate:
                if key in reference:
                    self.visit(candidate[key], reference[key], f"{path}.{key}")
            return
        if isinstance(candidate, (tuple, list)) or isinstance(reference, (tuple, list)):
            self.discrete["leaves"] += 1
            if type(candidate) is not type(reference):
                self.discrete_error(path, "sequence/type mismatch")
                return
            if len(candidate) != len(reference):
                self.discrete_error(path, "sequence length differs")
            for index, (a, b) in enumerate(zip(candidate, reference)):
                self.visit(a, b, f"{path}[{index}]")
            return
        self.discrete["leaves"] += 1
        if isinstance(candidate, float) and not math.isfinite(candidate) or isinstance(reference, float) and not math.isfinite(reference):
            self.nonfinite.append(path)
        if type(candidate) is not type(reference) or candidate != reference:
            # Python scalars here are configuration/clock/identity fields;
            # computed model values are tensors. Preserve exact semantics.
            self.discrete_error(path, f"scalar differs: {candidate!r} vs {reference!r}")

    def report(self):
        n = dict(self.numeric)
        ref_l2 = math.sqrt(n.pop("reference_sum_squares"))
        error_l2 = math.sqrt(n.pop("error_sum_squares"))
        n.update(reference_l2=ref_l2, error_l2=error_l2,
                 relative_l2_error=None if ref_l2 == 0 else error_l2/ref_l2,
                 zero_reference_scale=ref_l2 == 0)
        return dict(numerical=n, discrete=self.discrete, nonfinite=self.nonfinite)


def positive_support(result):
    """Discrete activity is checked independently of the control magnitudes."""
    import torch
    rows = []
    for event in result["trace"]:
        control = event["control"]
        if not isinstance(control, torch.Tensor) or control.numel() != 1:
            raise AssertionError("weighted control must be a scalar tensor")
        if not bool(torch.isfinite(control)) or bool(control < 0):
            raise AssertionError("invalid nonfinite/negative sending control")
        positive = bool(control > 0)
        if positive != event["active"]:
            raise AssertionError("active flag differs from positive sending support")
        rows.append((event["time"], event["batch"], event["node"], positive))
    return rows


def root_alias_check(model, features):
    from tidegraph.weighted import layered_inputs
    roots = {}
    rows = layered_inputs(model.encoder.spec, features)
    for row in rows:
        roots.setdefault((row.batch, row.position), []).append(id(row.value))
    if any(len(set(ids)) != 1 for ids in roots.values()):
        raise AssertionError("one physical input root must be shared across input ports")
    if len({ids[0] for ids in roots.values()}) != len(roots):
        raise AssertionError("independent sample/position roots were aliased")
    return dict(independent_roots=len(roots), physical_input_records=len(rows),
                scope="actual public input constructor used by the unchanged encoder")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "npu:0"), required=True,
                        help="cpu is for checker development; NPU must be admitted by the external site queue")
    args = parser.parse_args()
    # Register the accelerator before importing the model/checker or creating
    # autograd objects. This script never selects physical devices or queues.
    import torch
    if args.device == "npu:0":
        import torch_npu
        torch.npu.set_device(args.device)
    import check_core5 as helper
    import tide_lab.core5 as model_module
    import tide_lab.tasks as task_module
    from tide_lab.core5 import Core5Classifier
    from tide_lab.tasks import RetrievalTask
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(schema="tide-lab-core5-axes-v1", state="running", config_sha256=sha256(args.config),
        declared_mathematics="0770593838a37c790a568226930567ed60f99b83",
        declared_foundation_commit="85d161d5c9148eac2b164f89a0ba4204b4331898",
        checker_sha256=sha256(__file__), helper_sha256=sha256(helper.__file__),
        model_source_sha256=sha256(model_module.__file__), task_source_sha256=sha256(task_module.__file__),
        runtime=dict(device=args.device, dtype="float32", deterministic=True, threads=1,
            physical_visible=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"), torch=torch.__version__,
            tidegraph=importlib.metadata.version("tidegraph")),
        axes=dict(execution=dict(state="running", completed_training_steps=0),
                  strict_numerical=dict(state="not_checked", atol=ATOL, rtol=RTOL, comparisons=[]),
                  discrete_semantics=dict(state="not_checked"), mechanisms=dict(state="not_checked"),
                  recovery=dict(state="not_checked", required="separate fresh-process application checkpoint check"),
                  learning=dict(state="not_checked", reason="two updates on B2 are mechanism checks")),
        hard_failures=[], first_divergence=None, cause="unknown",
        interpretation="Finite numerical differences are not assumed harmless roundoff. No equivalence, learning or speedup claim.")
    def publish():
        helper.atomic_json(args.output_dir / "result.json", report)
    def hard(reason):
        report["hard_failures"].append(reason)
    def compare(name, candidate, reference, *, mechanism=False):
        audit = Audit()
        audit.visit(candidate, reference, name)
        row = dict(name=name, **audit.report())
        try:
            helper.equivalent(candidate, reference, ATOL, RTOL, path=name)
            row["strict_state"] = "passed"
        except AssertionError as error:
            row.update(strict_state="failed", strict_error=str(error))
        report["axes"]["strict_numerical"]["comparisons"].append(row)
        if row["numerical"]["first_failures"] and report["first_divergence"] is None:
            report["first_divergence"] = dict(comparison=name, **row["numerical"]["first_failures"][0],
                order="first failing observed leaf in step/component and canonical trace order; not an internal arithmetic root cause")
        if row["nonfinite"]:
            hard(dict(comparison=name, kind="nonfinite", paths=row["nonfinite"][:8]))
        if row["discrete"]["mismatches"]:
            hard(dict(comparison=name, kind="discrete_semantics", **row["discrete"]))
        if mechanism and row["strict_state"] != "passed":
            hard(dict(comparison=name, kind="causal_consistency_not_established",
                      reason="A failed causal mechanism gate is not waived by finite-error exploration."))
        publish()
        return row

    def train_step(model, optimizer, x, y, implementation, clip_norm):
        optimizer.zero_grad(set_to_none=True)
        before = {n: helper.snapshot(p) for n, p in model.named_parameters() if p.requires_grad}
        features = model.embedding(x)
        features.retain_grad()
        roots = root_alias_check(model, features)
        result = model.encoder.run_features(features, implementation=implementation, trace=True)
        logits = model.head(model.encoder.project(result, features))
        loss = torch.nn.functional.cross_entropy(logits, y)
        if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite forward logits/loss")
        loss.backward()
        raw_gradients = {n: helper.snapshot(p.grad) for n, p in model.named_parameters() if p.requires_grad}
        norm = torch.nn.utils.clip_grad_norm_(list(p for p in model.parameters() if p.requires_grad),
                                             clip_norm, error_if_nonfinite=True, foreach=False)
        optimizer.step()
        if args.device == "npu:0":
            torch.npu.synchronize()
        result_value = helper.canonical_result(result)
        # Time-major event records appear first, making the observation-order
        # divergence locator useful without pretending to localize arithmetic.
        result_value = {k: result_value[k] for k in ("trace", "outputs", "messages", "continuation")}
        observation = dict(result=result_value, positive_support=positive_support(result_value),
            logits=helper.snapshot(logits), loss=helper.snapshot(loss),
            input_gradient=helper.snapshot(features.grad), gradients=raw_gradients,
            gradient_norm=helper.snapshot(norm), parameters=helper.snapshot(model.state_dict()),
            optimizer=helper.snapshot(optimizer.state_dict()),
            parameter_updates={n: helper.snapshot(p)-before[n] for n, p in model.named_parameters() if p.requires_grad})
        connected = {n: g is not None for n, g in raw_gradients.items()}
        groups = {"graph": [], "embedding": [], "head": []}
        for name, grad in raw_gradients.items():
            if grad is not None:
                if not bool(torch.isfinite(grad).all()):
                    raise FloatingPointError("nonfinite gradient: " + name)
                group = "graph" if name.startswith("encoder.") else "embedding" if name.startswith("embedding.") else "head"
                groups[group].append(grad.double().square().sum())
        norms = {key: math.sqrt(sum(float(v) for v in values)) for key, values in groups.items()}
        if any(value == 0 for value in norms.values()) or features.grad is None:
            raise AssertionError("task loss must reach graph, embedding, head and shared input roots")
        workload = dict(candidates=len(result_value["trace"]),
            full=sum(e["active"] for e in result_value["trace"]),
            messages=len(result_value["messages"]), outputs=len(result_value["outputs"]),
            executor_stats=dict(result.stats))
        return observation, connected, dict(gradient_l2=norms, roots=roots, workload=workload)

    publish()
    try:
        config = json.loads(args.config.read_text())
        if config.get("schema") != "tide-lab-run-v1" or config["model"]["kind"] != "core5":
            raise ValueError("requires a project core5 run JSON")
        task = RetrievalTask(config["task"])
        if (task.config.n_keys, task.config.n_values, task.config.pairs, task.config.delay) != (8, 8, 4, 2):
            raise ValueError("this bounded checker is for the actual A task: 8 keys/8 values/4 pairs/delay 2")
        options = dict(config["model"])
        options.pop("kind")
        options.setdefault("seed", config["seed"])
        if options.get("width") != 16:
            raise ValueError("this bounded checker requires the actual A width 16")
        options["dtype"] = torch.float32
        training = config["training"]
        if training.get("optimizer", "adamw") != "adamw" or training["lr"] != 1e-3 or training.get("eps", 1e-6) != 1e-6:
            raise ValueError("bounded audit recipe requires AdamW lr=1e-3 eps=1e-6")
        recipe = dict(lr=1e-3, eps=1e-6, weight_decay=training.get("weight_decay", 0.), foreach=False)
        report.update(task=task.to_dict(), requested_dtype=config.get("dtype"),
                      resolved=dict(batch=2, updates=2, optimizer=dict(kind="AdamW", **recipe),
                                    clip_norm=training.get("clip_norm", 1.0), data_seed=training["data_seed"],
                                    seed=options["seed"], dtype="float32"))
        oracle = Core5Classifier(task.vocab_size, task.value_vocab_size, **options)
        candidate = Core5Classifier(task.vocab_size, task.value_vocab_size, **options).to(args.device)
        candidate.load_state_dict(oracle.state_dict(), strict=True)
        report["model"] = candidate.facts()
        aliases = alias_groups(oracle)
        if alias_groups(candidate) != aliases or not any(len(g) > 1 for g in aliases):
            raise AssertionError("parameter alias topology changed during device placement")
        optimizers = [torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], **recipe)
                      for m in (candidate, oracle)]
        for model, optimizer in zip((candidate, oracle), optimizers):
            expected = {id(p) for p in model.parameters() if p.requires_grad}
            actual = [p for group in optimizer.param_groups for p in group["params"]]
            if len(actual) != len(expected) or {id(p) for p in actual} != expected:
                raise AssertionError("optimizer must cover each declared unique parameter exactly once")
        report["axes"]["mechanisms"].update(parameter_alias_topology="passed", unique_optimizer_coverage="passed",
            shared_parameter_groups=sum(len(g) > 1 for g in aliases), steps=[],
            reused_evidence=[
                dict(path="artifacts/audit-npu-a/core5/result.json",
                     sha256="10488741d8f909b2ecfdcf9ef002559a639c6004f5d29e54e5b41167374a6a84",
                     check="shared_parameter_and_sample_gradient_sum", state="passed",
                     scope="historical NPU FP32 width8 B2; same public mechanisms, not qualification of width16"),
                dict(path="artifacts/core5-actual-a/result.json",
                     sha256="c2c2bb095090bf99cc0901e621465fdc50236050f29a4415d9e0c790054fb0ff",
                     check="shared_parameter_and_sample_gradient_sum", state="passed",
                     scope="historical CPU FP64 width16 actual task B2; not NPU numerical equivalence")])
        compare("initial_parameters", helper.snapshot(candidate.state_dict()), helper.snapshot(oracle.state_dict()))
        for step in range(2):
            x, y = task.batch("train", 2, step, training["data_seed"])
            actual, connectivity, observations = train_step(candidate, optimizers[0], x.to(args.device), y.to(args.device),
                                                           "eager", training.get("clip_norm", 1.0))
            expected, ref_connectivity, ref_observations = train_step(oracle, optimizers[1], x, y,
                                                                    "reference", training.get("clip_norm", 1.0))
            # Connectivity and routes are checked even when the first floating
            # component already disagrees. All remaining leaves are visited.
            compare(f"step{step}.gradient_connectivity", connectivity, ref_connectivity)
            for name in actual:
                compare(f"step{step}.{name}", actual[name], expected[name])
            if alias_groups(candidate) != aliases or alias_groups(oracle) != aliases:
                raise AssertionError("parameter alias topology changed after optimizer update")
            report["axes"]["mechanisms"]["steps"].append(dict(step=step, candidate=observations, reference=ref_observations))
            report["axes"]["execution"]["completed_training_steps"] = step+1
            publish()
        # Same shape and schedule, only future content is perturbed. A failure
        # of this separate causal gate blocks the path pending diagnosis.
        x, _ = task.batch("val", 2, 0, config["evaluation"]["seed"], args.device)
        candidate.eval()
        with torch.no_grad():
            features = candidate.embedding(x)
            changed = features.clone()
            changed[:, 1:] = features[:, 1:].flip(1) + .125
            ordinary = helper.canonical_result(candidate.encoder.run_features(features, trace=True))
            perturbed = helper.canonical_result(candidate.encoder.run_features(changed, trace=True))
            boundary = candidate.encoder.spec.stride
            prefix = lambda r: dict(trace=[e for e in r["trace"] if e["time"] < boundary],
                                    outputs=[e for e in r["outputs"] if e[1] < boundary],
                                    messages=[e for e in r["messages"] if e["position"] < boundary])
            compare("future_perturbation_causality", prefix(ordinary), prefix(perturbed), mechanism=True)
        rows = report["axes"]["strict_numerical"]["comparisons"]
        numerical_failed = any(row["strict_state"] == "failed" for row in rows)
        report["axes"]["strict_numerical"]["state"] = "failed" if numerical_failed else "passed"
        report["axes"]["discrete_semantics"] = dict(
            state="failed" if any(row["discrete"]["mismatches"] for row in rows) else "passed",
            compared_leaves=sum(row["discrete"]["leaves"] for row in rows),
            coverage="full result/state structure, clocks, integer slots, routes, active flags, emitted keys, control>0 masks, gradient connectivity and optimizer metadata")
        report["axes"]["mechanisms"]["state"] = "failed" if report["hard_failures"] else "passed"
        report["state"] = ("failed" if report["hard_failures"] else
                           "completed_with_strict_failures" if numerical_failed else "passed")
        report["axes"]["execution"]["state"] = "completed"
        return 1 if report["hard_failures"] else 0
    except BaseException as error:
        hard(dict(kind=type(error).__name__, error=str(error), traceback=traceback.format_exc()))
        report["state"] = "failed"
        report["axes"]["execution"]["state"] = "failed"
        return 1
    finally:
        report["elapsed_seconds"] = time.monotonic()-started
        publish()
        print(json.dumps(dict(state=report["state"], axes={k:v["state"] for k,v in report["axes"].items()},
                              hard_failure_count=len(report["hard_failures"]))))


if __name__ == "__main__":
    raise SystemExit(main())

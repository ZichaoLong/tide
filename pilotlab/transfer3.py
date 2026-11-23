"""Read-only, content-bound transfer of complete quad stage-A evidence.

Old tasks, results and checkpoints retain their old source identity. This
module authorizes reuse of their selected recipe/activity evidence only; it
never authorizes loading an old checkpoint into a different source snapshot.
Reports are returned as dictionaries for the caller's review JSON writer.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import re

from .io import sha256

SCHEMA = "pilot-r3-stage-a-transfer-v1"
REVIEW_SCHEMA = "pilot-r3-shannon-repair-review-v1"
TEST_SCHEMA = "pilot-r3-quad-transfer-validation-v1"
MAX_TRANSFER_MODELS = 24
TEST_PATHS = ("forward", "backward", "optimizer", "state")
SHANNON_HELPERS = frozenset({"_two_sum", "_compensated_sum", "_two_product",
                            "_shannon_bound_delta", "_shannon_solve", "_shannon_centered_parts"})


def _load(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _reference(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path)}


def _checked_reference(entry):
    if not isinstance(entry, dict) or not entry.get("path") or _reference(entry["path"]) != entry:
        raise ValueError("transfer evidence file is missing, changed or not content-bound")
    return _load(entry["path"])


class _QuadProjection(ast.NodeTransformer):
    """Conservative specialization of the reviewed allocation patch only.

    The new optional Shannon face argument is always None on quad calls. The
    new ctx.kind is private dispatch metadata, consumed only by backward. All
    other statements, imports, signatures and quad/pool operations must agree.
    This is a narrow structural check, not a general Python equivalence proof.
    """

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id in {"kind", "shannon_faces"}:
            return ast.Constant("quad" if node.id == "kind" else None)
        return node

    def visit_Attribute(self, node):
        if (isinstance(node.ctx, ast.Load) and isinstance(node.value, ast.Name)
                and node.value.id == "ctx" and node.attr == "kind"):
            return ast.Constant("quad")
        return self.generic_visit(node)

    def visit_Compare(self, node):
        node = self.generic_visit(node)
        if len(node.ops) != 1:
            return node
        try:
            left, right = ast.literal_eval(node.left), ast.literal_eval(node.comparators[0])
        except (ValueError, TypeError, SyntaxError):
            return node
        operator = node.ops[0]
        if isinstance(operator, ast.Is):
            if left is not None and right is not None:
                return node
            value = left is right
        elif isinstance(operator, ast.IsNot):
            if left is not None and right is not None:
                return node
            value = left is not right
        elif isinstance(operator, ast.Eq):
            value = left == right
        elif isinstance(operator, ast.NotEq):
            value = left != right
        elif isinstance(operator, ast.In):
            value = left in right
        elif isinstance(operator, ast.NotIn):
            value = left not in right
        else:
            return node
        return ast.Constant(value)

    def visit_BoolOp(self, node):
        node = self.generic_visit(node)
        remaining = []
        for value in node.values:
            if isinstance(value, ast.Constant) and type(value.value) is bool:
                if isinstance(node.op, ast.And) and value.value or isinstance(node.op, ast.Or) and not value.value:
                    continue
                # Preserve evaluation of preceding unknown expressions.
                if not remaining:
                    return value
                remaining.append(value)
                break
            remaining.append(value)
        if not remaining:
            return ast.Constant(isinstance(node.op, ast.And))
        return remaining[0] if len(remaining) == 1 else ast.BoolOp(node.op, remaining)

    def visit_If(self, node):
        node.test = self.visit(node.test)
        if isinstance(node.test, ast.Constant) and type(node.test.value) is bool:
            statements = node.body if node.test.value else node.orelse
            result = []
            for statement in statements:
                visited = self.visit(statement)
                if isinstance(visited, list):
                    result.extend(visited)
                elif visited is not None:
                    result.append(visited)
            return result
        return self.generic_visit(node)

    def visit_IfExp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.test, ast.Constant) and type(node.test.value) is bool:
            return node.body if node.test.value else node.orelse
        return node

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == "shannon_faces":
                if not isinstance(node.value, ast.Constant) or node.value.value is not None:
                    raise ValueError("quad path acquired nonempty Shannon faces")
                return None
            if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                    and target.value.id == "ctx" and target.attr == "kind"):
                if not isinstance(node.value, ast.Constant) or node.value.value != "quad":
                    raise ValueError("quad backward dispatch metadata changed")
                return None
        return node

    def visit_FunctionDef(self, node):
        if node.name == "_save_solution" and node.args.args[-1].arg == "shannon_faces":
            if (len(node.args.args) != 12 or len(node.args.defaults) != 1
                    or not isinstance(node.args.defaults[0], ast.Constant)
                    or node.args.defaults[0].value is not None):
                raise ValueError("unexpected allocation save-solution signature change")
            node.args.args.pop()
            node.args.defaults.pop()
        node = self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) \
                and isinstance(node.body[0].value.value, str):
            node.body.pop(0)
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "_save_solution" and len(node.args) == 12:
            if not isinstance(node.args[-1], ast.Constant) or node.args[-1].value is not None:
                raise ValueError("quad call carries Shannon reconstruction faces")
            node.args.pop()
        return node


def allocation_quad_contract(old_path, new_path):
    """Check the specific Shannon-only patch without executing either source."""
    old, new = (ast.parse(Path(path).read_text()) for path in (old_path, new_path))
    old_names = {node.name for node in old.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    extra = [node for node in new.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name not in old_names]
    if any(not isinstance(node, ast.FunctionDef) or node.name not in SHANNON_HELPERS
           or node.decorator_list or node.args.defaults or node.args.kw_defaults for node in extra):
        raise ValueError("unreviewed top-level allocation additions")
    extra_names = {node.name for node in extra}
    # New helper bodies are checked by their separate Shannon numerical review.
    # No helper may be referenced by the surviving quad/pool projection.
    new.body = [node for node in new.body if node not in extra]
    old, new = (_QuadProjection().visit(tree) for tree in (old, new))
    for tree in (old, new):
        if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) \
                and isinstance(tree.body[0].value.value, str):
            tree.body.pop(0)
    if any(isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in extra_names
           for node in ast.walk(new)):
        raise ValueError("Shannon helpers are reachable from projected quad/pool code")
    old_dump, new_dump = (ast.dump(tree, include_attributes=False) for tree in (old, new))
    if old_dump != new_dump:
        raise ValueError("allocation changes alter the reviewed quad/pool training contract")
    return {"method": "reviewed-shannon-patch-quad-specialization-v1",
            "projection_sha256": hashlib.sha256(old_dump.encode()).hexdigest(),
            "excluded_shannon_helpers": sorted(extra_names), "quad_pool_contract": "unchanged"}


def _source_comparison(old, new):
    from .prepare_campaign3 import CORE_FILES, source_inventory
    for prepared in (old, new):
        identity = prepared.get("source_identity", {})
        if (identity.get("schema") != "pilot-r3-source-v1"
                or not isinstance(identity.get("runtime_packages"), dict)
                or not {"torch", "torch-npu", "numpy"} <= identity["runtime_packages"].keys()
                or not identity.get("python_version") or not identity.get("data_manifest_sha256")):
            raise ValueError("complete frozen source/data/runtime identity required")
        if sha256(Path(prepared["data"]) / "manifest.json") != identity["data_manifest_sha256"]:
            raise ValueError("frozen data manifest changed")
    old_identity, new_identity = old["source_identity"], new["source_identity"]
    for field in ("data_manifest_sha256", "runtime_packages", "python_version"):
        if old_identity[field] != new_identity[field]:
            raise ValueError(f"stage-A transfer changed {field}")
    old_project, old_base = source_inventory(old)
    new_project, new_base = source_inventory(new)
    changed_core = sorted(name for name in CORE_FILES if old_project.get(name) != new_project.get(name))
    if changed_core:
        raise ValueError(f"stage-A training files changed: {changed_core}")
    differences = sorted(name for name in old_base.keys() | new_base.keys() if old_base.get(name) != new_base.get(name))
    if differences != ["allocation.py"]:
        raise ValueError("stage-A transfer requires exactly the reviewed allocation.py foundation change")
    allocation = allocation_quad_contract(Path(old["foundation"]) / "src/tide/allocation.py",
                                          Path(new["foundation"]) / "src/tide/allocation.py")
    return {"unchanged_training_files": {name: old_project[name] for name in sorted(CORE_FILES)},
            "foundation_changes": [{"file": "allocation.py", "old_sha256": old_base["allocation.py"],
                                    "new_sha256": new_base["allocation.py"]}],
            "lifecycle_changes": [{"file": name, "old_sha256": old_project.get(name), "new_sha256": new_project.get(name)}
                                  for name in sorted(old_project.keys() | new_project.keys())
                                  if old_project.get(name) != new_project.get(name)],
            "allocation_contract": allocation}


def _selected_models(selected, model_ids, identity):
    from .catalog3 import model_key
    from .gate3 import MI, combine_activity, continuation_matches, eligible, last_activity, read_evidence
    models, evidence, common_seeds = {}, [], None
    for model_id in model_ids:
        choice = selected.get(model_id)
        if not isinstance(choice, dict):
            raise ValueError(f"old selection does not contain {model_id}")
        groups, spec, activity_rows = {}, None, []
        for field, phase, target, count in (("selection_a1", "a1", MI//4, 2), ("selection_a2", "a2", MI, 2),
                                            ("confirmation", "a3", 2*MI, 2), ("diagnostics", "init", None, 3)):
            group = {}
            entries = choice.get(field)
            if not isinstance(entries, list) or len(entries) != count:
                raise ValueError(f"incomplete old {phase} evidence for {model_id}")
            for entry in entries:
                task, result, item = read_evidence(entry, checkpoint=phase != "init")
                if (phase == "init" and (result.get("checkpoint") or {}).get("path")
                        and Path(result["checkpoint"]["path"]).exists()):
                    task, result, item = read_evidence(entry, checkpoint=True)
                seed = task.get("seed")
                if (task.get("model_id") != model_id or task.get("phase") != phase
                        or target is not None and task.get("target_tokens") != target
                        or task.get("source_identity") != identity or task.get("init_recipe") != choice.get("init_recipe")
                        or phase != "init" and task.get("recipe_id") != choice.get("recipe_id")
                        or type(seed) is not int or seed < 0 or seed in group):
                    raise ValueError(f"old selected task identity/recipe/phase mismatch: {model_id}")
                if spec is None:
                    spec = task["model"]
                if task.get("model") != spec:
                    raise ValueError("old evidence mixes model specifications")
                reasons = eligible(task, result, require_resume=phase in {"a2", "a3"})
                if reasons:
                    raise ValueError(f"old stage-A evidence is not eligible: {model_id}: {reasons}")
                if phase in {"a2", "a3"}:
                    previous = groups["a1" if phase == "a2" else "a2"].get(seed)
                    if previous is None or not continuation_matches(previous, task, result):
                        raise ValueError("old selected evidence lacks exact same-source continuation")
                if phase == "a3":
                    activity_rows.append(last_activity(result, task))
                group[seed] = (task, result)
                evidence.append(item)
            groups[phase] = group
        tuning, diagnostics = sorted(groups["a1"]), sorted(groups["init"])
        if tuning != sorted(groups["a2"]) or tuning != sorted(groups["a3"]) or not set(tuning) <= set(diagnostics):
            raise ValueError("old selected evidence has inconsistent fixed seed groups")
        if common_seeds is None:
            common_seeds = (tuning, diagnostics)
        if common_seeds != (tuning, diagnostics):
            raise ValueError("old selected models use different fixed seed groups")
        if spec.get("family") != "tide" or spec.get("config", {}).get("allocation") != "quad":
            raise ValueError("only explicitly quad Tide subjects can transfer old stage-A evidence")
        models[model_id] = {"model": copy.deepcopy(spec), "model_key": model_key(spec),
                            "activity": combine_activity(activity_rows)}
    if len({row["model_key"] for row in models.values()}) != len(models):
        raise ValueError("transferred model IDs alias an effective model")
    return models, evidence, common_seeds


def _validate_review(review, test, changes, old_identity, new_identity, model_ids):
    difference = changes["foundation_changes"][0]
    if (review.get("schema") != REVIEW_SCHEMA or review.get("status") != "passed"
            or review.get("old_allocation_sha256") != difference["old_sha256"]
            or review.get("new_allocation_sha256") != difference["new_sha256"]
            or review.get("changed_nonshannon_paths") is not False or review.get("quad_pool_contract") != "unchanged"):
        raise ValueError("Shannon repair review does not bind this exact source change")
    if (test.get("schema") != TEST_SCHEMA or test.get("status") != "passed" or test.get("exact") is not True
            or test.get("old_source_identity") != old_identity or test.get("new_source_identity") != new_identity
            or not isinstance(test.get("model_ids"), list) or sorted(test["model_ids"]) != model_ids
            or not isinstance(test.get("tested_paths"), list) or sorted(test["tested_paths"]) != sorted(TEST_PATHS)
            or type(test.get("steps_per_model")) is not int or test["steps_per_model"] < 3):
        raise ValueError("independent quad training validation does not bind the identities/models/three-step contract")
    observations = test.get("observations")
    if not isinstance(observations, dict) or set(observations) != set(model_ids):
        raise ValueError("independent quad validation observations are incomplete")
    for model_id in model_ids:
        row = observations[model_id]
        if not isinstance(row, dict) or not set(TEST_PATHS) <= row.keys():
            raise ValueError("independent quad validation omits a training/state path")
        for path in TEST_PATHS:
            values = row[path]
            old_hash = values.get("old_sha256") if isinstance(values, dict) else None
            if (not isinstance(old_hash, str) or not re.fullmatch("[0-9a-f]{64}", old_hash)
                    or old_hash != values.get("new_sha256")):
                raise ValueError(f"independent quad validation is not exact: {model_id}:{path}")


def build_stage_a_transfer(old_prepared_path, new_prepared_path, old_selected_path, *, model_ids,
                           allocation_review_path, quad_test_evidence_path):
    """Build a verified report; the caller writes it with write_review_json."""
    model_ids = list(model_ids)
    if (not 1 <= len(model_ids) <= MAX_TRANSFER_MODELS or len(set(model_ids)) != len(model_ids)
            or any(not isinstance(mid, str) or not mid for mid in model_ids)):
        raise ValueError("transfer requires 1..24 explicitly listed distinct model IDs")
    model_ids.sort()
    references = {"old_prepared": _reference(old_prepared_path), "new_prepared": _reference(new_prepared_path),
                  "old_selected": _reference(old_selected_path), "allocation_review": _reference(allocation_review_path),
                  "quad_test_evidence": _reference(quad_test_evidence_path)}
    loaded = {name: _checked_reference(entry) for name, entry in references.items()}
    old, new = loaded["old_prepared"], loaded["new_prepared"]
    changes = _source_comparison(old, new)
    models, evidence, seeds = _selected_models(loaded["old_selected"], model_ids, old["source_identity"])
    _validate_review(loaded["allocation_review"], loaded["quad_test_evidence"], changes,
                     old["source_identity"], new["source_identity"], model_ids)
    # Detect mutations during the audit as well as changes seen at entry.
    for entry in references.values():
        _checked_reference(entry)
    return {"schema": SCHEMA, "status": "passed", **references,
            "old_source_identity": old["source_identity"], "new_source_identity": new["source_identity"],
            "model_ids": model_ids, "models": models, "tuning_seeds": seeds[0], "diagnostic_seeds": seeds[1],
            "source_comparison": changes, "evidence": evidence,
            "scope": "selected quad stage-A recipe/activity evidence only; old task/result/checkpoint identities are preserved",
            "formal_initialization": "new source identity, predeclared unseen seed, no stage-A checkpoint load"}


def verify_stage_a_transfer(entry, *, manifest, selected):
    """Revalidate every bound artifact; return the allowed per-model identity."""
    report = _checked_reference(entry)
    if report.get("schema") != SCHEMA:
        raise ValueError("unsupported stage-A transfer report")
    for name in ("old_prepared", "new_prepared", "old_selected", "allocation_review", "quad_test_evidence"):
        _checked_reference(report[name])
    rebuilt = build_stage_a_transfer(report["old_prepared"]["path"], report["new_prepared"]["path"],
        report["old_selected"]["path"], model_ids=report["model_ids"],
        allocation_review_path=report["allocation_review"]["path"],
        quad_test_evidence_path=report["quad_test_evidence"]["path"])
    if report != rebuilt:
        raise ValueError("stage-A transfer report disagrees with its independently re-read evidence")
    if (manifest.get("source_identity") != report["new_source_identity"]
            or sorted(manifest["tuning_seeds"]) != report["tuning_seeds"]
            or sorted(manifest["diagnostic_seeds"]) != report["diagnostic_seeds"]):
        raise ValueError("stage-A transfer source/seed contract differs from the new manifest")
    subjects = {row["id"]: row for row in manifest["subjects"]}
    old_selected = _checked_reference(report["old_selected"])
    for model_id, model in report["models"].items():
        subject = subjects.get(model_id, {})
        if (subject.get("model") != model["model"] or subject.get("effective_key") != model["model_key"]
                or subject.get("model_key", model["model_key"]) != model["model_key"]
                or selected.get(model_id) != old_selected[model_id]):
            raise ValueError("new manifest/selection differs from the explicitly transferred quad evidence")
    return {"model_source_identities": {mid: report["old_source_identity"] for mid in report["model_ids"]},
            "model_ids": report["model_ids"], "report": dict(entry),
            "source_comparison": report["source_comparison"]}

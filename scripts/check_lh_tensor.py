#!/usr/bin/env python3
"""Finite StaticKV → public/default LH column correctness diagnostic.

Same input/port material and unchanged parameters, independent forward/VJPs.
Checks controls, actual candidate/Full support, all terminal raw KV fields and
int64 metadata, per-port cG and one output+raw task derivative. This is not a
performance run, all-root qualification, optimizer or checkpoint-resume test.
Exit 0: strict numerical projection passed; 1: retained finite numerical
mismatch; 2: execution, capacity, nonfinite, structure or discrete failure.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import _planning_path  # noqa: F401
from durable_records import write_json
from source_identity import source_state, digest


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=("many-lh", "many-hb", "lh-1b", "lh-9b"), default="many-lh")
    p.add_argument("--width", type=int)
    p.add_argument("--depth", type=int)
    p.add_argument("--region-width", type=int)
    p.add_argument("--profile", choices=("baseline", "textbook"))
    p.add_argument("--batch", "--B", type=int, default=4)
    p.add_argument("--length", "--L", type=int, default=16)
    p.add_argument("--input-pattern", choices=("dense", "staggered-v1"), default="staggered-v1")
    p.add_argument("--plan", type=Path)
    p.add_argument("--level", choices=("public", "default", "both"), default="both")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    p.add_argument("--client", choices=("python", "native"), default="python")
    p.add_argument("--native-library", "--native-lib")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--max-materialize-bytes", type=int, default=512*2**20)
    p.add_argument("--max-workspace-bytes", type=int, default=2**30)
    p.add_argument("--max-comparison-bytes", type=int, default=2**30)
    p.add_argument("--max-node-rows", type=int, default=262144)
    p.add_argument("--max-pair-elements", type=int, default=2**26)
    p.add_argument("--max-differences", type=int, default=32)
    p.add_argument("--output-dir", "--output", type=Path, required=True)
    a = p.parse_args()
    limits = (a.batch, a.length, a.threads, a.max_materialize_bytes, a.max_workspace_bytes,
              a.max_comparison_bytes, a.max_node_rows, a.max_pair_elements, a.max_differences)
    if min(limits) < 1 or max(limits) >= 2**63 or a.length > 256:
        p.error("positive int64 dimensions/budgets and L<=256 required")
    if a.client == "native" and not a.native_library:
        p.error("native requires its explicit matching library")
    return a


def plan_for(a):
    from prefill_regression_config import CASES
    from tidegraph.planning.recipes import prepare
    if a.plan:
        return json.loads(a.plan.read_text())
    recipe = dict(deepcopy(CASES[a.case]), family="kv")
    for name in ("width", "depth", "region_width", "profile"):
        if getattr(a, name) is not None:
            recipe[name] = getattr(a, name)
    if a.width is not None:
        recipe.pop("target_parameters", None)
    return prepare(recipe, samples=a.batch, positions=a.length, seed=7,
                   input_pattern=None if a.input_pattern == "dense" else a.input_pattern)[0]


def main():
    a = arguments()
    root = Path(__file__).resolve().parents[1]
    output = a.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    source, dirty = source_state(root)
    upstream = json.loads((root/"docs"/"upstream.json").read_text())
    files = ("check_lh_tensor.py", "lh_tensor_check.py", "benchmark_lh_tensor.py", "lh_tensor_baseline.py",
             "lh_tensor_math.py", "lh_tensor_state.py")
    record = dict(schema="tide-lh-tensor-check-v1", source=source, dirty=bool(dirty), execution="running",
        mathematics={k: upstream[k] for k in ("semantic_version", "revision")},
        request={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        script_sha256={name: digest(root/"scripts"/name) for name in files},
        affinity=sorted(os.sched_getaffinity(0)), device_visibility=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        scope="Same parameters and benchmark-generated unit-strength physical inputs/port support. Controls, actual candidate/Full rows, complete raw terminal KV slots/int64 clocks, per-port cG and one output+raw numerical task VJP. Empty left state only. No all-root, edge-content, optimizer, continuation/restore or performance qualification.",
        raw_decode="Read actual candidate ColumnState banks; direct decodes its own padded banks. No recursive/reference fill. The direct inert KV field is an explicit constant-zero invariant.",
        task="1e-6 sum(per-port cG squared) + 1e-3 sum(all terminal floating raw fields squared)",
        gradient_scope="Named unique parameter owners and input; None-as-zero numeric projection. Connection differences reported separately.",
        budgets="Materialization bounds known persistent tensors; node/pair limits bound declared shape. Workspace is per candidate node/group; comparison bytes bound reference plus one candidate CPU capture, not full process RSS/vendor workspace.",
        variants={}, comparisons={})
    write_json(output/"result.json", record)
    exit_code = 2
    try:
        os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
        import torch
        from tidegraph.runtime import resolve_device, manifest
        from tidegraph.planning.materialize import materialize
        from tidegraph.settle import SettleGraph
        from layer_tensor_baseline import numerics
        from benchmark_lh_tensor import material, snapshot
        from lh_tensor_check import evaluate, compare
        torch.set_num_threads(a.threads); torch.set_num_interop_threads(1)
        device, reason = resolve_device(a.device)
        dtype = getattr(torch, a.dtype)
        record.update(runtime=manifest(device, reason, dtype), numerical_policy=numerics(device, "ieee"),
                      tolerance=dict(atol=1e-10 if dtype == torch.float64 else 1e-6,
                                     rtol=1e-8 if dtype == torch.float64 else 1e-5))
        plan = plan_for(a)
        rows = len(plan["graph"]["node_regions"])*plan["samples"]*plan["positions"]
        if not 1 <= plan["positions"] <= 256 or rows > a.max_node_rows or rows*plan["positions"] > a.max_pair_elements:
            raise MemoryError("LH diagnostic node-row/pair-element or L<=256 admission failed")
        record.update(plan=plan, admitted_node_rows=rows, admitted_pair_elements=rows*plan["positions"])
        binding, _, _ = materialize(plan, max_bytes=a.max_materialize_bytes, dtype=dtype, include_inputs=False)
        binding.model.to(device)
        initial, initial_hash, snapshot_bytes = snapshot(binding.model)
        del initial
        host_x, support = material(plan, dtype)
        spec = SettleGraph(binding.graph, tuple(plan["graph"]["ranks"]), binding="ports", time_stride=plan["stride"])
        record.update(initial_parameters_sha256=initial_hash, parameter_snapshot_bytes=snapshot_bytes,
                      parameter_initialization="same seeded FP32 materializer, then explicit requested dtype/device",
                      input_values_and_support_sha256=hashlib.sha256(host_x.numpy().tobytes()+support.numpy().tobytes()).hexdigest(),
                      active_input_positions=int(support.any(0).sum()),
                      trainable_parameters=sum(p.numel() for p in binding.model.parameters() if p.requires_grad))
        if a.native_library:
            record["native_build"] = json.loads((Path(a.native_library)/"build-manifest.json").read_text())
        options = dict(client=a.client, library=a.native_library, workspace_bytes=a.max_workspace_bytes,
                       capture_bytes=a.max_comparison_bytes//2)
        expected, info = evaluate(binding, spec, plan, host_x, support, level="direct", **options)
        record["variants"]["direct"] = info
        captured, after_direct, _ = snapshot(binding.model)
        del captured
        if after_direct != initial_hash:
            raise RuntimeError("direct diagnostic changed parameter material")
        info["unchanged_parameters_sha256"] = after_direct
        write_json(output/"result.json", record)
        levels = ("public", "default") if a.level == "both" else (a.level,)
        for level in levels:
            actual, info = evaluate(binding, spec, plan, host_x, support, level=level, **options)
            record["variants"][level] = info
            record["comparisons"][level+"-direct"] = compare(actual, expected, max_differences=a.max_differences)
            del actual
            captured, final_hash, _ = snapshot(binding.model)
            del captured
            if final_hash != initial_hash:
                raise RuntimeError("diagnostic changed parameter material between independent evaluations")
            record["variants"][level]["unchanged_parameters_sha256"] = final_hash
            write_json(output/"result.json", record)
        checks = record["comparisons"].values()
        exit_code = 2 if any(r["hard_failure"] for r in checks) else 1 if any(r["numerical_failure"] for r in checks) else 0
        record.update(execution="completed", strict_numerical_projection_passed=exit_code == 0,
                      qualification="passed" if exit_code == 0 else "finite_numerical_failure" if exit_code == 1 else "hard_failure")
    except BaseException as error:
        kind = "capacity" if isinstance(error, MemoryError) else "nonfinite" if isinstance(error, FloatingPointError) else getattr(error, "kind", "execution")
        record.update(execution="failed", failure_kind=kind, error=repr(error), strict_numerical_projection_passed=False)
    finally:
        record["exit_code"] = exit_code
        write_json(output/"result.json", record)
    print(json.dumps({k: record[k] for k in ("execution", "exit_code", "strict_numerical_projection_passed")}), flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

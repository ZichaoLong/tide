#!/usr/bin/env python3
"""Standalone query consumer: independent roots and fresh-process exact resume.

Run in the SDK environment, and inside the cooperative queue for NPU. The
fixture contains only static inputs/parameters, never a reference trajectory.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
from durable_records import write_json
from source_identity import digest


def check_query_selection(fixture, actual):
    """Assert default capabilities from static modules, including recurrent-only models."""
    memories = fixture["memories"]
    if len(memories) != len(fixture["nodes"]) or not memories:
        raise ValueError("fixture memory/node domain mismatch")
    kv = sum(memory.get("memory_family", "kv") == "kv" for memory in memories)
    expected = {"query_packed-v2_nodes": kv, "query_legacy_nodes": len(memories)-kv,
                "query_dense-v1_nodes": 0, "describe_batched_nodes": len(memories), "describe_scalar_nodes": 0}
    if any(actual.get(name, 0) != value for name, value in expected.items()):
        raise AssertionError(f"default query/Describe capability mismatch: expected {expected}, got {actual}")
    return expected


def check_selected_state_vjp(expected, actual):
    if actual.get("batched_state_events", 0) < 1:
        raise AssertionError("fixture did not exercise the batched state VJP")
    if expected["query_packed-v2_nodes"]:
        if actual.get("joint_state_vjp_events", 0) < 1:
            raise AssertionError("KV fixture did not exercise the hidden sequence VJP")
        return "kv-hidden-sequence" if not expected["query_legacy_nodes"] else "kv-hidden-and-recurrent-batched"
    if actual.get("joint_state_vjp_events", 0):
        raise AssertionError("recurrent-only fixture reported a KV hidden sequence VJP")
    return "recurrent-batched-state"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "npu:0"), required=True)
    p.add_argument("--dtype", choices=("float32", "float64"), required=True)
    p.add_argument("--return-mode", choices=("complete", "outputs"), default="complete")
    a = p.parse_args()
    fixture = json.loads(a.fixture.read_text())
    out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    base = [str(a.binary.resolve()), "--fixture", str(a.fixture.resolve()),
            "--device", a.device, "--dtype", a.dtype, "--scoring-dtype", a.dtype,
            "--placement", "native" if a.device == "cpu" else "mixed-c",
            "--mode", "training", "--warmup", "0", "--query-execution", "auto",
            "--threads", "1", "--interop-threads", "1", "--checkpoint-check", "true"]
    if a.return_mode != "complete":
        base += ["--return-mode", a.return_mode, "--validation", "none"]
    report = dict(state="running", binary_sha256=digest(a.binary), fixture_sha256=digest(a.fixture),
                  driver_sha256=digest(__file__), device=a.device, dtype=a.dtype, return_mode=a.return_mode,
                  visibility=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"), processes=[])
    def save():
        write_json(out/"result.json", report)
    save()
    try:
        for name, steps in (("whole", 3), ("prefix", 2), ("resumed", 1)):
            command = [*base, "--steps", str(steps), "--output-dir", str(out/name)]
            if name == "whole" and a.return_mode == "complete":
                command += ["--check-roots", "true"]
            if name == "resumed":
                command += ["--resume", str(out/"prefix/checkpoint")]
            with (out/(name+".log")).open("x") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=300)
            row = dict(name=name, command=command, exit_code=result.returncode)
            report["processes"].append(row); save()
            if result.returncode:
                raise RuntimeError(f"{name} failed; see its result.json/log")
            record = json.loads((out/name/"result.json").read_text())
            row["checks"] = {k: record[k] for k in ("state", "independent_comparison", "independent_roots",
                "exact_resume", "parameter_values_checked", "optimizer_values_checked", "final_parameter_gradients_checked")}
            if record["state"] != ("passed" if a.return_mode == "complete" else "completed") or record["exact_resume"]["state"] != "passed":
                raise AssertionError(row["checks"])
            if name == "whole" and a.return_mode == "complete" and record["independent_roots"]["state"] != "passed":
                raise AssertionError("independent query roots were not verified")
            row["query_selection"] = record["metrics"][0]["actual_stats"]
            row["expected_query_selection"] = check_query_selection(fixture, row["query_selection"])
            if a.return_mode != "complete":
                row["state_vjp_scope"] = check_selected_state_vjp(row["expected_query_selection"], row["query_selection"])
        def payload(name):
            path = out/name/"checkpoint"
            return path/json.loads((path/"latest.json").read_text())["generation"]/"state.pt"
        if digest(payload("whole")) != digest(payload("resumed")):
            raise AssertionError("fresh-process checkpoint bytes differ after next update")
        report["fresh_process_resume"] = dict(state="passed", checkpoint_sha256=digest(payload("whole")),
            scope="complete raw graph, parameters and SGD state bytes after next update; per-process roots checked separately")
        command = [*base, "--query-execution", "legacy", "--steps", "1", "--resume",
                   str(out/"prefix/checkpoint"), "--output-dir", str(out/"wrong-policy")]
        rejected = subprocess.run(command, capture_output=True, text=True, timeout=60)
        if not rejected.returncode or "resume model/workload/precision/task/codec identity differs" not in rejected.stderr:
            raise AssertionError("different query policy was not rejected before execution")
        report["cross_policy_rejection"] = dict(state="passed", command=command, exit_code=rejected.returncode)
        report["state"] = "passed"
    except BaseException as error:
        report.update(state="failed", error=repr(error)); raise
    finally:
        save()
    print(json.dumps(report))


if __name__ == "__main__":
    main()

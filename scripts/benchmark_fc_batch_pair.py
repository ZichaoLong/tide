#!/usr/bin/env python3
"""Finite FC16 before/after comparison plus the different-function Transformer.

Use two immutable source checkouts and the same published environment. Each
point runs in a new process; benchmark_dense_query owns the task and timing.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from durable_records import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    p.add_argument("--downstream-repo", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "npu:0"), required=True)
    p.add_argument("--threads", type=int, required=True)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--sdpa-training", action="store_true")
    a = p.parse_args()
    if not 1 <= a.repeat <= 3 or a.threads < 1:
        p.error("repeat must be 1..3 and threads positive")
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sources = {name: path.resolve() for name, path in (("before", a.before), ("after", a.after))}
    identities = {}
    for name, path in sources.items():
        identities[name] = dict(commit=subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip(),
            dirty=subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True).strip())
        if identities[name]["dirty"]:
            raise ValueError("comparison source must be clean: "+str(path))
    record = dict(sources=identities, device=a.device, threads=a.threads, cases=[],
                  scope="FP32 FC16 B8/L128/d128; independent processes, one reset warmup and one measured step per point")
    cases = [(name, mode, "manual") for mode in ("inference", "grad-forward", "training")
             for name in ("before", "after", "transformer")]
    if a.sdpa_training:
        cases.append(("after", "training", "sdpa4d"))
    for repeat in range(a.repeat):
        for name, mode, attention in cases:
            source = sources["before" if name == "before" else "after"]
            label = f"{name}-{mode}-{attention}-r{repeat}"
            argv = [sys.executable, str(source/"scripts/benchmark_dense_query.py"),
                    "--downstream-repo", str(a.downstream_repo.resolve()),
                    "--baseline", "transformer-16" if name == "transformer" else "fc-tensor",
                    "--client", "python", "--device", a.device, "--mode", mode,
                    "--batch", "8", "--length", "128", "--phase-fences", "step",
                    "--threads", str(a.threads), "--formula-attention", attention,
                    "--output-dir", str(out/label)]
            env = dict(os.environ, PYTHONPATH=str(source/"python")+":"+str(source/"scripts"),
                       PYTHONDONTWRITEBYTECODE="1")
            row = dict(label=label, command=argv)
            try:
                with (out/(label+".log")).open("w") as log:
                    result = subprocess.run(argv, env=env, cwd=out, stdout=log,
                                            stderr=subprocess.STDOUT, timeout=240)
                row["exit_code"] = result.returncode
            except subprocess.TimeoutExpired:
                row.update(exit_code=124, error="240-second point timeout")
            result_path = out/label/"result.json"
            if result_path.exists():
                row["result"] = json.loads(result_path.read_text())
            record["cases"].append(row)
            write_json(out/"matrix.json", record)
            phases = row.get("result", {}).get("phases", [])
            measured = next((x for x in phases if not x["warmup"]), {})
            print(json.dumps(dict(case=label, exit_code=row["exit_code"],
                seconds=measured.get("complete_wall_seconds"),
                positions_per_second=measured.get("input_positions_per_second"))), flush=True)
    return int(any(row["exit_code"] for row in record["cases"]))


if __name__ == "__main__":
    raise SystemExit(main())

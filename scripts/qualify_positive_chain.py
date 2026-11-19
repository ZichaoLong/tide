"""Finite positive-chain training/restore cases using the existing checker.

Two D candidates exercise a nontrivial pool/Shannon choice. Submit the immutable
source via scripts/launch.py; the companion primitive tests are separate jobs.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import re
import sys

from scripts import qualify_stable_mass as qualification
from scripts.job import atomic


def configurations():
    original = _original_configurations()
    result = []
    for name, config, precision in original:
        if name.startswith("hb-x-pool"):
            config = replace(config, topology=replace(config.topology, n=2))
        result.append((name, config, precision))
        if name == "hb-x-pool-fp32":
            result.append(("hb-x-shannon-fp32", replace(config, allocation="shannon"), "fp32"))
    return result


_original_configurations = qualification.configurations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "npu:0"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    qualification.configurations = configurations
    sys.argv = [sys.argv[0], "--stage", "training", "--device", args.device,
                "--threads", str(args.threads), "--output-dir", args.output_dir]
    status = qualification.main()
    if args.device.startswith("npu"):
        directory = Path(args.output_dir)
        # The project launcher owns parent C++ stderr; restored processes
        # have individual logs. A tensor on NPU alone is not a placement proof.
        logs = list(directory.rglob("*.log")) + [directory.parent / "stderr.log"]
        observed = [str(p) for p in logs if p.exists() and
                    re.search(r"(?:fall.?back|fall back)[^\n]*cpu", p.read_text().lower())]
        result = directory / "result.json"
        record = json.loads(result.read_text())
        record["fallback_audit"] = dict(logs=[str(p) for p in logs], observed=observed,
                                         parent_log_available=logs[-1].exists())
        if observed or not logs[-1].exists():
            record.update(state="failed", error="NPU placement gate: fallback or missing parent log")
        atomic(result, record)
        if record["state"] == "failed":
            raise RuntimeError(record["error"])
    return status


if __name__ == "__main__":
    raise SystemExit(main())

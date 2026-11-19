"""Finite positive-chain training/restore cases using the existing checker.

Two D candidates exercise a nontrivial pool/Shannon choice. Submit the immutable
source via scripts/launch.py; the companion primitive tests are separate jobs.
"""
import argparse
from dataclasses import replace
import sys

from scripts import qualify_stable_mass as qualification


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
    return qualification.main()


if __name__ == "__main__":
    raise SystemExit(main())

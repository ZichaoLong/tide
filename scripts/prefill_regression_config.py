#!/usr/bin/env python3
"""Generate one explicit prefill regression config for the existing public CLI.

No executor, hidden reference pass or Cartesian sweep lives here. Model scale,
batch, sequence, local modules and execution are separately configurable.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import json
import _planning_path  # noqa: F401
from tidegraph.experiment.config import resolve
from durable_records import write_json

CASES = {
    "wide-fc": dict(topology="fc", width=512, region_width=2, depth=2),
    "many-hb": dict(topology="hb", width=32, region_width=8, depth=3),
    "many-lh": dict(topology="lh", width=64, region_width=8, depth=3, profile="textbook"),
    "sum-fc": dict(topology="fc", width=64, region_width=4, depth=4),
    "layer-chain": dict(topology="fc", width=128, region_width=1, depth=4,
                        region_overrides=dict(region_control_mode="fixed", region_evidence=None,
                                              total_fraction=1., region_control_width=None)),
    "lh-1b": dict(topology="lh", target_parameters=1000000000, region_width=8, depth=3),
    "lh-9b": dict(topology="lh", target_parameters=9000000000, region_width=8, depth=3),
}


def config(case, *, family="kv", samples=4, positions=16, device="cpu", client="python-native",
           placement=None, threads=1, mode="training", schedule="auto", complete=False):
    recipe = dict(deepcopy(CASES[case]), family=family)
    return dict(schema="tide-experiment-v1", name=f"prefill-{case}-{family}",
        description="Finite regression observation. Small independent gates qualify mechanisms; this actual scale has separate numerical/continuation evidence.",
        model=dict(recipe=recipe), task=dict(samples=samples, positions=positions, seed=7,
            input_pattern="staggered-v1", objective=dict(kind="weighted-output-square", scale=.000001)),
        execution=dict(family="auto", client=client, device=device,
            placement=placement or ("cpu" if device == "cpu" else "mixed-c"),
            dtype="float32", scoring_dtype="float32", schedule="prefill", prefill_schedule=schedule,
            packing="packed", return_packing="scalar", return_mode="complete" if complete else "outputs",
            query_execution="auto", workers=1, intraop_threads=threads, interop_threads=1),
        measurement=dict(mode=mode, steps=1, warmup=1, validation="explore" if complete else "none",
                         checkpoint_check=True, timeout_seconds=1800),
        resources=dict(max_materialize_bytes=128*1024**3, checkpoint_max_bytes=128*1024**3))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=CASES, required=True)
    p.add_argument("--family", choices=("kv", "affine", "delta", "gated_delta", "ssd", "mixed"), default="kv")
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--positions", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--client", choices=("pytorch", "python-native", "libtorch"), default="python-native")
    p.add_argument("--placement", choices=("cpu", "mixed-a", "mixed-b", "mixed-c", "resident"))
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--mode", choices=("inference", "training"), default="training")
    p.add_argument("--schedule", choices=("auto", "greedy"), default="auto")
    p.add_argument("--complete", action="store_true", help="full diagnostic return and independent explore comparison")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); args = vars(a); output = args.pop("output")
    if output.exists():
        raise FileExistsError(output)
    value = config(**args)
    resolved = resolve(value)  # Pure planning validates all dimensions before allocation.
    write_json(output, value)
    print(json.dumps(dict(config_id=resolved["config_id"], output=str(output),
                          note="Use tidegraph plan/check/run with an explicit matching site/build.")))


if __name__ == "__main__":
    main()

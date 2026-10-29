#!/usr/bin/env python3
"""Resolve a core-5 model recipe and parameter scale without importing Torch."""
import argparse
import json
from pathlib import Path
import _planning_path  # noqa: F401
from tidegraph.planning.recipes import prepare, FAMILIES
from durable_records import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recipe", type=Path, help="JSON keyword arguments for public model_recipe/scaled_recipe")
    p.add_argument("--topology", choices=("fc", "hb", "lh"), default="hb")
    p.add_argument("--family", choices=FAMILIES, default="kv")
    p.add_argument("--profile", choices=("baseline", "textbook"), default="baseline")
    p.add_argument("--region-width", type=int, default=32)
    p.add_argument("--depth", type=int)
    p.add_argument("--sparse-denominator", type=int, default=8)
    scale = p.add_mutually_exclusive_group()
    scale.add_argument("--target-parameters", type=int, help="e.g. 1000000000, 8000000000, 16000000000")
    scale.add_argument("--width", type=int, default=None)
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--positions", type=int, default=1)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    if a.recipe:
        request = json.loads(a.recipe.read_text())
    else:
        request = {k: getattr(a, k) for k in ("topology", "family", "profile", "region_width", "sparse_denominator")}
        if a.depth is not None:
            request["depth"] = a.depth
        if a.target_parameters is not None:
            request["target_parameters"] = a.target_parameters
        else:
            request["width"] = 512 if a.width is None else a.width
    plan, inv = prepare(request, samples=a.samples, positions=a.positions, seed=a.seed)
    a.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(a.output_dir/"workload.json", plan)
    write_json(a.output_dir/"inventory.json", inv)
    write_json(a.output_dir/"request.json", request)
    print(json.dumps(dict(model_id=plan["model_id"], workload_id=plan["sha256"],
                         parameters=inv["totals"], scale=plan["recipe"]["scale"])))


if __name__ == "__main__":
    main()

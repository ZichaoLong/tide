#!/usr/bin/env python3
"""Generate a parameter-accounted core-5 workload without Torch or tensors."""
import argparse
from pathlib import Path
from durable_records import write_json
from weighted_topology import topology
from weighted_workload_plan import make_plan, LEGACY_TARGETS
from weighted_shapes import SENDING


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--topology", choices=("fc", "hb", "ranked", "dag", "pdg"), default="hb")
    p.add_argument("--family", choices=("kv", "affine", "delta", "gated_delta", "ssd"), default="kv")
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--positions", type=int, default=4)
    p.add_argument("--layers", type=int, default=15)
    p.add_argument("--region-width", type=int, default=2)
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--branches", type=int, default=2)
    for name in ("summary-width", "hidden-width", "ffn-width", "source-rank", "k",
                 "key-width", "value-width", "heads", "groups", "conv-size", "conv-stride"):
        p.add_argument("--"+name, type=int)
    p.add_argument("--adapter-rank", type=int, default=2)
    p.add_argument("--selection", choices=sorted(SENDING), default="regional_total")
    p.add_argument("--memory-read", choices=("sum", "qry", "state"))
    p.add_argument("--source-mode", choices=("region", "node", "source", "adapted"), default="region")
    p.add_argument("--class-policy", choices=("parent", "single"), default="parent")
    p.add_argument("--seed", type=int, default=7)
    targets = p.add_mutually_exclusive_group()
    targets.add_argument("--match", choices=sorted(LEGACY_TARGETS))
    targets.add_argument("--target-parameters", type=int)
    targets.add_argument("--match-core3", choices=("add", "attention"),
                         help="match the core-3 body at this topology and width, excluding task heads")
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    kwargs = vars(a).copy()
    graph = topology(kwargs.pop("topology"), **{k: kwargs.pop(k) for k in ("layers", "region_width", "branches", "depth")})
    out = Path(kwargs.pop("output_dir"))
    match = kwargs.pop("match")
    target = LEGACY_TARGETS[match] if match else kwargs.pop("target_parameters")
    kwargs.pop("target_parameters", None)
    plan, inv = make_plan(graph, **kwargs, target=target)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"workload.json", plan)
    write_json(out/"inventory.json", inv)
    print(plan["sha256"], inv["totals"], plan.get("parameter_match", {}))


if __name__ == "__main__":
    main()

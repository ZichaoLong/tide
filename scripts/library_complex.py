#!/usr/bin/env python3
"""Run exact-topology, reduced-tensor library gates with durable per-case reports."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import torch
from durable_records import write_json
from source_identity import source_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--case", action="append", help="development subset; default is the complete named suite")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "python"))
    os.environ["PYTHONPATH"] = str(root / "python")
    from tidegraph.qualification import qualify
    from library_cases import cases
    available = dict((name,(config,meta)) for name,config,meta in cases())
    selected = list(available) if args.case is None else args.case
    if not selected or set(selected)-available.keys():
        parser.error("unknown or empty case selection")
    source, dirty = source_state(root)
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    record = dict(schema="tide-complex-suite-v1", source=source, dirty=dirty, state="running",
                  scope="complete named suite" if args.case is None else "development subset", cases=[])
    write_json(out / "result.json", record)
    try:
        for name in selected:
            requested, meta = available[name]
            implementations = ("native", "python") if name.startswith(("active", "feedback")) else ("native",)
            for dtype in ("float64", "float32"):
                for implementation in implementations:
                    config = replace(requested, execution=replace(requested.execution, implementation=implementation))
                    case_id = f"{name}-{dtype}-{implementation}"
                    print(case_id, flush=True)
                    report = qualify(config, device="cpu", output_dir=out / case_id, width=4, dtype=dtype,
                                     batch_size=1, positions=2, steps=3,
                                     native_library=Path(args.build_dir).resolve() if implementation == "native" else None)
                    observed = report["coverage"]["observed_nodes"]
                    if "expected_observed_nodes" in meta and observed != meta["expected_observed_nodes"]:
                        raise AssertionError(f"{case_id}: expected complete active topology, got {observed} nodes")
                    record["cases"].append(dict(id=case_id, state="passed", provenance=meta,
                                                requested_sha256=report["requested_sha256"],
                                                effective_sha256=report["effective_sha256"], coverage=report["coverage"],
                                                tolerances=report["tolerances"], checks=report["checks"],
                                                elapsed_seconds=report["elapsed_seconds"]))
                    write_json(out / "result.json", record)
        if source_state(root) != (source, dirty):
            raise RuntimeError("source changed during qualification")
        record["state"] = "passed"
    except BaseException as error:
        record.update(state="failed", error=repr(error))
        raise
    finally:
        write_json(out / "result.json", record)
    print(json.dumps(dict(state=record["state"], cases=len(record["cases"]))))


if __name__ == "__main__":
    main()

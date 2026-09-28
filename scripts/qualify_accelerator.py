#!/usr/bin/env python3
"""Finite device-versus-CPU library gates; never certify an unavailable device."""
import argparse
import json
import os
from pathlib import Path
import sys
import torch
from durable_records import write_json
from source_identity import source_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--implementation", choices=("python", "native"), required=True)
    parser.add_argument("--native-library")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--case", action="append", help="development subset; default runs every named case")
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "python"))
    os.environ["PYTHONPATH"] = str(root / "python")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from tidegraph.qualification import qualify
    from accelerator_cases import cases
    available = dict(cases(args.implementation, args.dtype))
    selected = list(available) if args.case is None else args.case
    if not selected or set(selected) - available.keys():
        parser.error("unknown or empty case selection")
    source, dirty = source_state(root)
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    record = dict(schema="tide-accelerator-suite-v1", source=source, dirty=dirty, state="running",
                  device=args.device, implementation=args.implementation, dtype=args.dtype,
                  scope="complete named suite" if args.case is None else "development subset", cases=[])
    write_json(out / "result.json", record)
    try:
        for name in selected:
            print(name, flush=True)
            report = qualify(available[name], device=args.device, output_dir=out / name,
                             native_library=args.native_library, batch_size=1, positions=2, steps=3,
                             atol=args.atol, rtol=args.rtol)
            record["cases"].append(dict(id=name, state=report["state"], coverage=report["coverage"],
                                        runtime=report["candidate"]["runtime"], checks=report["checks"],
                                        tolerances=report["tolerances"], elapsed_seconds=report["elapsed_seconds"]))
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

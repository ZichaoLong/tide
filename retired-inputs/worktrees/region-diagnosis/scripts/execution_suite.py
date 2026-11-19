"""Finite isolated children for regional execution qualification candidates."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys

from benchmarks.programme import run_child
from scripts.job import atomic, enable_subreaper


VARIANTS = {"recompute": {"checkpoint_regions": True},
            "local_math": {"local_attention_backend": "math"},
            "batch4": {"region_batch_size": 4},
            "combined": {"checkpoint_regions": True, "region_batch_size": 4,
                         "local_attention_backend": "math"}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--configs", nargs="+", default=["configs/hb-multiscale-x-settle.json", "configs/hb-multiscale-x-timed.json"])
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    p.add_argument("--precisions", nargs="+", choices=("fp32","bf16","fp64"), default=["fp32","bf16"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--length", type=int, default=8)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--child-timeout", type=float, default=600)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    record = dict(schema="tide-execution-suite-v1", source=os.environ.get("TIDE_SOURCE_COMMIT"),
                  state="running", physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                  scope="independent numerical children; execution status and finite deviations are separate", children=[])
    persist = lambda: atomic(args.output_dir/"suite.json",record)
    def interrupt(signum, frame):
        raise KeyboardInterrupt(signum)
    signal.signal(signal.SIGTERM,interrupt)
    signal.signal(signal.SIGINT,interrupt)
    enable_subreaper()
    try:
        for config in args.configs:
            for precision in args.precisions:
                for name in args.variants:
                    identity = Path(config).stem+"-"+precision+"-"+name
                    output = args.output_dir/(identity+".json")
                    row = dict(identity=identity, state="planned", output=str(output),
                               log=str(args.output_dir/(identity+".log")), timeout_seconds=args.child_timeout,
                               max_rss_bytes=48*2**30,
                               command=[sys.executable,"-m","scripts.check_execution","--config",config,
                                        "--overrides",json.dumps(VARIANTS[name]),"--device",args.device,
                                        "--precision",precision,"--length",str(args.length),
                                        "--threads",str(args.threads),"--output",str(output)])
                    record["children"].append(row)
                    run_child(row,persist,supervisor_pid=os.getpid())
                    print(identity,row["state"],row.get("inner_state"),flush=True)
        record["state"] = "completed" if all(row["state"]=="completed" for row in record["children"]) else "failed"
    except BaseException as error:
        record.update(state="cancelled" if isinstance(error,KeyboardInterrupt) else "failed", error=repr(error))
        raise
    finally:
        persist()
    return int(record["state"]!="completed")


if __name__ == "__main__":
    raise SystemExit(main())

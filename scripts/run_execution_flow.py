#!/usr/bin/env python3
"""Run a continuous workload through Python, native adapter or standalone LibTorch."""
import argparse
import json
from pathlib import Path
import sys
from durable_records import write_json
from flow_protocol import validate_packet
from flow_failure import RecordedFailure
from flow_resident_options import add_arguments, validate as validate_resident_options


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--family", choices=("pdg", "timed-dag", "settle"), required=True)
    p.add_argument("--implementation", choices=("python", "native", "libtorch"), required=True)
    p.add_argument("--preset", choices=("cpu", "mixed-a", "mixed-b", "mixed-c", "resident"), required=True)
    p.add_argument("--schedule", choices=("streaming", "prefill"), required=True)
    p.add_argument("--dtype", choices=("float32", "float64", "float16"), default="float32")
    p.add_argument("--training", action="store_true")
    p.add_argument("--optimizer", choices=("sgd", "adamw"), default="sgd")
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--windows-per-step", type=int, default=2)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--workers", type=int, default=1, help="native node workers, separate from ATen --threads")
    p.add_argument("--packed-sources", action="store_true", help="use native packed source transport")
    p.add_argument("--batch-next", action="store_true", help="use native batched Next/reset")
    p.add_argument("--sample-chunk-rows", type=int, default=0,
                   help="physical sample maximum; 0 keeps the whole logical batch")
    p.add_argument("--parameter-budget", type=int, default=1024**3)
    p.add_argument("--native-library", type=Path)
    p.add_argument("--native-binary", type=Path)
    p.add_argument("--diagnostics", action="store_true")
    p.add_argument("--phase-timing", action="store_true",
                   help="synchronize before optimizer and record sample-work/update wall times; instrumentation cost is included")
    add_arguments(p)
    for name in ("read", "control", "selection", "events"):
        p.add_argument("--"+name, default="auto")
    p.add_argument("--scoring-dtype", choices=("profile", "payload", "float32", "float64"), default="profile")
    a = p.parse_args()
    if not 1 <= a.threads <= 1024 or not 1 <= a.workers <= 1024:
        p.error("threads and workers must be in [1,1024]")
    if (a.workers != 1 or a.packed_sources or a.batch_next) and (a.implementation == "python" or a.preset == "resident"):
        p.error("host workers/packed-sources/batch-next require an eager native consumer")
    packet = validate_packet(json.loads(a.packet.read_text()))
    if packet["schema"] != "tide-complete-flow-workload-v2":
        p.error("continuous consumer requires v2; legacy v1 declares reset windows")
    if (a.native_binary is not None) != (a.implementation == "libtorch"):
        p.error("--native-binary is required exactly for --implementation libtorch")
    a.output_dir = a.output_dir.resolve()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        validate_resident_options(a)
        if a.implementation == "libtorch":
            from run_flow_native import run as run_native
            result = run_native(packet,a)
        else:
            result = python_run(packet,a)
    except RecordedFailure as error:
        write_json(a.output_dir/"result.json", dict(error.record, state="failed", workload_sha256=packet["sha256"]))
        raise
    except Exception as error:
        write_json(a.output_dir/"result.json", dict(state="failed",workload_sha256=packet["sha256"],error=str(error)))
        raise
    write_json(a.output_dir/"result.json",dict(result,state="passed",threads=a.threads))
    print(json.dumps({k:result[k] for k in ("parameters","seconds","losses","outputs","final_cut")}))


def python_run(packet,a):
    # Help and the standalone launcher never import Torch or its vendor plugin.
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    import torch
    from tidegraph import ExecutionPlacement
    from tidegraph.runtime import resolve_device
    from tools.online_bench.host import run
    from tools.online_bench.records import observer
    from flow_resident_options import python_arguments
    from durable_records import replace_text
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
    device,_ = resolve_device(a.device)
    placement = ExecutionPlacement(a.preset,a.read,a.control,a.selection,a.events,a.scoring_dtype)
    rows=[]
    result=run(packet,family=a.family,implementation=a.implementation,device=device,dtype=a.dtype,
               schedule=a.schedule,preset=a.preset,training=a.training,optimizer=a.optimizer,
               steps=a.steps,warmup=a.warmup,windows_per_step=a.windows_per_step,
               native_library=a.native_library,diagnostics=a.diagnostics,placement=placement,
               workers=a.workers,packed_sources=a.packed_sources,batch_next=a.batch_next,
               phase_timing=a.phase_timing,
               sample_chunk_rows=a.sample_chunk_rows, context_memory_bytes=a.resident_context_bytes,
               parameter_budget=a.parameter_budget,observer=observer(rows) if a.diagnostics else None,
               **python_arguments(a,device))
    if a.diagnostics:
        replace_text(a.output_dir/"diagnostics.jsonl","".join(json.dumps(row,allow_nan=False)+"\n" for row in rows))
    return result


if __name__ == "__main__":
    main()

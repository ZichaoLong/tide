"""Bounded, paired KV compaction and query-bucketing candidates.

The bucket candidate is intentionally benchmark-owned until end-to-end benefit
is established. Neither implementation changes selection or its derivatives.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import types

import torch
import tide.kv as kv
from tide import ModelConfig, TideModel
from tide.precision import precision_context
from benchmarks.continuous import synchronize
from benchmarks.projections import projection_candidate
from scripts.job import atomic


def sorted_compaction(keep):
    positions = torch.arange(keep.shape[2], device=keep.device).expand_as(keep)
    return torch.where(keep, positions, -torch.ones_like(positions)).argsort(dim=2, stable=True)


@contextmanager
def candidate(model, name):
    original_order = kv.cache_order
    originals = []
    try:
        if name == "argsort":
            kv.cache_order = lambda keep, times: sorted_compaction(keep)
        else:
            kv.cache_order = lambda keep, times: kv.chronological_compaction(keep)
        if name == "bucketed":
            for memory in model.memories.values():
                if not isinstance(memory, kv.KVMemory):
                    continue
                original = memory._finish_tile
                originals.append((memory, original))
                def finish(self, u, times, owners, bank, lo, hi, control, _original=original):
                    counts = (control > 0).sum(-1).amax(-1)
                    previous, output = 0, torch.zeros_like(u)
                    # Geometric groups retain B/L vectorization within each
                    # node subset, but pay extra packing and graph submissions.
                    bound = 1
                    while previous < control.shape[2]:
                        upper = min(bound, control.shape[2])
                        selected = (counts > previous) & (counts <= upper)
                        masked = torch.where(selected[:, None, None], control, 0)
                        output = output + _original(u, times, owners, bank, lo, hi, masked)
                        previous, bound = upper, bound*4
                    return output
                memory._finish_tile = types.MethodType(finish, memory)
        projection = name if name in {"head_query", "gate_up", "both"} else "native"
        with projection_candidate(model, projection):
            yield
    finally:
        kv.cache_order = original_order
        for memory, original in originals:
            memory._finish_tile = original


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/fc-k.json")
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--precision",choices=("fp32","bf16"),default="fp32")
    parser.add_argument("--mode",choices=("inference","grad_forward","train"),default="train")
    parser.add_argument("--batch",type=int,default=8)
    parser.add_argument("--length",type=int,default=128)
    parser.add_argument("--threads",type=int,default=8)
    parser.add_argument("--warmup",type=int,default=2)
    parser.add_argument("--steps",type=int,default=5)
    parser.add_argument("--candidates",nargs="+",choices=("argsort","compact","bucketed","head_query","gate_up","both"),
                        default=["argsort","compact","bucketed"])
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    if Path(args.output).exists():
        parser.error("choose a new output identity")
    if args.device.startswith("npu"):
        import torch_npu
        torch.npu.set_device(args.device)
        torch.npu.matmul.allow_hf32=False
        torch.npu.conv.allow_hf32=False
    torch.set_num_threads(args.threads)
    device=torch.device(args.device)
    config=ModelConfig.from_dict(json.loads(Path(args.config).read_text()))
    x_cpu=torch.randn(args.batch,args.length,config.input_dim or config.d,
                      generator=torch.Generator().manual_seed(1108))
    x=x_cpu.to(device)
    source=os.environ.get("TIDE_SOURCE_COMMIT") or subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    record=dict(source=source,arguments=vars(args),torch=str(torch.__version__),
                physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                cpu_affinity=sorted(os.sched_getaffinity(0)),host_load=os.getloadavg(),
                timing="fresh state; forward and full loss/backward/AdamW/sync for train; no construction or audit",rows=[])
    for name in args.candidates:
        torch.manual_seed(1107)
        model=TideModel(config).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,foreach=True) if args.mode=="train" else None
        def step():
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(args.mode!="inference"), precision_context(device,args.precision):
                result=model(x)
                value=result.readout()
                loss=value.float().square().mean() if optimizer is not None else None
            if loss is not None:
                loss.backward(); optimizer.step()
            synchronize(device)
        with candidate(model,name):
            for _ in range(args.warmup):
                step()
            if device.type=="npu":
                torch.npu.reset_peak_memory_stats()
            samples=[]
            for _ in range(args.steps):
                before=time.perf_counter(); step(); samples.append(time.perf_counter()-before)
            with torch.no_grad(),precision_context(device,args.precision):
                observed=model(x,record_work=True)
            row=dict(candidate=name,seconds=samples,median_seconds=statistics.median(samples),
                     work=observed.work,finite=bool(torch.isfinite(observed.readout()).all()),
                     peak_device_bytes=torch.npu.max_memory_allocated() if device.type=="npu" else None)
        record["rows"].append(row); atomic(args.output,record); print(json.dumps(row),flush=True)
        del model,optimizer,observed
        if device.type=="npu":
            torch.npu.empty_cache()
    return int(not all(row["finite"] for row in record["rows"]))


if __name__=="__main__":
    raise SystemExit(main())

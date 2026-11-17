"""Finite, reproducible full-window timing; run each configuration once per workflow.

Example: python -m benchmarks.run --config configs/fc-k.json --device cpu
Train measures zero_grad + model + scalar loss + backward + AdamW + sync.
Inputs and model construction are outside timing. Every window starts from the
bound zero state; this is prefill throughput, not decode or first-token latency.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import time

import torch
from tide import ModelConfig, TideModel
from benchmarks.baselines import TransformerBaseline, SSDBaseline, parameter_count


def synchronize(device):
    if device.type == "npu":
        torch.npu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--length", type=int, default=128)
    p.add_argument("--modes", nargs="+", choices=("inference", "grad_forward", "train"),
                   default=["inference", "grad_forward", "train"])
    p.add_argument("--subjects", nargs="+", choices=("tide", "transformer_layers", "transformer_nodes",
                   "transformer_parameters", "ssd_layers", "ssd_parameters"), default=["tide", "transformer_parameters"])
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=1107)
    args = p.parse_args()
    if min(args.threads,args.batch,args.length,args.steps) < 1 or args.warmup < 0:
        p.error("counts must be positive, warmup nonnegative")
    output = Path(args.output)
    if output.exists():
        p.error("output already exists; choose a new observation identity")
    output.parent.mkdir(parents=True,exist_ok=True)
    if args.device.startswith("npu"):
        import torch_npu
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
    device = torch.device(args.device)
    if device.type=="npu":
        torch.npu.set_device(device)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    cfg = ModelConfig.from_dict(json.loads(Path(args.config).read_text()))
    tide = TideModel(cfg)
    target = parameter_count(tide)
    nodes, layers = len(tide.graph.nodes), len(tide.graph.widths)
    def factory(subject):
        if subject == "tide":
            return TideModel(cfg), {"family":"tide", "config":cfg.to_dict(),
                                   "nodes":nodes,"layers":layers,"edges":len(tide.graph.edges)}
        constructor = TransformerBaseline if subject.startswith("transformer") else SSDBaseline
        count = nodes if subject.endswith("nodes") else layers
        if subject.endswith("parameters"):
            with torch.device("meta"):
                one = constructor(cfg.d,1)
                two = constructor(cfg.d,2)
            per_block = parameter_count(two)-parameter_count(one)
            constant = 2*parameter_count(one)-parameter_count(two)
            count = max(1,round((target-constant)/per_block))
        model = constructor(cfg.d,count)
        return model,model.configuration()

    source = os.environ.get("TIDE_SOURCE_COMMIT") or subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    record = dict(source=source, arguments=vars(args),torch=torch.__version__,
                  physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                  timing="prefill_from_zero; forward includes state/packing/returns; train includes AdamW",
                  input_positions=args.batch*args.length, comparisons_equivalent=False,
                  precision="fp32", observations=[])
    if device.type=="npu":
        record["torch_npu"] = torch_npu.__version__
        record["allow_hf32"] = {"matmul":torch.npu.matmul.allow_hf32,"conv":torch.npu.conv.allow_hf32}
    def persist():
        temporary = output.with_suffix(output.suffix+".tmp")
        with temporary.open("w") as stream:
            json.dump(record,stream,indent=2); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        temporary.replace(output)
    persist()
    for subject in args.subjects:
        for mode in args.modes:
            torch.manual_seed(args.seed)
            model,spec = factory(subject)
            model = model.to(device)
            model.train(mode!="inference")
            input_width = (cfg.input_dim or cfg.d) if subject=="tide" else cfg.d
            x = torch.randn(args.batch,args.length,input_width,device=device)
            opt = torch.optim.AdamW(model.parameters(),lr=1e-4,foreach=True) if mode=="train" else None
            row = dict(subject=subject,mode=mode,spec=spec,parameters=parameter_count(model),state="running",
                       host_load_before=os.getloadavg(),cpu_affinity=sorted(os.sched_getaffinity(0)))
            record["observations"].append(row); persist()
            try:
                def step():
                    if opt is not None:
                        opt.zero_grad(set_to_none=True)
                    with torch.set_grad_enabled(mode!="inference"):
                        result = model(x)
                        value = result.readout() if subject=="tide" else result
                        loss = value.float().square().mean() if opt is not None else None
                    if opt is not None:
                        loss.backward(); opt.step()
                    synchronize(device)
                    return float(loss.detach()) if loss is not None else None
                for _ in range(args.warmup):
                    step()
                if device.type=="npu":
                    torch.npu.reset_peak_memory_stats()
                times=[]
                for _ in range(args.steps):
                    start=time.perf_counter(); loss=step(); times.append(time.perf_counter()-start)
                seconds=statistics.median(times)
                peak = torch.npu.max_memory_allocated() if device.type=="npu" else None
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                with torch.no_grad():
                    audit=model(x,record_work=True) if subject=="tide" else model(x)
                    value=audit.readout() if subject=="tide" else audit
                    if not bool(torch.isfinite(value).all()):
                        raise ArithmeticError("non-finite model output")
                    if opt is not None and (not all(bool(torch.isfinite(p).all()) for p in model.parameters()) or
                            not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())):
                        raise ArithmeticError("non-finite parameters or gradients")
                row.update(state="passed",seconds=times,median_seconds=seconds,
                           input_positions_per_second=args.batch*args.length/seconds,
                           window_latency_ms=1000*seconds,loss=loss,
                           work=audit.work if subject=="tide" else None,
                           process_peak_rss_kib=rss,peak_device_bytes=peak)
                del audit,value
            except Exception as exc:
                row.update(state="failed",error=repr(exc))
                import traceback
                traceback.print_exc()
            persist()
            print(json.dumps(row),flush=True)
            del model,opt,x
            if device.type=="npu":
                torch.npu.empty_cache()
    return int(any(row["state"]!="passed" for row in record["observations"]))


if __name__=="__main__":
    raise SystemExit(main())

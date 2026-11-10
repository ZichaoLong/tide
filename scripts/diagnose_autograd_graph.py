#!/usr/bin/env python3
"""Small public-API graph/FC-formula/Transformer autograd diagnostic.

Run each mode in a fresh process. This deterministic hidden-MSE task is not the
downstream language task: use benchmark_dense_query.py for that frozen task.
All times include observation overhead and are not throughput evidence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from durable_records import write_json


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--baseline", choices=("graph", "fc-tensor", "transformer"), default="graph")
    p.add_argument("--client", choices=("python", "native"), default="python")
    p.add_argument("--native-library")
    p.add_argument("--mode", choices=("nograd", "grad-forward", "backward", "training"), default="backward")
    p.add_argument("--device", default="cpu", help="explicit cpu, cuda[:N] or npu[:N]")
    p.add_argument("--placement", choices=("cpu", "native", "mixed-a", "mixed-b", "mixed-c"))
    p.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    p.add_argument("--width", type=int, default=8)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--region-width", type=int, default=2)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--length", type=int, default=4)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-parameter-bytes", type=int, default=2**30)
    p.add_argument("--packing", choices=("auto", "packed", "scalar"), default="auto")
    p.add_argument("--query-execution", choices=("auto", "legacy", "dense-v1", "packed-v2"), default="auto")
    p.add_argument("--prefill-schedule", choices=("auto", "window", "greedy"), default="auto")
    p.add_argument("--trace-records", choices=("complete", "minimal"), default="complete")
    a = p.parse_args()
    if min(a.width, a.depth, a.region_width, a.batch, a.length, a.threads, a.max_parameter_bytes) < 1:
        p.error("dimensions, capacity and threads must be positive")
    if a.device.split(":")[0] not in {"cpu", "cuda", "npu"}:
        p.error("device must be explicit; no automatic device substitution")
    if a.baseline == "graph" and a.client == "native" and not a.native_library:
        p.error("native graph requires --native-library")
    if a.baseline != "graph" and (a.client != "python" or a.native_library or a.placement):
        p.error("formula/Transformer baselines use Python ATen on --device, without a placement adapter")
    return a


def digest_tensors(values):
    h = hashlib.sha256()
    for name, value in values:
        h.update(name.encode()+b"\0")
        h.update(str((tuple(value.shape), value.dtype)).encode()+b"\0")
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    a = arguments()
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    record = dict(schema="tide-autograd-diagnostic-v1", execution="running", diagnostic_only=True,
                  request={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                  source=git("rev-parse", "HEAD"), worktree_status=git("status", "--porcelain"),
                  affinity=sorted(os.sched_getaffinity(0)), phases_seconds={},
                  correctness="not_checked; use independent semantic qualification",
                  scope="Single fresh step; deterministic hidden mean-square loss, Adam lr=0.001 only in training. Instrumented elapsed times are not formal throughput or allocator peaks.")
    write_json(out/"result.json", record)
    core = None
    try:
        import torch
        import tidegraph
        from tidegraph.autograd_census import Census, tensor_roots
        from tidegraph.runtime import resolve_device, synchronize, manifest
        torch.set_num_threads(a.threads)
        torch.set_num_interop_threads(1)
        device, reason = resolve_device(a.device)
        dtype = getattr(torch, a.dtype)
        record["runtime"] = manifest(device, reason, dtype)
        record["installed_package"] = str(Path(tidegraph.__file__).resolve())
        record["threads"] = dict(intraop=torch.get_num_threads(), interop=torch.get_num_interop_threads())
        def phase(name, function):
            synchronize(device)
            start = time.monotonic()
            value = function()
            synchronize(device)
            record["phases_seconds"][name] = time.monotonic()-start
            return value
        if a.baseline == "transformer":
            from fc_tensor_baseline import TransformerLayers
            torch.manual_seed(a.seed)
            model = TransformerLayers(width=a.width, heads=max(1, a.width//64),
                                      ffn=4*a.width, depth=a.depth, dtype=dtype).to(device)
            record["comparison_scope"] = "Different function: serial pre-RMS/causal SDPA/SwiGLU blocks; no graph/allocation/state equivalence claim."
        else:
            from tidegraph import Continuation, ExecutionPlacement
            from tidegraph.planning.recipes import prepare
            from tidegraph.planning.materialize import materialize
            from tidegraph.settle import SettleGraph
            from tidegraph.weighted import layered_inputs, readout
            from tidegraph.weighted.execution import eager_runner
            plan, _ = prepare(dict(topology="fc", family="kv", width=a.width, depth=a.depth,
                region_width=a.region_width, sparse_denominator=2,
                memory_overrides=dict(memory_read="qry"),
                region_overrides=dict(allocation_temperature=.25)), samples=a.batch, positions=a.length, seed=a.seed)
            write_json(out/"workload.json", plan)
            bound, _, _ = materialize(plan, max_bytes=a.max_parameter_bytes, dtype=dtype)
            model = bound.model.to(device)
            spec = SettleGraph(bound.graph, tuple(plan["graph"]["ranks"]), binding="ports", time_stride=plan["stride"])
            record["workload_sha256"] = plan["sha256"]
            if a.baseline == "graph":
                placement = ExecutionPlacement(preset=a.placement or ("cpu" if device.type == "cpu" else "mixed-c"), scoring_dtype=a.dtype)
                record["placement"] = dict(request=placement.to_dict(), resolved={k: str(v) for k, v in placement.resolve(device).items()})
                if a.client == "native":
                    from tidegraph.native_loader import load_native
                    core = load_native(a.native_library, backend=device.type)
                    if not all(hasattr(core, n) for n in ("reset_autograd_census", "register_autograd_roots", "autograd_census_metrics")):
                        raise RuntimeError("native census requires a matching build with private-graph instrumentation")
                    record["native_build"] = json.loads((Path(core.__file__).parent/"build-manifest.json").read_text())
                execute = eager_runner(bound, a.client, placement, native_library=a.native_library,
                    packing=a.packing, query_execution=a.query_execution, prefill_schedule=a.prefill_schedule,
                    trace=a.trace_records == "complete")
                record["comparison_scope"] = "Public weighted graph execution; all returned tensor fields and explicit private roots observed."
            else:
                record["comparison_scope"] = "Independent FC formula with same core parameters/input; output/terminal-state projection only, no full-record/None-root equivalence claim."
        model.train()
        record["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        record["initial_parameters_sha256"] = digest_tensors(model.state_dict().items())
        enabled = a.mode != "nograd"
        generator = torch.Generator(device="cpu").manual_seed(a.seed+1)
        x = (.2*torch.randn((a.batch, a.length, a.width), generator=generator, dtype=dtype)).to(device).requires_grad_(enabled)
        record["input_sha256"] = digest_tensors([("input", x)])
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.001) if a.mode == "training" else None
        if core is not None:
            core.reset_autograd_census(True)
        with torch.set_grad_enabled(enabled), Census() as census:
            if a.baseline == "graph":
                inputs = phase("input_records", lambda: layered_inputs(spec, x))
                result = phase("forward", lambda: execute(Continuation(bound.graph.identity, a.batch),
                    inputs, a.length*spec.stride, sealed_until=a.length*spec.stride))
                def project():
                    return torch.stack([torch.stack([readout(result.outputs,
                        [(p*spec.stride+spec.rank(n), port) for port, n in enumerate(bound.graph.outputs)],
                        x[b, p], batch=b) for p in range(a.length)]) for b in range(a.batch)])
                hidden = phase("readout", project)
                record["work"] = dict(result.stats)
            elif a.baseline == "fc-tensor":
                from fc_tensor_baseline import run
                result = phase("forward", lambda: run(bound, spec, x))
                hidden = result["hidden"]
                record["work"] = result["stats"]
            else:
                result = hidden = phase("forward", lambda: model(x))
                record["work"] = dict(block_positions=a.batch*a.length*a.depth)
            loss = phase("loss", lambda: hidden.square().mean())
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite diagnostic loss")
            def register():
                roots = list(tensor_roots((result, hidden, loss)))
                census.register(roots, label="outer/returned-and-task")
                if core is not None:
                    core.register_autograd_roots("outer/returned-and-task", roots)
            phase("census_register", register)
            record["before_backward"] = census.snapshot()
            if a.mode in {"backward", "training"}:
                phase("backward", loss.backward)
                record["parameter_gradients"] = dict(connected=sum(p.grad is not None for p in model.parameters()),
                    disconnected=sum(p.grad is None for p in model.parameters()))
            if optimizer is not None:
                phase("optimizer", optimizer.step)
            record["loss"] = float(loss.detach())
            record["after_step"] = census.snapshot()
            record["native"] = None if core is None else dict(core.autograd_census_metrics())
            record["census_scope"] = "Python/native universes overlap and cannot be summed. Native saved memory covers explicit saves only; Python saved fields cover visible Nodes only. All root owners remain live during the after_step snapshot."
        record["final_parameters_sha256"] = digest_tensors(model.state_dict().items())
        record["execution"] = "completed"
    except BaseException as error:
        record.update(execution="failed", error=repr(error))
        raise
    finally:
        if core is not None:
            core.reset_autograd_census(False)
        write_json(out/"result.json", record)
    print(json.dumps(dict(execution=record["execution"], output=str(out/"result.json"),
                         nodes=record["after_step"]["total"]["unique_nodes"])), flush=True)


if __name__ == "__main__":
    main()

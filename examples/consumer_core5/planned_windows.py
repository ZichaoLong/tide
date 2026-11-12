#!/usr/bin/env python3
"""Installed planned-graph client: real AdamW windows and fresh-process resume.

Accepts a resolved public planning record (FC/HB/LH, including heterogeneous
local configurations). Every child independently materializes the model and
uses only public default execution/checkpoint APIs. This validates lifecycle,
not mathematical equivalence, learning quality or timed throughput.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import torch
from tidegraph import ExecutionPlacement
from tidegraph.checkpoint import save, load
from tidegraph.comparison import equivalent
from tidegraph.finite import all_finite
from tidegraph.runtime import resolve_device, manifest
from tidegraph.settle import SettleGraph
from tidegraph.planning.materialize import materialize
from tidegraph.planning.input_pattern import coordinates
from tidegraph.weighted import eager_runner
from tidegraph.weighted.column_layout import SequenceLayout
from tidegraph.weighted.message_columns import MessageColumns
from recording import fingerprint, snapshot, json_record, tensor_record, sha256


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--client", choices=("python", "native"), default="python")
    p.add_argument("--native-library")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    p.add_argument("--placement", choices=("mixed-a", "mixed-b", "mixed-c"), default="mixed-c")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--max-materialize-bytes", type=int, default=2**30)
    p.add_argument("--max-workspace-bytes", type=int, default=2**30)
    p.add_argument("--child-timeout", type=int, default=900)
    p.add_argument("--deterministic-algorithms", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--save-checkpoint", action="store_true")
    p.add_argument("--child", action="store_true")
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    if min(a.steps, a.threads, a.max_materialize_bytes, a.max_workspace_bytes, a.child_timeout) < 1:
        p.error("positive steps, threads, byte and duration budgets required")
    if not 0 < a.learning_rate < 1:
        p.error("finite learning rate in (0,1) required")
    if a.client == "native" and not a.native_library:
        p.error("native requires its explicit matching library")
    if not a.child and (a.resume or a.steps < 2):
        p.error("parent needs at least two steps; resume is a child operation")
    return a


def input_window(plan, ref, index):
    """Physical port schedules are static input material, independent of execution."""
    b, length, d = plan["samples"], plan["positions"], plan["width"]
    x = .2*torch.sin(torch.arange(d)*.19+.3*torch.arange(length)[:, None]
                    +.1*torch.arange(b)[:, None, None]+.013*index)
    x = x.to(dtype=ref.dtype, device=ref.device).requires_grad_()
    support = [[] for _ in plan["graph"]["inputs"]]
    for sample, position, port, _ in coordinates(plan):
        support[port].append(sample*length+position)
    layout = SequenceLayout(torch.arange(b+1, dtype=torch.int64)*length,
                            (torch.arange(length, dtype=torch.int64)+index*length).repeat(b))
    ports = {}
    for p, entries in enumerate(support):
        rows = torch.tensor(sorted(entries), dtype=torch.int64)
        ports[p] = MessageColumns(rows, x.new_ones(len(rows)),
            x.reshape(-1, d).index_select(0, rows.to(x.device)))
    return x, layout, ports


def create(plan, a):
    binding, _, _ = materialize(plan, dtype=getattr(torch, a.dtype),
                                max_bytes=a.max_materialize_bytes, include_inputs=False)
    binding.model.to(a.device)
    spec = SettleGraph(binding.graph, tuple(plan["graph"]["ranks"]),
                       binding="ports", time_stride=plan["stride"])
    opt = torch.optim.AdamW(binding.model.parameters(), lr=a.learning_rate,
                           betas=(.9, .95), weight_decay=.01, foreach=False)
    execute = eager_runner(binding, a.client,
        ExecutionPlacement("cpu" if torch.device(a.device).type == "cpu" else a.placement,
                           scoring_dtype=a.dtype), native_library=a.native_library, trace=False)
    return binding, spec, opt, execute


def step(owner, plan, state, index, workspace, *, capture=True):
    binding, spec, opt, execute = owner
    x, layout, ports = input_window(plan, binding.model.nodes[0].bias, index)
    opt.zero_grad(set_to_none=True)
    result = execute.run_ports(spec, layout, ports, state=state,
        stop_position=(index+1)*plan["positions"], max_workspace_bytes=workspace)
    if result.columns is None or result.execution["schedule"] != "settle-columns-v1":
        raise AssertionError("planned lifecycle requires actual column execution: "+str(result.execution))
    terms = [(m.strength[:, None]*m.content).square().sum()*1e-6 for m in result.outputs.values()]
    loss = sum(terms)
    loss.backward()
    gradients = {name: p.grad for name, p in binding.model.named_parameters()}
    floating = [loss, *(g for g in (x.grad, *gradients.values()) if g is not None)]
    if not all_finite(floating):
        raise FloatingPointError("nonfinite planned training result")
    state = result.continuation.detach()
    opt.step()
    if not all_finite(binding.model.parameters()):
        raise FloatingPointError("nonfinite optimizer update")
    observed = (snapshot(dict(outputs=result.outputs, state=state, input_gradient=x.grad,
        gradients=gradients, parameters=binding.model.state_dict(), optimizer=opt.state_dict()))
        if capture else None)
    return state, observed, dict(step=index, loss=float(loss.detach().cpu()),
                                counts=result.stats, execution=result.execution)


def identity(plan, a, runtime):
    result = dict(schema="tide-planned-window-consumer-v1", plan_sha256=fingerprint(plan),
        client=a.client, device=a.device, dtype=a.dtype,
        placement="cpu" if torch.device(a.device).type == "cpu" else a.placement,
        deterministic_algorithms=a.deterministic_algorithms, threads=a.threads,
        optimizer=dict(name="AdamW", lr=a.learning_rate, betas=[.9, .95], weight_decay=.01, foreach=False),
        scripts={p.name: sha256(p) for p in (Path(__file__), Path(__file__).with_name("recording.py"))},
        runtime=runtime)
    if a.native_library:
        result["native_manifest_sha256"] = sha256(Path(a.native_library)/"build-manifest.json")
    return result


def child(a, plan, out):
    device, reason = resolve_device(a.device)
    runtime = manifest(device, reason, getattr(torch, a.dtype))
    ident = identity(plan, a, runtime)
    owner = create(plan, a)
    binding, spec, opt, _ = owner
    state, start = None, 0
    if a.resume:
        meta = json.loads(a.resume.with_suffix(".json").read_text())
        if meta["identity"] != ident:
            raise ValueError("planned resume identity differs")
        state = load(a.resume, binding.graph, binding.model, opt)
        start = meta["completed_steps"]
        if type(start) is not int or start < 1 or state.cut != start*plan["positions"]*spec.stride:
            raise ValueError("planned checkpoint step/cut differs")
    reports = []
    for index in range(start, start+a.steps):
        state, observed, report = step(owner, plan, state, index, a.max_workspace_bytes,
                                       capture=index == start+a.steps-1)
        reports.append(report)
    tensor_record(out/"observation.pt", observed)
    if a.save_checkpoint:
        save(out/"checkpoint.pt", binding.graph, binding.model, state, opt)
        json_record(out/"checkpoint.json", dict(identity=ident, completed_steps=start+a.steps))
    return dict(identity=ident, steps=reports, actual_parameters=sum(p.numel() for p in binding.model.parameters()
        if p.requires_grad), observation_sha256=sha256(out/"observation.pt"))


def parent(a, out):
    common = [sys.executable, str(Path(__file__).resolve()), "--child", "--plan", str(a.plan.resolve()),
        "--client", a.client, "--device", a.device, "--dtype", a.dtype, "--placement", a.placement,
        "--threads", str(a.threads), "--learning-rate", str(a.learning_rate),
        "--max-materialize-bytes", str(a.max_materialize_bytes), "--max-workspace-bytes", str(a.max_workspace_bytes)]
    if a.native_library:
        common += ["--native-library", str(Path(a.native_library).resolve())]
    if not a.deterministic_algorithms:
        common += ["--no-deterministic-algorithms"]
    actions = (("whole", ["--steps", str(a.steps)]),
               ("prefix", ["--steps", str(a.steps-1), "--save-checkpoint"]),
               ("resumed", ["--steps", "1", "--resume", str(out/"prefix/checkpoint.pt")]))
    children = {}
    for name, extra in actions:
        with (out/(name+".log")).open("w") as log:
            process = subprocess.run([*common, *extra, "--output-dir", str(out/name)],
                stdout=log, stderr=subprocess.STDOUT, timeout=a.child_timeout)
        children[name] = dict(exit_code=process.returncode,
                             result=json.loads((out/name/"result.json").read_text()))
        if process.returncode:
            raise RuntimeError("planned child failed: "+name)
    whole = torch.load(out/"whole/observation.pt", weights_only=True)
    resumed = torch.load(out/"resumed/observation.pt", weights_only=True)
    equivalent(whole, resumed, atol=0, rtol=0)
    return dict(children=children, exact_fresh_process_resume="passed",
                comparison="actual final outputs, raw continuation/pending/history/ledger, independent gradients including None, parameters and all AdamW slots")


def main():
    a = arguments()
    plan = json.loads(a.plan.read_text())
    torch.set_num_threads(a.threads); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(a.deterministic_algorithms)
    out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    record = dict(execution="running", plan=plan, request={k: str(v) if isinstance(v, Path) else v
        for k, v in vars(a).items()}, scope="public planned-column lifecycle; no independent math or throughput claim")
    code = 1
    try:
        record.update(child(a, plan, out) if a.child else parent(a, out))
        record["execution"], code = "completed", 0
    except BaseException as error:
        record.update(execution="failed", error=repr(error))
    finally:
        record["exit_code"] = code
        json_record(out/"result.json", record)
    print(json.dumps({k: record[k] for k in ("execution", "exit_code")}))
    return code


if __name__ == "__main__":
    sys.exit(main())

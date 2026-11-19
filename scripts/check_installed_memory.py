"""Five memory candidates through an isolated wheel and fresh-process recovery.

Run with the installed environment's ``python -I`` from outside the checkout.
This CPU FP32 mechanism check imports only public Tide APIs, Torch and stdlib;
it is not a throughput benchmark or a downstream learning-quality test.
"""
import argparse
from dataclasses import fields, is_dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback


TOLERANCE = dict(atol=1e-6, rtol=2e-3)
START = 2**53+17
CASES = dict(x={}, s=dict(memory="S"), reverse=dict(memory="X_reverse"),
             narrow32=dict(kv_width=32), memory_region=dict(memory_sharing="region_shared"))


def atomic(path, value):
    path = Path(path)
    descriptor, stage = tempfile.mkstemp(dir=path.parent, prefix=".installed-memory-")
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(stage, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(stage):
            os.unlink(stage)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def leaves(value, path="root"):
    if is_dataclass(value):
        for field in fields(value):
            yield from leaves(getattr(value, field.name), path+"."+field.name)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from leaves(item, path+"."+str(key))
    elif isinstance(value, tuple) and hasattr(value, "_fields"):
        for name, item in zip(value._fields, value):
            yield from leaves(item, path+"."+name)
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from leaves(item, path+"."+str(index))
    else:
        yield path, value


def fingerprint(value):
    digest = hashlib.sha256()
    for name, item in sorted(leaves(value)):
        digest.update(name.encode()+b"\0")
        if isinstance(item, torch.Tensor):
            digest.update(str((tuple(item.shape), item.dtype)).encode()+b"\0")
            digest.update(item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            digest.update(repr((type(item).__name__, item)).encode())
    return digest.hexdigest()


def compare(actual, expected, *, atol=0., rtol=0.):
    observed, reference = dict(leaves(actual)), dict(leaves(expected))
    result = dict(exact=True, finite=True, metadata_failures=[], support_failures=[],
                  numerical_failures=[], max_abs=0., atol=atol, rtol=rtol)
    if observed.keys() != reference.keys():
        result["metadata_failures"].append("leaf names")
    for name in sorted(observed.keys() & reference.keys()):
        a, b = observed[name], reference[name]
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if a.shape != b.shape or a.dtype != b.dtype:
                result["metadata_failures"].append(name+": shape/dtype")
                continue
            result["exact"] &= torch.equal(a, b)
            if not a.is_floating_point():
                if not torch.equal(a, b):
                    result["metadata_failures"].append(name+": integer/bool")
                continue
            finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
            result["finite"] &= finite
            if not finite:
                result["numerical_failures"].append(dict(path=name, nonfinite=True))
                continue
            field = name.rsplit(".", 1)[-1]
            if field in ("strengths", "weight", "short_weight", "long_weight") and (
                    field == "strengths" or ".state." in name):
                if not torch.equal(a > 0, b > 0):
                    result["support_failures"].append(name)
            if a.numel():
                absolute = float((a.double()-b.double()).abs().max())
                result["max_abs"] = max(result["max_abs"], absolute)
                if not torch.allclose(a, b, atol=atol, rtol=rtol):
                    result["numerical_failures"].append(dict(path=name, max_abs=absolute))
        elif type(a) is not type(b) or a != b:
            result["metadata_failures"].append(name+": scalar/None")
    result["exact"] &= not result["metadata_failures"]
    result["within_threshold"] = (result["finite"] and not result["metadata_failures"]
                                  and not result["numerical_failures"] and not result["support_failures"])
    return result


def installation(expected_root):
    import tide
    if not sys.flags.isolated:
        raise RuntimeError("run this checker using python -I")
    checkout = Path(__file__).resolve().parents[1]
    if Path.cwd().resolve().is_relative_to(checkout):
        raise RuntimeError("run from outside the source checkout")
    root = Path(expected_root or sys.prefix).resolve()
    package = Path(tide.__file__).resolve().parent
    if not package.is_relative_to(root):
        raise RuntimeError("loaded Tide package lies outside expected package root: "+str(package))
    modules = []
    for name, module in sorted(sys.modules.items()):
        if name == "tide" or name.startswith("tide."):
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(package):
                raise RuntimeError("loaded Tide module lies outside installed package: "+name)
            modules.append(dict(module=name, path=str(path), sha256=sha(path)))
    return dict(package_root=str(package), expected_root=str(root), loaded_modules=modules,
                package_files={str(path.relative_to(package)): sha(path) for path in sorted(package.glob("*.py"))},
                python=sys.executable, prefix=sys.prefix, torch_version=str(torch.__version__),
                isolated=sys.flags.isolated, cwd=str(Path.cwd()), threads=torch.get_num_threads(),
                vendor_imported="torch_npu" in sys.modules, precision="fp32", device="cpu")


def state_checks(model, result, previous, length):
    start = START if previous is None else previous.position
    expected = (torch.arange(length, dtype=torch.int64)+start)*model.graph.period+len(model.layer_nodes)
    if result.times.dtype != torch.int64 or not torch.equal(result.times, expected):
        raise AssertionError("public result lost original int64 times")
    clock_fields = []
    for name, item in leaves(result.state):
        if name.rsplit(".", 1)[-1] == "times":
            if not isinstance(item, torch.Tensor) or item.dtype != torch.int64:
                raise AssertionError("private memory/conv time is not int64: "+name)
            clock_fields.append(name)
    if not clock_fields or result.state.position != start+length:
        raise AssertionError("private clocks or continuation position are absent")
    return dict(position=result.state.position, time_first=int(result.times[0]),
                time_last=int(result.times[-1]), private_int64_fields=len(clock_fields))


def snapshot(model, optimizer, state):
    return dict(parameters=model.state_dict(), optimizer=optimizer.state_dict(), state=state)


def initial_state(model):
    from tide import ModelState
    # An empty private-memory dictionary per region also composes with regional
    # batching; None denotes each memory module's initial history.
    return ModelState(START, tuple({kind: None for kind in model.memories} for _ in model.graph.regions), ())


def binding_checks(model):
    c = model.config
    widths = dict(content=c.d, summary=c.d//4, ffn=c.ffn_width or 4*c.d,
                  kv=None if "K" not in model.memories else c.kv_width or c.d)
    ownership = {}
    for kind, memory in model.memories.items():
        if memory.summary.out_features != widths["summary"] or memory.ffn.gate.out_features != widths["ffn"]:
            raise AssertionError("installed summary/FFN width differs from serialized configuration")
        if memory.ffn.gate.shared or memory.ffn.up.shared or memory.ffn.down.shared:
            raise AssertionError("memory-only sharing changed independent FFN ownership")
        if kind == "K" and (memory.kv.out_features != 2*widths["kv"] or
                len(memory.query.heads) != widths["kv"]//32 or memory.output.in_features != widths["kv"]):
            raise AssertionError("installed KV binding did not narrow Q/K/V together")
        for name, module in memory.named_modules():
            if hasattr(module, "shared") and not name.startswith("ffn."):
                if module.shared != (c.memory_sharing == "region_shared"):
                    raise AssertionError("installed memory ownership differs from configuration")
                ownership[kind+"."+name] = dict(shared=module.shared, bases=len(set(module._host_groups)))
    if any((".left" in name or ".right" in name) and not name.startswith("memories.")
           for name, _ in model.named_parameters()):
        raise AssertionError("memory-only sharing added an adapter outside memory")
    return dict(widths=widths, memory_ownership=ownership,
                parameter_dtypes=sorted({str(parameter.dtype) for parameter in model.parameters()}))


def update(model, optimizer, data, state):
    from tide import detach_state
    from tide.precision import precision_context
    initial = initial_state(model) if state is None else state
    optimizer.zero_grad(set_to_none=True)
    x = data.detach().clone().requires_grad_()
    with precision_context("cpu", "fp32"):
        result = model(x, initial)
        value = result.readout()
        root = torch.cos(torch.arange(value.numel(), dtype=value.dtype)).reshape_as(value)
        loss = (value*root).square().mean()
    loss.backward()
    gradients = {name: None if parameter.grad is None else parameter.grad.detach().clone()
                 for name, parameter in model.named_parameters()}
    if x.grad is None or not bool(x.grad.abs().sum() > 0):
        raise AssertionError("input has no nonzero training gradient")
    connected = sum(gradient is not None and bool(gradient.abs().sum() > 0) for gradient in gradients.values())
    if not connected:
        raise AssertionError("no nonzero parameter gradient")
    optimizer.step()
    following = detach_state(result.state)
    observed = dict(values=result.values.detach(), strengths=result.strengths.detach(), times=result.times,
                    input_gradient=x.grad, parameter_gradients=gradients, **snapshot(model, optimizer, following))
    bad = [name for name, item in leaves(observed) if isinstance(item, torch.Tensor)
           and item.is_floating_point() and not bool(torch.isfinite(item).all())]
    if bad:
        raise ArithmeticError("nonfinite update fields: "+str(bad))
    clocks = state_checks(model, result, state, data.shape[1])
    return following, observed, dict(loss=float(loss.detach()), nonzero_parameter_gradient_fields=connected,
                                    input_gradient_max=float(x.grad.abs().max()), **clocks)


def child(args):
    from tide import load_checkpoint
    directory = args.output_dir.resolve()
    manifest = json.loads((directory/"boundary.json").read_text())
    model, state, slots = load_checkpoint(directory/"checkpoint.pt", device="cpu")
    binding = binding_checks(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
    optimizer.load_state_dict(slots)
    loaded = fingerprint(snapshot(model, optimizer, state))
    config_exact = model.config.to_dict() == manifest["config"]
    if loaded != manifest["snapshot_sha256"] or not config_exact:
        raise AssertionError("restored configuration or checkpoint data differ from saved parent boundary")
    data = torch.load(directory/"next-input.pt", map_location="cpu", weights_only=True)
    state, observed, summary = update(model, optimizer, data, state)
    expected = torch.load(directory/"expected-next-update.pt", map_location="cpu", weights_only=False)
    comparisons = {name: compare({name: observed[name]}, {name: expected[name]}, **TOLERANCE) for name in observed}
    structural = any(not row["finite"] or row["metadata_failures"] for row in comparisons.values())
    support = any(row["support_failures"] for row in comparisons.values())
    classification = ("structural_failed" if structural else "support_deviation" if support else
                      "within_threshold" if all(row["within_threshold"] for row in comparisons.values()) else "finite_deviation")
    report = dict(state=classification, checkpoint_data_exact=True, config_exact=config_exact,
                  checkpoint_sha256=sha(directory/"checkpoint.pt"), completed_update=4,
                  input_sha256=sha(directory/"next-input.pt"), update=summary, comparisons=comparisons,
                  binding=binding, installation=installation(args.expected_package_root))
    atomic(directory/"resume.json", report)
    return 1 if structural or support else 0


def parent(args):
    from tide import ClockConfig, ModelConfig, TideModel, TopologyConfig, save_checkpoint
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    report = dict(schema="tide-installed-memory-1", state="running", installation=installation(args.expected_package_root),
                  script_sha256=sha(__file__), scope="CPU FP32 isolated installed package; synthetic mechanism objective",
                  initial_position=START, batch=1, length=4, saved_updates=3, restored_update=4,
                  tolerance=TOLERANCE, cases=[])
    atomic(directory/"report.json", report)
    base = ModelConfig(topology=TopologyConfig(kind="multiscale", profile="compact-5-dd4", n=2, hubs=1),
                       d=192, memory="X", sharing="independent", adapter_rank=8,
                       clock=ClockConfig(kv_window=8), attention_backend="math", region_batch_size=2)
    for name, changes in CASES.items():
        case_dir = directory/name
        case_dir.mkdir()
        row = dict(candidate=name, state="running")
        report["cases"].append(row)
        try:
            torch.manual_seed(141)
            model = TideModel(replace(base, **changes))
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
            inputs = [torch.randn(1, 4, 192) for _ in range(4)]
            initial = initial_state(model)
            with torch.inference_mode():
                inference = model(inputs[0], initial)
            tracked = model(inputs[0].clone().requires_grad_(), initial)
            mode = compare(dict(values=tracked.values, strengths=tracked.strengths, times=tracked.times, state=tracked.state),
                           dict(values=inference.values, strengths=inference.strengths, times=inference.times, state=inference.state),
                           **TOLERANCE)
            if not tracked.values.requires_grad:
                raise AssertionError("gradient-enabled forward did not record a graph")
            state_checks(model, inference, None, 4)
            del inference, tracked
            row.update(config=model.config.to_dict(), parameters=sum(p.numel() for p in model.parameters()),
                       binding=binding_checks(model), inference_grad_forward=mode, updates=[])
            state = None
            for data in inputs[:3]:
                state, training_observation, summary = update(model, optimizer, data, state)
                row["updates"].append(summary)
                del training_observation
            save_checkpoint(case_dir/"checkpoint.pt", model, state=state, optimizer=optimizer)
            atomic(case_dir/"boundary.json", dict(config=model.config.to_dict(),
                   snapshot_sha256=fingerprint(snapshot(model, optimizer, state))))
            torch.save(inputs[3], case_dir/"next-input.pt")
            _, observed, summary = update(model, optimizer, inputs[3], state)
            torch.save(observed, case_dir/"expected-next-update.pt")
            row["parent_fourth_update"] = summary
            del model, optimizer, state, observed
            command = [sys.executable, "-I", str(Path(__file__).resolve()), "--child", "--output-dir", str(case_dir)]
            if args.expected_package_root:
                command += ["--expected-package-root", str(args.expected_package_root)]
            environment = dict(os.environ)
            environment.pop("PYTHONPATH", None)
            with (case_dir/"resume.log").open("w") as log:
                process = subprocess.run(command, cwd="/", env=environment, stdout=log, stderr=subprocess.STDOUT,
                                         timeout=120)
            row.update(resume_command=command, resume_exit_code=process.returncode)
            if not (case_dir/"resume.json").is_file():
                raise RuntimeError("fresh interpreter did not produce resume.json; inspect resume.log")
            row["resume"] = json.loads((case_dir/"resume.json").read_text())
            row["state"] = row["resume"]["state"]
            if not mode["finite"] or mode["metadata_failures"]:
                row["state"] = "structural_failed"
            elif process.returncode and row["state"] not in {"structural_failed", "support_deviation"}:
                row["state"] = "execution_failed"
            elif mode["support_failures"]:
                row["state"] = "support_deviation"
            elif not mode["within_threshold"] and row["state"] == "within_threshold":
                row["state"] = "finite_deviation"
        except Exception as error:
            row.update(state="execution_failed", error=str(error), traceback=traceback.format_exc())
        atomic(directory/"report.json", report)
    states = {row["state"] for row in report["cases"]}
    failed = bool(states & {"structural_failed", "execution_failed", "support_deviation"})
    report["state"] = "failed" if failed else "finite_deviation" if "finite_deviation" in states else "within_threshold"
    report["installation"] = installation(args.expected_package_root)
    atomic(directory/"report.json", report)
    print(json.dumps(dict(state=report["state"], report=str(directory/"report.json"))))
    return int(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path, help="new output directory (parent mode)")
    parser.add_argument("--expected-package-root", type=Path, help="required installation prefix; defaults to sys.prefix")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    global torch
    import torch
    torch.set_num_threads(1)
    installation(args.expected_package_root)
    return child(args) if args.child else parent(args)


if __name__ == "__main__":
    raise SystemExit(main())

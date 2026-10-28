#!/usr/bin/env python3
"""Original W32/B4/four-position PDG witness: four windows, native schedules.

Independent scalar Python remains the correctness oracle. No timing, tolerance
changes, trajectory replay or promotion other than an explicit --dtype float64.
"""
import argparse
from pathlib import Path
import sys
import torch
from durable_records import write_json
from source_identity import source_state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native-library", type=Path, required=True)
    p.add_argument("--dtype", choices=("float32", "float64"), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/"python"))
    from tidegraph import Continuation
    from tidegraph.compare import equivalent
    from tidegraph.reference import run as scalar
    from tidegraph.placement import ExecutionPlacement
    from tidegraph.checkpoint import save, load
    from weighted_benchmark_cases import Workload, fixture, fingerprint, window
    from weighted_benchmark_precision import promote
    from weighted_benchmark_runners import loss_from_outputs, work
    from weighted_device_checks import runner
    from weighted_numeric_differences import tensors, first_trace_difference
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    native = str(args.native_library.resolve())
    out = args.output_dir.resolve();out.mkdir(parents=True, exist_ok=False)
    config = Workload("pdg", "kv", width=32, samples=4, positions=4)
    source, dirty = source_state(root)
    report = dict(source=source, dirty=dirty, dtype=args.dtype, state="running", steps=[],
                  scope="original feedback workload, complete records/gradients/SGD slots and per-schedule exact resume; no throughput")

    def check(a, b, exact=False):
        try:
            equivalent(a, b, **(dict(atol=0, rtol=0) if exact else {}))
        except AssertionError as error:
            return str(error)
        return None

    def create(schedule):
        bound, xs, span = fixture(config)
        identity = fingerprint(bound, xs, config)
        bound, xs = promote(bound, xs, getattr(torch, args.dtype))
        execute = ((lambda *a, **kw: scalar(bound.graph, bound.model, *a, **kw)) if schedule == "reference" else
                   runner(bound, "native", ExecutionPlacement(), native, streaming=schedule == "streaming"))
        opt = torch.optim.SGD(bound.model.parameters(), lr=.001, momentum=.3, weight_decay=.01)
        return dict(binding=bound, inputs=xs, span=span, identity=identity, execute=execute,
                    optimizer=opt, continuation=Continuation(bound.graph.identity, config.samples))

    def step(owner, index):
        b, opt = owner["binding"], owner["optimizer"]
        opt.zero_grad(set_to_none=True)
        span = owner["span"]
        result = owner["execute"](owner["continuation"],
            window(owner["inputs"], span, config.positions, index, "cpu"),
            (index+1)*span, sealed_until=(index+1)*span)
        loss_from_outputs(result).backward()
        grads = {n: None if t.grad is None else t.grad.detach().clone() for n, t in b.model.named_parameters()}
        opt.step();owner["continuation"] = result.continuation.detach()
        return result, grads

    try:
        schedules = ("reference", "streaming", "prefill")
        owners = [create(s) for s in schedules]
        report["fixture_sha256"] = owners[0]["identity"]
        assert len({o["identity"] for o in owners}) == 1
        resumes = []
        for i in range(4):
            results = [step(o, i) for o in owners]
            row = dict(step=i, candidates=[])
            for name, owner, (actual, grads) in zip(schedules[1:], owners[1:], results[1:]):
                row["candidates"].append(dict(schedule=name, work=work(actual),
                    records=check(results[0][0], actual), gradients=check(results[0][1], grads),
                    parameters=check(owners[0]["binding"].model.state_dict(), owner["binding"].model.state_dict()),
                    optimizer=check(owners[0]["optimizer"].state_dict(), owner["optimizer"].state_dict()),
                    gradient_detail=tensors(results[0][1], grads),
                    first_exact_trace_difference=first_trace_difference(results[0][0], actual)))
            if resumes:
                row["exact_resume"] = []
                for original, resumed, (expected, grads) in zip(owners, resumes, results):
                    actual, rg = step(resumed, i)
                    row["exact_resume"].append(dict(records=check(expected, actual, True), gradients=check(grads, rg, True),
                        parameters=check(original["binding"].model.state_dict(), resumed["binding"].model.state_dict(), True),
                        optimizer=check(original["optimizer"].state_dict(), resumed["optimizer"].state_dict(), True)))
                resumes = []
            if i == 2:
                for name, owner in zip(schedules, owners):
                    b = owner["binding"]
                    save(out/(name+".pt"), b.graph, b.model, owner["continuation"], owner["optimizer"])
                    restored = create(name);r = restored["binding"]
                    restored["continuation"] = load(out/(name+".pt"), r.graph, r.model, restored["optimizer"])
                    equivalent(owner["continuation"], restored["continuation"], atol=0, rtol=0)
                    resumes.append(restored)
                row["exact_checkpoint_decode"] = True
            report["steps"].append(row);write_json(out/"result.json", report)
        failures = [c[k] for r in report["steps"] for c in r["candidates"]
                    for k in ("records", "gradients", "parameters", "optimizer")]
        failures += [v for r in report["steps"] for c in r.get("exact_resume", []) for v in c.values()]
        report["state"] = "strict_mismatch" if any(e is not None for e in failures) else "passed"
        assert source_state(root) == (source, dirty)
    except BaseException as error:
        report.update(state="failed", error=repr(error));raise
    finally:
        write_json(out/"result.json", report)
    raise SystemExit(0 if report["state"] == "passed" else 2)


if __name__ == "__main__":
    main()

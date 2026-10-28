#!/usr/bin/env python3
"""Compare two KV projection formulas on actual scalar-execution operands.

This observes local arithmetic only; no trace is replayed into a candidate.
FP64 is an explicit diagnostic for both FP32 formulas, not a replacement run.
"""
import argparse
import hashlib
from pathlib import Path
import sys
import torch
from durable_records import write_json
from source_identity import source_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/"python"))
    from weighted_plan_materialize import load, materialize
    from weighted_benchmark_cases import window
    from tidegraph import Continuation
    from tidegraph.reference import run
    from tidegraph.weighted.kv import KVMemory
    from tidegraph.weighted.tensors import Received, norm
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    plan, config = load(args.workload.resolve())
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    source, dirty = source_state(root)
    report = dict(schema="tide-kv-projection-diagnostic-v1", source=source, dirty=dirty,
        workload_sha256=plan["sha256"], state="running", dtype="float32", calls=0,
        scope="local equal operands, dense/factored formulas and explicit FP64 diagnostic; no throughput qualification",
        strict_local_mismatches=0, maxima=dict(formula_difference=0., dense_error=0., factored_error=0.))
    bound, xs, span = materialize(plan, max_bytes=1024**3)
    nodes = {id(w.kernel): v for v, w in enumerate(bound.model.nodes)}
    original = KVMemory.step
    def observe(memory, w, old, content, time):
        h = Received.decode(content.value)
        if h.mass > 0:
            x = norm(h.content, memory.config.epsilon)
            base, u, v = memory.base, memory.adapter_u, memory.adapter_v
            dense = (base+u@v)@x
            factored = base@x+u@(v@x)
            exact = base.double()@x.double()+u.double()@(v.double()@x.double())
            errors = dict(formula_difference=float((dense-factored).abs().max()),
                          dense_error=float((dense.double()-exact).abs().max()),
                          factored_error=float((factored.double()-exact).abs().max()))
            report["calls"] += 1
            report["maxima"] = {k: max(report["maxima"][k], e) for k, e in errors.items()}
            if not torch.allclose(dense, factored, atol=1e-6, rtol=1e-5):
                report["strict_local_mismatches"] += 1
                if "first" not in report:
                    report["first"] = dict(node=nodes[id(memory)], time=time, **errors)
                    torch.save(dict(base=base.detach().clone(), u=u.detach().clone(), v=v.detach().clone(),
                                    x=x.detach().clone(), dense=dense, factored=factored, fp64=exact), out/"operands.pt")
                    report["operands_sha256"] = hashlib.sha256((out/"operands.pt").read_bytes()).hexdigest()
        return original(memory, w, old, content, time)
    try:
        KVMemory.step = observe
        with torch.no_grad():
            run(bound.graph, bound.model, Continuation(bound.graph.identity, config.samples),
                window(xs, span, config.positions, 0, "cpu"), span, sealed_until=span)
        report["state"] = "diagnosed"
    except BaseException as error:
        report.update(state="failed", error=repr(error)); raise
    finally:
        KVMemory.step = original
        write_json(out/"result.json", report)
    print(report)


if __name__ == "__main__":
    main()

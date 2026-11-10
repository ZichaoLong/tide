#!/usr/bin/env python3
"""Bounded same-material first-Describe comparison for a full graph witness.

This diagnoses one root, not complete training/route/restore qualification.
Use a fixed installation/native build and the matching tests as --fixtures.
"""
import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixtures', type=Path, default=Path(__file__).resolve().parents[1]/'tests')
    p.add_argument('--native-library', type=Path, required=True)
    p.add_argument('--family', choices=('affine', 'delta', 'gated_delta', 'ssd', 'kv'), default='affine')
    p.add_argument('--topology', choices=('dag', 'pdg', 'hb'), default='dag')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    sys.path.insert(0, str(a.fixtures.resolve()))
    from test_component_window_weighted import fixture
    from test_weighted_execution import variables
    from isolated_cases import vjp
    from tidegraph import Continuation, ExecutionPlacement
    from tidegraph.reference import run as reference
    from tidegraph.weighted.execution import eager_runner
    from tidegraph.comparison import collect_comparison
    from durable_records import write_json
    import tidegraph
    base, inputs, stop = fixture(torch.float32, a.family, a.topology, 'cpu')
    values, gradients = {}, {}
    for precision in ('float32', 'float64'):
        for implementation in ('scalar', 'python', 'native'):
            b = replace(base, model=copy.deepcopy(base.model).to(getattr(torch, precision)))
            xs = [replace(x, value=x.value.detach().to(getattr(torch, precision)).clone().requires_grad_()) for x in inputs]
            q = Continuation(b.graph.identity, 2)
            if implementation == 'scalar':
                result = reference(b.graph, b.model, q, xs, stop, sealed_until=stop)
            else:
                run = eager_runner(b, implementation, ExecutionPlacement(scoring_dtype=precision),
                                   native_library=str(a.native_library))
                result = run(q, xs, stop, sealed_until=stop)
            name = implementation+'-'+precision
            root = result.trace[0]['descriptor']['evidence']
            values[name] = root.detach()
            gradients[name] = vjp(root, variables(b, xs))
    comparisons = {}
    distances = {}
    for precision in ('float32', 'float64'):
        for implementation in ('python', 'native'):
            comparisons[implementation+'-'+precision] = collect_comparison(
                gradients[implementation+'-'+precision], gradients['scalar-'+precision])
    for implementation in ('scalar', 'python', 'native'):
        low, high = gradients[implementation+'-float32'], gradients['scalar-float64']
        distances[implementation] = {n: float((g.double()-high[n]).abs().max())
                                    for n, g in low.items() if g is not None and g.numel()}
    checksums = {n: hashlib.sha256((a.fixtures/n).read_bytes()).hexdigest()
                 for n in ('test_component_window_weighted.py', 'test_weighted_recurrent_execution.py', 'test_weighted_execution.py')}
    record = dict(family=a.family, topology=a.topology, material='same FP32 parameters and input leaves promoted to FP64; int64 metadata unchanged',
                  scope='first trace evidence squared-norm VJP only; full training/restore checked separately',
                  package=str(Path(tidegraph.__file__).resolve()), fixtures=checksums,
                  native_build=json.loads((a.native_library/'build-manifest.json').read_text()),
                  comparisons=comparisons, fp32_distance_from_scalar64_by_root=distances)
    write_json(a.output, record)
    print(json.dumps(dict(comparisons=comparisons, max_fp32_distance={k:max(v.values()) for k,v in distances.items()})))


if __name__ == '__main__':
    main()

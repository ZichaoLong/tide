#!/usr/bin/env python3
"""Export actual static inputs/parameters for the independent C++ consumer.

This finite benchmark schema is not a graph/checkpoint interchange API. It
does no forward execution and exports no chosen action, route or batch.
"""
import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import torch
from durable_records import write_json
from source_identity import source_state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native-library", required=True)
    p.add_argument("--topology", choices=("hb", "dag", "pdg"), required=True)
    p.add_argument("--family", choices=("kv", "ssd"), required=True)
    p.add_argument("--size", choices=("small", "medium"), required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root/"python"))
    from tidegraph.native_loader import load_native
    from tidegraph.native_model import encode_model
    from weighted_benchmark_cases import Workload, fixture, fingerprint, limits
    shape = dict(width=4, samples=1, positions=2) if args.size == "small" else dict(width=32, samples=4, positions=4)
    config = Workload(args.topology, args.family, **shape)
    binding, inputs, span = fixture(config)
    core = load_native(args.native_library, backend="cpu")
    graph, model = encode_model(core, binding.graph, binding.model)
    tensors, identities = [], {}

    def tensor(x):
        key = x._cdata  # Actual TensorImpl alias, never equal-value deduplication.
        if key not in identities:
            identities[key] = len(tensors)
            tensors.append(dict(shape=list(x.shape), dtype="float32", trainable=x.requires_grad,
                                values=x.detach().reshape(-1).tolist()))
        return identities[key]

    weights = []
    for node in model.nodes:
        row = {name: tensor(getattr(node, name)) for name in ("decay", "weight", "bias", "read")}
        row["extra"] = {name: tensor(x) for name, x in node.extra.items()}
        weights.append(row)
    region_weights = [{name: tensor(x) for name, x in row.extra.items()} for row in model.regions]
    scales = {name: [tensor(x) for x in getattr(model, name)]
              for name in ("input_scale", "output_scale", "edge_scale", "agg_scale")}
    external = [dict(batch=x.batch, port=x.port, position=x.position, time=x.time, value=tensor(x.value)) for x in inputs]
    path = Path(args.output).resolve()
    if path.exists():
        raise FileExistsError(path)
    source, dirty = source_state(root)
    write_json(path, dict(schema="tide-weighted-benchmark-fixture-v1", source=source, dirty=dirty,
        config=asdict(config), span=span, fixture_sha256=fingerprint(binding, inputs, config),
        graph=dict(nodes=[asdict(x) for x in binding.graph.nodes], edges=[asdict(x) for x in binding.graph.edges],
                   regions=[asdict(x) for x in binding.graph.regions], inputs=list(binding.graph.inputs), outputs=list(binding.graph.outputs),
                   layout=asdict(binding.graph.ports), domain=asdict(binding.graph.domain), native_identity=graph.identity),
        memories=[asdict(x) for x in binding.memories], regions=[asdict(x) for x in binding.regions],
        classes=asdict(binding.source_classes), alpha_agg=.2, tensors=tensors, nodes=weights,
        region_weights=region_weights, scales=scales, inputs=external, limits=limits(binding, config, span)))
    print(path)


if __name__ == "__main__":
    main()

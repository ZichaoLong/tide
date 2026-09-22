#!/usr/bin/env python3
"""Freeze graph generator/config inputs and build a bounded original-LH graph."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types
from durable_records import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--profile', choices=('wide-add', 'narrow-add'), required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32:
        parser.error('seed must fit uint32')
    source, out = Path(args.source).resolve(), Path(args.output_dir).resolve()
    # Historical config bytes are clues, not claims that these were the user's runs.
    revision = 'ed4ba40' if args.profile == 'wide-add' else '81e702a'
    cfg_bytes = subprocess.check_output(['git', 'show', revision+':Connectome/cpp/test/cfg.json'], cwd=source)
    full_revision = subprocess.check_output(['git', 'rev-parse', revision], cwd=source, text=True).strip()
    model = json.loads(cfg_bytes)
    levels = [0, 1, 7, 224] if args.profile == 'wide-add' else [0, 1, 7, 56, 448, 57344]
    local = 32 if args.profile == 'wide-add' else 128
    graph_config = dict(base_num_or_hpnums=levels, d=2, levelnum=len(levels)-2,
                        localnum=local, region='circle', localscaling=1.5,
                        cross_level=2, edge_direction_strategy='bidirection')
    out.mkdir(parents=True, exist_ok=False)
    package = out/'generator'/'PyConnectome'
    package.mkdir(parents=True)
    hashes = {}
    for name in ('Graph.py', 'BaseUtils.py'):
        data = (source/'PyConnectome'/name).read_bytes()
        (package/name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    (out/'historical-model.json').write_bytes(cfg_bytes)
    # Load only graph generation and its configuration helper, not the LH Python interpreter.
    module = types.ModuleType('PyConnectome')
    module.__path__ = [str(package)]
    sys.modules['PyConnectome'] = module
    os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
    import numpy as np
    import scipy
    import sklearn
    import networkx
    import torch
    torch.set_num_threads(1)
    np.random.seed(args.seed)
    graph = importlib.import_module('PyConnectome.Graph')
    gd = graph.GraphData.create_from(graph.GraphConfig(**graph_config), enable_self_check=True)
    gd.savebytes(str(out/'graph'))
    matrices = {name: getattr(gd, name) for name in ('inputA', 'outputA', 'ioA', 'oiA')}
    edges = {name: int(value.nnz) for name, value in matrices.items()}
    n, width, vocab = sum(levels), model['input_']['emitD'], model['vocab_size']
    # Exact count for the recovered bias-free, RMSNorm, Add/allsoftmax recipe.
    for name in ('input_', 'output_'):
        cfg = model[name]
        if cfg['chals']['chal'] != 'add' or cfg['chals']['confluence'] != 'allsoftmax' or cfg['bias'] or cfg['norm'] != 'rms':
            raise ValueError('unsupported historical parameter-accounting profile')
    if model['pronounce']['chal'] != 'add' or model['pronounce']['confluence'] != 'allsoftmax':
        raise ValueError('unsupported readout accounting profile')
    count = sum(edges.values())*width**2 + 2*n*width + sum(edges.values())+1
    count += 2*vocab*width + model['n_layer']
    inventory = {str(p.relative_to(out/'graph')): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted((out/'graph').iterdir()) if p.is_file()}
    record = dict(schema='lh-benchmark-input-v1', profile=args.profile, graph_dir=str(out/'graph'),
                  model=model, graph_config=graph_config, selectnum=1 if local == 32 else 2,
                  with_lead_point=True, seed=args.seed,
                  historical_match='unconfirmed reconstructed configuration',
                  model_source_revision=full_revision,
                  model_source_sha256=hashlib.sha256(cfg_bytes).hexdigest(), generator_sha256=hashes,
                  graph_files_sha256=inventory, static_nodes=n, leaves=levels[-1], hubs=sum(levels[:-1]),
                  edges=edges, expected_parameters=count,
                  generator_versions=dict(numpy=np.__version__, scipy=scipy.__version__,
                    sklearn=sklearn.__version__, networkx=networkx.__version__, torch=torch.__version__))
    for name, digest in hashes.items():
        if hashlib.sha256((source/'PyConnectome'/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('graph generator changed during preparation')
    write_json(out/'input.json', record)
    print(json.dumps({k: record[k] for k in ('profile', 'static_nodes', 'edges', 'expected_parameters')}))


if __name__ == '__main__':
    main()

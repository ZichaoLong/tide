"""Capture the original pilot-r1 Shannon failure without changing its solver.

This is a bounded forward-only diagnostic. Device allocation is the caller's
responsibility. The default source paths are the immutable round-1 snapshots;
the known original allocation hash is checked before any device initialization.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_ALLOCATION_SHA256 = 'b4d0e130c1a87391606e6abedfb2a9b0ab870bfb9027dc31b8c91135bdacd4ca'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--cpu-threads', type=int)
    parser.add_argument('--repo-root', type=Path, default=ROOT,
                        help='repository root when this script is copied into a frozen launch directory')
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    source = repo_root / '.local/pilot-r1/foundation/src'
    adapter = repo_root / '.local/pilot-r1/frozen/source'
    allocation_path = source / 'tide/allocation.py'
    if sha256(allocation_path) != ORIGINAL_ALLOCATION_SHA256:
        parser.error('the original frozen allocation source has changed')
    for name in ('trace.json', 'failure.pt'):
        if (args.output / name).exists():
            parser.error(f'refusing to replace an existing {name}')
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(adapter), str(source)]
    # Running with -m would preload the working-tree pilotlab package. Reset
    # only its package search path so worker/data/models resolve to r1 copies.
    if 'pilotlab' in sys.modules:
        sys.modules['pilotlab'].__path__ = [str(adapter / 'pilotlab')]
    import numpy as np
    import torch
    import tide.allocation as allocation
    import tide.model as tide_model
    from tide.precision import precision_context
    import pilotlab.models as models
    from pilotlab.data import DocumentStore
    from pilotlab.worker import initialize, causality_check, synchronize
    from pilotlab.io import atomic_json

    if Path(allocation.__file__).resolve() != allocation_path.resolve():
        raise RuntimeError('trace did not load the frozen original foundation')
    if Path(models.__file__).resolve() != (adapter / 'pilotlab/models.py').resolve():
        raise RuntimeError('trace did not load the frozen original LM adapter')
    args.output.mkdir(parents=True, exist_ok=True)
    task = json.loads(args.task.read_text())
    threads = args.cpu_threads or task['cpu_threads']
    if threads < 1:
        parser.error('--cpu-threads must be positive')
    start = time.time()
    record = {
        'schema': 'pilot-r2-original-shannon-trace-v1',
        'task_id': task['id'], 'task_sha256': sha256(args.task),
        'repo_root': str(repo_root),
        'task': task, 'data_manifest_sha256': sha256(args.data / 'manifest.json'),
        'script': str(Path(__file__).resolve()), 'script_sha256': sha256(__file__),
        'allocation_source': str(allocation_path), 'allocation_sha256': sha256(allocation_path),
        'model_source_sha256': sha256(source / 'tide/model.py'),
        'adapter_source_sha256': sha256(models.__file__),
        'requested_device': args.device, 'cpu_threads': threads,
        'physical_visibility': os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
        'torch_version': torch.__version__, 'started_unix': start,
        'scope': 'Original source, seed and initial samples; forward only; original tolerances; first failure capture',
        'causality': None, 'validation': [], 'status': 'running',
    }
    last_bisect = {}

    def trace(frame, event, arg):
        if frame.f_code is allocation._bisect.__code__:
            if event == 'return':
                # Keep references only. No additional tensor kernels or host
                # copies occur before the original solver passes or raises.
                last_bisect.clear()
                last_bisect.update({key: value for key, value in frame.f_locals.items()
                    if key in {'shifted', 'lower', 'upper', 'target', 'solve', 'kind',
                               'c', 'inside', 'remainder', 'reconstructed', 'multiplier'}})
            return trace
        return None

    def cpu(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: cpu(item) for key, item in value.items()}
        return value

    stage, sample = 'initialization', None
    previous_trace = sys.gettrace()
    try:
        device = initialize(args.device, threads, task['seed'])
        store = DocumentStore(args.data)
        model = models.build_model(task['model'], store.vocab_size).to(device).eval()
        record['parameters'] = sum(p.numel() for p in model.parameters())
        sys.settrace(trace)
        stage = 'causality'
        record['causality'] = causality_check(model, store, task, device)
        rng = np.random.default_rng(17729)
        with torch.no_grad(), precision_context(device, task.get('precision', 'fp32')):
            for batch_index in range(task.get('eval_batches', 2)):
                stage = f'validation_batch_{batch_index}'
                sample = store.sample_batch('validation', min(task['batch_size'], 4),
                                            task['sequence_length'], rng)
                x = torch.from_numpy(sample[:, :-1].copy()).to(device)
                y = torch.from_numpy(sample[:, 1:].copy()).to(device)
                logits = model(x).logits.float()
                loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())
                synchronize(device)
                record['validation'].append({'batch_index': batch_index, 'nll': float(loss),
                    'input_sha256': hashlib.sha256(sample[:, :-1].copy().tobytes()).hexdigest()})
        record['status'] = 'completed_without_original_failure'
    except Exception as exc:
        sys.settrace(previous_trace)
        record.update(status='unexpected_failure', failure_stage=stage,
                      error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        frames = {}
        tb = exc.__traceback__
        while tb is not None:
            frame = tb.tb_frame
            if frame.f_code is allocation._Constrained.forward.__code__:
                frames['solver'] = frame.f_locals
            elif frame.f_code is allocation.allocate.__code__:
                frames['allocate'] = frame.f_locals
            elif frame.f_code is tide_model.TideModel._evaluate_region.__code__:
                frames['region'] = frame.f_locals
            tb = tb.tb_next
        if 'solver' in frames:
            solver = frames['solver']
            payload = {
                'stage': stage,
                'solver': cpu({key: solver[key] for key in (
                    'scores', 'mu', 'lower', 'upper', 'low_total', 'high_total', 'kind',
                    'c', 'target', 'tolerance', 'low_active', 'high_active', 'active')}),
                'bisect': cpu(last_bisect),
                'allocation_inputs': cpu({key: frames['allocate'][key]
                    for key in ('scores', 'w', 'mu', 'lower', 'upper', 'budget', 'kind')})
                    if 'allocate' in frames else None,
                'tokens': None if sample is None else torch.from_numpy(sample.copy()),
            }
            if 'region' in frames:
                regional = frames['region']
                owner, region = regional['self'], regional['region']
                members = owner.graph.regions[region]
                record['location'] = {
                    'region': region,
                    'layer_zero_based': next(i for i, regions in enumerate(owner.layer_regions) if region in regions),
                    'members': list(members),
                    'nodes': [dataclasses.asdict(owner.graph.nodes[v]) for v in members],
                    'kind': solver['kind'],
                }
                payload['location'] = record['location']
            residual = (payload['solver']['c'].sum(-1, keepdim=True) - payload['solver']['target']).abs()
            failed = payload['solver']['active'] & (residual > payload['solver']['tolerance'])
            record['allocation_shape'] = list(payload['solver']['scores'].shape)
            record['failing_indices'] = failed.squeeze(-1).nonzero().tolist()
            record['max_residual_from_saved_cpu_values'] = float(residual.max())
            record['failure_payload'] = str((args.output / 'failure.pt').resolve())
            torch.save(payload, args.output / 'failure.pt')
            record['failure_payload_sha256'] = sha256(args.output / 'failure.pt')
            if isinstance(exc, ArithmeticError) and str(exc) == 'allocation multiplier did not meet the total constraint':
                record['status'] = 'original_failure_captured'
    finally:
        sys.settrace(previous_trace)
        record['ended_unix'] = time.time()
        record['seconds'] = time.time() - start
        atomic_json(args.output / 'trace.json', record)
    print(json.dumps({key: record.get(key) for key in ('status', 'failure_stage', 'location',
          'error', 'failing_indices', 'max_residual_from_saved_cpu_values', 'seconds')}, ensure_ascii=False))
    return 2 if record['status'] == 'unexpected_failure' else 0


if __name__ == '__main__':
    raise SystemExit(main())

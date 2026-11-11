#!/usr/bin/env python3
"""Bounded replay and observation only; never replaces a foundation operator.

Imports the original frozen application's model/update/data implementation.
The diagnostic harness has a separate identity. Original run files are read-only.
"""
import argparse
from collections import Counter, deque
import contextlib
import copy
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
import torch
from torch.utils._python_dispatch import TorchDispatchMode


def tensor_record(value):
    value = value.detach().cpu()
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                values=[x if math.isfinite(x) else str(x) for x in value.flatten().tolist()])


class PositiveZeroObserver(TorchDispatchMode):
    """Read-only ATen observer; an event is not itself a semantic failure.

    Watch finite exp/sigmoid, positive products/ratios, and float casts. Store
    actual operand rows for small tensors; retain bounded samples otherwise.
    No tensor returned by the model is changed and no execution schedule is fed
    back. Absence of an event is NOT an exhaustive positive-support guarantee.
    """
    def __init__(self):
        super().__init__()
        self.calls = Counter()
        self.losses = Counter()
        self.events = deque(maxlen=48)
        self.ordinal = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = str(func)
        self.calls[name] += 1
        self.ordinal += 1
        if not isinstance(out, torch.Tensor) or not out.is_floating_point():
            return out
        mask = None
        inputs = []
        if name in ('aten.exp.default', 'aten.sigmoid.default'):
            inputs = [args[0]]
            mask = (out == 0) & torch.isfinite(args[0])
        elif name in ('aten.mul.Tensor', 'aten.div.Tensor') and all(
                isinstance(x, torch.Tensor) for x in args[:2]):
            inputs = list(args[:2])
            mask = (out == 0) & (args[0] > 0) & (args[1] > 0)
            mask &= torch.isfinite(args[0]) & torch.isfinite(args[1])
        elif name == 'aten._to_copy.default' and args[0].is_floating_point():
            if args[0].device.type == 'cpu' and out.device.type == 'cpu':
                inputs = [args[0]]
                mask = (args[0] > 0) & torch.isfinite(args[0]) & (out == 0)
        if mask is None or not bool(mask.any()):
            return out
        ids = mask.nonzero()
        self.losses[name] += int(mask.sum())
        row = dict(operator=name, ordinal=self.ordinal, output_shape=list(out.shape),
                   lost_coordinates=ids[:16].tolist(), lost_count=int(mask.sum()))
        # Broadcast first, so each captured coordinate refers to its operands.
        broadcast = torch.broadcast_tensors(*inputs, out)
        if out.numel() <= 512:
            row['inputs'] = [tensor_record(x) for x in broadcast[:-1]]
            row['output'] = tensor_record(out)
        else:
            indices = tuple(ids[:16].unbind(1))
            row['sample_inputs'] = [tensor_record(x[indices]) for x in broadcast[:-1]]
        self.events.append(row)
        return out

    def record(self):
        return dict(schema='tide-positive-zero-observation-v1', operator_calls=dict(self.calls),
                    lost_counts=dict(self.losses), last_events=list(self.events),
                    scope='forward ATen observer, finite exp/sigmoid, positive tensor products/ratios and casts; no semantic classification or gradient guarantee')


def main(a):
    start = time.monotonic()
    cfg = json.loads(a.config.read_text())
    root = a.root.resolve()
    original_source = root / cfg['original_source']
    # Import original application code, not this diagnostic snapshot's copies.
    sys.path.insert(0, str(original_source))
    from tide_lab.records import atomic_json, digest, now, Metrics
    from tide_lab.train import installed_identity, runtime
    from tide_lab.supervisor import verify_source
    from tide_lab.campaign_runtime import (make_model, optimizer_for, restore, checkpoint,
                                           logical_update, evaluate, memory)
    from tide_lab.campaign_data import Corpus
    from tide_lab.language_training import tensor_save

    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    original = root / cfg['original_run'] / 'train'
    manifest = json.loads((original/'manifest.json').read_text())
    c, ident = manifest['config'], manifest['identity']
    run_id = 'tl-v6-positive-' + cfg['id']
    diagnostic_source = Path(__file__).resolve().parents[1]
    source = json.loads((diagnostic_source/'.tide-source.json').read_text())
    result = dict(schema='tide-positive-path-diagnosis-v1', state='running', started_at=now(),
                  study_id='TL-POSITIVE-v6', original_run=cfg['original_run'],
                  original_identity=ident, diagnostic_source=dict(commit=source['commit'],
                  tree_sha256=source['content']['tree_sha256']), config_sha256=digest(a.config),
                  original_execution='failed', observations=[], test_access=False)
    metrics = Metrics(out, run_id)
    atomic_json(out/'manifest.json', dict(schema='tide-lab-run-v1', project='tide-learning-lab',
        name=run_id, run_id=run_id, group='TL-POSITIVE-v6', study_id='TL-POSITIVE-v6',
        stage='diagnostic-replay', created_at=now(), config=cfg,
        identity=result['diagnostic_source'], resolved=dict(device='cpu', threads=1,
            dtype='float32', seed=c['seed'], batch=c['training']['batch'])))
    atomic_json(out/'result.json', result)
    progress = None
    phase = 'admission'
    last_forward = {}
    try:
        fixed = verify_source(original_source)
        if fixed['commit'] != ident['source_commit'] or fixed['content']['tree_sha256'] != ident['source_sha256']:
            raise ValueError('original source identity mismatch')
        if installed_identity() != ident['installation']:
            raise ValueError('installed foundation identity mismatch')
        if (c['device'], c['dtype'], c['threads']) != ('cpu', 'float32', 1):
            raise ValueError('this replay preserves original CPU1/FP32 only')
        native = Path(os.environ['TIDE_LAB_NATIVE_LIBRARY'])
        for name, key in [('_tide_native.so', 'native_binary_sha256'), ('build-manifest.json', 'native_manifest_sha256')]:
            if digest(native/name) != ident[key]:
                raise ValueError('native identity changed')
        cp = original/'latest.pt'
        if digest(cp) != cfg['checkpoint_sha256']:
            raise ValueError('checkpoint identity changed')
        data = root/cfg['data']
        if digest(data/'data-lock.json') != ident['data_lock_sha256']:
            raise ValueError('data identity changed')
        device = runtime('cpu', 1, c['seed'])
        model = make_model(c, device)
        opt = optimizer_for(model, c)
        progress = restore(cp, model, opt, ident, device)
        corpus = Corpus(data, c['training']['length'])
        def capture_input(module, args):
            last_forward.update(input_ids=args[0].detach().clone(), training=module.training,
                                torch_rng=torch.get_rng_state(), python_rng=random.getstate())
        hook = model.register_forward_pre_hook(capture_input)
        original_metrics = {}
        for line in (original/'metrics.jsonl').read_text().splitlines():
            record = json.loads(line)
            if 'train/loss' in record['metrics']:
                original_metrics[record['step']] = record['metrics']
        last = json.loads((original/'progress.json').read_text())
        original_metrics[last['step']] = last['last_update']
        result.update(restored_progress=copy.deepcopy(progress), checkpoint_sha256=digest(cp),
                      identity_validation='original frozen application, installed package, native binary, data and checkpoint hashes matched')
        max_step = progress['step'] + cfg['max_updates']
        t = c['training']
        next_eval = (progress['tokens']//t['validation_every_tokens']+1)*t['validation_every_tokens']
        while progress['step'] < max_step:
            if time.monotonic()-start > cfg['timeout_seconds']-180:
                raise TimeoutError('bounded diagnostic deadline')
            phase = 'training_forward_or_backward'
            ids, cursor = corpus.sample_ids(progress['cursor'], t['global_batch'], c['seed'])
            milestone = min(m for m in t['milestones'] if m > progress['tokens'])
            batch = corpus.batch('train', ids, device, token_limit=milestone-progress['tokens'])
            model.encoder.observe = progress['step'] % t['log_every'] == 0
            row = logical_update(model, opt, [batch], c, progress['tokens'], device)
            progress.update(step=progress['step']+1, tokens=progress['tokens']+row['tokens'], cursor=cursor)
            if memory(device)['resources/peak_rss_mib'] > cfg['max_rss_gib']*1024:
                raise MemoryError('diagnostic RSS bound')
            values = {'train/loss': row['loss'], 'train/gradient_norm': row['gradient_norm'],
                      'train/tokens': progress['tokens'], 'train/lr': row['lr']}
            metrics.log(progress['step'], values)
            if progress['step'] in original_metrics:
                expected = original_metrics[progress['step']]
                check = dict(step=progress['step'], matches={k: v == expected[k] for k, v in values.items()})
                result['observations'].append(check)
                if not all(check['matches'].values()):
                    raise AssertionError('replay differs from original scalar observations: '+str(check))
            atomic_json(out/'progress.json', dict(progress, phase=phase, elapsed_seconds=time.monotonic()-start))
            print('replayed', progress['step'], values, flush=True)
            if progress['tokens'] >= next_eval:
                phase = 'validation_forward'
                model.encoder.observe = False
                val = evaluate(model, corpus, c, device, token_limit=t['eval_tokens'],
                               deadline=time.time()+max(1, cfg['timeout_seconds']-(time.monotonic()-start)-120))
                atomic_json(out/('validation-'+str(progress['tokens'])+'.json'), val)
                next_eval = (progress['tokens']//t['validation_every_tokens']+1)*t['validation_every_tokens']
        raise RuntimeError('original failure not reproduced within update bound')
    except Exception as error:
        result.update(error=repr(error), phase=phase, progress=copy.deepcopy(progress))
        expected = cfg['expected_error'] in str(error)
        if not expected:
            result.update(state='failed', reproduction='not_verified', traceback=traceback.format_exc())
        else:
            # Failed forward has not changed parameters. This boundary is a
            # diagnostic artifact, not an accepted continuation checkpoint.
            tensor_save(out/'failure-boundary.pt', checkpoint(model, opt, progress, ident, device))
            tensor_save(out/'failure-input.pt', last_forward)
            if callable(getattr(error, 'record', None)):
                atomic_json(out/'allocation-failure.json', error.record())
            hook.remove()
            observer = PositiveZeroObserver()
            torch.set_rng_state(last_forward['torch_rng'])
            random.setstate(last_forward['python_rng'])
            model.train(last_forward['training'])
            observed_error = None
            try:
                with (contextlib.nullcontext() if model.training else torch.no_grad()), observer:
                    model(last_forward['input_ids'])
            except Exception as observed:
                observed_error = repr(observed)
                if callable(getattr(observed, 'record', None)):
                    atomic_json(out/'observed-allocation-failure.json', observed.record())
            atomic_json(out/'operator-observation.json', dict(observer.record(), error=observed_error,
                original_error=repr(error), same_exception=observed_error == repr(error)))
            result.update(state='completed' if observed_error == repr(error) else 'failed',
                reproduction='original_failure_captured', observed_forward_same_exception=observed_error == repr(error),
                lost_counts=dict(observer.losses), boundary_sha256=digest(out/'failure-boundary.pt'),
                axes=dict(execution='diagnosis only; original execution failure retained',
                    strict_numerical='original thresholds/failures unchanged',
                    discrete_semantics='original positive-support guard failed',
                    recovery='checkpoint decoded; available historical scalar observations checked exactly; not full-state trajectory equivalence',
                    learning='not assessed; no candidate continuation or new quality comparison'))
            metrics.log(progress['step'], {'diagnostic/failure_reproduced': 1,
                'diagnostic/observed_same_exception': observed_error == repr(error),
                'diagnostic/positive_zero_coordinates': sum(observer.losses.values())})
    finally:
        metrics.close()
        result.update(finished_at=now(), elapsed_seconds=time.monotonic()-start)
        atomic_json(out/'result.json', result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result['state'] == 'completed' else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'root', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    raise SystemExit(main(p.parse_args()))

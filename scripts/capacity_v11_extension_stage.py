#!/usr/bin/env python3
"""Explicit endpoint-only migration and trained-state exact recovery for v11."""
import argparse
import copy
import json
from pathlib import Path
import random
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from scripts.campaign_stage import identity
from tide_lab.records import atomic_json, digest, now
from tide_lab.train import runtime, cpu_tree
from tide_lab.supervisor import verify_source
from tide_lab.campaign_runtime import make_model, optimizer_for, restore, checkpoint, logical_update, exact_tree
from tide_lab.campaign_data import Corpus
from tide_lab.language_training import tensor_save
from tide_lab.v10_repair import assert_finite_tree, training_batches, advance, gradient_signatures

MI = 2**20


def validate_extension(old, new):
    expected = copy.deepcopy(old)
    expected['training']['target_tokens'] = 256*MI
    expected['training']['milestones'] = sorted(set(old['training']['milestones']+[256*MI]))
    if (new != expected or old['training']['target_tokens'] != 128*MI
            or old['training']['lr_decay_tokens'] != 256*MI
            or old['kind'] not in ('transformer', 'moe')):
        raise ValueError('only declared 128-to256Mi endpoint/milestone extension is permitted')


def migrate(a, config, ident):
    old = json.loads(a.parent_config.read_text())
    validate_extension(old, config)
    parent_source = verify_source(a.parent_source)
    source = Path(__file__).resolve().parents[1]
    files = sorted(a.parent_source.glob('tide_lab/*.py'))+[a.parent_source/'scripts/campaign_stage.py']
    proof = []
    for path in files:
        relative = path.relative_to(a.parent_source)
        if digest(path) != digest(source/relative):
            raise ValueError('training implementation changed: '+str(relative))
        proof.append(dict(path=str(relative), sha256=digest(path)))
    state = torch.load(a.resume, map_location='cpu', weights_only=False)
    expected = dict(ident, config_sha256=digest(a.parent_config), source_commit=parent_source['commit'],
                    source_sha256=parent_source['content']['tree_sha256'])
    if state['identity'] != expected or state['progress']['tokens'] != 128*MI or state['progress'].get('probe'):
        raise ValueError('not the original qualified 128Mi parent')
    model = make_model(config, torch.device('cpu'))
    model.load_state_dict(state['model'], strict=True)
    opt = optimizer_for(model, config)
    opt.load_state_dict(state['optimizer'])
    exact_tree(state['model'], cpu_tree(model.state_dict()))
    exact_tree(state['optimizer'], cpu_tree(opt.state_dict()))
    assert_finite_tree(state['model'])
    assert_finite_tree(state['optimizer'])
    if model.embedding.weight is not model.head.weight:
        raise AssertionError('tied head alias lost')
    ancestry = dict(kind='explicit_endpoint_and_source_identity_migration',
        parent_checkpoint=str(a.resume), parent_sha256=digest(a.resume), parent_identity=state['identity'],
        change='target_tokens 128Mi ->256Mi; append256Mi milestone; original256Mi LR schedule unchanged',
        implementation_files_equal=proof,
        scope='Payload values preserved; new-source trained-state recovery is separately required.')
    tensor_save(a.output/'checkpoint.pt', dict(state, identity=ident, source_migration=ancestry))
    restored = torch.load(a.output/'checkpoint.pt', map_location='cpu', weights_only=False)
    for key in state:
        if key != 'identity':
            exact_tree(state[key], restored[key], key)
    return dict(state='completed', ancestry=ancestry, progress=state['progress'],
                checkpoint_sha256=digest(a.output/'checkpoint.pt'))


def segment(a, config, ident, device):
    model = make_model(config, device)
    opt = optimizer_for(model, config)
    progress = restore(a.resume, model, opt, ident, device)
    progress['probe'] = True
    corpus = Corpus(a.data, config['training']['length'])
    rows = []
    for _ in range(a.steps):
        if time.time() >= a.deadline_epoch-90:
            raise TimeoutError('recovery segment deadline')
        batches, cursor, raw = training_batches(corpus, config, progress, device, 256*MI)
        row = logical_update(model, opt, batches, config, progress['tokens'], device)
        advance(progress, row, cursor, raw, config)
        progress['rng_probe'] = [random.random(), float(torch.rand(())), float(torch.rand((), device=device))]
        rows.append(dict(loss=row['loss'], tokens=row['tokens'], gradient_norm=row['gradient_norm']))
    value = checkpoint(model, opt, progress, ident, device)
    assert_finite_tree(value['model'])
    assert_finite_tree(value['optimizer'])
    value['last_gradient_signatures'] = gradient_signatures(model)
    tensor_save(a.output/'checkpoint.pt', value)
    return dict(state='completed', progress=progress, updates=len(rows), observations=rows,
                checkpoint_sha256=digest(a.output/'checkpoint.pt'))


def main(a):
    config = json.loads(a.config.read_text())
    device = runtime(config['device'] if a.action=='segment' else 'cpu', config['threads'], config['seed'])
    ident = identity(a.config, config, a.data, 'train')
    a.output.mkdir(parents=True, exist_ok=False)
    result = dict(schema='tide-capacity-v11-extension-stage-v1', state='running', action=a.action,
                  identity=ident, started_at=now())
    atomic_json(a.output/'result.json', result)
    try:
        if a.action=='migrate':
            result.update(migrate(a, config, ident))
        elif a.action=='segment':
            result.update(segment(a, config, ident, device))
        else:
            left = torch.load(a.checkpoint_a, map_location='cpu', weights_only=False)
            right = torch.load(a.checkpoint_b, map_location='cpu', weights_only=False)
            if left['identity']!=ident or right['identity']!=ident or not left['progress']['probe']:
                raise ValueError('wrong trained-state recovery identity')
            exact_tree(left, right)
            recovery = dict(state='passed', identity=ident, at=now(),
                scope='Two full L256/global64 updates after128Mi; continuous versus1+1 fresh processes, model/Adam/cursor/gradients/CPU+NPU+Python RNG; no full-trajectory certification.',
                sources=[dict(path=str(p), sha256=digest(p)) for p in (a.checkpoint_a, a.checkpoint_b)])
            atomic_json(a.output/'recovery.json', recovery)
            result.update(state='completed', recovery=recovery)
    except BaseException as error:
        result.update(state='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        result['finished_at'] = now()
        atomic_json(a.output/'result.json', result)
        atomic_json(a.output/'completed.json', dict(producer_finished=True, state=result['state']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('migrate','segment','compare'))
    for name in ('config','data','output'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('parent-config','parent-source','resume','checkpoint-a','checkpoint-b'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--steps',type=int,default=2)
    p.add_argument('--deadline-epoch',type=float,required=True)
    main(p.parse_args())

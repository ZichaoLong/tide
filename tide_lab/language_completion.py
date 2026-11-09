"""Explicit application-state transfer for the finite v5 completion batch."""
import copy
import json
from pathlib import Path
import random
import torch
from .records import digest


def import_checkpoint(config, plan, model, optimizer, data_lock, device, kind, seed, batch, length, depth, schedule_steps):
    """Retain the parent file/identity; return a separately identified continuation."""
    path = Path(config['checkpoint'])
    if digest(path) != config['sha256']:
        raise ValueError('completion parent checkpoint changed')
    manifest_path = Path(config['parent_manifest'])
    if digest(manifest_path) != config['parent_manifest_sha256']:
        raise ValueError('completion parent manifest changed')
    parent = json.loads(manifest_path.read_text())
    old_plan = parent['config']
    for key in ('core', 'training'):
        old, new = copy.deepcopy(old_plan[key]), copy.deepcopy(plan[key])
        if key == 'core':
            old.pop('support_policy', None); new.pop('support_policy', None)
        if old != new:
            raise ValueError('completion changed model/training recipe: '+key)
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('schema') != 'tide-language-checkpoint-v1' or state['step'] != config['step']:
        raise ValueError('completion parent schema/step mismatch')
    saved = state['identity']
    expected = dict(seed=seed,batch=batch,length=length,schedule_steps=schedule_steps,
                    data_lock_sha256=data_lock)
    if config['format'] == 'support-acceptance':
        expected.update(schema='tide-policy-acceptance-identity-v1',policy=plan['core']['support_policy'],
                        model_id=model.encoder.model_id,graph_identity=model.encoder.binding.graph.identity)
        if kind != 'tide-dense':raise ValueError('acceptance transfer only covers dense')
    elif config['format'] == 'language-checkpoint':
        if saved != parent['identity']:raise ValueError('parent checkpoint/manifest identity mismatch')
        expected.update(route='a',kind=kind,depth=depth)
    else:raise ValueError('unknown completion transfer format')
    for key,value in expected.items():
        if saved.get(key) != value:raise ValueError('completion parent differs: '+key)
    model.load_state_dict(state['model'],strict=True)
    optimizer.load_state_dict(state['optimizer'])
    for slots in optimizer.state.values():
        for key,value in slots.items():
            if isinstance(value,torch.Tensor) and key != 'step':slots[key]=value.to(device)
    torch.set_rng_state(state['rng']);random.setstate(state['python_rng'])
    if device.type == 'npu' and state.get('accelerator_rng') is not None:
        torch.npu.set_rng_state_all(state['accelerator_rng'])
    if model.embedding.weight is not model.head.weight:raise AssertionError('tied head lost')
    return state['step'],state['valid_tokens'],dict(
        parent_checkpoint=str(path),parent_sha256=config['sha256'],parent_identity=saved,
        start_step=state['step'],mode='explicit application-state transfer into new source/dependency identity',
        exact_cross_identity_resume_claim=False,
        restored=['model','optimizer','CPU RNG','Python RNG','data cursor'],
        original_initial_validation_nll=config['original_initial_validation_nll'])

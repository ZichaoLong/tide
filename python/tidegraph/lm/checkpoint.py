"""Explicit small representative checkpoints, verified by a fresh owner/process."""
import hashlib
import os
from pathlib import Path
import time
from .runtime import setup, synchronize
from ..experiment.records import digest, read_json, write_json


def host_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: host_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(host_tree(v) for v in value)
    return value


def identity(value):
    import torch
    if isinstance(value, torch.Tensor):
        data = value.detach().contiguous().cpu()
        return dict(shape=list(data.shape), dtype=str(data.dtype),
                    sha256=hashlib.sha256(data.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
    if isinstance(value, dict):
        return {str(k): identity(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [identity(v) for v in value]
    return value


def tensor_bytes(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.numel()*value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(x) for x in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(x) for x in value)
    return 0


def publish(value, c, out):
    import torch
    begin = time.perf_counter(); path = out/'checkpoint.pt'
    estimated = tensor_bytes(value)
    if estimated*1.05+1024**2 > c['resources']['checkpoint_gib']*1024**3:
        raise RuntimeError('resources.checkpoint_gib: checkpoint tensors exceed explicit budget')
    temporary = path.with_suffix('.partial')
    try:
        with temporary.open('wb') as stream:
            torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        fd = os.open(out, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(path=path.name, bytes=path.stat().st_size, sha256=digest(path),
                save_and_hash_seconds=time.perf_counter()-begin, scope='new-process exact continuation, not independent correctness')


def cache_state(cache):
    return tuple((k, v) for k, v in cache.to_legacy_cache())


def inference_next(owner, state, c, device, backend):
    import torch
    from transformers import DynamicCache
    # Restore each layer on its actual device; no hidden full-model replica.
    if len(state['cache'])!=len(owner.model.layers):
        raise ValueError('checkpoint layer count differs from model')
    cached = []
    for index, (key, value) in enumerate(state['cache']):
        layer_device = next(owner.model.layers[index].parameters()).device
        cached.append((key.to(layer_device).clone(), value.to(layer_device).clone()))
    cache = DynamicCache.from_legacy_cache(tuple(cached))
    mask = state['mask'].to(device)
    mask = torch.cat((mask, torch.ones((mask.shape[0], 1), dtype=mask.dtype, device=device)), dim=1)
    with torch.inference_mode():
        result = owner(input_ids=state['last_token'].to(device), attention_mask=mask,
                       position_ids=mask.sum(-1, keepdim=True)-1, past_key_values=cache,
                       use_cache=True, logits_to_keep=1)
    synchronize(backend, c['execution']['devices'])
    if not bool(torch.isfinite(result.logits).all()):
        raise FloatingPointError('nonfinite resumed logits')
    return identity(dict(logits=result.logits, cache=cache_state(result.past_key_values),
                         mask=mask, next_token=result.logits[:, -1].argmax(-1)))


def prepare_inference(owner, result, c, out, device, backend):
    begin = time.perf_counter()
    state = host_tree(dict(cache=cache_state(result['cache']), mask=result['mask'], last_token=result['last_token']))
    record = publish(dict(kind='inference', state=state), c, out)
    expected = inference_next(owner, state, c, device, backend)
    record.update(expected=expected, preparation_seconds=time.perf_counter()-begin)
    write_json(out/'checkpoint-request.json', record)
    return record


def training_state(owner, optimizer):
    return dict(parameters={n: p.detach() for n, p in owner.named_parameters() if p.requires_grad},
                optimizer=optimizer.state_dict())


def prepare_training(owner, optimizer, ids, mask, c, out, device, backend):
    import torch
    from .training import update
    begin = time.perf_counter()
    state = host_tree(training_state(owner, optimizer))
    state.update(ids=ids, mask=mask, cpu_rng=torch.get_rng_state(),
                 device_rng=backend.get_rng_state_all() if backend else [])
    record = publish(dict(kind='training', state=state), c, out)
    update(owner, optimizer, ids, mask, c, device, backend)
    record.update(expected=identity(training_state(owner, optimizer)), preparation_seconds=time.perf_counter()-begin)
    write_json(out/'checkpoint-request.json', record)
    return record


def resume(run_dir, asset_root):
    import torch
    from .training import optimizer_for, update
    out = Path(run_dir); resolved = read_json(out/'resolved.json'); c = resolved['request']
    request = read_json(out/'checkpoint-request.json'); started = time.perf_counter()
    result = dict(state='running', process_id=os.getpid(), strict_exact=True)
    write_json(out/'resume.json', result)
    try:
        if digest(out/request['path']) != request['sha256']:
            raise ValueError('checkpoint bytes differ from saved identity')
        from ..experiment.identity import package_identity
        previous=read_json(out/'result.json')
        if package_identity()['installed_files_sha256']!=previous['package_identity']['installed_files_sha256']:
            raise ValueError('checkpoint exact resume requires the original installed/source content')
        owner, _, device, backend, runtime = setup(resolved, asset_root)
        if runtime['packages']!=previous['runtime']['packages']:
            raise ValueError('checkpoint exact resume requires the original dependency versions')
        saved = torch.load(out/request['path'], map_location='cpu', weights_only=True)
        state = saved['state']
        if saved['kind'] == 'inference':
            actual = inference_next(owner, state, c, device, backend)
        else:
            named = dict(owner.named_parameters())
            if set(state['parameters'])!={n for n,p in named.items() if p.requires_grad}:
                raise ValueError('checkpoint trainable parameter names differ')
            with torch.no_grad():
                for key, value in state['parameters'].items():
                    if named[key].shape!=value.shape or named[key].dtype!=value.dtype:
                        raise ValueError('checkpoint parameter shape/dtype differs: '+key)
                    named[key].copy_(value.to(named[key].device))
            opt = optimizer_for(owner); opt.load_state_dict(state['optimizer'])
            torch.set_rng_state(state['cpu_rng'])
            if backend:
                backend.set_rng_state_all(state['device_rng'])
            update(owner, opt, state['ids'], state['mask'], c, device, backend)
            actual = identity(training_state(owner, opt))
        result.update(state='passed' if actual == request['expected'] else 'failed',
                      exact=actual == request['expected'], actual=actual, runtime=runtime,
                      seconds=time.perf_counter()-started)
    except Exception as error:
        result.update(state='failed', error_kind=type(error).__name__, error=str(error)[:2000])
    write_json(out/'resume.json', result)
    previous = read_json(out/'result.json')
    previous['validation']['checkpoint'] = result['state']
    previous['validation']['checkpoint_scope'] = 'exact next step with newly loaded owner in another process'
    write_json(out/'result.json', previous)
    return result

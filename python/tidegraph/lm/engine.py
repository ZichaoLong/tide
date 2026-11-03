"""One recorded external-LM run; completion is not independent qualification."""
from contextlib import nullcontext
import gc
import os
from pathlib import Path
import resource
from .catalog import model
from .config import resolve
from .runtime import setup, peaks, synchronize
from .inputs import create
from ..experiment.records import now, write_json
from ..experiment.identity import package_identity


def run(config, asset_root, output):
    resolved = resolve(config); c = resolved['request']; e = c['execution']; m = c['measurement']
    if e['engine']=='llama_cpp':
        from .llama_adapter import run as execute
        return execute(c,asset_root,output)
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    write_json(out/'resolved.json', resolved)
    record = dict(schema='tide-lm-result-v1', config_id=resolved['config_id'], name=c['name'], model=c['model'],
        model_revision=resolved['model_revision'], configuration=c, started=now(), state='running',
        execution_status='running', validation=dict(numeric='not_checked', discrete='not_checked', checkpoint='not_checked'),
        package_identity=package_identity(), process_id=os.getpid(),
        performance_scope='external LM token/s; never divided by core-5 sample-windows/s',
        contention=dict(load_average=list(os.getloadavg()), affinity=sorted(os.sched_getaffinity(0)),
                        exclusivity='cooperative project serialization only; other users not excluded'),
        timing_scope='input upload, online scheduling, kernels, finite checks, token/statistic return and synchronization; training also backward and optimizer')
    def save():
        write_json(out/'result.json', record)
    save()
    try:
        from .resources import disk_admission
        additional = resolved['budget']['checkpoint_estimate_bytes']*1.1 if m['checkpoint'] else 16*1024**2
        disk_admission(Path(asset_root).parent, additional, project_gib=c['resources']['project_gib'],
                       account_gib=c['resources']['account_gib'])
        owner, tokenizer, device, backend, identity = setup(resolved, asset_root)
        record['runtime'] = identity; save()
        host_ids, host_mask, input_id = create(c['task'], model(c['model'])['config'], tokenizer)
        record.update(input=input_id, warmup=[], measurements=[]); save()
        if c['task']['mode'] == 'inference':
            from .inference import generate
            for _ in range(m['warmup']):
                warm = generate(owner, host_ids, host_mask, c, device, backend, output_tokens=min(8, c['task']['output_tokens']))
                record['warmup'].append(warm['metrics']); del warm
            gc.collect()
            if m['profile']:
                from .profiling import observe
            context = observe(owner, c, out) if m['profile'] else nullcontext()
            with context:
                for index in range(m['steps']):
                    result = generate(owner, host_ids, host_mask, c, device, backend)
                    record['measurements'].append(result['metrics']); record['work'] = result['work']; save()
                    if index+1 != m['steps']:
                        del result
            import torch
            torch.save(dict(token_ids=result['tokens'], input_identity=input_id), out/'outputs.pt')
            if m['checkpoint']:
                from .checkpoint import prepare_inference
                request = prepare_inference(owner, result, c, out, device, backend)
                record['checkpoint_request'] = {k: v for k, v in request.items() if k != 'expected'}
                record['validation']['checkpoint'] = 'awaiting_fresh_process'
        else:
            from .training import train
            record.update(train(owner, host_ids, host_mask, c, out, device, backend))
            if record.get('checkpoint_request'):
                record['checkpoint_request'].pop('expected', None)
                record['validation']['checkpoint'] = 'awaiting_fresh_process'
        synchronize(backend, e['devices'])
        record.update(state='completed', execution_status='completed', device_peak=peaks(backend, e['devices']),
                      peak_scope='whole process including construction, warmup and optional checkpoint preparation',
                      host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024, finished=now(),
                      performance_eligible=not m['profile'], independent_qualification='not_established')
        save()
    except Exception as error:
        record.update(state='failed', execution_status='failed', error_kind=type(error).__name__, error=str(error)[:2000], finished=now())
        save()
        raise
    return record

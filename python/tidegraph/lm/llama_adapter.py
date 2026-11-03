"""Public process adapter for an independent, fixed-source CPU llama.cpp client."""
from pathlib import Path
from types import SimpleNamespace
import subprocess
import time
from .catalog import model
from .config import resolve
from .inputs import create
from .resources import disk_admission
from ..experiment.records import read_json, write_json, replace_text, digest, now, fingerprint


def artifacts(c,assets):
    root=Path(assets).parent;engine=root/'engines/llama'
    build=read_json(engine/'ready.json');conversion=read_json(engine/c['model']/'conversion.json')
    binary=Path(build['binary']);weights=Path(conversion['gguf'])
    if digest(binary)!=build['binary_sha256'] or build['state']!='passed':raise ValueError('llama binary identity changed')
    for name,expected in build['shared_libraries'].items():
        if digest(name)!=expected:raise ValueError('llama shared library identity changed')
    if conversion['model_manifest_id']!=fingerprint(model(c['model'])):
        raise ValueError('GGUF source model identity differs')
    if weights.stat().st_size!=conversion['bytes'] or weights.stat().st_mtime_ns!=conversion['mtime_ns']:
        raise ValueError('GGUF bytes or timestamp changed; explicit revalidation required')
    return binary,weights,dict(build=build,conversion=conversion)


def material(c,out):
    ids,mask,identity=create(c['task'],model(c['model'])['config'],SimpleNamespace(pad_token_id=0,eos_token_id=0))
    t=c['task'];e=c['execution'];m=c['measurement']
    fields=[t['batch'],t['input_tokens'],t['output_tokens'],min(e['prefill_chunk'],t['input_tokens']),
            e['threads'],m['warmup'],int(t['batch']<=2 and t['input_tokens']<=32 and t['output_tokens']<=8),int(m['checkpoint'])]
    replace_text(out/'input.txt',' '.join(map(str,fields))+'\n'+'\n'.join(' '.join(map(str,row)) for row in ids.tolist())+'\n')
    return ids,mask,identity


def invoke(c,assets,out,action):
    binary,weights,runtime=artifacts(c,assets)
    with (out/(action+'.log')).open('ab') as log:
        subprocess.run([str(binary),str(weights),str(out/'input.txt'),str(out),action],stdout=log,stderr=log,
                       check=True,timeout=c['measurement']['timeout_seconds'])
    return runtime


def run(config,assets,output):
    c=resolve(config)['request'];out=Path(output);out.mkdir(parents=True,exist_ok=False)
    resolved=resolve(c);write_json(out/'resolved.json',resolved)
    record=dict(schema='tide-lm-result-v1',state='running',execution_status='running',configuration=c,
        config_id=resolved['config_id'],model=c['model'],started=now(),validation=dict(numeric='not_checked',discrete='not_checked',checkpoint='not_checked'),
        independent_qualification='not_established',
        timing_scope='CPU C++ batch construction, prefill/decode, full logit finite reductions and greedy token return; file persistence separate',
        performance_scope='exploratory GGML cost; GGUF retains FP32 1D tensors, not identical HF BF16 arithmetic or full-cache independent qualification')
    def save():write_json(out/'result.json',record)
    save()
    try:
        b=resolved['budget']
        available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        if b['host_estimate_bytes']>min(c['resources']['host_gib']*1024**3,available*.8):raise RuntimeError('llama host admission rejected')
        disk_admission(Path(assets).parent,b['checkpoint_estimate_bytes']*4+32*1024**2,
                       project_gib=c['resources']['project_gib'],account_gib=c['resources']['account_gib'])
        _,_,input_id=material(c,out)
        runtime=invoke(c,assets,out,'run');metrics=read_json(out/'metrics.json');t=c['task'];tokens=t['batch']*t['output_tokens']
        measurement={k:metrics[k] for k in ('total_seconds','prefill_seconds','decode_seconds')}
        measurement.update(output_tokens=tokens,output_tokens_per_s=tokens/metrics['total_seconds'],
                           decode_tokens_per_s=(tokens-t['batch'])/max(1e-12,metrics['decode_seconds']),input_tokens=t['batch']*t['input_tokens'])
        runtime.update(precision='GGUF mostly BF16 matrices, FP32 1D tensors; BF16 KV explicitly required; GGML CPU arithmetic',
            placement='single CPU',communication='none',construction_seconds=metrics['construction_seconds'],
            blas='GGML_BLAS=OFF; GGML native ARM/OpenMP',threads=c['execution']['threads'])
        record.update(state='completed',execution_status='completed',finished=now(),runtime=runtime,input=input_id,
            measurements=[measurement],warmup_seconds=metrics['warmup_seconds'],checkpoint_seconds=metrics['checkpoint_seconds'],
            host_peak_rss_bytes=metrics['host_peak_rss_bytes'],performance_eligible=True,
            work=dict(logical_batch=t['batch'],prefill_time_chunk=min(c['execution']['prefill_chunk'],t['input_tokens']),
                      requested_rows_per_prefill=t['batch']*min(c['execution']['prefill_chunk'],t['input_tokens']),
                      decode_rows_per_call=t['batch'],n_batch=metrics['n_batch'],n_ubatch=metrics['n_ubatch'],n_ctx=metrics['n_ctx'],
                      physical_split='GGML graph backend may split kernels; n_ubatch is capacity, not measured MM shape',
                      return_contract='all generated token IDs; model state remains in C++ context during generation'))
        if c['measurement']['checkpoint']:
            record['validation']['checkpoint']='awaiting_fresh_process'
            record['checkpoint_hashes']={n:digest(out/n) for n in ['state.bin','tokens.bin','expected-state.bin','expected-logits.bin']}
        save();return record
    except Exception as error:
        record.update(state='failed',execution_status='failed',error_kind=type(error).__name__,error=str(error)[:2000],finished=now());save();raise


def resume(run_dir,assets):
    out=Path(run_dir);previous=read_json(out/'result.json');c=previous['configuration'];begin=time.monotonic()
    result=dict(state='running',strict_exact=True,scope='fresh C++ owner; exact serialized full state and next logits, not independent numeric qualification')
    try:
        for name,expected in previous['checkpoint_hashes'].items():
            if digest(out/name)!=expected:raise ValueError('saved checkpoint identity changed')
        runtime=invoke(c,assets,out,'restore')
        if runtime['build']!=previous['runtime']['build'] or runtime['conversion']!=previous['runtime']['conversion']:
            raise ValueError('exact restore requires original C++ binary and GGUF identity')
        checks={kind:digest(out/('restored-'+kind+'.bin'))==digest(out/('expected-'+kind+'.bin')) for kind in ['state','logits']}
        result.update(state='passed' if all(checks.values()) else 'failed',checks=checks)
    except Exception as error:result.update(state='failed',error_kind=type(error).__name__,error=str(error)[:2000])
    result['seconds']=time.monotonic()-begin;write_json(out/'resume.json',result)
    previous['validation']['checkpoint']=result['state'];write_json(out/'result.json',previous);return result


def check(config,assets,output):
    import torch
    from transformers import AutoModelForCausalLM
    from .catalog import local_path
    from .verification import independent_prefix,compare
    c=resolve(config)['request'];t=c['task']
    if t['batch']>2 or t['input_tokens']>32 or t['output_tokens']>8:raise ValueError('llama check: B<=2 L<=32 output<=8')
    out=Path(output);out.mkdir(parents=True,exist_ok=False);write_json(out/'resolved.json',resolve(c))
    record=dict(schema='tide-lm-check-v1',state='running',started=now(),configuration=c,
                scope='CPU FP32 HF independent prefix vs C++ GGUF full logits/tokens; independent KV not exposed/checked')
    try:
        ids,mask,input_id=material(c,out);runtime=invoke(c,assets,out,'run')
        torch.set_num_threads(c['execution']['threads'])
        owner=AutoModelForCausalLM.from_pretrained(local_path(assets,c['model']),local_files_only=True,
                       dtype=torch.float32,attn_implementation='eager',low_cpu_mem_usage=True).eval()
        expected=independent_prefix(owner,ids,mask,t['output_tokens'])
        logits=torch.from_file(str(out/'logits.bin'),size=t['output_tokens']*t['batch']*model(c['model'])['config']['vocab_size'],dtype=torch.float32)
        logits=logits.reshape(t['output_tokens'],t['batch'],-1)
        tokens=torch.from_file(str(out/'tokens.bin'),size=t['batch']*t['output_tokens'],dtype=torch.int32).reshape(t['batch'],-1).long()
        checks={f'logits.{i}':compare(a,b) for i,(a,b) in enumerate(zip(expected['logits'],logits))}
        record.update(state='completed',finished=now(),numeric='passed' if all(x['passed'] for x in checks.values()) else 'failed',
            discrete='passed' if torch.equal(tokens,expected['tokens']) else 'failed',checks=checks,
            cache='not_checked',input=input_id,runtime=runtime)
    except Exception as error:
        record.update(state='failed',error_kind=type(error).__name__,error=str(error)[:2000]);write_json(out/'check.json',record);raise
    write_json(out/'check.json',record);return record

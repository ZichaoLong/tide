"""Reviewed finite continuation; the original 80-case manifest stays immutable."""
from copy import deepcopy
from .matrix import manifest as original, case
from .config import resolve
from ..experiment.records import fingerprint


def manifest(group='all'):
    if group not in {'ready','small','moe','all'}:raise ValueError('unknown recovery group')
    groups={k:[] for k in ('ready','small','moe')}
    old=original()
    for item in old['cases']:
        c=item['config'];key=c['model'];t=c['task'];e=c['execution']
        if key in {'olmoe','qwen7','qwenmoe'}:
            x=deepcopy(item)
            x['recovery']=dict(reason='previous dependency_failed; same workload',
                               original_config_id=resolve(c)['config_id'])
            groups['moe' if key=='qwenmoe' else 'small'].append(x)
        elif e['backend']=='cpu' and ((key=='qwen15' and item['kind']=='main' and
               ((t['batch']==512 and t['input_tokens']==512) or (t['batch']==8 and t['input_tokens']==8192))) or
               (key=='qwen05' and item['kind']=='train' and t['input_tokens']==2048)):
            x=deepcopy(item);x['name']='extended-'+item['name'];x['config']['name']=x['name']
            x['config']['measurement']['timeout_seconds']=7200
            x['recovery']=dict(reason='previous 1200s process timeout; explicit 7200s whole-process ceiling, unchanged workload',
                               original_config_id=resolve(c)['config_id'])
            groups['ready'].append(x)
    for backend in ('cpu','npu'):
        for key in ('qwen05','qwen15'):
            x=case(key,backend,2,16,kind='diagnostic',output_tokens=4,warmup=0,
                   dtype='float32' if backend=='cpu' else 'bfloat16')
            x['action']='diagnose';groups['ready'].append(x)
        x=case('qwen05',backend,1,16,kind='fp32-traincheck',mode='full',global_batch=1,
               dtype='float32',attention='eager',steps=1,warmup=0)
        x['action']='training-check';groups['ready'].append(x)
    x=case('qwen05','npu',4,2048,kind='memory-diagnostic',mode='full',microbatch=1,steps=1,warmup=0)
    x['action']='memory-check';groups['ready'].append(x)
    # Small checks first; unfinished expensive curves keep their own expanded limits.
    groups['ready'].sort(key=lambda x:x['action']=='run')
    selected=sum(groups.values(),[]) if group=='all' else groups[group]
    return dict(schema='tide-lm-matrix-v1',identity=fingerprint(selected),cases=selected,
                scope='32 dependency continuations + 4 explicit timeout extensions + 7 diagnostics; no completed curve reruns',
                prior_evidence='docs/evidence/llm-terminal-audit-20261103.json',
                stopped=[dict(case='train-qwen05-npu-b4-l2048-full-m4',
                              reason='73.23GiB conservative admission exceeds safe HBM; memory diagnostic does not waive guard')])

"""2026-11-04 authorized restart, generated without old run directories."""
from copy import deepcopy
from .recovery import manifest as recovery
from .engine_matrix import manifest as engines
from .config import resolve
from ..experiment.records import fingerprint


def manifest(group='all'):
    if group not in {'compute', 'llama', 'placement', 'all'}:
        raise ValueError('restart group: compute/llama/placement/all')
    groups = dict(compute=[], llama=[], placement=[])
    for old in recovery()['cases']:
        c = old['config']
        if c['model'] in {'qwen7', 'qwenmoe'}:
            groups['compute'].append(deepcopy(old))
        elif old['name'] == 'main-olmoe-cpu-b512-l512':
            x = deepcopy(old)
            x['name'] = x['config']['name'] = 'extended-' + old['name']
            x['config']['measurement']['timeout_seconds'] = 7200
            x['restart'] = dict(original_config_id=resolve(c)['config_id'],
                reason='1200s whole-process timeout; explicit 7200s limit, unchanged workload')
            groups['compute'].insert(0, x)
        elif old['name'] == 'main-olmoe-npu-b512-l512':
            x = deepcopy(old)
            x['name'] = x['config']['name'] = 'capacity-2-olmoe-npu-b512-l512'
            x['config']['execution'].update(devices=2, placement='layer_split')
            x['restart'] = dict(original_config_id=resolve(c)['config_id'],
                reason='single-device 63.08GiB rejected; explicit two-device alternative, not a single-device pass',
                prior_small_check='OLMoE two-device B2 executed with strict numeric/token failure; B1 exact resume passed')
            groups['placement'].append(x)
    for old in engines()['cases']:
        if old['config']['execution']['engine'] == 'llama_cpp':
            groups['llama'].append(deepcopy(old))
        elif old['config']['model'] == 'qwenmoe':
            groups['placement'].append(deepcopy(old))
    cases = sum(groups.values(), []) if group == 'all' else groups[group]
    return dict(schema='tide-lm-matrix-v1', identity=fingerprint(cases), cases=cases,
        scope='19 unexecuted dependencies + CPU B512 timeout extension + explicit dual-NPU B512 alternative',
        prior_evidence='docs/evidence/llm-terminal-audit-20261104.json',
        preserved='completed checks/curves and all strict failures; no retry of completed numerical disagreements')


def isolation_manifest(group='all'):
    """13 unexecuted cases and two bounded llama B32 successors; no B1/B512 reruns."""
    previous=manifest(group);cases=[]
    for item in previous['cases']:
        c=item['config']
        if c['model'] in {'qwen7','qwenmoe'}:
            cases.append(deepcopy(item))
        elif c['execution']['engine']=='llama_cpp' and c['task']['batch']==32:
            x=deepcopy(item)
            x['name']=x['config']['name']='extended-'+item['name']
            x['config']['measurement']['timeout_seconds']=7200
            x['restart']=dict(original_config_id=resolve(c)['config_id'],
                reason='1200s whole-process timeout; 7200s successor with stage progress; unchanged math/material/precision')
            cases.append(x)
    return dict(schema='tide-lm-matrix-v1',identity=fingerprint(cases),cases=cases,
        scope='13 unexecuted dependency/submission cases + 2 explicit llama B32 timeout successors',
        prior_evidence='docs/evidence/llm-restart-terminal-20261104.json',
        preserved='all completed B1/B512 costs, checks and exact resumes retain their original identities')

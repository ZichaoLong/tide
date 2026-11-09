#!/usr/bin/env python3
"""Freeze the finite measured backend/batch choices, without submitting jobs."""
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json,digest


def resolve(base,out):
    catalog=json.loads((base/'catalog.json').read_text());out.mkdir(parents=True,exist_ok=False);rows=[]
    for row in catalog['cases']:
        path=base/(row['id']+'.json');c=json.loads(path.read_text())
        c['base_config_sha256']=digest(path)
        # Full resolved module details remain in the base config and in the
        # public recipe; avoid duplicating them in every execution variant.
        c['planned_inventory']={k:v for k,v in c['planned_inventory'].items()
            if k in ('core_trainable','lm_trainable','transformer')}
        npu=c['kind']=='transformer';c['device']='npu:0' if npu else 'cpu';c['threads']=2 if npu else 1
        c['training']['global_batch']=8 if c['line']=='coverage' else 32
        c['training']['batch']=c['training']['global_batch'] if npu else 8
        c['training']['eval_batch']=c['training']['batch']
        c['training']['initial_eval_tokens']=2048
        if c['line']=='coverage':c['training']['validation_every_tokens']=32768
        positive_floor=c['core']['recipe'].get('region_overrides',{}).get('support_policy')=='positive-floor-v1'
        if c['kind']=='tide':c['core']['eager_implementation']='python' if positive_floor else 'native'
        c['resolution']=dict(reason='complete-step calibration plus related small-shape history; each actual recipe gets independent admission and fresh-process recovery',
            cpu_calibration='v6-cal-native2-kv-cpu-b8',npu_calibration='v6-cal-transformer-npu-b128',
            global_batch_reason='fixed independent samples per logical update within line; microbatch/accumulation is explicit',
            native_trace='public factory returns complete traces; no execution-core changes',selection='finite-exploration')
        if positive_floor:
            c['resolution']['reason']='published positive-floor-v1 is Python-eager only; original native rejection retained'
        atomic_json(out/(c['id']+'.json'),c)
        rows.append(dict(id=c['id'],line=c['line'],device=c['device'],threads=c['threads'],
            config=str(out/(c['id']+'.json')),parameters=c['planned_inventory']['lm_trainable']))
    for parent in ('scale-fc16-kv-quad','scale-fc16-ssd-quad'):
        c=json.loads((out/(parent+'.json')).read_text());c['id']=parent+'-npu'
        c['device']='npu:0';c['threads']=2;c['core']['eager_implementation']='python'
        c['resolution'].update(selection='named NPU learning exploration, not an equivalent-speedup claim',
            reason='NPU Python full updates measured on KV/SSD; intentionally retains two device trajectories within 16-run ceiling',
            npu_calibration='v6-cal-kv-npu-b8' if 'kv' in parent else 'v6-cal-ssd-npu-b8',
            matched_cpu_run='tl-v6-'+parent)
        atomic_json(out/(c['id']+'.json'),c)
        rows.append(dict(id=c['id'],line=c['line'],device=c['device'],threads=c['threads'],
            config=str(out/(c['id']+'.json')),parameters=c['planned_inventory']['lm_trainable']))
    atomic_json(out/'catalog.json',dict(schema='tide-language-campaign-catalog-v1',launchable=True,
        cases=rows,counts={line:sum(r['line']==line for r in rows) for line in ('coverage','scale')},
        limits='shared v6 budget remains authoritative; listed cases may fail admission or truncate before target'))
    print(json.dumps(dict(cases=len(rows),coverage=sum(r['line']=='coverage' for r in rows),scale=sum(r['line']=='scale' for r in rows))))


if __name__=='__main__':resolve(Path(sys.argv[1]),Path(sys.argv[2]))

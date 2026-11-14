#!/usr/bin/env python3
"""Resolve the finite v10 table from completed calibration, never confirmation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json,digest,now
from tide_lab.v10_config import hb,transformer,M
from scripts.review_io import write_review_json


def read(path):return json.loads(path.read_text())


def finalize(c):
    import torch
    from tide_lab.campaign_runtime import make_model
    if 'direct' in c:
        p=c['direct']['plan'];p.pop('sha256',None)
        p['sha256']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
    with torch.device('meta'):model=make_model(c,torch.device('meta'))
    count=sum(p.numel() for p in model.parameters())
    if count>200000000:raise ValueError('parameter ceiling exceeded')
    c['planned_inventory']=dict(lm_trainable=count)
    return c


def prepare(root,output,evidence):
    rows=[];calibrations=[];selection={};groups=[]
    control_path=root/'runs/v10-cal-controls/result.json';controls=read(control_path)
    if controls['state']!='completed':raise ValueError('control calibration incomplete')
    for d in (64,128):
        names=[f'v10-cal-hb-d{d}-r{r}' for r in (20,24,28,64)]+[f'v10-cal-hb-d{d}-r28-tau2']
        values={}
        for name in names:
            path=root/'runs/v10-calibration'/name/'result.json';r=read(path)
            if r['state']!='completed':raise ValueError('calibration incomplete: '+name)
            v=dict(id=name,result=str(path.relative_to(root)),sha256=digest(path),parameters=r['parameters'],
                temperature=r['selected_temperature'],region_width=r['config']['candidate']['region_width'],
                mean_tail=r['mean_tail'],peak_mib=max(x['resources/npu_peak_allocated_mib'] for x in r['updates']),
                nll=r['nll'],source_commit=r['source'].get('commit'))
            values[name]=v;calibrations.append(v)
        low=values[f'v10-cal-hb-d{d}-r64'];active=low['mean_tail']['activity/active_parameters_per_position']
        candidates=[v for name,v in values.items() if v['region_width']!=64
            and .08<=v['mean_tail']['activity/full_all_fraction']<=.12]
        high=min(candidates,key=lambda v:abs(v['mean_tail']['activity/active_parameters_per_position']/active-1))
        mismatch=abs(high['mean_tail']['activity/active_parameters_per_position']/active-1)
        selection[str(d)]=dict(q10=high['id'],q3=low['id'],active_parameter_relative_difference=mismatch,
            reason='minimum active-parameter mismatch within predeclared activity band; no NLL ranking',
            matching='within10pct calibration only' if mismatch<=.1 else 'unmatched; report actual difference')
        ids=[]
        for seed in (11,22):
            for tag,v,target,micro in [('q10',high,.1,32 if d==64 else 16),('q3',low,.03125,16 if d==64 else 8)]:
                c=hb(f'hb-d{d}-r{v["region_width"]}-{tag}',d,v['region_width'],target,seed,
                    micro=micro,temperature=v['temperature'])
                c['resource_admission']=dict(peak_mib=v['peak_mib'],targets_per_second=v['mean_tail']['train/tokens_per_second'],
                    source=v['result'],timing='real-data complete logical update, preparation/validation/checkpoint excluded')
                c['calibrated_active_parameters']=v['mean_tail']['activity/active_parameters_per_position']
                c['calibrated_full_all_fraction']=v['mean_tail']['activity/full_all_fraction']
                rows.append(finalize(c));ids.append(c['id'])
            # Density is the minimum-total-capacity point in the active-budget
            # comparison, not merely a same-Full-count construction.
            r,ffn=(4,288) if d==64 else (3,576)
            v=next(x for x in controls['cases'] if x['width']==d and x['region_width']==r and x['batch']==64)
            c=hb(f'hb-d{d}-r{r}-dense-match',d,r,seed=seed,micro=64,ffn=ffn)
            c['resource_admission']=dict(peak_mib=v['peak_mib'],targets_per_second=v['positions_per_second']*.9,
                source=str(control_path.relative_to(root)),timing='synthetic complete update; effective-target estimate uses0.9 factor')
            rows.append(finalize(c));ids.append(c['id'])
            width,layers=(192,6) if d==64 else (256,12)
            candidates=[]
            for path in sorted((root/'runs/v10-calibration').glob(f'v10-cal-tf-d{width}-*/result.json')):
                rtf=read(path)
                if rtf['state']=='completed':candidates.append((rtf['nll'],path,rtf))
            if len(candidates)!=6:raise ValueError('TF bounded six-recipe table incomplete')
            _,path,tf=min(candidates,key=lambda v:v[0]);cfg=tf['config']
            c=transformer(f'tf-d{width}-l{layers}',width,layers,seed,initialization=cfg['transformer']['initialization'],lr=cfg['training']['lr'])
            c['resource_admission']=dict(peak_mib=max(x['resources/npu_peak_allocated_mib'] for x in tf['updates']),
                targets_per_second=tf['mean_tail']['train/tokens_per_second'],source=str(path.relative_to(root)),
                timing='real-data complete logical update; selected among6 declared recipes by validation NLL')
            rows.append(finalize(c));ids.append(c['id'])
            if d==128:
                for v in (high,low):
                    r=v['region_width'];micro=16 if r!=64 else 8
                    cp=next(x for x in controls['cases'] if x['width']==d and x['region_width']==r and x['batch']==micro)
                    c=hb(f'hb-d{d}-r{r}-pool',d,r,seed=seed,micro=micro);c['study_id']='TL-CONTROL-v10'
                    c['resource_admission']=dict(peak_mib=cp['peak_mib'],targets_per_second=cp['positions_per_second']*.9,
                        source=str(control_path.relative_to(root)),timing='synthetic complete update; effective-target estimate uses0.9 factor')
                    rows.append(finalize(c));ids.append(c['id'])
        groups.append(dict(id='active-budget-d'+str(d),ids=ids))
    if len(rows)!=20 or len({c['id'] for c in rows})!=20:raise ValueError('finite table changed')
    output.mkdir(parents=True,exist_ok=False);(output/'main').mkdir()
    for c in rows:atomic_json(output/'main'/(c['id']+'.json'),c)
    # Main3 milestone weights + latest optimizer; optional256M adds one weight
    # and one optimizer checkpoint while retaining the main128M endpoint.
    params=sum(c['planned_inventory']['lm_trainable'] for c in rows)
    catalog=dict(schema='tide-v10-frozen-table-v1',created_at=now(),cases=[c['id'] for c in rows],
        main_target_tokens=128*M,extension_target_tokens=256*M,continuation_groups=groups,selection=selection,
        expected_parameter_checkpoint_gib=dict(main=params*24/2**30,with_extensions=params*40/2**30,
            scope='tensor bytes before serialization, metadata, recordings and transient recovery probes'),
        confirmation=dict(tokens=262144,endpoints=['all eligible128M','all eligible256M'],
            comparisons='width within activity; q10/q3/dense within active budget; d128 same-shape pools; calibrated TF',
            selection_feedback=False,test_access=False))
    atomic_json(output/'catalog.json',catalog)
    write_review_json(evidence,dict(schema='tide-v10-selection-review-v1',created_at=now(),catalog=catalog,
        hb_calibrations=calibrations,controls_result_sha256=digest(control_path),
        table=[dict(id=c['id'],group=c['study_id'],parameters=c['planned_inventory']['lm_trainable'],
            microbatch=c['training']['batch'],global_batch=64,peak_mib=c['resource_admission']['peak_mib'],
            rate=c['resource_admission']['targets_per_second'],calibrated_active_parameters=c.get('calibrated_active_parameters'),
            calibrated_full_all_fraction=c.get('calibrated_full_all_fraction')) for c in rows],
        limitations=['calibration matching is not a guarantee of later activity matching',
            'dense physical Full with masks; no achieved sparse compute saving',
            'TF is a near-active-budget language control; its role mix differs',
            'new attention omits historical normal-tiny attention flooring; strict failures retained']))
    print(json.dumps(catalog,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd())
    p.add_argument('--output',type=Path,default=Path('configs/studies/language/direct-v10/table'))
    p.add_argument('--evidence',type=Path,default=Path('docs/evidence/direct-v10-selection.json'))
    a=p.parse_args();prepare(a.root.resolve(),a.output,a.evidence)

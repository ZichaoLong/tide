#!/usr/bin/env python3
"""Freeze the approved candidate table using completed full-update calibration."""
import argparse
from itertools import zip_longest
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.moe_config import config,screening_configs,SHAPES,M
from tide_lab.records import atomic_json,digest,now


def main(a):
    a.output.mkdir(parents=True,exist_ok=False);dest=a.output/'configs';dest.mkdir()
    measurements={};micros={}
    for name in SHAPES:
        path=a.root/'runs'/('moe-batch-'+name)/'result.json';r=json.loads(path.read_text())
        projection=json.loads(path.with_name('projection.json').read_text())
        if r['state']!='completed' or not projection.get('readback_matches') or projection.get('project')!='tide-learning-lab':
            raise ValueError('batch/Lens admission incomplete: '+name)
        rows=[v for v in r['cases'] if v['state']=='completed' and v['peak_mib']<24576]
        best=max(rows,key=lambda x:x['positions_per_second']);micros[name]=best['microbatch']
        measurements[name]=dict(peak_mib=best['peak_mib'],targets_per_second=best['positions_per_second'],
            measured_source=r['source']['commit'],record=str(path.relative_to(a.root)),record_sha256=digest(path),
            selection='fastest measured full global64 update with >=8GiB extra HBM allowance; peak below24GiB',
            implementation_note='integer-sort baseline; new lossless FP32-key sort has separate scope check and E64 full-update recheck')
    rows=screening_configs(micros);grouped=[[c for c in rows if c['selection_key']==s] for s in SHAPES]
    rows=[c for batch in zip_longest(*grouped) for c in batch if c is not None]
    mapping={}
    for c in rows:
        c['resource_admission']=measurements[c['selection_key']]
        atomic_json(dest/(c['id']+'.json'),c);mapping[c['id']]=[]
        for seed in (11,22):
            v=config(c['selection_key'],c['training']['lr'],c['training']['balance_coefficient'],seed,micros[c['selection_key']])
            v['resource_admission']=measurements[c['selection_key']]
            atomic_json(dest/(v['id']+'.json'),v);mapping[c['id']].append(v['id'])
    catalog=dict(schema='tide-moe-controls-table-v1',created_at=now(),screen_cases=[c['id'] for c in rows],
        main_by_recipe=mapping,main_target_tokens=128*M,screen_targets=[8*M,32*M],
        max_main_trajectories=12,max_selected32=12,automatic_extension=False,
        unused_configs='Possible fresh seeds only; controller submits one winning recipe per shape, not the whole set',
        data='runs/v9-data-r3/data-lock.json',data_lock_sha256=digest(a.root/'runs/v9-data-r3/data-lock.json'),
        selection='per-shape complete finite validation NLL; tie exact equality by ID; no confirmation feedback',
        measurements=measurements)
    atomic_json(a.output/'catalog.json',catalog)
    print(json.dumps(dict(screen_configs=len(rows),conditional_seed_configs=sum(len(v) for v in mapping.values()),micros=micros)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())

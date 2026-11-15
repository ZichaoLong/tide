#!/usr/bin/env python3
"""Resolve physical batches using measured shape/implementation calibration only."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json,digest

def calibration_key(variant):
    if variant in ('base-q10','share8-q10'):return 'base-q10'
    if variant=='base128-q3':return variant
    if variant.startswith('share8-d'):return variant
    if variant.startswith(('share','fullshare')):return 'share8-q3'
    return variant if variant in ('top2-q3','ports4-q3') else 'base-q3'

def main():
    root=Path.cwd();base=root/'configs/studies/language/hb-v12';table=json.loads((base/'table.json').read_text())
    for case in table['cases']:
        key=calibration_key(case['variant']);path=root/'runs'/('v12-cal-'+key)/'result.json';r=json.loads(path.read_text())
        if r['state']!='completed':raise ValueError('failed calibration: '+key)
        projection=json.loads((path.parent/'projection.json').read_text())
        if projection.get('status')!='verified' or not projection.get('readback_matches'):raise ValueError('Lens calibration readback not verified')
        selected=r['selected'];p=root/case['config'];c=json.loads(p.read_text())
        c['training'].update(batch=selected['micro'],eval_batch=min(16,selected['micro']))
        c['resource_admission']=dict(selected,source=str(path.relative_to(root)),sha256=digest(path),
            reuse_scope='same d/region shape; sharing variants use share8 except q10 conservative base; additional read projection uses base peak plus safety margin',
            timing='complete synthetic L256 global64 logical update; separately observed physical microbatch and accumulation')
        atomic_json(p,c)
    table['calibration_resolved']=True;atomic_json(base/'table.json',table)

if __name__=='__main__':main()

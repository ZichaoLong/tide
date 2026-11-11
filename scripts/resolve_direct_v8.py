#!/usr/bin/env python3
"""Public planning metadata only; the direct training source imports no executor."""
import argparse
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tidegraph.planning.recipes import prepare
from tide_lab.campaign_config import local_candidates
from tide_lab.records import atomic_json,digest,now
from scripts.prepare_direct_v8 import common


def recipe(row,topology,width=128):
    r=copy.deepcopy(row['recipe']);r.update(topology=topology,width=width,depth=4 if topology=='fc' else 2)
    if r['family']=='mixed' and topology=='fc':r['depth']=8
    mem=r['memory_overrides']
    if r['family']=='kv' and r['profile']!='textbook':
        mem.setdefault('heads',1)
        if mem['heads']==1:
            mem.setdefault('full_frequencies',[[]]);mem.setdefault('distance_slopes',[0.])
    if row['id']=='ffn2x':mem['ffn_width']=2*width
    if row['id']=='ffn8x':mem['ffn_width']=8*width
    if row['id'] in ('rope-half','rope-full'):
        rotation=width//2 if row['id']=='rope-half' else width
        mem['full_frequencies']=[[10000.**(-2*j/rotation) for j in range(rotation//2)]]
    if row['id']=='rope-read':mem['read_frequencies']=[10000.**(-2*j/(width//4)) for j in range(width//8)]
    return r


def resolve(row,topology,width=128):
    r=recipe(row,topology,width);p,inventory=prepare(r,samples=1,positions=1,seed=11)
    # Export declarative metadata, not materialized weights or execution code.
    keys=('graph','memories','regions','classes','source_binding','source_hidden_width','alpha_agg','stride','width','upstream','sha256','recipe')
    plan={k:p[k] for k in keys}
    cid=topology+'-'+row['id']+'-s11';c=common(cid,'direct_semantic')
    c['direct']=dict(plan=plan);c['initialization']='role-keyed-fan-in-v1'
    c['candidate']=dict(id=row['id'],group=row['group'],recipe=r)
    c['comparison_scope']='Independent direct static local-function screening; dense physical Full plus explicit support masks. Public continuation equivalence and real sparse execution unchecked.'
    c['numerical_policy']=dict(name='direct-v8-normal-tiny',
        positive_exp='floor mathematically positive exponentials at dtype normal-tiny; zero derivative below floor; preserve mathematical zero masks',
        allocation='optimal-map implicit VJP; active branch endpoints saturated; singleton upper set follows mu; zero support has zero Full-boundary VJP',
        recurrence='finite nilpotent innovation solve, checked against serial recurrence; effective states at every grid position, not canonical raw continuation',
        strict_equivalence='not certified; independent numerical checks retained separately')
    return c


def main(a):
    a.output.mkdir(parents=True,exist_ok=False);rows=[]
    for topo in ('fc','hb'):
        for row in local_candidates():
            c=resolve(row,topo);atomic_json(a.output/(c['id']+'.json'),c);rows.append(c)
    atomic_json(a.output/'catalog.json',dict(schema='tide-direct-v8-local-table-v1',created_at=now(),
        cases=[c['id'] for c in rows],candidate_count=len(local_candidates()),
        scaling_notes=['KV default heads explicitly 1 so heads2 remains an intervention at d128.',
            'ffn2x/8x and RoPE rotate widths rescaled; window16/128 remain logical-time limits.',
            'mixed FC has depth8; both configurations and actual inventories must state structural difference.']))
    print(json.dumps(dict(cases=len(rows))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args())

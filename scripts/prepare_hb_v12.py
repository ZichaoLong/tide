#!/usr/bin/env python3
"""Freeze the authorized union of HB mechanism comparisons, not a Cartesian grid."""
import argparse,copy,json,time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.v10_config import hb,M
from scripts.prepare_direct_v10 import finalize
from tide_lab.records import atomic_json,now


VARIANTS=[
 ('base-q3',64,64,.03125,{},'CONTROL'),
 ('base-q10',64,28,.1,{},'CONTROL'),
 ('base128-q3',128,64,.03125,{},'WIDTH'),
 ('share0-q3',64,64,.03125,dict(share='ffn',rank=0),'SHARING'),
 ('share8-q3',64,64,.03125,dict(share='ffn',rank=8),'SHARING'),
 ('fullshare8-q3',64,64,.03125,dict(share='full',rank=8),'SHARING'),
 ('share8-q10',64,28,.1,dict(share='ffn',rank=8),'SHARING'),
 ('share8-d128-q3',128,64,.03125,dict(share='ffn',rank=8),'WIDTH'),
 ('share8-d192-q3',192,64,.03125,dict(share='ffn',rank=8),'WIDTH'),
 ('budget0-q3',64,64,.03125,dict(budget_fraction=0.),'ROUTING'),
 ('top2-q3',64,64,.03125,dict(route='top2'),'ROUTING'),
 ('share8-budget0-q3',64,64,.03125,dict(share='ffn',rank=8,budget_fraction=0.),'INTERACTION'),
 ('share8-top2-q3',64,64,.03125,dict(share='ffn',rank=8,route='top2'),'INTERACTION'),
 ('readcontent-q3',64,64,.03125,dict(read_content=True),'INFORMATION'),
 ('untied-q3',64,64,.03125,dict(readout='untied_mean'),'READOUT'),
 ('ports4-q3',64,64,.03125,dict(readout='ports4'),'READOUT'),
 ('position-q3',64,64,.03125,dict(position=True),'INFORMATION'),
 ('residual-q3',64,64,.03125,dict(residual=True),'INFORMATION'),
 ('share8-lr3-q3',64,64,.03125,dict(share='ffn',rank=8,lr=.003),'RECIPE'),
]


def build(name,d,r,target,options,group,seed):
    options=copy.deepcopy(options)
    c=hb(name,d,r,target,seed,micro=16 if d==64 else 8 if d==128 else 4,temperature=1.)
    # Keep original selected starting temperatures for exact shape controls.
    root=Path(__file__).resolve().parents[1]
    old=root/f'configs/studies/language/direct-v10/table/main/hb-d{min(d,128)}-r{r}-q{10 if target==.1 else 3}-s11.json'
    oldc=json.loads(old.read_text())
    for s,old_s in zip(c['direct']['plan']['regions'],oldc['direct']['plan']['regions']):
        s['allocation_temperature']=old_s['allocation_temperature']
        if 'budget_fraction' in options:s['total_fraction']=options['budget_fraction']
    options.pop('budget_fraction',None)
    c['training']['lr']=options.pop('lr',.001)
    if options.get('route')=='top2':
        c['implementation'].pop('activity_controller',None)
        for s in c['direct']['plan']['regions']:s['allocation_temperature']=1.
    c['implementation'].update(aggregate_vjp='ordered-v1',mechanism=dict(center_quad=True,**options))
    c['campaign']='hb-mechanisms-v12';c['study_id']='TL-HB-'+group+'-v12'
    c['head_tied']=options.get('readout','tied_mean')=='tied_mean'
    c['training'].update(target_tokens=128*M,milestones=[8*M,32*M,128*M],lr_decay_tokens=256*M,
        full_probe_batch=4,full_probe_length=32)
    c['comparison_scope']='V12 independent direct HB; explicit changed sharing/route/information function. Frozen graph, L256/global64,128Mi exposure; no foundation equivalence or achieved sparse speedup claim.'
    c['numerical_policy']['quad_origin']='subtract detached eligible row maximum before fixed-total solve; unchanged real-arithmetic optimum/VJP; original solver threshold2e-5'
    c['numerical_policy']['top2']='if selected: hard top2 after2Mi; capped sigmoid split with8*tanh logits; min(original budget,selected caps); different function'
    return finalize(c)


def main(a):
    a.output=a.output.resolve()
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'configs').mkdir()
    cases=[]
    for spec in VARIANTS:
        for seed in (11,22):
            c=build(*spec,seed);atomic_json(a.output/'configs'/(c['id']+'.json'),c)
            cases.append(dict(id=c['id'],variant=spec[0],config=str((a.output/'configs'/(c['id']+'.json')).relative_to(Path.cwd())),
                study_id=c['study_id'],parameters=c['planned_inventory']['lm_trainable'],width=spec[1],seed=seed))
    started=time.time();budget=dict(schema='tide-hb-v12-budget-v1',started_epoch=started,deadline_epoch=started+48*3600,
        max_parameters=200000000,max_npu_workers=12,aggregate_compute_cap=None,max_variants=19,max_training=38,
        max_new_disk_gib=96,target_new_disk_gib=64,minimum_quota_headroom_gib=200,
        target_tokens=128*M,maximum_tokens_per_trajectory=128*M)
    atomic_json(a.output/'budget.json',budget)
    resource=json.loads(Path('configs/studies/language/capacity-v11/resource.json').read_text())
    resource.update(deadline_epoch=budget['deadline_epoch'],latest_authorization='user approval HB sharing/width/routing and fuller controlled diagnosis',minimum_free_hbm_mib=56000)
    atomic_json(a.output/'resource.json',resource)
    atomic_json(a.output/'table.json',dict(schema='tide-hb-v12-table-v1',created_at=now(),cases=cases,
        selection='all19 named variants x seeds11/22; execution/pilot gates only, no quality ranking exclusion or automatic extension',
        confirmation='new262K whole-document split excluding prior v6/v9/v11 and official documents; test sealed',
        estimated_checkpoint_gib=sum(x['parameters'] for x in cases)*24/2**30,
        scope='dense masked physical execution; weight-sharing unique active parameters and application work separately interpreted'))
    print(json.dumps(dict(variants=19,runs=len(cases),budget=budget,parameters=[(c['id'],c['parameters']) for c in cases if c['seed']==11]),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('configs/studies/language/hb-v12'));main(p.parse_args())

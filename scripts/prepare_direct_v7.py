#!/usr/bin/env python3
"""Prepare the bounded direct-DAG learning table, never submit work implicitly."""
import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json,digest,now

ROOT=Path(__file__).resolve().parents[1]


def config(kind,model,*,initialization='fan_in',norm=True,position='balanced',seed=11,batch=32,residual=False):
    cost=json.loads((ROOT/'configs/checks/hb-cost-20261111.json').read_text())
    tag=(f'{model}-{initialization}-'+('rms' if norm else 'raw')) if kind=='direct_graph' else f'{model}-{position}'
    if residual:tag+='-depth-residual'
    tag+=f'-s{seed}'
    c=dict(schema='tide-direct-language-config-v1',campaign='direct-v7',id=tag,
        study_id='TL-DIRECT-LM-v7',kind=kind,seed=seed,dtype='float32',device='npu:0',threads=2,
        core={},transformer=None,
        training=dict(target_tokens=4194304,milestones=[262144,1048576,4194304],length=128,
            batch=batch,global_batch=32,eval_batch=32,lr=.001,lr_floor_fraction=.1,warmup_tokens=65536,
            clip_norm=1.,eps=1e-6,weight_decay=.01,eval_tokens=32768,initial_eval_tokens=32768,
            checkpoint_seconds=900,validation_every_tokens=262144,timeout_seconds=3600,log_every=16),
        checks=dict(core_atol=1e-6,core_rtol=1e-5,finite_exploration=True),
        evaluation=dict(test_policy='sealed; frozen finalist review only',inference_length=16),
        resources=dict(max_rss_gib=24),
        comparison_scope='Ordinary dense Torch DAG; not the complete TIDE function or a fix/equivalence claim for historical TIDE runs.',
        topology_source=dict(path='configs/checks/hb-cost-20261111.json',sha256=digest(ROOT/'configs/checks/hb-cost-20261111.json')))
    if kind=='direct_graph':
        graph=copy.deepcopy(cost['models'][model]['graph'])
        c['direct']=dict(graph=graph,width=128,attention=True,initialization=initialization,final_norm=norm,
            residual_scale=1/math.sqrt(2*max(graph['ranks'])) if residual else 1.)
    else:
        c['transformer']=dict(width=256,heads=4,ffn=1024,layers=int(model.removeprefix('tf')),
            position_scale=math.sqrt(2)*.02 if position=='balanced' else 1.)
    return c


def main(a):
    a.output.mkdir(parents=True,exist_ok=False);cases=[]
    for model in ('hb40','hb88'):
        for initialization in ('std015','fan_in'):
            for norm in (False,True):cases.append(config('direct_graph',model,initialization=initialization,norm=norm,batch=a.batch))
    for model in ('tf12','tf28'):
        for position in ('original','balanced'):cases.append(config('transformer',model,position=position,batch=a.batch))
    for c in cases:atomic_json(a.output/(c['id']+'.json'),c)
    catalog=dict(schema='tide-direct-v7-catalog-v1',created_at=now(),screen=[c['id'] for c in cases],
        screen_stop_tokens=262144,main_models=['hb40','hb88','fc16','tf12','tf28'],main_seeds=[11,22],
        main_target_tokens=4194304,conditional_max_cases=2,
        selection='Same initialization/norm for both HB sizes: mean complete 262K validation NLL, tie <=0.02 prefer fan_in+rms. Same position scale for both Transformer sizes: mean validation, tie <=0.02 prefer balanced.',
        conditional='If mean seed11/22 HB88-minus-HB40 NLL is >0.05 at both complete 1M and 4M, run two seed11 depth-residual controls with the chosen recipe to 4M.',
        non_goals=['No new TIDE execution implementation or quad/proportional imitation.',
                   'No test access, semantic Cartesian matrix, additional data or size beyond this table.'])
    atomic_json(a.output/'catalog.json',catalog)
    budget=dict(schema='tide-direct-v7-budget-v1',deadline_epoch=a.deadline_epoch,cpu_core_seconds=32*3600,
        npu_device_seconds=8*3600,max_npu_workers=4,max_new_disk_gib=32,minimum_quota_headroom_gib=100,
        minimum_available_ram_gib=128,max_total_rss_gib=128,max_total_threads=16)
    atomic_json(a.output/'budget.json',budget)
    policy=dict(schema='tide-shared-resource-policy-v1',authorization='user-2026-11-09-shared-low-load',
        latest_authorization='2026-11-11 execute accepted learning plan using direct implementations',
        candidate_ids=[0,1,2,3,4,5,6,7,8,9,10,11,14,15],minimum_free_hbm_mib=32768,
        max_aicore_percent=30,max_new_concurrent_npus=4,slot_namespace='tide-direct-v7',
        exclusive_device=False,deadline_epoch=a.deadline_epoch,prerequisites={})
    atomic_json(a.output/'resource.json',policy)
    print(json.dumps(dict(screen_cases=len(cases),main_trajectories=10,conditional_cases=2,budget=budget)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch',type=int,default=32);p.add_argument('--deadline-epoch',type=float,default=time.time()+12*3600)
    main(p.parse_args())

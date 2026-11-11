#!/usr/bin/env python3
"""Resolve the finite direct coverage table; never submit implicitly."""
import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json,now


def common(cid,kind,seed=11):
    return dict(schema='tide-direct-coverage-config-v1',campaign='direct-v8',id=cid,
        study_id='TL-DIRECT-TF-v8' if kind=='transformer' else 'TL-DIRECT-COVER-v8',
        kind=kind,seed=seed,dtype='float32',device='npu:0',threads=2,core={},transformer=None,
        training=dict(target_tokens=1048576,milestones=[1048576],length=128,batch=32,global_batch=32,
            eval_batch=32,lr=.001,lr_floor_fraction=.1,warmup_tokens=65536,clip_norm=1.,eps=1e-8,
            weight_decay=.1,decay_policy='matrix_only',betas=[.9,.95],eval_tokens=32768,
            initial_eval_tokens=32768,checkpoint_seconds=1800,validation_every_tokens=262144,
            timeout_seconds=8*3600,log_every=16),
        checks=dict(core_atol=1e-6,core_rtol=1e-5,finite_exploration=True),
        evaluation=dict(test_policy='sealed',inference_length=16),resources=dict(max_rss_gib=28),
        revisions=dict(mathematics='0770593838a37c790a568226930567ed60f99b83',
            foundation_metadata='74e7bbd62341d2ccd382194a2c52ad4fea033665'))


def transformers():
    rows=[]
    for layers in (6,12):
        for init in ('gpt2','fan_in_depth','legacy'):
            for lr in ((.001,) if init=='legacy' else (.0003,.001,.003)):
                cid=f'tf{layers}-{init.replace("_","-")}-lr{lr:g}-s11'
                c=common(cid,'transformer');c['training'].update(lr=lr,timeout_seconds=3600)
                c['transformer']=dict(width=256,heads=4,ffn=1024,layers=layers,
                    position_scale=math.sqrt(2)*.02,initialization=init)
                if init=='legacy':c['training'].update(eps=1e-6,weight_decay=.01,decay_policy='all',betas=[.9,.999])
                rows.append(c)
    return rows


def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    rows=transformers()
    for c in rows:atomic_json(a.output/(c['id']+'.json'),c)
    atomic_json(a.output/'tf-catalog.json',dict(schema='tide-direct-v8-tf-table-v1',created_at=now(),
        cases=[c['id'] for c in rows],selection='Per size, minimum complete finite 1M validation NLL; tie within 0.01 prefers lower LR; seed22 top two per size. Validation selection is not independent test acceptance.'))
    atomic_json(a.output/'budget.json',dict(schema='tide-direct-v8-budget-v1',deadline_epoch=a.deadline_epoch,
        cpu_core_seconds=768*3600,npu_device_seconds=240*3600,max_npu_workers=10,max_new_disk_gib=128,
        minimum_quota_headroom_gib=200,minimum_available_ram_gib=256,max_total_rss_gib=384,max_total_threads=64))
    atomic_json(a.output/'resource.json',dict(schema='tide-shared-resource-policy-v1',
        authorization='user-2026-11-09-shared-low-load',latest_authorization='2026-11-11 direct coverage and calibrated Transformer',
        candidate_ids=list(range(12))+[14,15],minimum_free_hbm_mib=32768,max_aicore_percent=35,
        max_new_concurrent_npus=10,slot_namespace='tide-direct-v8',exclusive_device=False,
        deadline_epoch=a.deadline_epoch,prerequisites={}))
    print(json.dumps(dict(transformer_cases=len(rows),deadline_epoch=a.deadline_epoch)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deadline-epoch',type=float,default=time.time()+48*3600);main(p.parse_args())

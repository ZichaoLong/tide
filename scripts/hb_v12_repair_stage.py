#!/usr/bin/env python3
"""Explicit, unchanged-config v12 source migration and a finite learning prefix."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from tide_lab.records import atomic_json,digest,now,Metrics
from tide_lab.train import runtime,cpu_tree
from tide_lab.supervisor import verify_source
from tide_lab.campaign_runtime import make_model,optimizer_for,exact_tree
from tide_lab.language_training import tensor_save
from tide_lab.v10_repair import assert_finite_tree
from scripts.campaign_stage import identity


def migrate(a,c,ident):
    if digest(a.config)!=digest(a.parent_config):raise ValueError('v12 repair cannot change configuration')
    old_source=verify_source(a.parent_source)
    if old_source['commit']!='5147701447dd15a092cc4b052f828b037ed98df4':raise ValueError('only original v12 source parents')
    proof=json.loads(a.qualification.read_text());source=Path(__file__).resolve().parents[1]
    if proof['state']!='completed' or proof['device']!='npu:0':raise ValueError('NPU qualification not completed')
    for p,h in proof['implementation_sha256'].items():
        if digest(source/p)!=h:raise ValueError('qualification implementation identity mismatch: '+p)
    parent_sha=digest(a.resume)
    if parent_sha!=a.parent_sha256:raise ValueError('parent checkpoint changed')
    state=torch.load(a.resume,map_location='cpu',weights_only=False)
    expected=dict(ident,source_commit=old_source['commit'],source_sha256=old_source['content']['tree_sha256'])
    if state['identity']!=expected or state['schema']!='tide-language-campaign-checkpoint-v1':
        raise ValueError('original source/config/data/install/backend identity mismatch')
    if state['progress'].get('probe') or not 0<state['progress']['tokens']<128*2**20:
        raise ValueError('not an unfinished scientific checkpoint')
    model=make_model(c,torch.device('cpu'));model.load_state_dict(state['model'],strict=True)
    opt=optimizer_for(model,c);opt.load_state_dict(state['optimizer'])
    if (model.embedding.weight is model.head.weight)!=c.get('head_tied',True):raise AssertionError('head alias changed')
    exact_tree(state['model'],cpu_tree(model.state_dict()));exact_tree(state['optimizer'],cpu_tree(opt.state_dict()))
    assert_finite_tree(state['model']);assert_finite_tree(state['optimizer'])
    ancestry=dict(kind='explicit_quad_pivot_source_migration',parent_checkpoint=str(a.resume),parent_sha256=parent_sha,
        parent_identity=state['identity'],qualification_sha256=digest(a.qualification),
        change='quad-pivot-v1 and race-tolerant disk observation; config/model/Adam/RNG/cursor/control values unchanged',
        limit='Numerical implementation changes future trajectory; not bitwise continuation of old source. New-source fresh-process recovery required.')
    tensor_save(a.output/'checkpoint.pt',dict(state,identity=ident,source_migration=ancestry))
    reread=torch.load(a.output/'checkpoint.pt',map_location='cpu',weights_only=False)
    for key in state:
        if key!='identity':exact_tree(state[key],reread[key],key)
    return dict(state='completed',ancestry=ancestry,progress=state['progress'],
        checkpoint_sha256=digest(a.output/'checkpoint.pt'),payload_values_preserved=True)


def main(a):
    if a.action in ('segment','compare'):
        from scripts.capacity_v11_extension_stage import main as recovery
        return recovery(a)
    c=json.loads(a.config.read_text());device=runtime(c['device'] if a.action=='prefix' else 'cpu',c['threads'],c['seed'])
    ident=identity(a.config,c,a.data,'train');a.output.mkdir(parents=True,exist_ok=False)
    r=dict(schema='tide-hb-v12-repair-stage-v1',state='running',action=a.action,config_id=c['id'],identity=ident,started_at=now())
    atomic_json(a.output/'result.json',r);metrics=None
    try:
        if a.action=='migrate':r.update(migrate(a,c,ident))
        else:
            from scripts.v10_repair_stage import segment
            saved=torch.load(a.resume,map_location='cpu',weights_only=False)
            exposure=saved['progress']['tokens'];del saved
            a.steps=min(a.steps,max(1,(128*2**20-exposure)//(c['training'].get('global_batch',c['training']['batch'])*c['training']['length'])))
            r['expected_updates']=a.steps
            atomic_json(a.output/'manifest.json',dict(project='tide-learning-lab',run_id=a.run_id,name=a.run_id,
                group='TL-HB-REPAIR-v12',study_id='TL-HB-REPAIR-v12',identity=ident,config=c,
                ancestry=dict(checkpoint=str(a.resume),sha256=digest(a.resume))))
            metrics=Metrics(a.output,a.run_id);a.action='pilot'
            r.update(segment(a,c,ident,device,metrics))
            if r['updates']!=r['expected_updates']:raise ValueError('learning prefix incomplete')
    except BaseException as e:r.update(state='failed',error=repr(e));raise
    finally:
        if metrics:metrics.close()
        r['finished_at']=now();atomic_json(a.output/'result.json',r)
        atomic_json(a.output/'completed.json',dict(producer_finished=True,state=r['state']))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=('migrate','segment','compare','prefix'))
    for n in ('config','data','output'):p.add_argument('--'+n,type=Path,required=True)
    for n in ('parent-config','parent-source','resume','qualification','checkpoint-a','checkpoint-b'):p.add_argument('--'+n,type=Path)
    p.add_argument('--parent-sha256');p.add_argument('--steps',type=int,default=2);p.add_argument('--run-id')
    p.add_argument('--deadline-epoch',type=float,required=True);main(p.parse_args())

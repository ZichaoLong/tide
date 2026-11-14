#!/usr/bin/env python3
"""Finite real-data calibration: activity/capacity matching or TF recipe."""
import argparse
import copy
import json
import math
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json,now,Metrics,project_to_lens


def main(a):
    import torch
    from tide_lab.train import runtime,source_identity,installed_identity
    from tide_lab.campaign_runtime import make_model,optimizer_for,logical_update,evaluate,memory
    from tide_lab.campaign_data import Corpus
    from tide_lab.v10_config import hb,transformer,M
    device=runtime('npu:0',2,11);a.output.mkdir(parents=True,exist_ok=False)
    if a.kind=='hb':
        c=hb(a.id,a.width,a.region,a.target,micro=a.micro,controller=True,temperature=a.temperature)
        for s in c['direct']['plan']['regions']:s.pop('activity_schedule',None)
        c['implementation']['activity_controller']['start_tokens']=0
        limit=M
    else:
        c=transformer(a.id,a.width,a.layers,initialization=a.initialization,lr=a.lr);limit=2*M
    c['id']=a.id;c['study_id']='TL-CALIBRATION-v10';c['training'].update(eval_tokens=32768,initial_eval_tokens=32768,warmup_tokens=65536)
    atomic_json(a.output/'config.json',c);run_id='tl-'+a.id
    report=dict(schema='tide-v10-learning-calibration-v1',state='running',source=source_identity(),installation=installed_identity(),
        started_at=now(),config=c,updates=[],scope='finite recipe/activity selection, not independent learning confirmation; fresh-process formal recovery follows separately')
    atomic_json(a.output/'result.json',report)
    atomic_json(a.output/'manifest.json',dict(schema='tide-lab-run-v1',project='tide-learning-lab',run_id=run_id,name=run_id,
        group='TL-CALIBRATION-v10',study_id='TL-CALIBRATION-v10',config=c,created_at=now()))
    metrics=Metrics(a.output,run_id);corpus=Corpus(a.data,256);model=make_model(c,device);opt=optimizer_for(model,c)
    report['parameters']=sum(p.numel() for p in model.parameters());tokens=0;cursor=0;step=0;started=time.monotonic()
    try:
        while tokens<limit and time.time()<a.deadline_epoch-60:
            ids,cursor=corpus.sample_ids(cursor,64,11);batches=[];remain=limit-tokens
            for i in range(0,len(ids),c['training']['batch']):
                if remain<=0:break
                b=corpus.batch('train',ids[i:i+c['training']['batch']],device,token_limit=remain)
                remain-=int((b['labels']!=-100).sum());batches.append(b)
            row=logical_update(model,opt,batches,c,tokens,device);tokens+=row['tokens'];step+=1
            # Only bounded scalar records; no per-step checkpoint or tensor dump.
            selected={k:v for k,v in row['work'].items() if k.startswith(('activity/','work/active_parameter','work/router_multi'))}
            grad=[p.grad.detach().square().sum() for p in model.selectors.parameters() if p.grad is not None] if hasattr(model,'selectors') else []
            values={'train/tokens':tokens,'train/loss':row['loss'],'train/step_seconds':row['seconds'],
                'train/tokens_per_second':row['tokens']/row['seconds'],'gradient/router_l2':float(torch.stack(grad).sum().sqrt()) if grad else 0.,**selected,**memory(device)}
            metrics.log(step,values);report['updates'].append(values)
            report.update(tokens=tokens,steps=step);atomic_json(a.output/'result.json',report)
        if tokens!=limit:raise TimeoutError('finite calibration exposure not reached')
        val=evaluate(model,corpus,c,device,deadline=a.deadline_epoch-20)
        if val['state']!='completed':raise TimeoutError('calibration validation incomplete')
        atomic_json(a.output/'validation.json',val);metrics.log(step,{'validation/nll':val['nll'],'train/tokens':tokens})
        tail=report['updates'][-16:]
        report.update(state='completed',nll=val['nll'],tokens=tokens,seconds=time.monotonic()-started,
            mean_tail={k:statistics.mean(x[k] for x in tail) for k in tail[0] if k.startswith(('activity/','gradient/','train/tokens_per_second','work/router_multi'))})
        if a.kind=='hb':
            report['selected_temperature']=max(.005,min(32.,a.temperature*model.activity_state['scale']))
            report['controller_state']=model.activity_state
    except BaseException as error:
        report.update(state='failed',error=repr(error),tokens=tokens,seconds=time.monotonic()-started);raise
    finally:
        report['finished_at']=now();atomic_json(a.output/'result.json',report);metrics.close()
        atomic_json(a.output/'completed.json',dict(producer_finished=True,state=report['state']));project_to_lens(a.output,a.lens_cli)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=('hb','tf'),required=True);p.add_argument('--id',required=True)
    for n in ('output','data'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--width',type=int,required=True);p.add_argument('--region',type=int,default=20);p.add_argument('--target',type=float,default=.1)
    p.add_argument('--micro',type=int,default=8);p.add_argument('--layers',type=int,default=6);p.add_argument('--initialization',default='fan_in_depth')
    p.add_argument('--temperature',type=float,default=.25)
    p.add_argument('--lr',type=float,default=.001);p.add_argument('--deadline-epoch',type=float,required=True);p.add_argument('--lens-cli',required=True)
    main(p.parse_args())

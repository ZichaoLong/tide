#!/usr/bin/env python3
"""Predeclared independent confirmation, never the sealed test split."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json,digest,now,Metrics,project_to_lens


def main(a):
    import torch
    from tide_lab.train import runtime,source_identity,installed_identity
    from tide_lab.campaign_data import Corpus
    from tide_lab.campaign_runtime import make_model,evaluate,memory
    c=json.loads(a.config.read_text());device=runtime(c['device'],c['threads'],c['seed'])
    a.output.mkdir(parents=True,exist_ok=False);run_id='tl-v10-confirm-'+c['id']
    report=dict(schema='tide-v10-confirmation-v1',state='running',started_at=now(),source=source_identity(),installation=installed_identity(),
        config_sha256=digest(a.config),checkpoint=str(a.checkpoint),checkpoint_sha256=digest(a.checkpoint),test_access=False)
    atomic_json(a.output/'result.json',report)
    metrics=Metrics(a.output,run_id)
    atomic_json(a.output/'manifest.json',dict(schema='tide-lab-run-v1',project='tide-learning-lab',run_id=run_id,name=run_id,
        group='TL-CONFIRMATION-v10',study_id='TL-CONFIRMATION-v10',config=c,created_at=now()))
    try:
        state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        if state['identity']['config_sha256']!=digest(a.config) or state['identity']['data_lock_sha256']!=digest(a.data/'data-lock.json'):
            raise ValueError('checkpoint config/data identity mismatch')
        model=make_model(c,device);model.load_state_dict(state['model'],strict=True)
        exposure=state['progress']['tokens'];report['training_identity']=state['identity'];report['exposure']=exposure
        if hasattr(model,'set_progress'):model.set_progress(exposure)
        del state
        corpus=Corpus(a.data,c['training']['length'],extra_splits=('confirmation',))
        value=evaluate(model,corpus,c,device,token_limit=262144,deadline=a.deadline_epoch-20,split='confirmation')
        atomic_json(a.output/'confirmation.json',value)
        if value['state']!='completed':raise TimeoutError('confirmation incomplete')
        metrics.log(0,{'confirmation/nll':value['nll'],'confirmation/tokens':value['tokens'],'train/tokens':exposure,**memory(device)})
        report.update(state='completed',nll=value['nll'],tokens=value['tokens'])
    except BaseException as e:report.update(state='failed',error=repr(e));raise
    finally:
        report['finished_at']=now();atomic_json(a.output/'result.json',report);metrics.close()
        atomic_json(a.output/'completed.json',dict(producer_finished=True,state=report['state']));project_to_lens(a.output,a.lens_cli)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('config','checkpoint','data','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--lens-cli',required=True);p.add_argument('--deadline-epoch',type=float,required=True);main(p.parse_args())

#!/usr/bin/env python3
"""One finite topology/control study; independent studies may run concurrently."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
from tide_lab.language_driver import StageRunner


def run(a):
    import torch
    from scripts.run_capability_fourth import exact_tree
    plan=json.loads(a.plan.read_text());cfg=plan['depth_batch'];torch.set_num_threads(1)
    if a.deadline_epoch!=cfg['deadline_epoch']:raise ValueError('changed shared deadline')
    limits=cfg['limits'];runner=StageRunner(a,plan,limits,cfg['compute_seconds']-45)
    runner.state.update(study_id=plan['study_id'],batch_id=plan['package_id'],training=[],admission={},
        axes=dict(execution='running',strict_numerical='original thresholds; inspect checks',
            discrete_semantics='pending',recovery='pending',learning='unchecked',performance='shared-resource observations'))
    runner.publish();cache_path=a.cache
    depth=plan['core']['depth']
    def updates(kind):return cfg.get('candidate_updates',{}).get(kind,cfg['updates'])
    def horizon(kind):return cfg.get('candidate_schedule_steps',{}).get(kind,cfg['schedule_steps'])
    def flags(kind,batch=8,length=32):
        out=['--kind',kind,'--batch',str(batch),'--length',str(length),'--depth',str(depth),'--seed','11']
        if a.route=='b':out+=['--cache',str(cache_path)]
        return out
    def checked_stage(*args,**kwargs):
        path,result=runner.stage(*args,**kwargs)
        if result['state']!='completed':raise RuntimeError('required stage failed: '+str(args[0]))
        return path,result
    def admit(kind):
        _,result=checked_stage('checks-'+kind,'checks','admission',600,flags(kind,2,3),record=False)
        if not result.get('admitted_for_finite_exploration'):raise RuntimeError('mechanism admission failed')
        common=flags(kind,2,3)+['--steps','3','--probe','--retain-step','1']
        if a.route=='a':common+=['--schedule-steps',str(horizon(kind))]
        paths=[]
        for label,extra in [('full',[]),('split',['--max-updates','1']),('continued',None)]:
            if extra is None:extra=['--resume',str(paths[1]/'checkpoint.pt')]
            path,_=checked_stage('resume-'+kind+'-'+label,'train','admission',240,common+extra,record=False);paths.append(path)
        full=torch.load(paths[0]/'checkpoint.pt',map_location='cpu',weights_only=False)
        continued=torch.load(paths[2]/'checkpoint.pt',map_location='cpu',weights_only=False)
        exact_tree(full,continued)
        exact_tree(torch.load(paths[0]/'checkpoint-step1.pt',map_location='cpu',weights_only=False),
                   torch.load(paths[1]/'checkpoint.pt',map_location='cpu',weights_only=False))
        runner.state['admission'][kind]=dict(mechanisms='passed',recovery='passed',
            scope='B2/L3 three steps in fresh processes; exact full/continued and saved prefix',
            strict_numerical='read checks; finite disagreements do not disappear');runner.publish()
    def evaluation(kind,checkpoint,label,steps=None):
        extra=flags(kind)+['--checkpoint',str(checkpoint)]
        checked_stage('test-'+label,'evaluate','final',900,extra)
        if steps is None:
            checked_stage('inference-'+label,'inference','final',450,extra)
            if a.route=='a':
                checked_stage('context-reset-'+label,'evaluate','final',900,extra+['--intervention','context_reset'])
    try:
        if a.route=='b' and cache_path is None:
            path,_=checked_stage('cache','cache','admission',420,['--length','32','--reuse-cache',str(a.parent_cache)],record=False)
            cache_path=path/'features.pt'
        if cfg.get('requires_learning'):
            for path in cfg['requires_learning']:
                row=json.loads(Path(path).read_text())
                if row.get('state')!='completed' or not row.get('learning_signal'):
                    runner.state['skipped'].append(dict(name='conditional-nesting',reason='HB16 validation learning prerequisite absent'))
                    runner.finish('completed');return 0
        cals={}
        for kind in cfg['candidates']:
            admit(kind)
            _,cal=checked_stage('calibrate-'+kind,'calibrate','admission',600,flags(kind),record=False)
            cals[kind]=cal
        # Actual time constrains this pair; sparse/dense throughput ratios do not.
        forecast=sum(1.2*updates(k)*r['mean_step_seconds']+180 for k,r in cals.items())
        runner.state['decisions'].append(dict(name='fixed-B8-comparison',forecast_training_seconds=forecast,
            remaining_training_seconds=runner.remaining('training'),batch=8,accumulation=1,
            reason='same sampler/exposure and schedule prefix as earlier FC4; see separate batch/backend calibration'))
        runner.publish()
        if forecast>runner.remaining('training'):
            raise RuntimeError('complete pair training does not fit measured remaining budget; learning not adjudicated')
        for kind in cfg['candidates']:
            extra=flags(kind)+['--steps',str(updates(kind))]
            if a.route=='a':extra+=['--schedule-steps',str(horizon(kind))]
            for step in cfg.get('retain_steps',[]):extra+=['--retain-step',str(step)]
            path,result=checked_stage('train-'+kind+'-s11','train','training',runner.remaining('training'),extra)
            runner.state['training'].append(dict(kind=kind,output=str(path),steps=result['steps'],
                valid_tokens=result['valid_tokens'],learning_signal=result['learning_signal']))
            runner.publish();evaluation(kind,path/'checkpoint.pt',kind)
            for step in cfg.get('evaluate_prefixes',[]):evaluation(kind,path/f'checkpoint-step{step}.pt',kind+f'-prefix{step}',steps=step)
        runner.state['axes'].update(execution='completed',discrete_semantics='passed in affected checks',
            recovery='three-step exact; long trajectory untested',learning='heldout records; human review pending')
        runner.finish();return 0
    except BaseException as error:
        runner.state['axes']['execution']='failed';runner.finish('failed',repr(error));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','data','output','asset-path','recording-cli'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--route',choices=['a','b'],required=True)
    p.add_argument('--cache',type=Path);p.add_argument('--parent-cache',type=Path)
    p.add_argument('--deadline-epoch',type=float,required=True);p.add_argument('--cpu-only',action='store_true')
    raise SystemExit(run(p.parse_args()))

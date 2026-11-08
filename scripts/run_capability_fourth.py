#!/usr/bin/env python3
"""Sequential B4 dependency chain inside one bounded NPU allocation.

Actual results only determine the predeclared dependent stages. No prompts,
thresholds, width, alpha, optimizer or candidate lists are tuned by the driver.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json,digest,now


def exact_tree(left,right,path='root'):
    import torch
    if isinstance(left,torch.Tensor):
        if not isinstance(right,torch.Tensor) or left.dtype!=right.dtype or not torch.equal(left,right):
            raise AssertionError('exact tensor recovery differs: '+path)
    elif isinstance(left,dict):
        if not isinstance(right,dict) or left.keys()!=right.keys():raise AssertionError('recovery keys: '+path)
        for key in left:exact_tree(left[key],right[key],path+'.'+str(key))
    elif isinstance(left,(list,tuple)):
        if type(left)!=type(right) or len(left)!=len(right):raise AssertionError('recovery sequence: '+path)
        for index,(a,b) in enumerate(zip(left,right)):exact_tree(a,b,path+f'[{index}]')
    elif left!=right:
        raise AssertionError('recovery value differs: '+path)


def run(a):
    plan=json.loads(a.plan.read_text());a.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();deadline=min(a.deadline_epoch,time.time()+plan['budget']['compute_seconds'])
    state=dict(schema='tide-b4-stage-chain-v1',started_at=now(),state='running',
        plan_sha256=digest(a.plan),deadline_epoch=a.deadline_epoch,compute_deadline_epoch=deadline,
        steps=[],skipped=[],decisions=[],training=[])
    atomic_json(a.output/'stage.json',state)
    def remaining():return deadline-time.time()
    def stage(name,action,cap,extra=(),record=True):
        if remaining()<min(cap,60)+10:raise TimeoutError('remaining common budget insufficient for next stage')
        destination=a.output/name
        argv=[sys.executable,'scripts/capability_stage.py',action,'--plan',str(a.plan.resolve()),
              '--output',str(destination),'--run-id','tl-b4-'+name,'--asset-path',str(a.asset_path),*map(str,extra)]
        follow=record and action=='train'
        if record and not follow:argv+=['--recording-cli',str(a.recording_cli)]
        wrapper=a.output/(name+'-wrapper')
        command=[sys.executable,'scripts/bounded_workload.py','--max-seconds',str(min(cap,max(1,int(remaining())-10))),
                 '--deadline-epoch',str(deadline),'--output',str(wrapper),'--',*argv]
        row=dict(name=name,action=action,started_at=now(),output=str(destination),state='running')
        state['steps'].append(row);atomic_json(a.output/'stage.json',state)
        recorder=None;recording_log=None
        with (a.output/(name+'.log')).open('x') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            try:
                while child.poll() is None:
                    if follow and recorder is None and (destination/'metrics.jsonl').exists():
                        recording_log=(a.output/(name+'-recording.log')).open('x')
                        recorder=subprocess.Popen([str(a.recording_cli),'record','--events',str(destination/'metrics.jsonl'),
                            '--manifest',str(destination/'manifest.json'),'--report',str(destination/'projection.json'),
                            '--follow','--done-file',str(destination/'completed.json')],stdout=recording_log,stderr=subprocess.STDOUT,
                            env={k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','VIRTUAL_ENV')})
                    time.sleep(.2)
                exit_code=child.returncode
                if recorder is not None:
                    if not (destination/'completed.json').exists():
                        atomic_json(destination/'completed.json',dict(producer_finished=True,state='failed',exit_code=exit_code))
                    try:recorder.wait(timeout=max(1,min(45,remaining())))
                    except subprocess.TimeoutExpired:
                        recorder.terminate()
                        try:recorder.wait(timeout=5)
                        except subprocess.TimeoutExpired:recorder.kill();recorder.wait()
            finally:
                for process in (child,recorder):
                    if process is not None and process.poll() is None:
                        process.terminate()
                        try:process.wait(timeout=5)
                        except subprocess.TimeoutExpired:process.kill();process.wait()
                if recording_log is not None:recording_log.close()
        result=json.loads((destination/'result.json').read_text()) if (destination/'result.json').exists() else {}
        row.update(state=result.get('state','failed'),exit_code=exit_code,finished_at=now(),
                   result_sha256=digest(destination/'result.json') if result else None)
        atomic_json(a.output/'stage.json',state)
        print(json.dumps(row),flush=True)
        if exit_code or row['state'] not in ('completed','passed'):
            raise RuntimeError(f'{name} failed; no dependent work started')
        if record:
            projection=destination/'projection.json'
            # Recording is a separate axis; preserve a failure but do not turn
            # it into a numerical or quality result.
            row['lens']=json.loads(projection.read_text()) if projection.exists() else {'status':'failed'}
        return destination,result
    try:
        base,baseline=stage('baseline','prepare',plan['budget']['baseline_seconds'])
        if not baseline['selected_families']:
            state['skipped'].append(dict(stages=['information','checks','calibration','shadow','preservation'],
                                         reason='no task met preregistered baseline validation gate'))
            state['state']='completed_with_unmet_gate';return state
        cache=base/'features.pt';fixtures=base/'dataset.json'
        cache.chmod(0o444);fixtures.chmod(0o444)
        common=['--cache',cache]
        stage('information','information',plan['budget']['information_seconds'],common)
        stage('mechanisms','mechanisms',plan['budget']['admission_seconds'],common)
        # New application state is the shadow learner, not the immutable Qwen
        # teacher. Every declared dynamic state component is compared below.
        import torch
        resume_records=[]
        admission_start=time.monotonic()
        for kind in plan['training']['candidates']:
            if time.monotonic()-admission_start>plan['budget']['admission_seconds']:
                raise TimeoutError('shared resume admission budget exhausted')
            outputs=[]
            for part,limit,resume in [('full',3,None),('split',1,None),('resumed',3,'split')]:
                args=[*common,'--kind',kind,'--seed','11','--batch','2','--limit',str(limit)]
                if resume:args+=['--resume',a.output/f'check-{kind}-split'/'checkpoint.pt']
                path,result=stage(f'check-{kind}-{part}','train',180,args,record=False)
                outputs.append(path)
            full=torch.load(outputs[0]/'checkpoint.pt',map_location='cpu',weights_only=False)
            resumed=torch.load(outputs[2]/'checkpoint.pt',map_location='cpu',weights_only=False)
            exact_tree(full,resumed)
            resume_records.append(dict(kind=kind,state='passed',scope='fresh-process 3-step/B2 shadow learner, AdamW, normalization, data cursor, CPU/accelerator/Python RNG and full runtime identity',
                full_sha256=digest(outputs[0]/'checkpoint.pt'),resumed_sha256=digest(outputs[2]/'checkpoint.pt')))
        atomic_json(a.output/'resume-checks.json',dict(state='passed',checks=resume_records))
        calibration_path,calibration=stage('batch-calibration','calibrate',plan['budget']['calibration_seconds'],common)
        batch=calibration['batch'];atomic_json(a.output/'resolved.json',dict(batch=batch,plan_sha256=digest(a.plan),
            cache_sha256=digest(cache),calibration_sha256=digest(calibration_path/'result.json'),
            steps=plan['training']['steps'],optimizer=plan['training']['optimizer']))
        (a.output/'resolved.json').chmod(0o444)
        trained=[];tide_pilot_pass=False
        for seed_index,seed in enumerate(plan['training']['seeds']):
            if seed_index and not tide_pilot_pass:
                state['skipped'].append(dict(stage=f'seed{seed}',reason='TIDE seed11 local fit or validation preservation gate unmet'))
                break
            for kind in plan['training']['candidates']:
                # Reserve final heldout evaluation; calibrated estimate does not
                # silently expand the global or per-run caps.
                projected=calibration['projected_training_step_seconds'][kind]*1.35+120
                if remaining()<projected+plan['budget']['evaluation_seconds']:
                    state['skipped'].append(dict(stage=f'shadow-{kind}-s{seed}',reason='budget forecast leaves insufficient heldout evaluation reserve'))
                    continue
                name=f'shadow-{kind}-s{seed}'
                path,result=stage(name,'train',plan['training']['max_seconds_per_run']+90,
                    [*common,'--kind',kind,'--seed',seed,'--batch',batch])
                item=dict(kind=kind,seed=seed,path=str(path),fit_gate=result['learning_gate_met'],validation_preservation=None)
                trained.append(item);state['training'].append(item)
                if result['learning_gate_met']:
                    ev,assessment=stage(f'validation-{kind}-s{seed}','evaluate',360,
                        [*common,'--dataset',fixtures,'--kind',kind,'--seed',seed,'--checkpoint',path/'checkpoint.pt'])
                    item['validation_preservation']=assessment['preservation_screen_met']
                    if kind=='tide' and seed==11:tide_pilot_pass=assessment['preservation_screen_met']
                else:
                    state['skipped'].append(dict(stage=f'validation-{kind}-s{seed}',reason='local teacher fit gate unmet'))
                atomic_json(a.output/'stage.json',state)
        state['decisions'].append(dict(at=now(),training_decisions_locked=True,trained=trained,
            independent_test_used_for_decision=False))
        atomic_json(a.output/'stage.json',state)
        eval_start=time.monotonic()
        stage('test-baseline','evaluate',300,[*common,'--dataset',fixtures,'--split','test','--baseline-only'])
        stage('test-mean-control','evaluate',300,[*common,'--dataset',fixtures,'--split','test'])
        for item in trained:
            if not item['fit_gate']:
                state['skipped'].append(dict(stage=f"test-{item['kind']}-s{item['seed']}",reason='local fit gate unmet; no nonzero integration qualification'))
                continue
            if time.monotonic()-eval_start>plan['budget']['evaluation_seconds']:
                raise TimeoutError('final evaluation stage budget exhausted')
            stage(f"test-{item['kind']}-s{item['seed']}",'evaluate',360,
                [*common,'--dataset',fixtures,'--split','test','--kind',item['kind'],'--seed',item['seed'],
                 '--checkpoint',Path(item['path'])/'checkpoint.pt'])
        state['state']='completed'
    except BaseException as error:
        state.update(state='failed',error=repr(error));raise
    finally:
        state.update(finished_at=now(),elapsed_seconds=time.monotonic()-start)
        if state['state']=='running':state['state']='interrupted'
        atomic_json(a.output/'stage.json',state)
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('plan','output','asset-path','recording-cli'):parser.add_argument('--'+key,required=True,type=Path)
    parser.add_argument('--deadline-epoch',required=True,type=float)
    args=parser.parse_args();result=run(args)
    return 0 if result['state'] in ('completed','completed_with_unmet_gate') else 1


if __name__=='__main__':
    raise SystemExit(main())

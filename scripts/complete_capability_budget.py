#!/usr/bin/env python3
"""One finite completion after the frozen B4 attempt's diagnosed time cap.

The original service is never changed or interrupted. Wait for its actual
termination; only a declared per-training time-cap failure permits completion.
Workers execute the ORIGINAL frozen source/config on the ORIGINAL NPU mapping.
At most one exact continuation per TIDE candidate, within the original batch
deadline and aggregate allocated-compute budget. No extra optimizer updates.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.job import inspect
from tide_lab.records import atomic_json,digest,now
from tide_lab.supervisor import verify_source


def budget_failure(result):
    return result.get('state')=='failed' and result.get('error')=="TimeoutError('fixed per-run training budget exhausted')"


def read(path):return json.loads(Path(path).read_text())


def completion_worker(a):
    old=a.original_root.resolve();source=a.original_source.resolve();plan=source/'configs/studies/fourth/b-preserve.json'
    cfg=read(plan);budget=read(a.original_budget);decision=read(a.output/'completion-decision.json')
    compute_end=min(time.time()+decision['remaining_compute_seconds'],budget['deadline_epoch']-45)
    start=time.monotonic();record=dict(state='running',started_at=now(),steps=[],candidates=[],skipped=[],
        original_source_commit=verify_source(source)['commit'],original_plan_sha256=digest(plan),
        original_deadline_epoch=budget['deadline_epoch'],compute_deadline_epoch=compute_end,
        scope='same original candidates/128 optimizer steps; prior failed time-cap attempts retained; not extra training exposure selection')
    def publish():atomic_json(a.output/'completion.json',record)
    def remain():return compute_end-time.time()
    def execute(name,action,cap,extra=(),allow_budget=False):
        if remain()<30:raise TimeoutError('original aggregate budget exhausted')
        directory=a.output/name;wrapper=a.output/(name+'-wrapper')
        command=[sys.executable,str(source/'scripts/capability_stage.py'),action,'--plan',str(plan),'--output',str(directory),
            '--run-id','tl-b4-'+name,'--cache',str(old/'baseline/features.pt'),'--asset-path',str(a.asset_path),
            *map(str,extra)]
        follow=action=='train'
        if not follow:command+=['--recording-cli',str(a.recording_cli)]
        argv=[sys.executable,str(source/'scripts/bounded_workload.py'),'--max-seconds',str(min(cap,int(remain())-10)),
            '--deadline-epoch',str(compute_end),'--output',str(wrapper),'--',*command]
        row=dict(name=name,state='running',started_at=now(),record=str(directory/'result.json'))
        record['steps'].append(row);publish();recorder=None;writer_log=None
        with (a.output/(name+'.log')).open('x') as log:
            child=subprocess.Popen(argv,cwd=source,stdout=log,stderr=subprocess.STDOUT)
            try:
                while child.poll() is None:
                    if follow and recorder is None and (directory/'metrics.jsonl').exists():
                        writer_log=(a.output/(name+'-recording.log')).open('x')
                        recorder=subprocess.Popen([str(a.recording_cli),'record','--events',str(directory/'metrics.jsonl'),
                            '--manifest',str(directory/'manifest.json'),'--report',str(directory/'projection.json'),
                            '--follow','--done-file',str(directory/'completed.json')],stdout=writer_log,stderr=subprocess.STDOUT,
                            env={k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','VIRTUAL_ENV')})
                    time.sleep(.2)
                if recorder is not None:
                    if not (directory/'completed.json').exists():
                        atomic_json(directory/'completed.json',dict(producer_finished=True,state='failed',exit_code=child.returncode))
                    try:recorder.wait(timeout=max(1,min(45,remain())))
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
                if writer_log is not None:writer_log.close()
        result=read(directory/'result.json') if (directory/'result.json').exists() else {'state':'failed','error':'missing result'}
        row.update(state=result['state'],exit_code=child.returncode,finished_at=now(),
                   result_sha256=digest(directory/'result.json') if (directory/'result.json').exists() else None)
        publish();print(json.dumps(row),flush=True)
        cleanup=read(wrapper/'result.json') if (wrapper/'result.json').exists() else {}
        if not cleanup.get('termination_verified'):
            raise RuntimeError('stage descendants not verified terminated: '+name)
        if child.returncode or result['state'] not in ('completed','passed'):
            if allow_budget and budget_failure(result) and (directory/'checkpoint.pt').exists():return directory,result
            raise RuntimeError('completion stage failed: '+name+': '+result.get('error','unknown'))
        return directory,result
    try:
        if os.getenv('ASCEND_RT_VISIBLE_DEVICES')!=str(decision['physical_id']):
            raise ValueError('resume requires original physical mapping')
        batch=read(old/'resolved.json')['batch'];base_args=['--batch',batch]
        completed=[];tide_pilot=False
        for seed in (11,22):
            if seed==22 and not tide_pilot:
                record['skipped'].append(dict(seed=seed,reason='original TIDE pilot fit/preservation gate unmet'));break
            if seed==22 and remain()<3000:
                record['skipped'].append(dict(seed=seed,reason='reserve two bounded TIDE attempts and final evaluation within original aggregate budget'));break
            for kind in ('tide','mlp'):
                candidate=dict(kind=kind,seed=seed,attempts=[],state='pending')
                record['candidates'].append(candidate);publish()
                if kind=='tide' and seed==11:
                    original=old/'shadow-tide-s11'
                    if not budget_failure(read(original/'result.json')):raise ValueError('original failure changed')
                    candidate['attempts'].append(dict(path=str(original),state='failed',kind='retained_original_time_cap'))
                    path,result=execute('shadow-tide-s11-resume1','train',1300,
                        [*base_args,'--kind',kind,'--seed',seed,'--resume',original/'checkpoint.pt'])
                else:
                    path,result=execute(f'shadow-{kind}-s{seed}-completion','train',1300 if kind=='tide' else 120,
                        [*base_args,'--kind',kind,'--seed',seed],allow_budget=kind=='tide')
                    if budget_failure(result):
                        candidate['attempts'].append(dict(path=str(path),state=result['state'],kind='time_cap'))
                        if remain()<1500:raise TimeoutError('no remaining budget for one bounded exact continuation and evaluation')
                        path,result=execute(f'shadow-{kind}-s{seed}-resume1','train',1300,
                            [*base_args,'--kind',kind,'--seed',seed,'--resume',path/'checkpoint.pt'])
                candidate.update(state=result['state'],path=str(path),learning_gate_met=result['learning_gate_met'],
                                 logical_steps=result['steps'],logical_examples=result['trained_examples'])
                candidate['attempts'].append(dict(path=str(path),state=result['state'],kind='final'))
                completed.append(candidate)
                if result['learning_gate_met']:
                    ev,quality=execute(f'validation-{kind}-s{seed}-completion','evaluate',300,
                        ['--dataset',old/'baseline/dataset.json','--kind',kind,'--seed',seed,'--checkpoint',path/'checkpoint.pt'])
                    candidate['validation_preservation_screen_met']=quality['preservation_screen_met']
                    if kind=='tide' and seed==11:tide_pilot=quality['preservation_screen_met']
                else:candidate['validation_preservation_screen_met']=None
                publish()
        record['training_decisions_locked_at']=now();record['independent_test_used_for_decisions']=False;publish()
        execute('test-baseline-completion','evaluate',300,['--dataset',old/'baseline/dataset.json','--split','test','--baseline-only'])
        execute('test-mean-control-completion','evaluate',300,['--dataset',old/'baseline/dataset.json','--split','test'])
        for item in completed:
            if not item['learning_gate_met']:
                record['skipped'].append(dict(stage='test-'+item['kind']+str(item['seed']),reason='local fit gate unmet'));continue
            execute(f"test-{item['kind']}-s{item['seed']}-completion",'evaluate',300,
                ['--dataset',old/'baseline/dataset.json','--split','test','--kind',item['kind'],'--seed',item['seed'],
                 '--checkpoint',Path(item['path'])/'checkpoint.pt'])
        record['state']='completed'
    except BaseException as error:
        record.update(state='failed',error=repr(error));raise
    finally:
        record.update(finished_at=now(),elapsed_seconds=time.monotonic()-start);publish()
    return 0


def coordinate(a):
    a.output.mkdir(parents=True,exist_ok=False)
    budget=read(a.original_budget);state=dict(state='waiting_dependency',started_at=now(),
        deadline_epoch=budget['deadline_epoch'],original_job=str(a.original_job))
    atomic_json(a.output/'coordinator.json',state)
    while time.time()<budget['deadline_epoch']-90:
        original=inspect(a.original_job/'job.json')
        if (a.original_job/'stop-request.json').exists():
            state.update(state='cancelled',reason='user stopped original service; no automatic continuation');break
        if original['termination_verified_now'] and original['state']['state'] in ('completed','failed','interrupted','stopped'):
            break
        time.sleep(2)
    else:state.update(state='deadline_exhausted',reason='original wall deadline reached')
    if state['state']!='waiting_dependency':
        atomic_json(a.output/'coordinator.json',state);return 0
    original_result=a.original_root/'shadow-tide-s11/result.json'
    if not original_result.exists() or not budget_failure(read(original_result)):
        state.update(state='no_action',reason='original did not end with the diagnosed candidate time cap')
        atomic_json(a.output/'coordinator.json',state);return 0
    checkpoint=a.original_root/'shadow-tide-s11/checkpoint.pt'
    if not checkpoint.exists():raise ValueError('no published complete checkpoint; do not restart candidate silently')
    original_elapsed=original['state']['elapsed_seconds']
    remaining=budget['compute_seconds']-original_elapsed-30
    if remaining<1800:
        state.update(state='budget_exhausted',reason='insufficient original aggregate compute to finish candidate and evaluate')
        atomic_json(a.output/'coordinator.json',state);return 0
    physical=original['queue']['physical_id']
    decision=dict(observed_at=now(),physical_id=physical,original_service_seconds_charged=original_elapsed,
        remaining_compute_seconds=remaining,original_deadline_epoch=budget['deadline_epoch'],
        checkpoint_sha256=digest(checkpoint),checkpoint_metadata=read(checkpoint.with_suffix('.json')),
        reason='calibration chose B128 by throughput but estimated 128-step time exceeds 1200s single-attempt cap; preserve fixed source and resume same trajectory, maximum one continuation per candidate')
    atomic_json(a.output/'completion-decision.json',decision)
    wait=max(0,int(budget['deadline_epoch']-time.time()-remaining-60))
    if wait<1:raise TimeoutError('no remaining original queue window')
    worker=[sys.executable,str(Path(__file__).resolve()),'worker',
        '--original-root',str(a.original_root),'--original-source',str(a.original_source),
        '--original-job',str(a.original_job),'--original-budget',str(a.original_budget),
        '--output',str(a.output),'--asset-path',str(a.asset_path),
        '--recording-cli',str(a.recording_cli),'--queue-helper',str(a.queue_helper)]
    bounded=[sys.executable,str(a.original_source/'scripts/bounded_workload.py'),'--max-seconds',str(remaining),
        '--deadline-epoch',str(budget['deadline_epoch']-45),'--output',str(a.output/'allocated-wrapper'),'--',*worker]
    command=[sys.executable,str(a.queue_helper),'--project','tide-learning-lab','--job-id','s4-capability-completion-npu',
        '--priority','0','--npu-count','1','--candidate-ids',str(physical),'--max-wait-seconds',str(wait),
        '--state-file',str(a.output/'queue-state.json'),'--',*bounded]
    state.update(state='queued_for_original_mapping',queue_command=command,remaining_compute_seconds=remaining)
    atomic_json(a.output/'coordinator.json',state)
    result=subprocess.run(command)
    state.update(state='completed' if result.returncode==0 else 'failed',exit_code=result.returncode,finished_at=now())
    atomic_json(a.output/'coordinator.json',state)
    return result.returncode


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['coordinate','worker'])
    for key in ('original-root','original-source','original-job','original-budget','output','asset-path','recording-cli','queue-helper'):
        p.add_argument('--'+key,required=True,type=Path)
    a=p.parse_args()
    for key,value in vars(a).items():
        if isinstance(value,Path):setattr(a,key,value.resolve())
    return coordinate(a) if a.mode=='coordinate' else completion_worker(a)


if __name__=='__main__':raise SystemExit(main())

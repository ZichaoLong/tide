#!/usr/bin/env python3
"""Finite follow-up: separate admission axes, calibrate batch, then one learning exploration."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json,digest,now
from tide_lab.supervisor import group_processes


class ChildFailure(RuntimeError):
    pass


class ChildTimeout(ChildFailure):
    pass


class CleanupFailure(RuntimeError):
    """Never downgraded to an optional observation failure."""


def bounded_command(argv, log_path, timeout, record, publish, grace_seconds=5):
    """Own the complete phase process group, including training and Lens children.

    These children inherit the service cgroup but have a dedicated process group.
    Do not return to the device-queue wrapper until their termination is verified.
    The caller reserves two grace intervals inside its phase/stage wall budget.
    """
    child = None
    failure = None
    started = time.monotonic()
    try:
        with Path(log_path).open('x') as log:
            child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record.update(process_group=child.pid, state='running', started_at=now())
            publish()
            try:
                code = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                raise ChildTimeout(f'phase exceeded {timeout:.3f}s execution window')
            record['exit_code'] = code
            if code:
                raise ChildFailure(f'child exit {code}; retained log/result')
            if group_processes(child.pid):
                raise ChildFailure('phase leader exited with live descendants')
    except BaseException as error:
        failure = error
    finally:
        if child is not None:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                if not group_processes(child.pid):
                    break
                try:
                    os.killpg(child.pid, sig)
                except ProcessLookupError:
                    pass
                deadline = time.monotonic() + grace_seconds
                while group_processes(child.pid) and time.monotonic() < deadline:
                    child.poll()
                    time.sleep(.05)
            child.poll()
            remaining = group_processes(child.pid)
            record.update(exit_code=child.returncode, remaining_descendants=remaining,
                          termination_verified=not remaining)
            if remaining:
                failure = CleanupFailure(f'phase group {child.pid} still has descendants: {remaining}')
        record.update(state='failed' if failure else 'completed', elapsed_seconds=time.monotonic()-started,
                      finished_at=now())
        if failure:
            record['error'] = repr(failure)
        publish()
    if failure:
        raise failure


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--recording-cli',required=True)
    a=p.parse_args();plan=json.loads(Path(a.plan).read_text());root=Path(a.output).resolve()
    root.mkdir(parents=True,exist_ok=False)
    if (plan['schema']!='tide-lab-npu-learning-stage-v1' or not 0<plan['max_seconds']<=10800
        or plan['learning_steps']!=80 or plan['learning_max_seconds']>7200
        or plan['calibration_max_seconds']>1200 or plan['profile_max_seconds']>180):
        raise ValueError('outside the declared first-batch bounded stage')
    start=time.monotonic()
    state=dict(schema='tide-lab-npu-stage-record-v1',state='running',started_at=now(),plan=plan,
               plan_sha256=digest(a.plan),calibration=[],commands=[],axes={
                   'execution':'running','strict_numerical':'not_checked','discrete_semantics':'not_checked',
                   'mechanisms':'not_checked','recovery':'not_checked','learning':'not_checked'})
    def publish():
        state['elapsed_seconds']=time.monotonic()-start;atomic_json(root/'stage.json',state)
    def interrupted(signum, _frame):
        if not state.get('stop_signal'):
            state['stop_signal']=signum
            raise InterruptedError(f'stage received signal {signum}')
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,interrupted)
    def run(label,argv,limit):
        remaining=plan['max_seconds']-(time.monotonic()-start)
        wall_limit=min(limit,remaining)
        if wall_limit<=10: raise TimeoutError('insufficient stage budget, including descendant cleanup')
        row=dict(label=label,argv=argv,wall_budget_seconds=wall_limit,cleanup_reserve_seconds=10)
        state['phase']=label;state['commands'].append(row);publish()
        bounded_command(argv,root/(label+'.log'),wall_limit-10,row,publish)
    def batch(label,entries,limit):
        if limit<=90: raise TimeoutError('insufficient batch budget including writer/cleanup')
        matrix=root/(label+'-matrix.json')
        atomic_json(matrix,dict(schema='tide-lab-matrix-v1',study_id=plan['study_id'],max_seconds=limit-90,
                               stop_on_failure=True,runs=entries));matrix.chmod(0o444)
        run(label,[sys.executable,'-m','tide_lab.batch','--matrix',str(matrix),'--output',str(root/label),
                   '--recording-cli',a.recording_cli],limit)
        record=json.loads((root/label/'batch.json').read_text())
        if record['state']!='completed' or any(r['state']!='completed' for r in record['runs']):
            raise RuntimeError(f'{label}: execution is not completed')
    entries=[]
    publish()
    try:
        try:
            run('mechanisms',[sys.executable,'scripts/assess_core5_axes.py','--config',plan['admission_config'],
                              '--output-dir',str(root/'mechanisms'),'--device','npu:0'],600)
        finally:
            if (root/'mechanisms/result.json').exists():
                audit=json.loads((root/'mechanisms/result.json').read_text())
                for axis in ['strict_numerical','discrete_semantics','mechanisms']:
                    state['axes'][axis]=dict(audit['axes'][axis],source='mechanisms/result.json')
                publish()
        if audit['hard_failures'] or audit['axes']['discrete_semantics']['state']!='passed' or audit['axes']['mechanisms']['state']!='passed':
            raise RuntimeError('a hard mechanism/discrete failure cannot be waived by exploration')
        try:
            run('resume',[sys.executable,'scripts/check_resume.py','--config',plan['resume_config'],
                          '--output',str(root/'resume')],600)
        finally:
            if (root/'resume/result.json').exists():
                restore=json.loads((root/'resume/result.json').read_text())
                state['axes']['recovery']=dict(state=restore['state'],source='resume/result.json',scope='fresh processes, B2, same stack/device, full model/AdamW/data step/RNG')
                publish()
        if restore['state']!='passed': raise RuntimeError('declared exact resume failed')
        publish()
        calibration_deadline=time.monotonic()+plan['calibration_max_seconds'];previous=None
        for item in plan['batches']:
            remaining_calibration=calibration_deadline-time.monotonic()
            if remaining_calibration<=120:
                state['calibration_stop']='wall budget';break
            if previous and item['batch']>=256:
                first=state['calibration'][0]
                gain=previous['examples_per_second']/first['examples_per_second']-1
                predicted=previous['median_step_seconds']*item['batch']/previous['batch']*plan['learning_steps']*1.2+previous['predicted_evaluation_seconds']+180
                if gain<plan['minimum_throughput_gain'] or predicted>plan['learning_max_seconds']-120:
                    state['calibration_stop']='throughput flattened or larger batch cannot fit fixed update budget'
                    break
            label=f"calibrate-b{item['batch']}";rid=f"tl-a1-npu-cal-b{item['batch']}-s11"
            batch(label,[dict(config=str(Path(item['config']).resolve()),run_id=rid)],min(600,remaining_calibration))
            directory=root/label/rid
            result=json.loads((directory/'result.json').read_text())
            events=[json.loads(s) for s in (directory/'metrics.jsonl').read_text().splitlines()]
            timed=[e for e in events if 'time/step_seconds' in e['metrics']]
            if len(timed)!=3: raise RuntimeError('calibration requires one warmup and two measured updates')
            times=[e['metrics']['time/step_seconds'] for e in timed[1:]]
            if any(not math.isfinite(t) or t<=0 for t in times): raise RuntimeError('invalid calibration timing')
            median=statistics.median(times)
            evaluation_per_sample=max(result[k]['seconds']/result[k]['samples']
                                      for k in ('initial_validation','last_validation'))
            # Initial + four validation windows + history intervention, each 128 samples.
            predicted_evaluation=evaluation_per_sample*128*6
            previous=dict(batch=item['batch'],config=item['config'],measured_seconds=times,
                independent_sample_batch=item['batch'],per_device_microbatch=item['batch'],
                global_batch_per_update=item['batch'],gradient_accumulation=1,
                median_step_seconds=median,examples_per_second=item['batch']/median,
                predicted_learning_seconds=median*plan['learning_steps']*1.2+predicted_evaluation+180,
                predicted_evaluation_seconds=predicted_evaluation,
                calibration_process_seconds=result['elapsed_seconds'],
                observed_shapes={k:v for k,v in timed[-1]['metrics'].items() if k.startswith(('work/max_','resource/'))},
                scope='two post-warmup forward/backward/clip/gradient-observation/optimizer steps with device synchronization; excludes data construction/transfers, validation/checkpoint and Lens; not a global batch optimum')
            state['calibration'].append(previous);publish()
        eligible=[r for r in state['calibration'] if r['predicted_learning_seconds']<=plan['learning_max_seconds']-120]
        if not eligible: raise RuntimeError('no calibrated batch fits the learning update/time budget')
        # Prefer throughput, but do not pay a much longer update for a flat result.
        fastest=max(r['examples_per_second'] for r in eligible)
        selected=min((r for r in eligible if r['examples_per_second']>=fastest/(1+plan['minimum_throughput_gain'])),key=lambda r:r['batch'])
        state['selection']=dict(selected,reason='smallest admissible batch within 10% of best observed throughput; fixed 80-update/time budget retained')
        publish()
        try:
            run('profile',[sys.executable,'scripts/profile_core5_batch.py','--config',selected['config'],
                           '--output',str(root/'operator-profile.json')],plan['profile_max_seconds'])
            state['profile']='recorded; CPU dispatch timing and shapes only'
        except ChildTimeout as error:
            state['profile']=dict(state='failed',error=str(error),impact='observation incomplete; not a numerical or learning pass')
        template=json.loads(Path(plan['learning_template']).read_text())
        for kind in ['query','fc-dense']:
            c=json.loads(json.dumps(template));c['training']['batch_size']=selected['batch']
            c['training']['max_seconds']=plan['learning_max_seconds']
            c['evidence_axes']['strict_numerical']=dict(state=audit['axes']['strict_numerical']['state'],
                source_sha256=digest(root/'mechanisms/result.json'),original_atol=1e-6,original_rtol=1e-5,
                note='applies to the core5 admission; not reinterpreted as same-model strict equivalence')
            c['evidence_axes']['discrete_semantics']=dict(state='passed',scope='B2 admission, full support/structural checks; formal steps only observed workload')
            c['evidence_axes']['recovery']=dict(state='passed',source_sha256=digest(root/'resume/result.json'),scope='B2 same-stack exact full application')
            if kind=='query':
                c['model']=dict(kind='no_memory',width=16)
                for axis in ('strict_numerical','discrete_semantics','recovery'):
                    c['evidence_axes'][axis]=dict(state='not_checked',note='the separate core5 admission does not qualify this query-only control')
            config=root/f'resolved-{kind}.json';atomic_json(config,c);config.chmod(0o444)
            entries.append(dict(config=str(config),run_id=f'tl-a1-npu-{kind}-s11-explore'))
        batch('learning',entries,plan['learning_max_seconds'])
        state['state']='completed';state['axes']['execution']='completed'
        return 0
    except BaseException as error:
        state['state']='failed';state['axes']['execution']='failed';state['error']=repr(error)
        if isinstance(error,CleanupFailure):
            state['phase']='cleanup_failed_waiting_for_descendants';publish()
            # Keep the queue allocation held. The finite service timeout is the
            # final guard; an unkillable process is never certified terminated.
            while any(group_processes(row.get('process_group')) for row in state['commands']):
                time.sleep(.2)
        return 1
    finally:
        quality={}
        for item in entries:
            result_path=root/'learning'/item['run_id']/'result.json'
            if result_path.exists():
                r=json.loads(result_path.read_text())
                quality[item['run_id']]=dict(state=r['state'],validation=r.get('last_validation'),
                    without_history=r.get('without_history'),trained_examples=r.get('trained_examples'))
        if quality:
            state['axes']['learning']=dict(state='observed_unreviewed',runs=quality,
                scope='possibly partial one-seed exploration; review required; different batch/precision is a different recipe')
        state['finished_at']=now();publish();print(json.dumps(dict(state=state['state'],phase=state.get('phase'),output=str(root))))


if __name__=='__main__': raise SystemExit(main())

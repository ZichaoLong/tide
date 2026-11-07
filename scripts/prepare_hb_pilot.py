#!/usr/bin/env python3
"""Bounded admission and common-batch calibration for the predeclared HB pair."""
import argparse
import json
import math
from pathlib import Path
import signal
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json, digest, now
from npu_learning_stage import bounded_command, CleanupFailure
from tide_lab.supervisor import group_processes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--recording-cli',required=True)
    args=parser.parse_args()
    plan=json.loads(Path(args.plan).read_text())
    if plan['schema']!='tide-lab-hb-prepare-v1' or plan['max_seconds']>1200:
        raise ValueError('outside bounded HB preparation')
    prerequisite=json.loads(Path(plan['prerequisite_evidence']).read_text())
    if len(prerequisite['rows'])!=3 or not all(
        row['validation_accuracy']>=.5 and row['validation_accuracy']-row['query_validation_accuracy']>=.2
        for row in prerequisite['rows']):
        raise ValueError('reviewed A1 validation prerequisite not met')
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    state=dict(schema='tide-lab-hb-preparation-v1',state='running',started_at=now(),
               plan=plan,plan_sha256=digest(args.plan),prerequisite_sha256=digest(plan['prerequisite_evidence']),
               commands=[],admission={},calibration=[],learning='not_checked')
    def publish():
        state['elapsed_seconds']=time.monotonic()-start;atomic_json(root/'preparation.json',state)
    def stop(sig,_frame):
        if not state.get('stop_signal'):
            state['stop_signal']=sig;raise InterruptedError(f'signal {sig}')
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,stop)
    def run(label,argv,limit):
        wall=min(limit,plan['max_seconds']-(time.monotonic()-start))
        if wall<=10: raise TimeoutError('preparation wall budget exhausted')
        record=dict(label=label,argv=argv,wall_budget_seconds=wall)
        state['phase']=label;state['commands'].append(record);publish()
        bounded_command(argv,root/(label+'.log'),wall-10,record,publish)
    publish()
    try:
        import torch
        from tide_lab.tasks import RetrievalTask
        from tide_lab.train import make_model, runtime
        runtime('cpu',1,11)
        models=[]
        for send in plan['sends']:
            config=json.loads(Path(f'configs/studies/a2-hb-{send}-pilot-template.json').read_text())
            models.append(make_model(config,RetrievalTask(config['task']),None))
        left,right=[model.state_dict() for model in models]
        if left.keys()!=right.keys() or any(not torch.equal(left[k],right[k]) for k in left):
            raise AssertionError('paired HB initialization must match every parameter')
        state['paired_initialization']=dict(state='passed',model_facts=[model.facts() for model in models],
                                           scope='identical CPU FP32 parameter values/keys; sending regularizer differs')
        del models,left,right
        publish()
        for send in plan['sends']:
            config=f'configs/checks/resume-hb-{send}.json'
            admission=root/f'{send}-mechanisms';resume=root/f'{send}-resume'
            try:
                run(f'{send}-mechanisms',[sys.executable,'scripts/assess_core5_axes.py','--config',config,
                    '--output-dir',str(admission),'--device','cpu'],120)
            finally:
                if (admission/'result.json').exists():
                    audit=json.loads((admission/'result.json').read_text())
                    state['admission'][send]=dict(axes=audit['axes'],source=str(admission/'result.json'))
                    publish()
            if audit['hard_failures'] or any(audit['axes'][k]['state']!='passed' for k in ['discrete_semantics','mechanisms']):
                raise RuntimeError(f'{send}: mechanism/discrete gate failed')
            try:
                run(f'{send}-resume',[sys.executable,'scripts/check_resume.py','--config',config,
                                     '--output',str(resume)],120)
            finally:
                if (resume/'result.json').exists():
                    recovery=json.loads((resume/'result.json').read_text())
                    state['admission'][send]['axes']['recovery']=dict(state=recovery['state'],source=str(resume/'result.json'))
                    publish()
            if recovery['state']!='passed': raise RuntimeError(f'{send}: exact resume failed')
        entries=[]
        for batch in plan['batches']:
            for send in plan['sends']:
                entries.append(dict(run_id=f'tl-a2-hb-{send}-cal-b{batch}-s11',
                    config=str(Path(f'configs/checks/calibrate-hb-{send}-b{batch}.json').resolve())))
        matrix=root/'calibration-matrix.json'
        atomic_json(matrix,dict(schema='tide-lab-matrix-v1',study_id='TL-A2',
                               max_seconds=plan['calibration_max_seconds']-90,stop_on_failure=True,runs=entries))
        matrix.chmod(0o444)
        run('calibration',[sys.executable,'-m','tide_lab.batch','--matrix',str(matrix),
            '--output',str(root/'calibration'),'--recording-cli',args.recording_cli],plan['calibration_max_seconds'])
        batch_record=json.loads((root/'calibration/batch.json').read_text())
        if batch_record['state']!='completed' or len(batch_record['runs'])!=4:
            raise RuntimeError('incomplete paired batch calibration')
        for batch in plan['batches']:
            variants={}
            for send in plan['sends']:
                directory=root/'calibration'/f'tl-a2-hb-{send}-cal-b{batch}-s11'
                result=json.loads((directory/'result.json').read_text())
                if result['state']!='completed': raise RuntimeError('calibration workload failed')
                events=[json.loads(s) for s in (directory/'metrics.jsonl').read_text().splitlines()]
                timed=[e for e in events if 'time/step_seconds' in e['metrics']]
                if len(timed)!=3: raise RuntimeError('one warmup/two measured steps required')
                times=[e['metrics']['time/step_seconds'] for e in timed[1:]]
                if any(not math.isfinite(t) or t<=0 for t in times): raise RuntimeError('invalid timings')
                step=statistics.median(times)
                eval_cost=max(result[k]['seconds']/result[k]['samples'] for k in ['initial_validation','last_validation'])*128*6
                variants[send]=dict(median_step_seconds=step,measured_seconds=times,
                    predicted_learning_seconds=step*80*1.2+eval_cost+120,
                    observed_work=timed[-1]['metrics'],process_seconds=result['elapsed_seconds'])
            state['calibration'].append(dict(batch=batch,variants=variants,
                paired_examples_per_second=2*batch/sum(v['median_step_seconds'] for v in variants.values()),
                eligible=all(v['predicted_learning_seconds']<=plan['learning_max_seconds']-120 for v in variants.values()),
                scope='common independent batch; microbatch=global batch, accumulation1; post-warmup forward/backward/clip/gradient observation/optimizer, excludes data/validation/checkpoint/writer'))
        eligible=[r for r in state['calibration'] if r['eligible']]
        if not eligible: raise RuntimeError('no common calibrated batch fits both fixed-update budgets')
        best=max(r['paired_examples_per_second'] for r in eligible)
        selected=min((r for r in eligible if r['paired_examples_per_second']>=best/1.1),key=lambda r:r['batch'])
        state['selection']=dict(selected,reason='common batch, within 10% of best paired throughput; each 80-step pilot fits time budget')
        state['resolved']=[]
        for send in plan['sends']:
            config=json.loads(Path(f'configs/studies/a2-hb-{send}-pilot-template.json').read_text())
            config['training']['batch_size']=selected['batch']
            config['evidence_axes']=dict(state['admission'][send]['axes'])
            config['evidence_axes']['learning']=dict(state='not_checked')
            config['evidence_axes']['execution']=dict(state='see separate training result')
            path=root/f'resolved-hb-{send}.json';atomic_json(path,config);path.chmod(0o444)
            state['resolved'].append(dict(send=send,config=str(path),sha256=digest(path)))
        state['state']='completed';return 0
    except BaseException as error:
        state.update(state='failed',error=repr(error));publish()
        if isinstance(error,CleanupFailure):
            while any(group_processes(row.get('process_group')) for row in state['commands']): time.sleep(.2)
        return 1
    finally:
        state['finished_at']=now();publish();print(json.dumps(dict(state=state['state'],phase=state.get('phase'))))


if __name__=='__main__': raise SystemExit(main())

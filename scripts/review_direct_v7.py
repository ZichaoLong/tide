#!/usr/bin/env python3
"""Read-only v7 observations and paired validation; no automatic scientific claims."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.job import inspect
from scripts.review_hb_learning import paired
from scripts.review_io import write_review_json
from tide_lab.records import digest, now

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def summarize_case(case, phase):
    directory = Path(case['output']); run = directory/'train'
    result = read(run/'result.json'); progress = read(run/'progress.json')
    admission = read(Path(case.get('reuse_admission', directory/'check/result.json')))
    recovery = read(Path(case.get('reuse_recovery', directory/'recovery.json')))
    manifest = read(run/'manifest.json'); projection = read(run/'projection.json')
    metrics = run/'metrics.jsonl'
    events = [json.loads(line) for line in metrics.read_text().splitlines()] if metrics.exists() else []
    updates = [row['metrics'] for row in events if 'train/loss' in row['metrics']]
    validations = []
    for path in sorted(run.glob('validation-*.json')):
        value = read(path)
        if value.get('test_access') or value.get('split') != 'validation':
            raise ValueError('v7 must not open test')
        validations.append(dict(record=relative(path), sha256=digest(path), state=value['state'],
            training_tokens=value['progress']['tokens'], validation_tokens=value['tokens'], nll=value['nll'],
            ppl=math.exp(value['nll']) if value['nll'] is not None and value['nll']<700 else None))
    checks = admission.get('checks', {})
    numerical = [checks.get('forward', {})]+list(checks.get('parameter_vjp', {}).values())
    row = dict(id=case['id'], phase=phase, config=relative(case['config']), output=relative(directory),
        run_id=manifest.get('run_id'), group=manifest.get('group'),
        state=result.get('state', 'not_started'), error=result.get('error'),
        progress=result.get('progress', progress), identity=result.get('identity'), ancestry=result.get('ancestry'),
        parameters=result.get('inventory', {}).get('trainable_parameters'),
        physical_device=result.get('physical_visibility'),
        admission_state=admission.get('state','unchecked'), recovery_state=recovery.get('state','unchecked'),
        mechanisms=admission.get('axes',{}).get('mechanisms','unchecked'),
        causal=checks.get('causal', {}).get('state', 'unchecked'),
        strict_numerical=dict(passed=sum(c.get('state')=='passed' for c in numerical),
                              failed=sum(c.get('state')=='failed' for c in numerical),
                              forward=checks.get('forward'),
                              worst_parameter_vjp_relative_l2=max((v.get('relative_l2',0) for v in checks.get('parameter_vjp',{}).values()),default=None),
                              worst_parameter_vjp_max_abs=max((v.get('max_abs',0) for v in checks.get('parameter_vjp',{}).values()),default=None),
                              scope='CPU/NPU B2 L3 forward and parameter VJP, atol1e-6 rtol1e-5; no TIDE equivalence claim'),
        validations=validations, inference=result.get('inference', {}), axes=result.get('axes', {}),
        lens=dict(status=projection.get('status','missing'), run_id=projection.get('source_run_id'),
            record=relative(run/'projection.json'), events=len(events),
            verified=(projection.get('status')=='verified' and projection.get('readback_matches') is True
                      and projection.get('project')=='tide-learning-lab'
                      and projection.get('source_run_id')==manifest.get('run_id')
                      and projection.get('event_count')==len(events))),
        curve=[dict(step=e['step'], **e['metrics']) for e in events])
    if updates:
        row['training_observations']=dict(logged_updates=len(updates), first=updates[0],last=updates[-1],
            max_gradient_norm=max(r['train/gradient_norm'] for r in updates),
            median_logged_step_seconds=statistics.median(r['train/step_seconds'] for r in updates),
            median_logged_tokens_per_second=statistics.median(r['train/tokens_per_second'] for r in updates))
    return row


def collect():
    controller=read(ROOT/'runs/v7-campaign/state.json');cases=[];jobs=[]
    for name,phase in controller.get('phases',{}).items():
        cases.extend(summarize_case(c,name) for c in phase['cases'])
    for path in sorted((ROOT/'artifacts/jobs').glob('v7-*/job.json')):
        observation=inspect(path);spec=read(path);state=observation['state'];queue=observation.get('queue') or {}
        jobs.append(dict(id=spec['job_id'],state=state.get('state'),source=spec['source'],
            state_record=relative(path.parent/'state.json'), record=relative(path), resources=spec['resources'],
            slice=(observation.get('systemd') or {}).get('Slice'), physical_ids=queue.get('physical_ids'),
            terminated=observation['termination_verified_now'], elapsed_seconds=state.get('elapsed_seconds'),
            cgroup=state.get('control_group'), command=spec['command']))
    pairs=[]
    def compare(label,left,right):
        if not left.exists() or not right.exists():return
        if any(read(p).get('state')!='completed' or read(p).get('tokens')!=32768 for p in (left,right)):return
        value=paired(left,right)
        if not value['complete_same_blocks'] or value['tokens']!=32768:raise ValueError('comparison target mismatch')
        value.update(label=label,left=relative(left),right=relative(right));pairs.append(value)
    by={(c['phase'],c['id']):ROOT/c['output']/'train' for c in cases}
    for model in ('hb40','hb88'):
        for init in ('fan_in','std015'):
            a=by.get(('screen',f'{model}-{init}-rms-s11'));b=by.get(('screen',f'{model}-{init}-raw-s11'))
            if a and b:compare(f'{model} {init}: RMS-minus-raw',a/'validation-final.json',b/'validation-final.json')
        for norm in ('rms','raw'):
            a=by.get(('screen',f'{model}-fan_in-{norm}-s11'));b=by.get(('screen',f'{model}-std015-{norm}-s11'))
            if a and b:compare(f'{model} {norm}: fan_in-minus-std015',a/'validation-final.json',b/'validation-final.json')
    selection=controller.get('selection',{});hb=selection.get('hb');tf=selection.get('transformer')
    if hb and tf:
        suffix=hb['initialization']+'-'+('rms' if hb['norm'] else 'raw')
        for seed in (11,22):
            lookup={m:by.get(('main',f'{m}-{suffix if m in ("fc16","hb40","hb88") else tf["position"]}-s{seed}'))
                    for m in ('fc16','hb40','hb88','tf12','tf28')}
            for tokens in (1048576,4194304):
                for left,right in [('hb88','hb40'),('hb40','fc16'),('hb40','tf12'),('hb88','tf28'),('tf28','tf12')]:
                    if lookup[left] and lookup[right]:compare(f'{tokens} s{seed} {left}-minus-{right}',
                        lookup[left]/f'validation-{tokens}.json',lookup[right]/f'validation-{tokens}.json')
                for model in ('hb40','hb88'):
                    control=by.get(('residual',f'{model}-{suffix}-depth-residual-s{seed}'))
                    if control and lookup[model]:compare(f'{tokens} s{seed} {model} depth-residual-minus-original',
                        control/f'validation-{tokens}.json',lookup[model]/f'validation-{tokens}.json')
    return dict(schema='tide-direct-v7-observations-v1',at=now(),controller=controller,
        jobs=jobs,cases=cases,comparisons=pairs,budget=read(ROOT/'runs/v7-budget-v2/state.json'),
        review_script_sha256=digest(Path(__file__)),
        scope='Ordinary dense graph, not TIDE. Observations and descriptive validation intervals; scientific interpretation requires separate review. Test sealed.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();report=collect()
    if args.output:write_review_json(args.output,report)
    print('controller',report['controller'].get('state','not_started'))
    for c in report['cases']:
        final=next((v for v in c['validations'] if v['record'].endswith('validation-final.json')), {})
        print(c['phase'],c['id'],c['state'],c['progress'].get('tokens',0),final.get('nll'),
              'recovery='+c['recovery_state'],'strict_failed='+str(c['strict_numerical']['failed']),
              'lens='+str(c['lens']['verified']))

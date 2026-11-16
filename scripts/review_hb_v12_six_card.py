#!/usr/bin/env python3
"""Read-only six-card placement, administrative ancestry and continuation state."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.job import inspect
from scripts.review_io import write_review_json
from tide_lab.records import now


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def review(root, policy_name='six-card'):
    spec = read(root/'configs/studies/language/hb-v12'/(policy_name+'.json'))
    prefix = spec.get('job_prefix', 'v12-six'); limit = spec['max_new_concurrent_npus']
    submissions = read(root/'runs'/(prefix+'-submit')/'state.json')
    jobs = [(job, 'retained') for job in spec['retained_jobs']]
    jobs += [(prefix+'-'+item['id'], item['mode']) for item in spec['cases']]
    rows = []
    for job, mode in jobs:
        record = root/'artifacts/jobs'/job/'job.json'
        if not record.is_file():
            rows.append(dict(job=job, mode=mode, lifecycle='not_submitted')); continue
        obs = inspect(record); worker = read(root/'runs'/(job+'-worker')/'state.json')
        cases = worker.get('cases', []); stages = cases[-1]['stages'] if cases else []
        stage = stages[-1]['name'] if stages else None
        output = root/'runs'/job
        result = read(output/stage/'result.json') if stage else {}
        progress = read(output/stage/'progress.json') if stage else {}
        ck = read(output/'train/checkpoint.json')
        rows.append(dict(job=job, mode=mode, lifecycle=obs['state']['state'], stage=stage,
                         stage_state=result.get('state'), termination_verified=obs['termination_verified_now'],
                         physical_device=(obs.get('queue') or {}).get('physical_id'),
                         tokens=progress.get('tokens', result.get('progress', {}).get('tokens')),
                         saved_tokens=ck.get('progress', {}).get('tokens'),
                         recovery=read(output/'recovery-compare/recovery.json').get('state'),
                         preparation_admission=worker.get('preparation_admission'),
                         lens=read(output/'train/projection.json').get('status'),
                         error=result.get('error', worker.get('error'))))
    live = []
    for path in (root/'artifacts/jobs').glob('v12-*/job.json'):
        if read(path)['device'] != 'npu': continue
        obs = inspect(path)
        if not obs['termination_verified_now']:
            live.append(dict(job=path.parent.name, lifecycle=obs['state']['state'],
                             physical_device=(obs.get('queue') or {}).get('physical_id')))
    devices = sorted({row['physical_device'] for row in live if row['physical_device'] is not None})
    result = dict(schema='tide-hb-v12-'+policy_name+'-review-v1', observed_at=now(), deadline=spec['deadline'],
                allowed_devices=spec['candidate_ids'], active_physical_devices=devices,
                card_limit=limit, within_card_limit=len(devices)<=limit and set(devices)<=set(spec['candidate_ids']),
                counts=dict(Counter(row['lifecycle'] for row in rows)), jobs=rows, all_live_v12_npu_jobs=live,
                submissions_state=submissions.get('state', 'not_submitted'),
                administrative_pauses=read(root/'docs/evidence'/('hb-v12-'+policy_name+'-transition.json')).get('pauses_verified', []),
                scope='Administrative pauses retain raw budget_truncated/outer failed records; not new numerical failures. Scientific source62 unchanged; recovery/quality separately reviewed.')
    if policy_name == 'six-card': result['within_six_card_limit'] = result['within_card_limit']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd()); parser.add_argument('--output', type=Path)
    parser.add_argument('--policy', choices=('six-card', 'four-card', 'all-cards'), default='six-card')
    args = parser.parse_args(); result = review(args.root.resolve(), args.policy)
    if args.output: write_review_json(args.output, result)
    else: print(json.dumps(result, indent=2, ensure_ascii=False))

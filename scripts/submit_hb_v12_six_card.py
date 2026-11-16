#!/usr/bin/env python3
"""Submit a fixed administrative manifest; preserve the original scientific source."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.job import inspect
from tide_lab.records import atomic_json, digest, now
from tide_lab.supervisor import verify_source
from tide_lab.disk_usage import campaign_usage
from tide_lab.quota import account_quota


def read(path):
    return json.loads(path.read_text())


def main(args):
    root = args.root.resolve(); source = Path(__file__).resolve().parents[1]
    orchestration = verify_source(source)
    model = root/'.local/source-hb-v12-repair-v2'
    model_id = verify_source(model)
    spec_path = source/args.spec
    spec = read(spec_path); deadline = spec['deadline_epoch']
    prefix = spec.get('job_prefix', 'v12-six')
    limit = spec['max_new_concurrent_npus']
    if model_id['commit'] != spec['model_source'] or time.time() > deadline-3600:
        raise ValueError('wrong model source or insufficient budget')
    expected = spec.get('administrative_case_count', 14)
    if not 0 < expected <= 38 or len(spec['cases']) != expected or len({x['id'] for x in spec['cases']}) != expected:
        raise ValueError('only fixed unique original administrative descendants')
    quota = account_quota(); disk = campaign_usage(root)
    if quota['hard_quota_headroom_gib'] < 200 or disk['gib'] > 96:
        raise RuntimeError('original disk/headroom limits')
    if not inspect(root/'artifacts/jobs/v12-repair-controller/job.json')['termination_verified_now']:
        raise ValueError('old submission controller still alive')
    for path in (root/'artifacts/jobs').glob('v12-*/job.json'):
        job = read(path); obs = inspect(path)
        if job['device'] == 'npu' and not obs['termination_verified_now']:
            if job['job_id'] not in spec['retained_jobs']:
                raise ValueError('unexpected live NPU job: '+job['job_id'])
            if (obs.get('queue') or {}).get('physical_id') not in spec['candidate_ids']:
                raise ValueError('retained device is outside six-card set')
    for item in spec['cases']:
        if not inspect(root/'artifacts/jobs'/item['previous_job']/'job.json')['termination_verified_now']:
            raise ValueError('previous service still alive: '+item['id'])
        if (root/'artifacts/jobs'/(prefix+'-'+item['id'])).exists():
            raise FileExistsError('never automatically retry an existing administrative descendant')
    out = root/'runs'/(prefix+'-submit'); out.mkdir(exist_ok=False)
    state = dict(state='submitting', started_at=now(), orchestration_source=orchestration['commit'],
                 model_source=model_id['commit'], deadline_epoch=deadline, jobs=[], retained_jobs=spec['retained_jobs'],
                 resources=dict(quota=quota, disk=disk))
    atomic_json(out/'state.json', state)
    for item in spec['cases']:
        name = item['id']; job = prefix+'-'+name
        dest = out/name; dest.mkdir()
        original = (root/item['base_plan']).parent if item.get('base_plan') else root/'runs/v12-repair-controller'/name
        plan = copy.deepcopy(read(original/'plan.json')); case = plan['cases'][0]
        if case['id'] != name: raise ValueError('original case identity mismatch')
        plan.update(administrative_mode=item['mode'], model_source=model_id['commit'], deadline_epoch=deadline,
                    original_config_sha256=digest(case['config']), previous_job=item['previous_job'],
                    card_limit=limit, run_prefix='tl-'+prefix+'-')
        case['output'] = str(root/'runs'/job); case['administrative_parent'] = item['previous_job']
        extra_inputs = []
        if item['mode'] == 'resume_train':
            parent_run = root/'runs'/item.get('resume_parent_job', item['previous_job'])
            parent_observation = inspect(root/'artifacts/jobs'/parent_run.name/'job.json')
            if not parent_observation['termination_verified_now']:
                raise ValueError('scientific parent still alive')
            result = read(parent_run/'train/result.json')
            if result['state'] != 'budget_truncated' or result['stop_signal'] != [15]:
                raise ValueError('only user administrative boundary stop can resume here')
            if digest(item['checkpoint']) != item['sha256']:
                raise ValueError('administrative checkpoint changed')
            case.update(parent_checkpoint=item['checkpoint'], parent_sha256=item['sha256'],
                        admission=str(parent_run/'check/result.json'))
            extra_inputs = [Path(case['admission']), parent_run/'train/result.json']
        elif (root/'runs'/item['previous_job']).exists():
            raise ValueError('unstarted repair unexpectedly produced stage output')
        policy = read(original/'resource.json')
        policy.update(candidate_ids=spec['candidate_ids'], max_new_concurrent_npus=limit, deadline_epoch=deadline,
                      latest_authorization=spec['authorization'])
        pp = dest/'plan.json'; rp = dest/'resource.json'
        atomic_json(pp, plan); atomic_json(rp, policy); pp.chmod(0o444); rp.chmod(0o444)
        cap = int(deadline-time.time()-30); runner = source/'scripts/resume_hb_v12_six_card.py'
        # cwd and every model stage remain62; only this orchestration input is new.
        cmd = [sys.executable, str(model/'scripts/job.py'), 'submit', '--name', job, '--source', str(model),
               '--output', str(root/'artifacts/jobs'/job), '--env-script', str(model/'scripts/env.sh'),
               '--env', 'TIDE_LAB_VENV='+str(root/'.venv-positive-floor-v1'), '--device', 'npu',
               '--threads', '4', '--memory-gib', '72', '--timeout', str(cap), '--env',
               'TIDE_LAB_RESOURCE_POLICY='+str(rp), '--queue-helper', str(model/'scripts/run_on_light_device.py'),
               '--wait-seconds', str(cap-300)]
        inputs = [spec_path, runner, source/'.tide-source.json', pp, rp, Path(case['config']),
                  Path(case['parent_checkpoint']), Path(plan['qualification']),
                  root/'runs/v9-data-r3/data-lock.json', root/'runs/v12-data/data-lock.json', *extra_inputs]
        for path in inputs: cmd += ['--input', str(path)]
        cmd += ['--', 'python', str(runner), '--plan', str(pp), '--data', str(root/'runs/v9-data-r3'),
                '--output', str(root/'runs'/(job+'-worker')), '--recording-cli', str(args.lens_cli),
                '--deadline-epoch', str(deadline)]
        reply = subprocess.run(cmd, capture_output=True, text=True)
        atomic_json(dest/'submission.json', dict(argv=cmd, returncode=reply.returncode, stdout=reply.stdout, stderr=reply.stderr))
        if reply.returncode: raise RuntimeError('submission failed: '+job+' '+reply.stderr)
        state['jobs'].append(dict(id=name, job=job, mode=item['mode'], previous_job=item['previous_job'],
                                  plan=str(pp), resource=str(rp), submitted_at=now()))
        atomic_json(out/'state.json', state); print(job, 'submitted', flush=True)
    atomic_json(out/'job-list.json', dict(jobs=spec['retained_jobs']+[row['job'] for row in state['jobs']]))
    state.update(state='submitted', finished_at=now()); atomic_json(out/'state.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--lens-cli', type=Path, required=True)
    parser.add_argument('--spec', type=Path, default=Path('configs/studies/language/hb-v12/six-card.json'))
    main(parser.parse_args())

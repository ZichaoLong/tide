#!/usr/bin/env python3
"""Finite v11 MoE follow-up using unchanged, exact-parent execution source."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.job import inspect
from scripts.run_moe_controls import eligible_result
from tide_lab.records import atomic_json, now
from tide_lab.quota import account_quota
from tide_lab.supervisor import TERMINAL


def read(path):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else {}


class Driver:
    def __init__(self, args):
        self.a = args
        self.root = args.root.resolve()
        self.source = Path(__file__).resolve().parents[1]
        self.cfg = self.source/'configs/studies/language/capacity-v11'
        self.budget = read(self.cfg/'budget.json')
        self.deadline = self.budget['deadline_epoch']
        self.stopped = False
        self.last_guard = 0
        args.output.mkdir(parents=True, exist_ok=False)
        self.state = dict(schema='tide-capacity-v11-moe-controller-v1', state='running',
            started_at=now(), source=read(self.source/'.tide-source.json')['commit'],
            deadline_epoch=self.deadline, jobs=[], phases={}, scientific_review='awaiting')
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)
        self.publish()

    def stop(self, *_):
        self.stopped = True

    def publish(self):
        atomic_json(self.a.output/'state.json', self.state)

    def guard(self):
        if self.stopped or time.time() >= self.deadline-180:
            raise TimeoutError('v11 stopped or shared deadline reached')
        if time.time()-self.last_guard < 60:
            return
        self.last_guard = time.time()
        paths = [p for base in ('runs', 'artifacts/jobs') for p in (self.root/base).glob('v11-*')]
        used = 0.
        if paths:
            raw = subprocess.check_output(['du', '-s', '-B1', *map(str, paths)], text=True)
            used = sum(int(s.split()[0]) for s in raw.splitlines())/2**30
        quota = account_quota()
        self.state['resources'] = dict(at=now(), new_disk_gib=used, quota=quota)
        self.publish()
        if used > self.budget['max_new_disk_gib'] or quota['hard_quota_headroom_gib'] < 200:
            raise RuntimeError('v11 shared disk/headroom limit')

    def observe(self, row):
        value = inspect(self.root/'artifacts/jobs'/row['job']/'job.json')
        return dict(job=row['job'], state=value['state'].get('state'),
            terminal=value['termination_verified_now'] and value['state'].get('state') in TERMINAL,
            physical_id=(value.get('queue') or {}).get('physical_id'))

    def launch(self, job, source, inputs, resource, command):
        self.guard()
        cap = int(self.deadline-time.time()-30)
        args = [sys.executable, str(self.source/'scripts/job.py'), 'submit', '--name', job,
            '--source', str(source), '--output', str(self.root/'artifacts/jobs'/job),
            '--env-script', str(source/'scripts/env.sh'), '--env', 'TIDE_LAB_VENV='+str(self.root/'.venv-positive-floor-v1'),
            '--env', 'TIDE_LAB_RESOURCE_POLICY='+str(resource), '--device', 'npu', '--threads', '4',
            '--memory-gib', '72', '--timeout', str(cap), '--wait-seconds', str(cap-120),
            '--queue-helper', str(source/'scripts/run_on_light_device.py')]
        for path in inputs:
            args += ['--input', str(path)]
        result = subprocess.run(args+['--', *command], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr+' '+result.stdout)
        self.state['jobs'].append(job)
        self.publish()
        print(json.dumps(dict(submitted=job, at=now())), flush=True)

    def train(self, case):
        cid = case['id']
        job = 'v11-'+case['phase']+'-'+cid
        source = self.root/case['execution_source']
        config = self.root/case['config']
        out = self.root/'runs'/job
        control = self.a.output/job
        control.mkdir()
        arguments = ['--until-tokens', str(case['until_tokens']), '--run-id', 'tl-'+job,
                     '--recording-group', case['group']]
        row = dict(id=cid, config=str(config), output=str(out), max_rss_gib=64,
            timeout_seconds=self.deadline-time.time()-240, retire_probe_checkpoints=True,
            train_arguments=arguments)
        inputs = [config, self.cfg/'budget.json', self.cfg/'moe-table.json', self.a.data/'data-lock.json']
        if case.get('parent'):
            parent = self.root/case['parent']
            row.update(reuse_admission=str(parent/'check/result.json'), reuse_recovery=str(parent/'recovery.json'))
            arguments += ['--resume', str(parent/'train/latest.pt')]
            inputs += [parent/'check/result.json', parent/'recovery.json', parent/'train/latest.pt']
        plan = control/'worker.json'
        atomic_json(plan, dict(cases=[row]))
        plan.chmod(0o444)
        policy = read(self.cfg/'resource.json')
        policy['minimum_free_hbm_mib'] = max(16384, int(case['peak_mib']*1.5+8192))
        resource = control/'resource.json'
        atomic_json(resource, policy)
        resource.chmod(0o444)
        self.launch(job, source, inputs+[plan, resource], resource,
            ['python', 'scripts/campaign_worker.py', '--plan', str(plan), '--data', str(self.a.data),
             '--output', str(self.root/'runs'/(job+'-worker')), '--recording-cli', self.a.lens_cli,
             '--deadline-epoch', str(self.deadline)])
        return dict(case, job=job, output=str(out), resource=str(resource))

    def confirm(self, row):
        job = 'v11-confirm-seen-'+row['id']
        out = self.root/'runs'/job
        config = self.root/row['config']
        checkpoint = Path(row['output'])/'train/latest.pt'
        resource = Path(row['resource'])
        self.launch(job, self.root/row['execution_source'],
            [config, checkpoint, resource, self.a.data/'data-lock.json'], resource,
            ['python', 'scripts/evaluate_direct_v10.py', '--config', str(config), '--checkpoint', str(checkpoint),
             '--data', str(self.a.data), '--output', str(out), '--lens-cli', self.a.lens_cli,
             '--run-id', 'tl-'+job, '--recording-group', 'TL-CAPACITY-CONFIRM-v11',
             '--deadline-epoch', str(self.deadline)])
        return dict(id=row['id'], job=job, output=str(out), scope='previously reviewed confirmation; descriptive bridge only')

    def wait_phase(self, phase):
        while True:
            self.guard()
            phase['observed'] = [self.observe(r) for r in phase['cases']]
            self.publish()
            if all(r['terminal'] for r in phase['observed']):
                break
            time.sleep(20)

    def run(self):
        cases = read(self.cfg/'moe-table.json')['cases']
        if len(cases) != 8:
            raise ValueError('expected six extensions and two balance controls')
        phase = dict(state='running', cases=[], results=[])
        self.state['phases']['train'] = phase
        for case in cases:
            phase['cases'].append(self.train(case))
            self.publish()
        self.wait_phase(phase)
        for row in phase['cases']:
            out = Path(row['output'])
            result = read(out/'train/result.json')
            validation = read(out/'train/validation-final.json')
            worker = read(self.root/'runs'/(row['job']+'-worker')/'state.json')
            inner = (worker.get('cases') or [{}])[-1]
            phase['results'].append(dict(row, eligible=eligible_result(result, validation, row['until_tokens']),
                state=result.get('state', inner.get('state', 'unchecked')), error=result.get('error', inner.get('reason'))))
        phase['state'] = 'terminal'
        confirmation = dict(state='running', cases=[], results=[])
        self.state['phases']['confirm-seen'] = confirmation
        for row in phase['results']:
            if row['eligible']:
                confirmation['cases'].append(self.confirm(row))
                self.publish()
        self.wait_phase(confirmation)
        for row in confirmation['cases']:
            result = read(Path(row['output'])/'result.json')
            confirmation['results'].append(dict(row, **{k: result[k] for k in ('state', 'nll', 'error') if k in result}))
        confirmation['state'] = 'terminal'
        self.state.update(state='completed', finished_at=now(),
            scope='finite phases terminal; failures retained; fresh confirmation and scientific review remain separate')
        self.publish()

    def stop_children(self):
        for job in self.state['jobs']:
            if not self.observe(dict(job=job))['terminal']:
                subprocess.run([sys.executable, str(self.source/'scripts/job.py'), 'stop',
                    str(self.root/'artifacts/jobs'/job)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'data', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--lens-cli', required=True)
    args = parser.parse_args()
    driver = Driver(args)
    try:
        driver.run()
    except BaseException as error:
        driver.state.update(state='failed', error=repr(error), finished_at=now())
        driver.publish()
        driver.stop_children()
        raise
    finally:
        atomic_json(args.output/'submission-closed.json', dict(closed=True, state=driver.state['state'], at=now()))

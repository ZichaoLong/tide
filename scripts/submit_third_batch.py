#!/usr/bin/env python3
"""Submit the explicitly authorized third batch with one shared 12-hour deadline."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json, now
from tide_lab.supervisor import verify_source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'project-root', 'venv', 'asset-path', 'recording-cli', 'lens-info', 'queue-helper'):
        p.add_argument('--'+name, required=True, type=Path)
    a = p.parse_args()
    source, project = a.source.resolve(), a.project_root.resolve()
    identity = verify_source(source)
    environment = json.loads(a.lens_info.read_text())['environment']
    if set(environment) != {'TRACKIO_DIR', 'TRACKIO_STORAGE_MODE'}:
        raise ValueError('unexpected Lens destination environment; resolve before writing')
    local = project/'.local'
    budget_path, submission_path = local/'third-batch-budget.json', local/'third-batch-submission.json'
    names = ['s3-growth', 's3-npu-sampling'] + [f's3-hb-{send}-s{seed}'
            for seed in (22, 33) for send in ('dense', 'sparse')] + ['s3-hb-evaluation']
    if budget_path.exists() or submission_path.exists() or any((project/'artifacts/jobs'/n).exists() for n in names):
        raise FileExistsError('never overwrite or automatically repeat a batch submission')
    begin = time.time(); deadline = begin+12*3600
    budget = dict(schema='tide-lab-third-budget-v1', submitted_at=now(), started_epoch=begin,
                  deadline_epoch=deadline, wall_seconds=43200, source_commit=identity['commit'],
                  source_tree_sha256=identity['content']['tree_sha256'],
                  added_disk_gib=20, maximum_compute_npus=2, active_compute_cpu_threads=12,
                  active_compute_ram_gib=64, queue_wait_cap_seconds=28800)
    atomic_json(budget_path, budget); budget_path.chmod(0o444)
    record = dict(budget=budget, submissions=[])
    def submit(name, device, threads, memory, timeout, wait, command, inputs=()):
        remaining = int(deadline-time.time())-30
        timeout = min(timeout, remaining)
        if timeout <= wait+30:
            raise TimeoutError('common deadline exhausted before submission')
        args = [sys.executable, str(source/'scripts/job.py'), 'submit', '--name', name,
            '--source', str(source), '--output', str(project/'artifacts/jobs'/name),
            '--env-script', str(source/'scripts/env.sh'), '--env', 'TIDE_LAB_VENV='+str(a.venv.resolve()),
            '--device', device, '--threads', str(threads), '--memory-gib', str(memory),
            '--timeout', str(timeout), '--wait-seconds', str(wait), '--input', str(budget_path)]
        for k, v in environment.items():
            args += ['--env', k+'='+v]
        for item in inputs:
            args += ['--input', str(item)]
        if device == 'npu':
            args += ['--queue-helper', str(a.queue_helper.resolve()), '--priority', '0']
        args += ['--', *command]
        row = dict(job_id=name, argv=args, state='submitting')
        record['submissions'].append(row); atomic_json(submission_path, record)
        result = subprocess.run(args, text=True, capture_output=True)
        row.update(exit_code=result.returncode, state='submitted' if result.returncode==0 else 'failed',
                   stdout=result.stdout, stderr=result.stderr, observed_at=now())
        atomic_json(submission_path, record)
        print(json.dumps(dict(job_id=name, state=row['state'], exit_code=result.returncode)), flush=True)
        # Submission failure does not cancel unrelated authorized work.
    planroot = 'configs/studies/third/'
    submit('s3-growth', 'npu', 4, 24, 39660, 28800,
        ['python', 'scripts/run_growth_third.py', '--plan', planroot+'b-stage.json',
         '--output', str(project/'runs/s3-growth'), '--asset-path', str(a.asset_path.resolve()),
         '--recording-cli', str(a.recording_cli.resolve()), '--deadline-epoch', str(deadline)])
    submit('s3-npu-sampling', 'npu', 4, 8, 29760, 28800,
        ['python', 'scripts/bounded_workload.py', '--max-seconds', '900', '--deadline-epoch', str(deadline),
         '--output', str(project/'runs/s3-npu-sampling-wrapper'), '--',
         'python', 'scripts/sample_core5_runtime.py', '--config', planroot+'npu-sampling.json',
         '--output', str(project/'runs/s3-npu-sampling'), '--recording-cli', str(a.recording_cli.resolve())])
    for seed in (22, 33):
        for send in ('dense', 'sparse'):
            name = f's3-hb-{send}-s{seed}'
            submit(name, 'cpu', 1, 8, 10770, 0,
                ['python', 'scripts/bounded_workload.py', '--max-seconds', '10740',
                 '--deadline-epoch', str(deadline), '--output', str(project/'runs'/(name+'-wrapper')), '--',
                 'python', '-m', 'tide_lab.batch', '--matrix', planroot+f'hb-{send}-s{seed}-matrix.json',
                 '--output', str(project/'runs'/name), '--recording-cli', str(a.recording_cli.resolve())])
    seed11 = [project/f'runs/s2-hb-{send}/tl-a2-long-{send}-s11/checkpoint.pt' for send in ('dense', 'sparse')]
    submit('s3-hb-evaluation', 'cpu', 1, 4, 14490, 0,
        ['python', 'scripts/evaluate_hb_third.py', '--plan', planroot+'hb-evaluation.json',
         '--project-root', str(project), '--output', str(project/'runs/s3-hb-evaluation'),
         '--recording-cli', str(a.recording_cli.resolve()), '--deadline-epoch', str(deadline)], seed11)
    return 0 if all(r['exit_code']==0 for r in record['submissions']) else 1


if __name__ == '__main__':
    raise SystemExit(main())

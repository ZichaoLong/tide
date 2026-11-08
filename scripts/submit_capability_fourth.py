#!/usr/bin/env python3
"""Freeze-budget submission for the approved multi-stage B capability batch."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json,now
from tide_lab.supervisor import verify_source


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('source','project-root','venv','asset-path','recording-cli','lens-info','queue-helper'):
        parser.add_argument('--'+key,required=True,type=Path)
    a=parser.parse_args();source=a.source.resolve();project=a.project_root.resolve()
    identity=verify_source(source);plan=source/'configs/studies/fourth/b-preserve.json'
    configuration=json.loads(plan.read_text());limits=configuration['budget']
    environment=json.loads(a.lens_info.read_text())['environment']
    if set(environment)!={'TRACKIO_DIR','TRACKIO_STORAGE_MODE'}:raise ValueError('unexpected Lens environment')
    local=project/'.local';budget_path=local/'fourth-batch-budget.json';job_output=project/'artifacts/jobs/s4-capability'
    if budget_path.exists() or job_output.exists():raise FileExistsError('no duplicate fourth-batch submission')
    started=time.time();deadline=started+limits['wall_seconds']
    budget=dict(schema='tide-b4-budget-v1',submitted_at=now(),started_epoch=started,deadline_epoch=deadline,
        source_commit=identity['commit'],source_tree_sha256=identity['content']['tree_sha256'],
        compute_seconds=limits['compute_seconds'],wall_seconds=limits['wall_seconds'],
        queue_seconds=limits['queue_seconds'],npu_count=1,cpu_threads=4,memory_gib=24,added_disk_gib=4)
    atomic_json(budget_path,budget);budget_path.chmod(0o444)
    command=[sys.executable,str(source/'scripts/job.py'),'submit','--name','s4-capability','--source',str(source),
        '--output',str(job_output),'--env-script',str(source/'scripts/env.sh'),'--env','TIDE_LAB_VENV='+str(a.venv.resolve()),
        '--device','npu','--threads','4','--memory-gib','24','--timeout',str(limits['wall_seconds']-30),
        '--wait-seconds',str(limits['queue_seconds']),'--queue-helper',str(a.queue_helper.resolve()),
        '--priority','0','--input',str(budget_path),'--input',str(plan)]
    for key,value in environment.items():command+=['--env',key+'='+value]
    command+=['--','python','scripts/bounded_workload.py','--max-seconds',str(limits['compute_seconds']),
        '--deadline-epoch',str(deadline-30),'--output',str(project/'runs/s4-capability-wrapper'),'--',
        'python','scripts/run_capability_fourth.py','--plan',str(plan),'--output',str(project/'runs/s4-capability'),
        '--asset-path',str(a.asset_path.resolve()),'--recording-cli',str(a.recording_cli.resolve()),
        '--deadline-epoch',str(deadline-30)]
    submitted=subprocess.run(command,capture_output=True,text=True)
    atomic_json(local/'fourth-batch-submission.json',dict(budget=budget,command=command,
        exit_code=submitted.returncode,stdout=submitted.stdout,stderr=submitted.stderr,observed_at=now()))
    print(submitted.stdout);print(submitted.stderr,file=sys.stderr)
    return submitted.returncode


if __name__=='__main__':raise SystemExit(main())

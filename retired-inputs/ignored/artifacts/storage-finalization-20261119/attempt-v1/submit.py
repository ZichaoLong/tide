"""Launch only this task's frozen cleanup or post-cleanup CPU smoke."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('frozen_job', HERE/'frozen-job.py')
job = importlib.util.module_from_spec(spec); spec.loader.exec_module(job)
kind = sys.argv[1]
assert kind in {'cleanup', 'smoke'}
plan = json.loads((HERE/'plan.json').read_text())
script = HERE/('cleanup.py' if kind == 'cleanup' else 'installed_smoke.py')
python = Path('/usr/bin/python3') if kind == 'cleanup' else ROOT/'.venv-instances/bin/python'
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
extra = ['--execute-sha', sha(HERE/'plan.json')] if kind == 'cleanup' else []
unit = f'tide-storage-finalization-{kind}-v1'
timeout, rss, cpus = (2700,16,'176 177 178 179 180 181 182 183') if kind == 'cleanup' else (300,4,'176 177')
command = [str(python), '-B', str(HERE/'frozen-job.py'), '--status', str(HERE/f'{kind}-status.json'),
           '--source',plan['source'], '--max-rss-gib',str(rss),'--',str(python),'-B',str(script),*extra]
argv = ['systemd-run','--user',f'--unit={unit}','--service-type=exec','--slice=background.slice',
        '--working-directory=/', '--property=KillMode=control-group', f'--property=RuntimeMaxSec={timeout}',
        '--property=TimeoutStopSec=15', f'--property=CPUAffinity={cpus}', f'--property=MemoryMax={rss}G',
        '--property=StandardOutput=append:'+str(HERE/f'{kind}.log'),
        '--property=StandardError=append:'+str(HERE/f'{kind}.log'), *command]
record = dict(state='prepared', source=plan['source'], unit=unit, command=command, launch=argv,
              frozen_files={p.name:sha(p) for p in [script,HERE/'frozen-job.py',HERE/'plan.json',Path(__file__)]},
              resources=dict(wall_seconds=timeout,max_rss_gib=rss,cpus=cpus),
              scope='CPU filesystem maintenance' if kind == 'cleanup' else 'CPU isolated installed public training/resume smoke')
target = HERE/f'{kind}-launch.json'
assert not target.exists()
job.atomic(target, record)
subprocess.run(argv,check=True)
record['unit_observation'] = subprocess.check_output(['systemctl','--user','show',unit,'-p','MainPID',
    '-p','Slice','-p','ControlGroup','-p','ActiveState','-p','SubState','-p','MemoryMax','-p','CPUAffinity'],text=True)
record['state']='submitted'; job.atomic(target,record)
print(record['unit_observation'])

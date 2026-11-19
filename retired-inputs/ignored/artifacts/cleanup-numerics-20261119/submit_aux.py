import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from job import atomic

kind=sys.argv[1]
assert kind in ('finish','smoke')
revision='97c52960358e6d5414fc987dc3861b1e85d997a9'
script=HERE/('finish_cleanup.py' if kind=='finish' else 'installed_smoke.py')
python=Path('/usr/bin/python3') if kind=='finish' else ROOT/'.venv-instances/bin/python'
unit=f'tide-cleanup-numerics-{kind}-v1'
extra=['--execute-sha',hashlib.sha256((HERE/'finish-plan.json').read_bytes()).hexdigest()] if kind=='finish' else []
command=[str(python),'-B',str(HERE/'frozen-job.py'),'--status',str(HERE/f'{kind}-status.json'),
         '--source',revision,'--max-rss-gib','4','--',str(python),'-B',str(script),*extra]
argv=['systemd-run','--user',f'--unit={unit}','--service-type=exec','--slice=background.slice',
      '--working-directory=/','--property=KillMode=control-group','--property=RuntimeMaxSec=300',
      '--property=TimeoutStopSec=15','--property=CPUAffinity=176 177',
      '--property=StandardOutput=append:'+str(HERE/f'{kind}.log'),
      '--property=StandardError=append:'+str(HERE/f'{kind}.log'),*command]
record=dict(state='prepared',source=revision,unit=unit,command=command,launch=argv,
    script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
    supervisor_sha256=hashlib.sha256((HERE/'frozen-job.py').read_bytes()).hexdigest())
atomic(HERE/f'{kind}-launch.json',record)
subprocess.run(argv,check=True)
record['unit_observation']=subprocess.check_output(['systemctl','--user','show',unit,'-p','MainPID',
    '-p','Slice','-p','ControlGroup','-p','ActiveState','-p','SubState'],text=True)
record['state']='submitted';atomic(HERE/f'{kind}-launch.json',record)
print(record['unit_observation'])

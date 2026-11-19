"""Finite installed-wheel dependency check after project artifact retirement."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[2]
RUN=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
from job import atomic

record=dict(state='running',started=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    source='0992b467584c1f603314ed04d8efe8596a14f576',
    scope='post-retirement installed-public-API dependency smoke, not new numerical qualification',
    cwd=str(Path.cwd()),affinity=sorted(os.sched_getaffinity(0)),commands=[],
    preflight={key:subprocess.check_output(command,text=True) for key,command in {
        'quota':['quota','-s'],'memory':['free','-h'],
        'units':['systemctl','--user','list-units','--state=running','tide-*.service','--no-pager']}.items()})
output=RUN/'post-cleanup-driver-result.json'
atomic(output,record)
started=time.monotonic()
try:
    inventory=json.loads((RUN/'inventory.json').read_text())
    retired=json.loads((RUN/'retirement.json').read_text())
    assert retired['state']=='completed'
    assert all(not (ROOT/item['path']).exists() for item in inventory['candidates'])
    record['retired_candidates_absent']=len(inventory['candidates'])
    wheel=ROOT/'artifacts/wheel-installed-cpu-v15/wheelhouse/tide_instances-0.1.0-py3-none-any.whl'
    assert hashlib.sha256(wheel.read_bytes()).hexdigest()=='1b3349526257e87daf1177d88c5a97ea30484b402f88080d00a72b4f521eb0e4'
    example=RUN/'train.py'
    example.write_bytes(subprocess.check_output(['git','show',record['source']+':examples/train.py'],cwd=ROOT))
    smoke=ROOT/'artifacts/wheel-installed-cpu-v15-inputs/smoke.py'
    assert hashlib.sha256(smoke.read_bytes()).hexdigest()=='ebc190070bb913db9d4841d44baca313c25aaa78d7f29ea28c0e0259a3533d88'
    record['input_hashes']={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in (Path(__file__),smoke,example,wheel)}
    venv=RUN/'post-cleanup-venv'
    assert not venv.exists()
    temporary=RUN/'post-cleanup-tmp';temporary.mkdir()
    environment=os.environ.copy()
    environment.pop('PYTHONPATH',None)
    environment.update(TORCH_DEVICE_BACKEND_AUTOLOAD='0',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
        PIP_NO_INDEX='1',PIP_CONFIG_FILE='/dev/null',PIP_DISABLE_PIP_VERSION_CHECK='1',
        TMPDIR=str(temporary),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    def execute(command,name,cwd):
        item=dict(command=[str(x) for x in command],cwd=str(cwd),log=str(RUN/(name+'.log')))
        record['commands'].append(item);atomic(output,record)
        then=time.monotonic()
        with open(item['log'],'xb') as log:
            result=subprocess.run(item['command'],cwd=cwd,env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        item.update(exit_code=result.returncode,elapsed_seconds=time.monotonic()-then)
        atomic(output,record)
        assert result.returncode==0,item
    execute([ROOT/'.venv-instances/bin/python','-I','-m','venv','--system-site-packages',venv],'post-cleanup-venv-create',RUN)
    python=venv/'bin/python'
    environment['VIRTUAL_ENV']=str(venv)
    environment['PATH']=str(venv/'bin')+os.pathsep+environment['PATH']
    execute([python,'-I','-m','pip','install','--ignore-installed','--no-deps','--no-index','--no-cache-dir','--no-compile',wheel],'post-cleanup-wheel-install',RUN)
    record['venv_config']=(venv/'pyvenv.cfg').read_text()
    execute([python,'-I',smoke,'--output',RUN/'post-cleanup-smoke-result.json','--example',example,'--wheel',wheel,'--source',record['source']],'post-cleanup-installed-smoke',Path('/'))
    record['state']='passed'
except BaseException as error:
    record.update(state='failed',error=repr(error))
    raise
finally:
    record.update(elapsed_seconds=time.monotonic()-started,finished=datetime.datetime.now(datetime.timezone.utc).isoformat())
    atomic(output,record)
print(json.dumps(dict(state=record['state'],seconds=record['elapsed_seconds'])))

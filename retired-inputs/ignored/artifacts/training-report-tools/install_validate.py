"""Frozen-input offline build and isolated installed public-API validation."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys

source=Path.cwd()
run=Path(sys.argv[1])
sys.path.insert(0,str(source))
from scripts.job import atomic

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

files={str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
manifest=dict(source=os.environ['TIDE_SOURCE_COMMIT'],files=files)
atomic(run/'source-manifest.json',manifest)
build=run/'build'
shutil.copytree(source,build,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
dist=run/'dist';dist.mkdir()
env=run/'env'
record=dict(source=manifest['source'],driver_sha256=sha(Path(__file__)),commands=[],state='running')
clean_env={k:v for k,v in os.environ.items() if k!='PYTHONPATH'}
clean_env['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'

def call(label,command,cwd=run):
    with (run/(label+'.stdout')).open('w') as out,(run/(label+'.stderr')).open('w') as err:
        p=subprocess.run([str(x) for x in command],cwd=cwd,env=clean_env,stdout=out,stderr=err,timeout=300)
    record['commands'].append(dict(label=label,argv=[str(x) for x in command],cwd=str(cwd),exit_code=p.returncode,
        stdout=label+'.stdout',stderr=label+'.stderr'))
    atomic(run/'installed.json',record)
    if p.returncode:
        raise RuntimeError(label+' failed: '+(run/(label+'.stderr')).read_text()[-2000:])

try:
    call('build-wheel',[sys.executable,'-m','pip','wheel','--no-deps','--no-build-isolation','--no-index','--no-cache-dir','-w',dist,build])
    wheel,=list(dist.glob('*.whl'))
    call('new-venv',[sys.executable,'-m','venv','--system-site-packages',env])
    python=env/'bin/python'
    call('install-wheel',[python,'-m','pip','install','--no-deps','--no-index',wheel])
    call('installed-check',[python,'-I',source/'scripts/check_installed_capacity.py','--execution-variant','training',
        '--wheel',wheel,'--venv',env,'--source-manifest',run/'source-manifest.json','--output-dir',run/'results'],cwd=Path('/'))
    record['installed_result']=json.loads((run/'results/result.json').read_text())['state']
    call('entry-default',[python,'-I',source/'examples/train.py'],cwd=Path('/'))
    for preset in sorted((source/'configs/recommended').glob('npu-*.json')):
        call('dry-'+preset.stem,[python,'-I',source/'examples/train.py','--preset',preset.stem,'--device','npu:0','--dry-run'],cwd=Path('/'))
    checkpoint=run/'entry-checkpoint.pt'
    call('entry-training',[python,'-I',source/'examples/train.py','--preset','hb-settle','--batch','1','--length','4',
        '--steps','2','--checkpoint-reception','--ssd-summary-backend','prefix','--region-batch-size','4','--checkpoint',checkpoint],cwd=Path('/'))
    call('entry-resume',[python,'-I',source/'examples/train.py','--resume',checkpoint,'--batch','1','--length','4','--steps','1'],cwd=Path('/'))
    assert all(sha(source/name)==value for name,value in files.items())
    record.update(state='completed',wheel=dict(path=str(wheel),sha256=sha(wheel)),
        source_unchanged=True,source_files=len(files),checkpoint_sha256=sha(checkpoint))
except BaseException as error:
    record.update(state='failed',error=repr(error));atomic(run/'installed.json',record);raise
atomic(run/'installed.json',record)
print(json.dumps({k:record[k] for k in ['source','state','installed_result','source_files']}))

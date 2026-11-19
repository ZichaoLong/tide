"""Frozen one-shot installed wheel smoke driver; writes only its artifact run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile


def atomic(path, value):
    path=Path(path)
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.installed.')
    try:
        with os.fdopen(fd,'w') as stream:
            json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.replace(tmp,path)
        fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--smoke',type=Path,required=True)
    args=parser.parse_args()
    started=time.monotonic()
    output=args.run/'build-result.json'
    record=dict(state='running',source=os.environ['TIDE_SOURCE_COMMIT'],
                source_path=str(args.source),driver_sha256=sha(__file__),smoke_sha256=sha(args.smoke),
                cpu_affinity=sorted(os.sched_getaffinity(0)),commands=[])
    atomic(output,record)
    try:
        # Preserve the immutable archive; setuptools may write build/egg-info.
        build=args.run/'build-copy'
        shutil.copytree(args.source,build)
        before={str(p.relative_to(args.source)):sha(p) for p in args.source.rglob('*') if p.is_file()}
        wheelhouse=args.run/'wheelhouse';wheelhouse.mkdir()
        temporary=args.run/'tmp';temporary.mkdir()
        environment=os.environ.copy()
        environment.pop('PYTHONPATH',None)
        environment.update(PIP_CONFIG_FILE='/dev/null',PIP_NO_INDEX='1',PIP_DISABLE_PIP_VERSION_CHECK='1',
                           PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(temporary),
                           TORCH_DEVICE_BACKEND_AUTOLOAD='0')
        def execute(command, name, cwd):
            item=dict(command=[str(x) for x in command],cwd=str(cwd),log=str(args.run/(name+'.log')))
            record['commands'].append(item);atomic(output,record)
            began=time.monotonic()
            with open(item['log'],'xb') as log:
                result=subprocess.run(item['command'],cwd=cwd,env=environment,stdout=log,stderr=subprocess.STDOUT)
            item.update(exit_code=result.returncode,elapsed_seconds=time.monotonic()-began)
            atomic(output,record)
            if result.returncode:
                raise RuntimeError(f'{name} returned {result.returncode}; see {item["log"]}')
        execute([sys.executable,'-I','-m','pip','wheel','--no-deps','--no-build-isolation',
                 '--no-index','--no-cache-dir','--wheel-dir',wheelhouse,build],'wheel-build',args.run)
        wheels=list(wheelhouse.glob('*.whl'))
        if len(wheels)!=1:raise RuntimeError('expected one project wheel')
        wheel=wheels[0]
        with zipfile.ZipFile(wheel) as archive:
            package_hashes={name:hashlib.sha256(archive.read(name)).hexdigest()
                            for name in archive.namelist() if name.startswith('tide/') and name.endswith('.py')}
        record['wheel']=dict(path=str(wheel),sha256=sha(wheel),bytes=wheel.stat().st_size,
                             package_file_sha256=package_hashes)
        venv=args.run/'installed-venv'
        execute([sys.executable,'-I','-m','venv','--system-site-packages',venv],'venv-create',args.run)
        python=venv/'bin/python'
        environment['VIRTUAL_ENV']=str(venv)
        environment['PATH']=str(venv/'bin')+os.pathsep+environment['PATH']
        execute([python,'-I','-m','pip','install','--ignore-installed','--no-deps','--no-index',
                 '--no-cache-dir','--no-compile',wheel],'wheel-install',args.run)
        record['venv_config']=(venv/'pyvenv.cfg').read_text()
        record['clean_environment']={k:environment.get(k) for k in (
            'PYTHONPATH','PYTHONNOUSERSITE','PYTHONDONTWRITEBYTECODE','TORCH_DEVICE_BACKEND_AUTOLOAD',
            'OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PIP_CONFIG_FILE','TMPDIR','VIRTUAL_ENV')}
        atomic(output,record)
        execute([python,'-I',args.smoke,'--output',args.run/'smoke-result.json',
                 '--example',args.source/'examples/train.py','--wheel',wheel,
                 '--source',record['source']], 'installed-smoke',Path('/'))
        record['frozen_source_unchanged']=all((args.source/p).exists() and sha(args.source/p)==digest
                                              for p,digest in before.items())
        record['frozen_source_file_count']=len(before)
        if not record['frozen_source_unchanged']:raise RuntimeError('frozen source changed')
        record['state']='passed'
    except BaseException as error:
        record.update(state='failed',error=repr(error))
        raise
    finally:
        record['elapsed_seconds']=time.monotonic()-started
        atomic(output,record)


if __name__=='__main__':main()

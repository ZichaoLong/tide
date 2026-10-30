#!/usr/bin/env python3
"""Freeze/copy/install and check the actual core-5 downstream starter on CPU.

A directed installed-client gate, not a full backend or performance matrix.
Run under job.py in a bounded detached service with a frozen source checkout.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import venv
from build_identity import source_hash
from durable_records import write_json
from source_identity import digest, source_state


def source_files(root):
    names = subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=root).split(b'\0')
    return {name.decode():digest(root/name.decode()) for name in sorted(set(names)-{b''}) if (root/name.decode()).is_file()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--build-dir',type=Path,required=True,help='matching unchanged CPU build; not rebuilt')
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1];out=a.output_dir.resolve();build=a.build_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    source,dirty=source_state(root);files=source_files(root)
    r=dict(schema='tide-downstream-consumer-evidence-v1',source=source,dirty=dirty,state='running',
           scope='installed CPU starter + lock/checkpoint rejection; no accelerator, standalone or performance qualification',
           commands=[],checks=[],source_files_sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest())
    write_json(out/'result.json',r)
    env=dict(os.environ,TORCH_DEVICE_BACKEND_AUTOLOAD='0',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
             OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    env.pop('PYTHONPATH',None)
    def run(name,command,cwd=out,expected=0):
        command=list(map(str,command))
        log=out/(name+'.log')
        with log.open('w') as stream:
            proc=subprocess.run(command,cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=240)
        r['commands'].append(dict(name=name,argv=command,exit_code=proc.returncode,log_sha256=digest(log)))
        write_json(out/'result.json',r)
        if expected==0 and proc.returncode != 0 or expected!=0 and proc.returncode==0:
            raise RuntimeError(name+' unexpected exit: '+str(proc.returncode))
    try:
        with tarfile.open(out/'source.tar.gz','w:gz') as archive:
            for name in files:archive.add(root/name,arcname=name,recursive=False)
        r['source_archive_sha256']=digest(out/'source.tar.gz')
        manifest=json.loads((build/'build-manifest.json').read_text())
        if manifest['cpp_source_sha256']!=source_hash(root) or manifest['backend']!='cpu':
            raise ValueError('reused build source/backend mismatch')
        for name,expected in manifest['binary_sha256'].items():
            if digest(build/name)!=expected:raise ValueError('reused binary changed: '+name)
        r['native_build']=manifest;r['native_build_manifest_sha256']=digest(build/'build-manifest.json')
        run('unit-tests',[sys.executable,'-m','pytest',root/'tests/test_downstream_consumer.py','-q',
                          '--junitxml='+str(out/'tests.xml')],root)
        package=out/'package';package.mkdir()
        shutil.copy2(root/'pyproject.toml',package)
        shutil.copytree(root/'python',package/'python',ignore=shutil.ignore_patterns('__pycache__','*.egg-info'))
        run('wheel',[sys.executable,'-m','pip','wheel','--no-index','--no-deps','--no-build-isolation',package,'-w',out/'wheels'])
        wheel,=(out/'wheels').glob('*.whl');r['wheel_sha256']=digest(wheel)
        venv.EnvBuilder(with_pip=True,system_site_packages=True).create(out/'env')
        site,=(out/'env/lib').glob('python*/site-packages')
        (site/'selected-dependencies.pth').write_text(sysconfig.get_path('purelib')+'\n')
        python=out/'env/bin/python'
        run('install',[python,'-m','pip','install','--no-index','--no-deps',wheel])
        app=out/'application';shutil.copytree(root/'examples/consumer_core5',app,ignore=shutil.ignore_patterns('__pycache__'))
        native=out/'native';native.mkdir()
        shutil.copy2(build/'_tide_native.so',native);shutil.copy2(build/'build-manifest.json',native)
        upstream=json.loads((root/'docs/upstream.json').read_text())
        # The package static plan independently checks this revision at startup.
        math_revision=upstream.get('revision',upstream.get('commit'))
        if math_revision is None:
            raise ValueError('upstream lock revision missing')
        command=[python,app/'dependency.py','--wheel',wheel,'--source-commit',source,'--math-commit',math_revision,
                 '--native-library',native,'--output',app/'foundation.lock.json']
        if dirty:command+=['--development-dirty']
        run('capture-lock',command)
        r['foundation_lock']=json.loads((app/'foundation.lock.json').read_text())
        fc=app/'recipe.json'
        hb=app/'hb-mixed.json';hb.write_text(json.dumps(dict(topology='hb',family='mixed',width=8,region_width=2,depth=1)))
        cases=[(f'fc-{client}-{dtype}',fc,client,dtype,'prefill','packed') for client in ('pytorch','python-native') for dtype in ('float32','float64')]
        cases += [(f'hb-mixed-{client}',hb,client,'float64','prefill','packed') for client in ('pytorch','python-native')]
        cases += [(f'fc-streaming-{client}',fc,client,'float64','streaming','scalar') for client in ('pytorch','python-native')]
        for name,recipe,client,dtype,schedule,packing in cases:
            command=[python,app/'check.py','--device','cpu','--foundation-lock',app/'foundation.lock.json',
                     '--recipe',recipe,'--client',client,'--dtype',dtype,'--schedule',schedule,'--packing',packing,'--output-dir',out/name]
            if client=='python-native':command+=['--native-library',native]
            run(name,command)
            report=json.loads((out/name/'result.json').read_text())
            if report['state']!='passed' or not all(report['checks'].values()):raise ValueError('incomplete check')
            if not Path(report['dependency']['library_path']).is_relative_to(out/'env'):
                raise ValueError('consumer imported source instead of installed wheel')
            r['checks'].append(dict(name=name,state=report['state'],checks=report['checks'],facts=report['facts'],
                                    result_sha256=digest(out/name/'result.json')))
        common=[python,app/'train.py','--device','cpu','--foundation-lock',app/'foundation.lock.json']
        bad=out/'bad-checkpoint.pt'
        import torch
        r['torch']=str(torch.__version__)
        torch.save({'schema':'not-the-application'},bad)
        for name,extra in [('bad-resume',['--resume',bad]),('missing-native',['--client','python-native']),
                           ('unsupported-packing',['--schedule','streaming','--packing','packed'])]:
            run(name,[*common,*extra,'--output-dir',out/name],expected=1)
            if (out/name).exists():raise ValueError('invalid request left final directory: '+name)
        badlock=out/'bad-lock.json';lock=json.loads((app/'foundation.lock.json').read_text())
        lock['foundation']['installed_files_sha256']='0'*64;badlock.write_text(json.dumps(lock))
        run('bad-lock',[python,app/'train.py','--device','cpu','--foundation-lock',badlock,'--output-dir',out/'bad-lock'],expected=1)
        if (out/'bad-lock').exists():raise ValueError('bad lock left final directory')
        r['rejections']=['bad-resume','missing-native','unsupported-packing','bad-lock']
        if source_state(root)!=(source,dirty) or source_files(root)!=files:
            raise ValueError('source changed during verification')
        for name,expected in manifest['binary_sha256'].items():
            if digest(build/name)!=expected:raise ValueError('build changed during verification')
        r.update(state='passed',junit_sha256=digest(out/'tests.xml'))
    except BaseException as error:
        r.update(state='failed',error=repr(error));raise
    finally:
        write_json(out/'result.json',r)
    print(json.dumps(dict(state=r['state'],installed_cases=len(r['checks']),rejections=r['rejections'])))


if __name__=='__main__':main()

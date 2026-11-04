#!/usr/bin/env python3
"""Private fixed llama.cpp build/conversion. Caller serializes heavy work."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import urllib.request
from tidegraph.lm.engine_matrix import LLAMA_REVISION
from tidegraph.lm.catalog import model,local_path
from tidegraph.lm.resources import disk_admission
from tidegraph.experiment.records import write_json,read_json,digest,fingerprint,now


def prepare(root):
    out=root/'engines/llama';out.mkdir(parents=True,exist_ok=True)
    foundation=Path(__file__).resolve().parents[1]
    client={name:digest(foundation/'examples/external_lm/llama'/name) for name in ('main.cpp','CMakeLists.txt')}
    if (out/'ready.json').exists():
        record=read_json(out/'ready.json')
        if record['revision']!=LLAMA_REVISION or digest(record['binary'])!=record['binary_sha256']:
            raise ValueError('existing llama build differs from fixed identity')
        if record.get('client_files_sha256')==client:return record
        (out/'build-records').mkdir(exist_ok=True)
        write_json(out/'build-records'/(record['binary_sha256']+'.json'),record)
    disk_admission(root,6*1024**3,project_gib=160,account_gib=300)
    archive=out/(LLAMA_REVISION+'.tar.gz')
    if not archive.exists():
        url=f'https://codeload.github.com/ggml-org/llama.cpp/tar.gz/{LLAMA_REVISION}'
        partial=archive.with_suffix('.part');size=0
        with urllib.request.urlopen(url,timeout=60) as response,partial.open('wb') as target:
            while block:=response.read(1024**2):
                size+=len(block)
                if size>256*1024**2:raise ValueError('source archive exceeds 256MiB budget')
                target.write(block)
            target.flush();os.fsync(target.fileno())
        partial.replace(archive)
    source=out/('llama.cpp-'+LLAMA_REVISION)
    if not source.exists():
        with tarfile.open(archive) as bundle:bundle.extractall(out,filter='data')
    if not (source/'include/llama.h').is_file():raise ValueError('source archive root differs from pinned revision')
    build=out/('build-'+fingerprint(client)[:16])
    subprocess.run(['cmake','-S',str(foundation/'examples/external_lm/llama'),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release',
                    '-DLLAMA_SOURCE='+str(source)],check=True)
    jobs=len(os.sched_getaffinity(0))
    subprocess.run(['cmake','--build',str(build),'--target','tide-lm-llama','--parallel',str(jobs)],check=True)
    binary=build/'tide-lm-llama'
    env=root/'env/llama-convert'
    if not env.exists():subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(env)],check=True)
    python=env/'bin/python'
    subprocess.run([str(python),'-m','pip','install','--no-cache-dir','transformers==4.57.1','tokenizers==0.22.1',
                    'huggingface-hub==0.36.0','sentencepiece==0.2.1','tiktoken==0.12.0'],check=True)
    record=dict(state='passed',finished=now(),revision=LLAMA_REVISION,client_files_sha256=client,archive_sha256=digest(archive),
                binary=str(binary),binary_sha256=digest(binary),build_jobs=jobs,
                shared_libraries={str(p):digest(p) for p in build.rglob('*.so*') if p.is_file()},
                compiler=subprocess.check_output(['c++','--version'],text=True).splitlines()[0],
                foundation_source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=foundation,text=True).strip(),
                cmake_flags='Release; CPU native ARM/OpenMP; GGML_BLAS OFF; no GPU',
                conversion_python=str(python),source_directory=str(source))
    write_json(out/'ready.json',record);return record


def convert(root,key):
    out=root/'engines/llama';build=read_json(out/'ready.json');destination=out/key;destination.mkdir(exist_ok=True)
    weights=destination/'model-bf16.gguf';record=destination/'conversion.json';spec=model(key)
    if record.exists():
        old=read_json(record)
        if old['model_manifest_id']!=fingerprint(spec) or digest(weights)!=old['sha256']:raise ValueError('old GGUF differs')
        return old
    disk_admission(root,sum(f['bytes'] for f in spec['files'])+1024**3,project_gib=160,account_gib=300)
    if weights.exists():raise ValueError('unpublished GGUF exists; inspect previous conversion, no silent overwrite')
    begin=now()
    subprocess.run([build['conversion_python'],'-B',str(Path(build['source_directory'])/'convert_hf_to_gguf.py'),
                    str(local_path(root/'models',key)),'--outfile',str(weights),'--outtype','bf16'],cwd=destination,check=True)
    value=dict(model=key,model_manifest_id=fingerprint(spec),gguf=str(weights),sha256=digest(weights),
               bytes=weights.stat().st_size,mtime_ns=weights.stat().st_mtime_ns,started=begin,finished=now(),
               dtype='MOSTLY_BF16; converter keeps 1D tensors FP32',llama_revision=LLAMA_REVISION)
    value['conversion_packages']=subprocess.check_output([build['conversion_python'],'-I','-m','pip','freeze'],text=True).splitlines()
    write_json(record,value);return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--convert',choices=('qwen15','olmoe'))
    args=p.parse_args();convert(args.root,args.convert) if args.convert else prepare(args.root)

#!/usr/bin/env python3
"""Finite FC comparison suites. Run inside a bounded frozen-source job service.

The suites answer separate questions; this is not a Cartesian benchmark grid.
Profiles and task-root omissions are diagnostic, never production qualification.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from durable_records import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',choices=('main','threads','boundaries','profiles','axes','checks','blocks','windows','window-profiles'),required=True)
    p.add_argument('--device',choices=('cpu','npu:0'),required=True)
    p.add_argument('--threads',type=int,required=True)
    p.add_argument('--native-library',type=Path,required=True)
    p.add_argument('--downstream-repo',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--profile-scratch',type=Path)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--length',type=int,default=128)
    a=p.parse_args()
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    commands=[]

    def invoke(name,args,timeout=600):
        command=[sys.executable,*map(str,args)];begin=time.monotonic()
        with (out/(name+'.log')).open('x') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
                code=124
        row=dict(name=name,command=command,exit_code=code,seconds=time.monotonic()-begin)
        commands.append(row);write_json(out/'commands.json',commands);print(json.dumps(row),flush=True)
        if code==124:raise SystemExit(124)

    def cost(shape,baseline,mode,*,client='native',boundary='public',b=8,length=128,profile='none',fences='step',attention='manual',query='auto',census=False):
        name=f'{shape}-{baseline}-{client}-{boundary}-{mode}-b{b}l{length}-{fences}-{attention}-{profile}-{query}'
        args=['scripts/benchmark_dense_query.py','--downstream-repo',a.downstream_repo,
              '--device',a.device,'--threads',a.threads,'--fc-shape',shape,'--baseline',baseline,
              '--client',client if baseline in ('graph','columns','window') else 'python','--mode',mode,'--trace-records','minimal',
              '--phase-fences',fences,'--diagnostic-boundary',boundary,'--formula-attention',attention,'--query-execution',query,
              '--batch',b,'--length',length,'--profile',profile,'--output-dir',out/name]
        if baseline in ('graph','columns','window') and client=='native':args+=['--native-library',a.native_library]
        if census:args+=['--autograd-census']
        if profile in ('cpu','npu'):args+=['--profile-shapes']
        if profile=='npu':
            if a.profile_scratch is None:raise ValueError('NPU profiles require explicit scratch storage')
            args+=['--profile-output',a.profile_scratch/name]
        invoke(name,args)

    if a.suite=='windows':
        for shape,depth in (('fc4',4),('fc16',16)):
            for mode in ('inference','grad-forward','training'):
                for client in ('native','python'):
                    cost(shape,'window',mode,client=client,b=a.batch,length=a.length)
                for baseline in ('fc-tensor',f'transformer-{depth}'):
                    cost(shape,baseline,mode,b=a.batch,length=a.length)
        cost('fc16','columns','training',client='python',b=a.batch,length=a.length)
    elif a.suite=='window-profiles':
        for baseline,client in (('window','native'),('window','python'),('fc-tensor','python'),('transformer-16','python')):
            cost('fc16',baseline,'training',client=client,b=2,length=32,census=True,
                 profile='cpu' if a.device=='cpu' else 'npu')
    elif a.suite=='blocks':
        for shape,depth in (('fc4',4),('fc16',16)):
            for mode in ('inference','grad-forward','training'):
                for query in ('auto','packed-v2'):
                    cost(shape,'graph',mode,query=query,b=a.batch,length=a.length)
                for baseline in ('fc-tensor',f'transformer-{depth}'):
                    cost(shape,baseline,mode,b=a.batch,length=a.length)
            cost(shape,'graph','training',client='python',b=a.batch,length=a.length)
    elif a.suite=='main':
        for shape,transformers in [('fc4',(2,4)),('fc16',(4,16))]:
            for baseline in ('graph','fc-tensor',*(f'transformer-{d}' for d in transformers)):
                for mode in ('inference','grad-forward','training'):cost(shape,baseline,mode)
        for baseline in ('graph','fc-tensor','transformer-16'):
            cost('fc16',baseline,'nograd')
    elif a.suite=='threads':
        for baseline in ('graph','fc-tensor','transformer-16'):
            for mode in ('inference','grad-forward','training'):cost('fc16',baseline,mode)
    elif a.suite=='boundaries':
        for boundary in ('public','query-direct','region-direct','both-direct'):
            cost('fc16','graph','training',client='python',boundary=boundary)
        for mode in ('inference','training'):
            cost('fc16','fc-tensor',mode,attention='sdpa')
    elif a.suite=='profiles':
        for baseline in ('graph','fc-tensor','transformer-4','transformer-16'):
            cost('fc16',baseline,'training',b=2,length=32,profile='cpu' if a.device=='cpu' else 'npu')
        cost('fc16','graph','training',profile='stages',fences='phase')
        for baseline in ('fc-tensor','transformer-16'):
            cost('fc16',baseline,'training',fences='phase')
    elif a.suite=='axes':
        for baseline in ('graph','fc-tensor'):
            for b,length in ((1,128),(8,16)):cost('fc16',baseline,'training',b=b,length=length)
    else:
        for shape,b,length,trace,dtype in [('fc4',2,8,True,'float64' if a.device=='cpu' else 'float32'),
                                         ('fc16',8,128,False,'float32')]:
            name=f'projection-{shape}-b{b}l{length}-{dtype}'
            args=['scripts/check_fc_comparison.py','--downstream-repo',a.downstream_repo,
                  '--native-library',a.native_library,'--device',a.device,'--fc-shape',shape,
                  '--batch',b,'--length',length,'--dtype',dtype,'--output-dir',out/name]
            if trace:args+=['--trace']
            invoke(name,args)
    raise SystemExit(int(any(row['exit_code'] for row in commands)))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Fresh-process exact application resume; all comparisons are bitwise."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json,digest,now


def equal(a,b,path='root'):
    import torch
    if isinstance(a,torch.Tensor):
        if not torch.equal(a,b): raise AssertionError(f'{path}: tensor differs')
    elif isinstance(a,dict):
        if a.keys()!=b.keys(): raise AssertionError(f'{path}: keys differ')
        for k in a: equal(a[k],b[k],f'{path}.{k}')
    elif isinstance(a,(tuple,list)):
        if type(a)!=type(b) or len(a)!=len(b): raise AssertionError(f'{path}: sequence differs')
        for i,(x,y) in enumerate(zip(a,b)): equal(x,y,f'{path}.{i}')
    elif a!=b: raise AssertionError(f'{path}: value differs')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--asset-path')
    p.add_argument('--retire-large-checkpoints',action='store_true')
    a=p.parse_args();root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=False)
    report=dict(schema='tide-lab-exact-resume-v1',state='running',started_at=now(),config_sha256=digest(a.config),checks={})
    atomic_json(root/'result.json',report)
    try:
        for label,extra in [('full',[]),('split',['--stop-after','2']),('resumed',['--resume',str(root/'split/checkpoint.pt')])]:
            command=[sys.executable,'-m','tide_lab.train','--config',a.config,'--output',str(root/label),*extra]
            if a.asset_path: command+=['--asset-path',a.asset_path]
            with (root/(label+'.log')).open('x') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
        import torch
        full=torch.load(root/'full/checkpoint.pt',map_location='cpu',weights_only=False)
        resumed=torch.load(root/'resumed/checkpoint.pt',map_location='cpu',weights_only=False)
        for component in ['model','optimizer','step','config_sha256','identity','rng','python_rng','accelerator_rng','sequence_state']:
            equal(full[component],resumed[component],component)
            report['checks'][component]='exact'
        report.update(state='passed',checkpoint_sha256={str(p.relative_to(root)):digest(p) for p in root.glob('*/*.pt')})
        if a.retire_large_checkpoints:
            for p in root.glob('*/*.pt'): p.unlink()
            report['retention']='checkpoints retired after exact comparison; hashes/logs/results retained; rerun reproduces'
    except BaseException as error:
        report.update(state='failed',error=repr(error));raise
    finally:
        report['finished_at']=now();atomic_json(root/'result.json',report)
    print(json.dumps(report))


if __name__=='__main__': main()

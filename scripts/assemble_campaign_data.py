#!/usr/bin/env python3
"""Recover already-downloaded fixed ranges; never retries a network request."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json,digest,now


def main(a):
    row=next(r for r in json.loads(a.tree.read_text()) if r['path'].endswith('train-00000-of-00002.parquet'))
    a.output.mkdir(parents=True,exist_ok=False);target=a.output/Path(row['path']).name;ranges=[]
    with target.open('xb') as out,a.prefix.open('rb') as prefix:
        for i,start in enumerate(range(0,row['size'],8*2**20)):
            n=min(8*2**20,row['size']-start);part=a.parts/str(i)
            if part.is_file() and part.stat().st_size==n:
                value=part.read_bytes();origin=str(part)
            else:
                if a.prefix.stat().st_size<start+n:raise ValueError('missing bytes cannot be recovered locally')
                prefix.seek(start);value=prefix.read(n);origin=str(a.prefix)+f'[{start}:{start+n}]'
            if len(value)!=n:raise ValueError('short recovered range')
            out.write(value);ranges.append(dict(index=i,origin=origin,bytes=n))
    actual=digest(target)
    if actual!=row['lfs']['oid']:raise ValueError('assembled file does not match fixed public LFS hash')
    target.chmod(0o444)
    atomic_json(a.output/'assembly.json',dict(schema='tide-data-byte-recovery-v1',created_at=now(),
        sha256=actual,size=row['size'],ranges=ranges,prefix_sha256=digest(a.prefix),
        note='reuse only already downloaded bytes; full fixed source digest passed; failed download attempts remain unchanged'))
    print(json.dumps(dict(sha256=actual,size=row['size'],recovered=True)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('tree','parts','prefix','output'):p.add_argument('--'+n,type=Path,required=True)
    main(p.parse_args())

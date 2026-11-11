#!/usr/bin/env python3
"""Extend the frozen WikiText pool without putting any old holdout into training."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json, digest, now
from tide_lab.language_data import parse_documents, text_hash


def prepare(a):
    import numpy as np
    from tokenizers import Tokenizer
    a.output.mkdir(parents=True, exist_ok=False)
    state = dict(state='running', started_at=now(), schema='tide-v9-data-preparation-v1')
    atomic_json(a.output/'state.json', state)
    start = time.monotonic()
    try:
        raw = a.output/'raw'; raw.mkdir()
        files = []
        for row in json.loads(a.tree.read_text()):
            if '/train-' not in row['path']: continue
            path = raw/Path(row['path']).name
            cached = a.old/'raw'/path.name
            if cached.exists(): shutil.copyfile(cached, path)
            else:
                url = 'https://huggingface.co/datasets/Salesforce/wikitext/resolve/'+a.revision+'/'+row['path']+'?download=true'
                subprocess.run(['curl','--fail','--location','--silent','--show-error',
                    '--connect-timeout','20','--max-time','1200','--speed-time','60',
                    '--speed-limit','16384','--output',str(path),url],check=True,timeout=1230)
            if path.stat().st_size != row['size'] or digest(path) != row['lfs']['oid']:
                raise ValueError('fixed source hash/size mismatch: '+path.name)
            path.chmod(0o444)
            files.append(dict(path=row['path'],sha256=digest(path),size=path.stat().st_size))
            print(json.dumps(dict(acquired=path.name)),flush=True)
        extracted = a.output/'raw-rows.jsonl'
        code = 'import pyarrow.parquet as pq,json,sys;from pathlib import Path\nwith open(sys.argv[2],"x") as f:\n for p in sorted(Path(sys.argv[1]).glob("*.parquet")):\n  for t in pq.read_table(p)["text"].to_pylist():f.write(json.dumps(t)+"\\n")\n'
        subprocess.run([str(a.arrow_python),'-c',code,str(raw),str(extracted)],check=True,timeout=600)
        docs = parse_documents([json.loads(s) for s in extracted.read_text().splitlines()], 'w103-train')
        old = json.loads((a.old/'data-lock.json').read_text())
        title = lambda d: ' '.join(d['title'].lower().split())
        # Historical official heldouts and all v6 heldouts remain sealed from train.
        official = [json.loads(s) for s in a.documents.read_text().splitlines()]
        held = [d for s in ('validation','test') for d in old['splits'][s]['documents']]
        held += [d for d in official if d['split'] in ('validation','test')]
        hashes = {d['normalized_sha256'] for d in held}; titles = {title(d) for d in held}
        known = {d['normalized_sha256'] for d in official}
        known |= {d['normalized_sha256'] for s in old['splits'].values() for d in s['documents']}
        known_titles = {title(d) for d in official}
        known_titles |= {title(d) for s in old['splits'].values() for d in s['documents']}
        pools = dict(train=[], confirmation=[]); seen=set(); seen_titles=set(); removed=0
        for d in docs:
            h=d['normalized_sha256']; t=title(d)
            if h in seen or t in seen_titles or h in hashes or t in titles:
                removed += 1; continue
            seen.add(h);seen_titles.add(t)
            bucket=int(text_hash('v6-split:'+h)[:8],16)%100
            if bucket in (0,1):
                if h not in known and t not in known_titles:pools['confirmation'].append(d)
                # Do not recycle unused historical holdout-pool articles into train.
            else:pools['train'].append(d)
        tokenizer=Tokenizer.from_file(str(a.tokenizer));splits={}
        for split,pool in pools.items():
            pool.sort(key=lambda d:text_hash('v9-order:'+d['normalized_sha256']))
            target=a.train_tokens if split=='train' else 262144
            spans=[]; count=0;path=a.output/(split+'.u16')
            with path.open('xb') as f:
                for d in pool:
                    if count>=target:break
                    ids=tokenizer.encode(d['text']).ids+[2]
                    if split!='train':ids=ids[:2048]
                    ids=ids[:target-count]
                    f.write(np.asarray(ids,dtype='<u2').tobytes())
                    spans.append({k:d[k] for k in ('id','sha256','normalized_sha256','title')} |
                                 dict(offset=count,tokens=len(ids),source_utf8_bytes=d['utf8_bytes']))
                    count+=len(ids)
            if count<target:raise ValueError(f'insufficient independent {split}: {count} < {target}')
            path.chmod(0o444);splits[split]=dict(file=path.name,sha256=digest(path),tokens=count,documents=spans)
        for split in ('validation','test'):
            info=old['splits'][split]
            shutil.copyfile(a.old/info['file'],a.output/info['file'])
            if digest(a.output/info['file'])!=info['sha256']:raise ValueError('old split changed')
            (a.output/info['file']).chmod(0o444);splits[split]=info
        hsets={s:{d['normalized_sha256'] for d in info['documents']} for s,info in splits.items()}
        if any(hsets[s]&hsets[t] for s in hsets for t in hsets if s<t):raise AssertionError('document split overlap')
        lock=dict(old,created_at=now(),splits=splits,source=dict(old['source'],files=files),
            tokenizer_sha256=digest(a.tokenizer),parent_data_lock_sha256=digest(a.old/'data-lock.json'),
            preparation='v9: full two-shard document reconstruction; normalized-hash and title exclusions; 128M unique-token pool',
            excluded_documents=removed,confirmation_policy='new whole documents absent from old train/heldouts; final review only',
            split_method='preserve validation/test bytes; reserve old hash buckets0/1; new train order v9-order',
            limitations='Exact normalized hash/title dedup; near duplicate contamination unknown. Train tail article may be truncated.')
        atomic_json(a.output/'data-lock.json',lock);(a.output/'data-lock.json').chmod(0o444)
        state.update(state='completed',data_lock_sha256=digest(a.output/'data-lock.json'),
            splits={s:dict(tokens=i['tokens'],documents=len(i['documents'])) for s,i in splits.items()})
    except BaseException as error:
        state.update(state='failed',error=repr(error));raise
    finally:
        state.update(finished_at=now(),seconds=time.monotonic()-start);atomic_json(a.output/'state.json',state)
        print(json.dumps(state),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('tree','old','documents','tokenizer','arrow-python','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--train-tokens',type=int,default=128*2**20)
    p.add_argument('--revision',default='b08601e04326c79dfdd32d625aee71d232d685c3')
    prepare(p.parse_args())

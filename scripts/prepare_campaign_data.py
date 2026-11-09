#!/usr/bin/env python3
"""Acquire pinned WikiText103 and prepare a document-disjoint mmap corpus."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import time
import urllib.request
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json,digest,now
from tide_lab.language_data import parse_documents,text_hash


def prepare(a):
    import numpy as np
    from tokenizers import Tokenizer
    a.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();raw=a.output/'raw';raw.mkdir()
    state=dict(schema='tide-v6-data-preparation-v1',state='running',started_at=now())
    atomic_json(a.output/'state.json',state)
    tree=json.loads(a.tree.read_text());files=[]
    try:
        for row in tree:
            if '/train-' not in row['path']:continue
            target=raw/Path(row['path']).name
            url='https://huggingface.co/datasets/Salesforce/wikitext/resolve/'+a.revision+'/'+row['path']+'?download=true'
            with urllib.request.urlopen(url,timeout=60) as response,target.with_suffix('.partial').open('xb') as f:
                while chunk:=response.read(2**20):f.write(chunk)
            temp=target.with_suffix('.partial')
            if temp.stat().st_size!=row['size'] or digest(temp)!=row['lfs']['oid']:
                raise ValueError('fixed remote file size/hash mismatch: '+row['path'])
            temp.rename(target);target.chmod(0o444)
            files.append(dict(path=row['path'],sha256=digest(target),size=target.stat().st_size))
            print(json.dumps(dict(downloaded=row['path'],bytes=target.stat().st_size)),flush=True)
        old={json.loads(s)['normalized_sha256'] for s in a.exclude.read_text().splitlines()}
        # Reuse the two already-qualified private data interpreters without
        # mixing their site-packages or changing either environment.
        extracted=a.output/'raw-rows.jsonl'
        code=('import json,sys; from pathlib import Path; import pyarrow.parquet as pq; '
              'out=open(sys.argv[2],"x"); '
              '[out.write(json.dumps(t)+"\\n") for p in sorted(Path(sys.argv[1]).glob("train-*.parquet")) '
              'for t in pq.read_table(p)["text"].to_pylist()]; out.close()')
        subprocess.run([str(a.arrow_python),'-c',code,str(raw),str(extracted)],check=True,timeout=600)
        rows=[json.loads(line) for line in extracted.read_text().splitlines()]
        documents=parse_documents(rows,'w103-train');del rows
        # The official small heldout files have already been observed in v1-v5.
        # Reserve fresh whole documents from the large training split instead.
        pools={s:[] for s in ('train','validation','test')};seen=set();duplicates=0
        for d in documents:
            h=d['normalized_sha256']
            if h in seen:duplicates+=1;continue
            seen.add(h);bucket=int(text_hash('v6-split:'+h)[:8],16)%100
            split='validation' if bucket==0 else 'test' if bucket==1 else 'train'
            if h in old:split='train'
            pools[split].append(d)
        tokenizer=Tokenizer.from_file(str(a.tokenizer));splits={}
        for split,pool in pools.items():
            pool.sort(key=lambda d:text_hash('v6-order:'+d['normalized_sha256']))
            spans=[];count=0;limit=a.train_tokens if split=='train' else 524288
            path=a.output/(split+'.u16')
            with path.open('xb') as f:
                for d in pool:
                    if count>=limit:break
                    ids=tokenizer.encode(d['text']).ids+[2]
                    # Cap each heldout article's contribution, spreading the
                    # estimate over documents instead of only long openings.
                    if split!='train':ids=ids[:2048]
                    ids=ids[:limit-count]
                    if not ids:continue
                    f.write(np.asarray(ids,dtype='<u2').tobytes())
                    spans.append(dict(id=d['id'],sha256=d['sha256'],normalized_sha256=d['normalized_sha256'],
                        offset=count,tokens=len(ids),source_utf8_bytes=d['utf8_bytes']))
                    count+=len(ids)
            if count<(a.train_tokens if split=='train' else 65536):raise ValueError('insufficient independent '+split+' tokens')
            path.chmod(0o444)
            splits[split]=dict(file=path.name,sha256=digest(path),tokens=count,documents=spans)
        hs={s:{d['normalized_sha256'] for d in r['documents']} for s,r in splits.items()}
        if any(hs[x]&hs[y] for x,y in [('train','validation'),('train','test'),('validation','test')]):
            raise AssertionError('whole-document split overlap')
        if (hs['validation']|hs['test'])&old:raise AssertionError('confirmation contains an old document')
        lock=dict(schema='tide-language-corpus-v6',created_at=now(),dtype='<u2',vocab_size=4096,bos=1,eos=2,pad=0,
            source=dict(repository='Salesforce/wikitext',revision=a.revision,subset='wikitext-103-raw-v1',
                licenses=['cc-by-sa-3.0','gfdl'],files=files),tokenizer_sha256=digest(a.tokenizer),
            excluded_old_documents_sha256=digest(a.exclude),splits=splits,duplicates_removed=duplicates,
            split_method='hash whole normalized documents, seed v6-split; old documents never enter confirmation',
            sample_method='epoch permutation of document-bounded blocks; full pool before repeats',
            limitations='exact normalized document dedup; near duplicate contamination not excluded; heldout per-document prefix <=2048 tokens; no test-based selection')
        atomic_json(a.output/'data-lock.json',lock);(a.output/'data-lock.json').chmod(0o444)
        state.update(state='completed',data_lock_sha256=digest(a.output/'data-lock.json'),
            splits={s:dict(tokens=r['tokens'],documents=len(r['documents'])) for s,r in splits.items()})
    except BaseException as error:
        state.update(state='failed',error=repr(error));raise
    finally:
        state.update(finished_at=now(),elapsed_seconds=time.monotonic()-started)
        atomic_json(a.output/'state.json',state);print(json.dumps(state),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('tree','tokenizer','exclude','output','arrow-python'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--revision',default='b08601e04326c79dfdd32d625aee71d232d685c3')
    p.add_argument('--train-tokens',type=int,default=20971520)
    prepare(p.parse_args())

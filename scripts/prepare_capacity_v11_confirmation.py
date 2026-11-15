#!/usr/bin/env python3
"""Fresh whole-document confirmation; historical data locks remain unchanged."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json,digest,now
from tide_lab.language_data import parse_documents,text_hash


def main(a):
    import numpy as np
    from tokenizers import Tokenizer
    root=a.root.resolve();a.output.mkdir(parents=True,exist_ok=False)
    report=dict(schema='tide-v11-confirmation-preparation-v1',state='running',started_at=now(),test_access=False)
    atomic_json(a.output/'state.json',report)
    try:
        locks=[root/'runs/v9-data-r3/data-lock.json',root/'runs/v6-data-final/data-lock.json',*getattr(a,'exclude_lock',[])]
        old=json.loads(locks[0].read_text());official=root/'runs/lm-data-v1/documents/documents.jsonl'
        known=[json.loads(s) for s in official.read_text().splitlines()]
        for p in locks:
            lock=json.loads(p.read_text());known.extend(d for v in lock['splits'].values() for d in v['documents'])
        title=lambda d:' '.join(d['title'].lower().split())
        hs={d['normalized_sha256'] for d in known};ts={title(d) for d in known}
        raw=root/'runs/v9-data-r3/raw-rows.jsonl'
        docs=parse_documents([json.loads(s) for s in raw.read_text().splitlines()],'w103-train')
        pool=[];seen_h=set();seen_t=set()
        for d in docs:
            h=d['normalized_sha256'];t=title(d)
            if h in hs or t in ts or h in seen_h or t in seen_t:continue
            seen_h.add(h);seen_t.add(t);pool.append(d)
        salt=getattr(a,'salt','v11-fresh-confirmation')
        pool.sort(key=lambda d:text_hash(salt+':'+d['normalized_sha256']))
        tokenizer_path=root/'runs/lm-data-v1/tokens/tokenizer-a.json';tok=Tokenizer.from_file(str(tokenizer_path))
        file=a.output/'confirmation.u16';spans=[];count=0;target=262144
        with file.open('xb') as f:
            for d in pool:
                if count>=target:break
                if time.time()>a.deadline_epoch-30:raise TimeoutError('data preparation deadline')
                ids=(tok.encode(d['text']).ids+[2])[:min(2048,target-count)]
                f.write(np.asarray(ids,dtype='<u2').tobytes())
                spans.append({k:d[k] for k in ('id','title','sha256','normalized_sha256')}|dict(offset=count,tokens=len(ids)))
                count+=len(ids)
        if count!=target:raise ValueError(f'insufficient unused articles: {count}')
        if any(d['normalized_sha256'] in hs or title(d) in ts for d in spans):raise AssertionError('prior split overlap')
        splits={s:old['splits'][s] for s in ('train','validation')}
        for s,v in splits.items():(a.output/v['file']).symlink_to(root/'runs/v9-data-r3'/v['file'])
        file.chmod(0o444);splits['confirmation']=dict(file=file.name,sha256=digest(file),tokens=count,documents=spans)
        lock=dict(old,splits=splits,created_at=now(),training_parent_lock_sha256=digest(locks[0]),
            tokenizer_sha256=digest(tokenizer_path),preparation=salt+' whole-document normalized hash AND title exclusion; unused W103 train articles',
            input_locks=[dict(path=str(p.relative_to(root)),sha256=digest(p)) for p in locks],
            official_documents_sha256=digest(official),raw_sha256=digest(raw),
            limitations='Exact hash/title exclusions include entire truncated train-tail article. Near-duplicate paraphrases not certified absent. Original test tokens never opened.')
        atomic_json(a.output/'data-lock.json',lock);(a.output/'data-lock.json').chmod(0o444)
        report.update(state='completed',tokens=count,documents=len(spans),unused_pool_documents=len(pool),data_lock_sha256=digest(a.output/'data-lock.json'))
    except BaseException as e:report.update(state='failed',error=repr(e));raise
    finally:report['finished_at']=now();atomic_json(a.output/'state.json',report)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--exclude-lock',type=Path,action='append',default=[]);p.add_argument('--salt',default='v11-fresh-confirmation')
    p.add_argument('--deadline-epoch',type=float,required=True);main(p.parse_args())

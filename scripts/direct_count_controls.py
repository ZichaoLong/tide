#!/usr/bin/env python3
"""Fixed train-count controls on the exact v7 sample stream; test stays sealed."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from tide_lab.campaign_data import Corpus
from tide_lab.records import atomic_json,digest,now,Metrics,project_to_lens
from tide_lab.train import source_identity,installed_identity


def stream_counts(corpus,seed,milestones=(262144,1048576,4194304)):
    counts=np.zeros(4096,dtype=np.int64)
    # Corpus is seed-local in training; this evaluator deliberately reuses it.
    # Discard epoch-only cached permutations before changing the seed.
    corpus.permutations.clear()
    cursor=tokens=0
    # Accumulate at most 4M targets in a temporary ~32MiB packed index vector.
    pair_chunks=[];boundaries=[]
    while tokens<milestones[-1]:
        ids,cursor=corpus.sample_ids(cursor,32,seed)
        remaining=min(m for m in milestones if m>tokens)-tokens
        for i in ids:
            offset,n,_,_=corpus.blocks['train'][i];n=min(int(n),remaining)
            if n<=0:break
            targets=np.asarray(corpus.arrays['train'][offset:offset+n],dtype=np.int64)
            inputs=np.concatenate(([corpus.lock['bos']],targets[:-1]))
            counts+=np.bincount(targets,minlength=4096)
            pair_chunks.append(inputs*4096+targets)
            tokens+=n;remaining-=n
        if tokens in milestones:boundaries.append(dict(tokens=tokens,cursor=cursor))
    bigrams=np.bincount(np.concatenate(pair_chunks),minlength=4096*4096).reshape(4096,4096)
    assert counts.sum()==tokens and bigrams.sum()==tokens
    return counts,bigrams,boundaries


def evaluate(corpus,counts,bigrams):
    unigram=(counts.astype(np.float64)+1)/(counts.sum()+len(counts))
    row_counts=bigrams.sum(1);records=[];remaining=32768
    totals={'unigram':0.,'bigram':0.}
    for i in corpus.evaluation_ids('validation',remaining):
        offset,n,_,_=corpus.blocks['validation'][i];n=min(int(n),remaining)
        targets=np.asarray(corpus.arrays['validation'][offset:offset+n],dtype=np.int64)
        inputs=np.concatenate(([corpus.lock['bos']],targets[:-1]))
        losses=dict(unigram=-np.log(unigram[targets]),
            bigram=-np.log((bigrams[inputs,targets]+10*unigram[targets])/(row_counts[inputs]+10)))
        row=dict(corpus.metadata('validation',i),tokens=n)
        for key,loss in losses.items():
            row[key+'_nll_sum']=float(loss.sum());totals[key]+=float(loss.sum())
        records.append(row);remaining-=n
    if remaining:raise ValueError('incomplete validation control')
    return dict(tokens=32768,nll={k:v/32768 for k,v in totals.items()},records=records)


def main(a):
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    corpus=Corpus(a.data,128);results=[]
    for seed in (11,22):
        run_id=f'tl-v7-count-controls-r2-s{seed}';directory=a.output/f'seed{seed}';directory.mkdir()
        atomic_json(directory/'manifest.json',dict(schema='tide-lab-run-v1',project='tide-learning-lab',
            run_id=run_id,name=run_id,study_id='TL-DIRECT-SCALE-v7',group='TL-DIRECT-SCALE-v7',
            stage='analytic-count-control',created_at=now(),config=dict(seed=seed,length=128,global_batch=32,
                target_tokens=4194304,unigram_add_one=True,bigram_unigram_prior_strength=10),
            source=source_identity(),installation=installed_identity(),data_lock_sha256=digest(a.data/'data-lock.json')))
        counts,bigrams,boundaries=stream_counts(corpus,seed);value=evaluate(corpus,counts,bigrams)
        value.update(state='completed',seed=seed,train_tokens=int(counts.sum()),boundaries=boundaries,
            test_access=False,scope='Fixed analytic unigram and last-token bigram controls, exact same sampled 4M training targets; no optimizer or checkpoint ancestry. No validation tuning.')
        atomic_json(directory/'result.json',value);metrics=Metrics(directory,run_id)
        metrics.log(0,{'validation/unigram_nll':value['nll']['unigram'],'validation/bigram_nll':value['nll']['bigram'],
                       'train/tokens':counts.sum(),'validation/tokens':value['tokens']});metrics.close()
        atomic_json(directory/'completed.json',dict(producer_finished=True,state='completed'))
        project_to_lens(directory,a.lens_cli);results.append(dict(run_id=run_id,**value))
        print(json.dumps(dict(run_id=run_id,nll=value['nll'],boundaries=boundaries)),flush=True)
        del bigrams
    atomic_json(a.output/'result.json',dict(state='completed',cases=results,elapsed_seconds=time.monotonic()-start))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lens-cli',required=True);main(p.parse_args())

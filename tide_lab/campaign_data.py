"""Read-only document-bounded mmap data, with a batch-independent token stream."""
import json
from pathlib import Path
import numpy as np
from .records import digest


class Corpus:
    def __init__(self,path,length,*,verify=True,allow_test=False,max_train_documents=None,extra_splits=()):
        self.path=Path(path);self.lock=json.loads((self.path/'data-lock.json').read_text())
        if self.lock['schema']!='tide-language-corpus-v6':raise ValueError('wrong corpus schema')
        self.length=length;self.arrays={};self.blocks={};self.permutations={}
        self.documents={};self.allow_test=allow_test
        if any(s!='confirmation' for s in extra_splits):raise ValueError('only explicitly requested confirmation may be added')
        for split in ('train','validation')+ (('test',) if allow_test else ()) + tuple(extra_splits):
            info=self.lock['splits'][split];file=self.path/info['file']
            if verify and digest(file)!=info['sha256']:raise ValueError('data content hash changed')
            self.arrays[split]=np.memmap(file,mode='r',dtype=self.lock['dtype'])
            blocks=[];docs=[]
            documents=info['documents'][:max_train_documents] if split=='train' and max_train_documents is not None else info['documents']
            for di,d in enumerate(documents):
                for offset in range(0,d['tokens'],length):
                    blocks.append((d['offset']+offset,min(length,d['tokens']-offset),di,offset))
                docs.append(d)
            self.blocks[split]=np.asarray(blocks,dtype=np.int64);self.documents[split]=docs

    def sample_ids(self,cursor,count,seed):
        total=len(self.blocks['train']);result=[]
        while count:
            epoch,offset=divmod(cursor,total)
            if epoch not in self.permutations:
                self.permutations={epoch:np.random.default_rng(seed+epoch*104729).permutation(total)}
            n=min(count,total-offset);result.extend(self.permutations[epoch][offset:offset+n].tolist())
            cursor+=n;count-=n
        return result,cursor

    def batch(self,split,ids,device='cpu',token_limit=None):
        import torch
        if split=='test' and not self.allow_test:raise ValueError('test is sealed during screening/training')
        x=np.full((len(ids),self.length),self.lock['pad'],dtype=np.int64)
        y=np.full((len(ids),self.length),-100,dtype=np.int64);remain=token_limit
        for row,i in enumerate(ids):
            offset,n,_,_=self.blocks[split][i]
            if remain is not None:n=min(n,max(0,remain));remain-=n
            if not n:continue
            vals=self.arrays[split][offset:offset+n];y[row,:n]=vals
            x[row,0]=self.lock['bos'];x[row,1:n]=vals[:-1]
        return dict(input_ids=torch.from_numpy(x).to(device),labels=torch.from_numpy(y).to(device))

    def evaluation_ids(self,split,token_limit):
        # Round-robin over document position, not a long first-document prefix.
        blocks=self.blocks[split]
        order=np.lexsort((blocks[:,2],blocks[:,3]))
        result=[];tokens=0
        for i in order:
            if tokens>=token_limit:break
            result.append(int(i));tokens+=int(blocks[i,1])
        return result

    def metadata(self,split,i):
        _,n,di,offset=self.blocks[split][i];d=self.documents[split][int(di)]
        return dict(block_id=d['id']+':'+str(offset),document_id=d['id'],tokens=int(n))

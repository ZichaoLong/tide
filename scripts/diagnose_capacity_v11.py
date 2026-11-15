#!/usr/bin/env python3
"""Read-only parent diagnostics; one disposable in-memory optimizer update."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from tide_lab.records import atomic_json,digest,now,Metrics,project_to_lens
from tide_lab.train import runtime,source_identity,installed_identity
from tide_lab.campaign_runtime import make_model,optimizer_for,logical_update,exact_tree,memory
from tide_lab.campaign_data import Corpus


def vector_stats(t):
    t=t.detach().float().cpu()
    v=t.flatten(1).norm(dim=1) if t.ndim>1 else t.abs()
    return dict(l2=float(t.norm()),per_first_axis_l2=v.tolist(),zero_first_axis=int((v==0).sum()),elements=t.numel())


class Observe:
    def __init__(self,model):
        self.model=model;self.handles=[];self.rows={};self.grads={};self.valid=None;self.enabled=True;self.backward=False
        if hasattr(model,'selectors'):
            for j,s in enumerate(model.selectors):self.handles.append(s.register_forward_hook(self.hb(j)))
        elif hasattr(model,'inventory'):
            for j,b in enumerate(model.blocks):self.handles.append(b.ffn.register_forward_hook(self.moe(j)))
    def accumulate(self,j,key,v):
        v=v.detach().float().cpu();r=self.rows.setdefault(str(j),{})
        r[key]=v if key not in r else r[key]+v
    def hb(self,j):
        def hook(module,args,out):
            if not self.enabled:return
            r,m=args;c=out[0];valid=self.valid[None,:,:]
            for key,value in [('receipt_all',r>0),('full_all',c>0),('receipt_valid',(r>0)&valid),('full_valid',(c>0)&valid)]:
                self.accumulate(j,key,value.sum((1,2)))
            self.accumulate(j,'send_valid',(c*valid).sum((1,2)))
            self.accumulate(j,'content_sq_valid',(m['content'].detach().square()*valid[...,None]).sum((1,2,3)))
            self.accumulate(j,'read_sq_valid',sum((v.detach().square()*valid[...,None]).sum((1,2,3)) for v in m['memory']))
            if self.backward:
                for key,tensor in [('input',m['x']),('content',m['content']),*[(f'read{i}',v) for i,v in enumerate(m['memory'])]]:
                    if tensor.requires_grad:
                        def save(g,k=str(j)+'/'+key):
                            value=g.detach().float().square().sum((1,2,3)).cpu()
                            self.grads[k]=self.grads.get(k,torch.zeros_like(value))+value
                        tensor.register_hook(save)
        return hook
    def moe(self,j):
        def hook(module,args,out):
            if self.enabled:
                # Parent objective supplies the valid mask during a real update;
                # evaluation below supplies it explicitly as well.
                self.accumulate(j,'assignments_valid',module.stats['counts'])
        return hook
    def export(self):
        rows={j:{k:v.tolist() for k,v in r.items()} for j,r in self.rows.items()}
        for j,r in rows.items():
            counts=r.get('full_valid',r.get('assignments_valid'))
            if counts is not None:
                c=np.asarray(counts);p=c/max(1,c.sum())
                r['coverage_summary']=dict(units=len(c),zero=int((c==0).sum()),max_share=float(p.max()),
                    effective_units=float(np.exp(-(p*np.log(np.maximum(p,1e-30))).sum())))
        return rows
    def remove(self):
        for h in self.handles:h.remove()


def forward(model,c,b):
    return model(b['input_ids'],valid_mask=b['labels']!=-100) if c['kind']=='moe' else model(b['input_ids'])


def main(a):
    root=a.root.resolve();case=json.loads(a.table.read_text())['cases'][a.index]
    c=json.loads((root/case['config']).read_text());a.output.mkdir(parents=True,exist_ok=False)
    report=dict(schema='tide-capacity-diagnostic-v11',state='running',started_at=now(),case=case,source=source_identity(),installation=installed_identity(),
        scope='512 fixed train sequences plus one disposable global64 update; no lifetime-use inference, no confirmation/test access; parameter deltas include weight decay and optimizer history')
    atomic_json(a.output/'result.json',report);run='tl-v11-diag-'+case['id'];metrics=Metrics(a.output,run)
    atomic_json(a.output/'manifest.json',dict(project='tide-learning-lab',run_id=run,name=run,group='TL-CAPACITY-DIAG-v11',study_id='TL-CAPACITY-DIAG-v11',config=c))
    try:
        device=runtime(c['device'],c['threads'],c['seed']);state=torch.load(root/case['checkpoint'],map_location='cpu',weights_only=False)
        if digest(root/case['config'])!=state['identity']['config_sha256'] or digest(a.data/'data-lock.json')!=state['identity']['data_lock_sha256']:
            raise ValueError('parent config/data mismatch')
        if digest(root/case['checkpoint'])!=case['sha256']:raise ValueError('parent checkpoint drift')
        model=make_model(c,device);model.load_state_dict(state['model'],strict=True);model.eval()
        exposure=state['progress']['tokens'];report['parent_identity']=state['identity']
        if hasattr(model,'set_progress'):model.set_progress(exposure)
        corpus=Corpus(a.data,256);ids=np.random.default_rng(110011).choice(len(corpus.blocks['train']),512,replace=False).tolist()
        report['block_ids']=ids;micro=c['training']['batch'];b=corpus.batch('train',ids[:micro],device)
        with torch.no_grad():baseline=forward(model,c,b).detach()
        obs=Observe(model);obs.valid=b['labels']!=-100
        with torch.no_grad():instrumented=forward(model,c,b).detach()
        exact_tree(baseline,instrumented);report['observer_output_equality']='passed_exact_same_batch'
        del baseline,instrumented,b;obs.rows={};work={};targets=0;nllsum=0.
        with torch.no_grad():
            for start in range(0,512,micro):
                if time.time()>a.deadline_epoch-60:raise TimeoutError('diagnostic deadline')
                b=corpus.batch('train',ids[start:start+micro],device);obs.valid=b['labels']!=-100
                y=forward(model,c,b);targets+=int(obs.valid.sum())
                nllsum+=float(torch.nn.functional.cross_entropy(y.flatten(0,1),b['labels'].flatten(),ignore_index=-100,reduction='sum'))
                for k,v in model.last_metrics.items():
                    if k.startswith('work/'):work[k]=work.get(k,0)+v
        report.update(observations=obs.export(),work=work,valid_targets=targets,model_positions=512*256,train_window_nll=nllsum/targets)
        atomic_json(a.output/'forward.json',report)
        obs.rows={};obs.backward=True
        # label context follows every physical microbatch through a pre-hook.
        batches=[corpus.batch('train',ids[i:i+micro],device) for i in range(0,64,micro)]
        cursor=[0]
        def set_valid(module,args):
            i=cursor[0]%len(batches);obs.valid=batches[i]['labels']!=-100;cursor[0]+=1
        h=model.register_forward_pre_hook(set_valid)
        opt=optimizer_for(model,c);opt.load_state_dict(state['optimizer'])
        row=logical_update(model,opt,batches,c,exposure,device);h.remove();obs.remove()
        report['single_update']=row;report['tensor_gradient_l2']={k:v.sqrt().tolist() for k,v in obs.grads.items()}
        report['parameters']={}
        for name,p in model.named_parameters():
            if p.grad is not None and not bool(torch.isfinite(p.grad).all()):raise FloatingPointError(name)
            report['parameters'][name]=dict(gradient=vector_stats(p.grad) if p.grad is not None else None,
                update=vector_stats(p.detach().cpu()-state['model'][name]),weight=vector_stats(state['model'][name]))
        report['historical_changes']={}
        for label,path in case['history'].items():
            old=torch.load(root/path,map_location='cpu',weights_only=False)
            report['historical_changes'][label]=dict(path=path,sha256=digest(root/path),exposure=old['progress']['tokens'],
                to_128mi={n:vector_stats(state['model'][n]-v) for n,v in old['model'].items() if isinstance(v,torch.Tensor) and v.is_floating_point()})
            del old
        metrics.log(0,{'diag/train_window_nll':report['train_window_nll'],'diag/valid_targets':targets,
            'diag/update_loss':row['loss'],'diag/gradient_norm':row['gradient_norm'],**memory(device)})
        report['state']='completed'
    except BaseException as e:report.update(state='failed',error=repr(e),traceback=traceback.format_exc());raise
    finally:
        report['finished_at']=now();atomic_json(a.output/'result.json',report);metrics.close()
        atomic_json(a.output/'completed.json',dict(producer_finished=True,state=report['state']));project_to_lens(a.output,a.lens_cli)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','table','data','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--index',type=int,required=True);p.add_argument('--deadline-epoch',type=float,required=True)
    p.add_argument('--lens-cli',required=True);main(p.parse_args())

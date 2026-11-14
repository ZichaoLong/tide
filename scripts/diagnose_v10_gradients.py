#!/usr/bin/env python3
"""Bounded, independent replay of a failed v10 update; never resumes a main run.

The model/update are imported from the ORIGINAL frozen training source. This
diagnostic has a separate frozen identity and writes only its own new output.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('training-source','config','data','checkpoint','original-train','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seconds',type=int,default=3300)
    p.add_argument('--mode',choices=('replay','checkpoint-failed-batch'),default='replay')
    return p.parse_args()


def main(a):
    diagnostic_source=Path(__file__).resolve().parents[1]
    sys.path[:0]=[str(a.training_source.resolve()),str(a.training_source.resolve()/'scripts')]
    import torch
    from torch.nn import functional as F
    from tide_lab.records import atomic_json,digest,now
    from tide_lab.train import runtime,cpu_tree
    from tide_lab.campaign_runtime import make_model,optimizer_for,logical_update,restore,checkpoint
    from tide_lab.campaign_data import Corpus
    from tide_lab.language_training import tensor_save
    from tide_lab.supervisor import verify_source
    from campaign_stage import identity

    a.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();deadline=started+a.seconds
    c=json.loads(a.config.read_text());original=json.loads((a.original_train/'progress.json').read_text())
    source=verify_source(a.training_source);diag=verify_source(diagnostic_source)
    device=runtime(c['device'],c['threads'],c['seed'])
    ident=identity(a.config,c,a.data,'train')
    result=dict(schema='tide-v10-gradient-diagnosis-v1',state='running',started_at=now(),
        config_id=c['id'],mode=a.mode,original_failure_step=original['step']+1,
        training_source=dict(commit=source['commit'],tree_sha256=source['content']['tree_sha256']),
        diagnostic_source=dict(commit=diag['commit'],tree_sha256=diag['content']['tree_sha256']),
        identity=ident,checkpoint_sha256=digest(a.checkpoint),
        physical_visibility=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
        budget=dict(seconds=a.seconds,new_disk_gib=4),
        axes=dict(execution='running',strict_numerical='original_failures_retained',
                  recovery='replay_pending',learning='not_a_new_learning_trajectory'),
        scope='Original model/optimizer/batches; no validation, no main output writes or formal continuation.')
    atomic_json(a.output/'result.json',result)

    def stats(t):
        if t is None:return None
        x=t.detach().cpu();finite=torch.isfinite(x);v=x[finite].double()
        return dict(shape=list(x.shape),dtype=str(x.dtype),elements=x.numel(),
                    nan=int(x.isnan().sum()),inf=int(x.isinf().sum()),
                    max_abs_finite=float(v.abs().max()) if v.numel() else None,
                    min_positive=float(v[v>0].min()) if (v>0).any() else None,
                    finite_l2=float(torch.linalg.vector_norm(v)) if v.numel() else 0.)

    def gradients():
        rows={n:stats(p.grad) for n,p in model.named_parameters() if p.grad is not None}
        bad=[n for n,v in rows.items() if v['nan'] or v['inf']]
        norm=math.sqrt(math.fsum(v['finite_l2']**2 for v in rows.values()))
        return dict(parameters=rows,nonfinite_parameters=bad,
                    all_parameter_gradients_finite=not bad,
                    finite_elements_norm_fp64=norm,
                    interpretation='true_nonfinite_gradients' if bad else 'finite_gradients; inspect norm_reduction')

    def batches_for(progress):
        t=c['training'];target=min([m for m in t['milestones'] if m>progress['tokens']]+[t['target_tokens']])
        ids,cursor=corpus.sample_ids(progress['cursor'],t['global_batch'],c['seed'])
        remain=target-progress['tokens'];batches=[];raw=0
        for i in range(0,len(ids),t['batch']):
            subset=ids[i:i+t['batch']];raw+=sum(int(corpus.blocks['train'][j,1]) for j in subset)
            if remain<=0:continue
            b=corpus.batch('train',subset,device,token_limit=remain)
            remain-=int((b['labels']!=-100).sum());batches.append(b)
        return batches,cursor,raw,ids

    class Incident(Exception):pass

    def inspect_backward(batches,progress):
        """After failure only, inspect graph boundaries without modifying VJPs."""
        count=sum(int((b['labels']!=-100).sum()) for b in batches);reports=[]
        for micro,b in enumerate(batches):
            if time.monotonic()>deadline:break
            model.zero_grad(set_to_none=True);model.set_progress(progress['tokens']);model.train()
            incident={};handles=[]
            def hook_for(node):
                def inspect(grad_inputs,grad_outputs):
                    # Only risky arithmetic nodes are watched in this first
                    # pass; a finite-input/nonfinite-output boundary is proof.
                    if not any(g is not None and not bool(torch.isfinite(g).all()) for g in grad_inputs):return
                    incident.update(node=node.name(),microbatch=micro,
                        grad_inputs=[stats(x) for x in grad_inputs],
                        grad_outputs=[stats(x) for x in grad_outputs],
                        upstream_all_finite=all(g is None or bool(torch.isfinite(g).all()) for g in grad_outputs),
                        forward_trace=node.metadata.get('traceback_',[]))
                    saved={}
                    for name in dir(node):
                        if name.startswith('_saved_'):
                            value=getattr(node,name)
                            if isinstance(value,torch.Tensor):saved[name]=value.detach().cpu()
                    incident['saved']={k:stats(v) for k,v in saved.items()}
                    tensor_save(a.output/'operator-witness.pt',dict(node=node.name(),saved=saved,
                        grad_inputs=cpu_tree(grad_inputs),grad_outputs=cpu_tree(grad_outputs),microbatch=micro))
                    atomic_json(a.output/'operator-incident.json',incident)
                    raise Incident(node.name())
                return inspect
            try:
                with torch.autograd.set_detect_anomaly(True,check_nan=False):
                    logits=model(b['input_ids'])
                    loss=F.cross_entropy(logits.flatten(0,1),b['labels'].flatten(),ignore_index=-100,reduction='sum')/count
                    pending=[loss.grad_fn];seen=set()
                    while pending:
                        node=pending.pop()
                        if node is None or node in seen:continue
                        seen.add(node)
                        if any(x in node.name().lower() for x in ('divbackward','logbackward','rsqrtbackward','expbackward','fusionattention')):
                            handles.append(node.register_hook(hook_for(node)))
                        pending.extend(parent for parent,_ in node.next_functions)
                    loss.backward()
                reports.append(dict(microbatch=micro,loss=float(loss.detach()),watched_nodes=len(handles),incident=False))
            except Incident:
                reports.append(dict(microbatch=micro,loss=float(loss.detach()),watched_nodes=len(handles),incident=incident))
                return reports
            finally:
                for h in handles:h.remove()
        return reports

    try:
        model=make_model(c,device);opt=optimizer_for(model,c)
        progress=restore(a.checkpoint,model,opt,ident,device)
        result['checkpoint_progress']=copy.deepcopy(progress)
        corpus=Corpus(a.data,c['training']['length'])
        historical={}
        for line in (a.original_train/'metrics.jsonl').read_text().splitlines():
            row=json.loads(line)
            if 'train/loss' in row.get('metrics',{}):historical[row['step']]=row['metrics']
        historical[original['step']]=original['last_update']
        comparisons=[];result['common_step_comparisons']=comparisons
        if a.mode=='checkpoint-failed-batch':
            # Explicitly NOT the failed weights: isolate the next input batch.
            progress={k:copy.deepcopy(original[k]) for k in ('step','tokens','cursor','probe','milestones','dropped_tail_tokens')}
        while progress['step']<=original['step'] and time.monotonic()<deadline-180:
            batches,cursor,raw,ids=batches_for(progress)
            before=copy.deepcopy(progress)
            try:
                if a.mode=='checkpoint-failed-batch':
                    # No optimizer update in this non-trajectory check.
                    model.set_progress(progress['tokens']);model.train();opt.zero_grad(set_to_none=True)
                    count=sum(int((b['labels']!=-100).sum()) for b in batches)
                    for b in batches:
                        loss=F.cross_entropy(model(b['input_ids']).flatten(0,1),b['labels'].flatten(),ignore_index=-100,reduction='sum')/count
                        loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),c['training']['clip_norm'],error_if_nonfinite=True)
                    result['state']='failed_batch_at_old_weights_finite';break
                row=logical_update(model,opt,batches,c,progress['tokens'],device)
            except (RuntimeError,FloatingPointError) as exc:
                result.update(state='failure_reproduced',failure_step=before['step']+1,error=repr(exc),
                              failure_progress=before,failed_batch_ids=ids)
                summary=gradients();atomic_json(a.output/'gradients.json',summary)
                result['gradient_summary']={k:v for k,v in summary.items() if k!='parameters'}
                saved_progress=copy.deepcopy(before)
                if a.mode!='replay':
                    # A failed-batch input probe uses OLD weights. Do not mint
                    # an apparently resumable formal trajectory at newer tokens.
                    saved_progress.update(probe=True,diagnostic_not_formal_resume=True,
                        weights_progress=result['checkpoint_progress'],input_progress=before)
                tensor_save(a.output/'failure-checkpoint.pt',checkpoint(model,opt,saved_progress,ident,device))
                tensor_save(a.output/'failure-batches.pt',cpu_tree(batches))
                # Persist proof even if deeper instrumentation reaches its budget.
                atomic_json(a.output/'result.json',result)
                result['backward_inspection']=inspect_backward(batches,before)
                break
            progress.update(step=progress['step']+1,tokens=progress['tokens']+row['tokens'],cursor=cursor,
                dropped_tail_tokens=progress['dropped_tail_tokens']+max(0,raw-row['tokens']))
            if progress['tokens'] in c['training']['milestones']:progress['milestones'].append(progress['tokens'])
            if progress['step'] in historical:
                h=historical[progress['step']]
                comparisons.append(dict(step=progress['step'],loss=row['loss'],original_loss=h['train/loss'],
                    loss_delta=row['loss']-h['train/loss'],gradient_norm=row['gradient_norm'],
                    original_gradient_norm=h['train/gradient_norm'],
                    gradient_norm_delta=row['gradient_norm']-h['train/gradient_norm'],
                    tokens_equal=progress['tokens']==h['train/tokens']))
            atomic_json(a.output/'progress.json',dict(progress,updated_at=now(),last_loss=row['loss'],
                gradient_norm=row['gradient_norm'],elapsed_seconds=time.monotonic()-started))
            if progress['step']%16==0:print(json.dumps(dict(step=progress['step'],tokens=progress['tokens'],loss=row['loss'])),flush=True)
        if result['state']=='running':result['state']='failure_not_reproduced' if progress['step']>original['step'] else 'budget_exhausted'
        result['axes'].update(execution='completed',recovery='see_common_step_comparisons; no_final_parameter_equality_claim')
    except BaseException as exc:
        result.update(state='diagnostic_error',error=repr(exc),traceback=traceback.format_exc())
        result['axes']['execution']='failed'
        raise
    finally:
        result.update(ended_at=now(),seconds=time.monotonic()-started)
        atomic_json(a.output/'result.json',result)
        print(json.dumps({k:result[k] for k in ('state','config_id','seconds')}),flush=True)


if __name__=='__main__':main(arguments())

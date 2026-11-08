#!/usr/bin/env python3
"""One finite stage of TL-B4; the durable driver owns the whole budget."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import resource
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import torch
from tide_lab.capability import (ShadowBranch, dataset, intervention, normalize,
                                paired_summary, preservation_gate, statistics)
from tide_lab.growth import CHECKPOINT_REPOSITORY, CHECKPOINT_REVISION, verify_checkpoint
from tide_lab.records import Metrics, atomic_json, digest, now, project_to_lens
from tide_lab.train import cpu_tree, installed_identity, runtime, source_identity, synchronize


def save_tensor(path, value):
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('wb') as stream:
        torch.save(cpu_tree(value), stream)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def load_teacher(asset, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hashes = verify_checkpoint(Path(asset))
    tokenizer = AutoTokenizer.from_pretrained(asset, local_files_only=True, trust_remote_code=False)
    lm = AutoModelForCausalLM.from_pretrained(asset, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float32, attn_implementation='eager').to(device).eval()
    lm.requires_grad_(False)
    if lm.config.model_type != 'qwen2' or lm.config.hidden_size != 896:
        raise ValueError('fixed checkpoint architecture changed')
    return lm, tokenizer, hashes


def encode(rows, tokenizer, plan, device):
    sequences, answers = [], []
    for row in rows:
        ids = tokenizer.apply_chat_template([{'role':'user', 'content':row['prompt']}],
                                            tokenize=True, add_generation_prompt=True)
        answer = tokenizer.encode(row['answer'], add_special_tokens=False)
        if len(answer) != 1 or not 0 < len(ids) <= plan['data']['max_tokens']:
            raise ValueError('fixed native task needs one-token answer and bounded prompt')
        sequences.append(ids); answers.append(answer[0])
    width = max(map(len,sequences)); pad = tokenizer.pad_token_id or tokenizer.eos_token_id
    tokens = torch.tensor([[pad]*(width-len(s))+s for s in sequences],device=device)
    mask = torch.tensor([[0]*(width-len(s))+[1]*len(s) for s in sequences],device=device)
    positions = (mask.cumsum(-1)-1).clamp_min(0)
    return dict(input_ids=tokens, attention_mask=mask, position_ids=positions), answers, list(map(len,sequences))


def last_logits(lm, inputs, **kwargs):
    logits = lm(**inputs, logits_to_keep=1, return_dict=True, **kwargs).logits[:, -1].float()
    if not torch.isfinite(logits).all():
        raise FloatingPointError('nonfinite native LM logits')
    return logits


@torch.no_grad()
def teacher_rows(lm, tokenizer, rows, plan, device, collect=True):
    x, y, observed = [], [], []
    batch = plan['baseline']['batch_size']
    for offset in range(0,len(rows),batch):
        subset = rows[offset:offset+batch]
        inputs, answers, lengths = encode(subset,tokenizer,plan,device)
        captured = {}
        with intervention(lm, capture=captured) as calls:
            logits = last_logits(lm,inputs,use_cache=False)
        if len(calls)!=1:
            raise AssertionError('expected one final MLP call per teacher forward')
        probs = logits.log_softmax(-1).cpu(); predictions = probs.argmax(-1).tolist()
        if collect:
            x.append(captured['x']); y.append(captured['y'])
        for i,row in enumerate(subset):
            observed.append(dict(id=row['id'],family=row['family'],answer_id=answers[i],
                prediction=predictions[i],prediction_text=tokenizer.decode([predictions[i]]),
                correct=predictions[i]==answers[i],answer_nll=-float(probs[i,answers[i]]),
                prompt_tokens=lengths[i]))
    return (torch.cat(x) if x else None, torch.cat(y) if y else None, observed)


def prepare(a, plan, device, metrics):
    fixtures = dataset(plan)
    atomic_json(a.output/'dataset.json', fixtures)
    lm, tokenizer, hashes = load_teacher(a.asset_path, device)
    validation, families, observations = {}, [], {}
    for index,family in enumerate(plan['tasks']):
        x,y,rows = teacher_rows(lm,tokenizer,fixtures[family+'/validation'],plan,device)
        accuracy = sum(r['correct'] for r in rows)/len(rows)
        observations[family] = dict(accuracy=accuracy,samples=len(rows),rows=rows,
            admitted=accuracy>=plan['baseline']['minimum_validation_accuracy'])
        metrics.log(index,{f'baseline/{family}/accuracy':accuracy})
        if observations[family]['admitted']:
            families.append(family); validation[family] = dict(x=x,y=y,rows=rows)
    # Initial alpha=0 neutrality and independent full/cached numerical axis.
    row = fixtures[plan['tasks'][0]+'/validation'][:1]
    inputs, _, _ = encode(row,tokenizer,plan,device)
    with torch.no_grad():
        ordinary = last_logits(lm,inputs,use_cache=False)
        with intervention(lm,alpha=0.):
            neutral = last_logits(lm,inputs,use_cache=False)
        exact = torch.equal(ordinary,neutral)
        if not exact: raise AssertionError('alpha=0 altered native output')
        prefix = {k:v[:,:-1] for k,v in inputs.items()}
        past = lm(**prefix,use_cache=True,return_dict=True,logits_to_keep=1).past_key_values
        tail = dict(input_ids=inputs['input_ids'][:,-1:],attention_mask=inputs['attention_mask'],
                    position_ids=inputs['position_ids'][:,-1:])
        cached = last_logits(lm,tail,use_cache=True,past_key_values=past)
        atol,rtol=plan['checks']['atol'],plan['checks']['rtol']
        passed = torch.allclose(ordinary,cached,atol=atol,rtol=rtol)
        cache_check=dict(state='passed' if passed else 'strict_failed_finite',atol=atol,rtol=rtol,
                         max_abs=float((ordinary-cached).abs().max()),
                         same_argmax=bool(torch.equal(ordinary.argmax(-1),cached.argmax(-1))))
        if not cache_check['same_argmax']:
            raise AssertionError('native cache check changed discrete next-token prediction')
    if families:
        tx,ty=[],[]
        for family in families:
            x,y,_ = teacher_rows(lm,tokenizer,fixtures[family+'/train'],plan,device)
            tx.append(x);ty.append(y)
        train=dict(x=torch.cat(tx),y=torch.cat(ty))
        stats=statistics(train['x'],train['y'])
        cache=dict(schema='tide-shadow-features-v1',families=families,train=train,
                   validation=validation,stats=stats,dataset_sha256=digest(a.output/'dataset.json'),
                   plan_sha256=digest(a.plan),checkpoint_files=hashes,
                   checkpoint_repository=CHECKPOINT_REPOSITORY,checkpoint_revision=CHECKPOINT_REVISION)
        save_tensor(a.output/'features.pt',cache)
    return dict(state='completed',observations=observations,selected_families=families,
        initial_alpha_zero_exact=exact,full_cache_check=cache_check,
        teacher_frozen=all(not p.requires_grad and p.grad is None for p in lm.parameters()),
        checkpoint_files=hashes,dataset_sha256=digest(a.output/'dataset.json'),
        features_sha256=digest(a.output/'features.pt') if families else None,
        independent_test='not_checked; held until all training/confirmation decisions fixed',
        scope='native full-vocabulary next-token accuracy, two one-word tasks; not broad language capability')


def cache_data(a,plan):
    cache=load(a.cache)
    if cache['plan_sha256']!=digest(a.plan): raise ValueError('feature/plan mismatch')
    return cache


def information(a,plan,device,metrics):
    cache=cache_data(a,plan)
    x,y=normalize(cache['train']['x'],cache['train']['y'],cache['stats'])
    x,y=x.double(),y.double()
    covariance=x.T@x/max(1,len(x)-1)
    values,vectors=torch.linalg.eigh(covariance)
    basis=vectors[:,-plan['interface']['width']:]
    fraction=float(values[-plan['interface']['width']:].sum()/values.sum())
    rows=[]
    for label,features in [('full',x),('pca16',x@basis)]:
        # Regularization is fixed, train-derived; no heldout choice of strength/rank.
        gram=features.T@features
        ridge=1e-3*gram.diag().mean().clamp_min(1e-12)
        weights=torch.linalg.solve(gram+ridge*torch.eye(gram.shape[0],dtype=gram.dtype),features.T@y)
        for family,data in cache['validation'].items():
            vx,vy=normalize(data['x'],data['y'],cache['stats']);vx,vy=vx.double(),vy.double()
            if label=='pca16':vx=vx@basis
            nmse=float((vx@weights-vy).square().sum()/vy.square().sum())
            rows.append(dict(readout=label,family=family,relative_mse=nmse))
            metrics.log(len(rows),{f'information/{label}/{family}/relative_mse':nmse})
    return dict(state='completed',pca16_variance_fraction=fraction,readouts=rows,
        scope='CPU FP64 ridge diagnostic on actual teacher MLP input/output; linear readout error is not proof of absent information')


def branch_data(cache,device):
    stats={k:v.to(device) for k,v in cache['stats'].items()}
    x,y=normalize(cache['train']['x'].to(device),cache['train']['y'].to(device),stats)
    validation={f:normalize(d['x'].to(device),d['y'].to(device),stats) for f,d in cache['validation'].items()}
    return x,y,validation


def optimizer_for(model,plan):
    cfg=plan['training']
    parameters=[p for p in model.parameters() if p.requires_grad]
    if len(parameters)!=len({id(p) for p in parameters}):raise AssertionError('duplicate optimizer owner')
    return torch.optim.AdamW(parameters,lr=cfg['lr'],eps=cfg['eps'],weight_decay=cfg['weight_decay'],foreach=False)


@torch.no_grad()
def validate_branch(model,validation,batch):
    model.eval();result={}
    for family,(x,y) in validation.items():
        error=0.
        for i in range(0,len(x),batch):
            pred=model(x[i:i+batch])
            if not torch.isfinite(pred).all():raise FloatingPointError('nonfinite shadow prediction')
            error+=float((pred-y[i:i+batch]).square().sum().item())
        result[family]=error/float(y.square().sum().item())
    return result


def update(model,optimizer,x,y,plan,device):
    model.train();optimizer.zero_grad(set_to_none=True);synchronize(device);start=time.monotonic()
    prediction=model(x);loss=(prediction-y).square().mean()
    if not torch.isfinite(loss):raise FloatingPointError('nonfinite shadow loss')
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                    plan['training']['clip_norm'],error_if_nonfinite=True,foreach=False)
    optimizer.step();synchronize(device)
    return float(loss.detach()),float(norm),time.monotonic()-start


def index_batch(n,batch,seed,step):
    generator=torch.Generator().manual_seed(100003*seed+step)
    return torch.randint(n,(batch,),generator=generator)


def calibrate(a,plan,device,metrics):
    cache=cache_data(a,plan);x,y,_=branch_data(cache,device);rows=[]
    for batch in plan['training']['batch_candidates']:
        for kind in plan['training']['candidates']:
            model=ShadowBranch(kind,x.shape[-1],plan['interface'],11).to(device)
            optimizer=optimizer_for(model,plan);ids=index_batch(len(x),batch,11,0).to(device)
            elapsed=[]
            for step in range(3):
                loss,norm,seconds=update(model,optimizer,x[ids],y[ids],plan,device)
                if step:elapsed.append(seconds)
            rows.append(dict(kind=kind,batch=batch,warmup_steps=1,timed_steps=2,step_seconds=elapsed,
                             last_loss=loss,npu_peak_allocated_mib=torch.npu.max_memory_allocated()/2**20 if device.type=='npu' else None))
            metrics.log(len(rows),{f'calibration/{kind}/batch':batch,
                                  f'calibration/{kind}/samples_per_second':2*batch/sum(elapsed)})
            del model,optimizer
    scores={b:b/sum(sum(r['step_seconds'])/2 for r in rows if r['batch']==b) for b in plan['training']['batch_candidates']}
    best=max(scores.values());chosen=min(b for b,score in scores.items() if score>=.9*best)
    projected={k:sum(r['step_seconds'])/2*plan['training']['steps'] for r in rows if r['batch']==chosen for k in [r['kind']]}
    return dict(state='completed',batch=chosen,rows=rows,scores=scores,projected_training_step_seconds=projected,
        reason='smallest common batch within 90% of maximum paired measured samples/second; finite calibration, not universal optimum',
        independent_sample_batch=chosen,microbatch=chosen,global_batch=chosen,accumulation=1,positions=1)


def training_identity(a,plan):
    return dict(source=source_identity(),installation=installed_identity(),device=plan['device'],
                dtype=plan['dtype'],threads=plan['threads'],physical_visible=os.getenv('ASCEND_RT_VISIBLE_DEVICES'),
                plan_sha256=digest(a.plan),cache_sha256=digest(a.cache),kind=a.kind,seed=a.seed,batch=a.batch,
                deterministic=torch.are_deterministic_algorithms_enabled())


def checkpoint(path,model,optimizer,step,identity,cache,device):
    saved=dict(schema='tide-shadow-checkpoint-v1',model=cpu_tree(model.state_dict()),
        optimizer=cpu_tree(optimizer.state_dict()),step=step,identity=identity,stats=cache['stats'],
        rng=torch.get_rng_state(),python_rng=random.getstate(),
        accelerator_rng=cpu_tree(torch.npu.get_rng_state_all()) if device.type=='npu' else None,
        data_cursor='stateless (seed,step) batch index',sequence_state='empty; one-position episodes')
    save_tensor(path,saved)
    atomic_json(path.with_suffix('.json'),dict(step=step,sha256=digest(path),identity=identity))


def train(a,plan,device,metrics):
    cache=cache_data(a,plan);x,y,validation=branch_data(cache,device)
    model=ShadowBranch(a.kind,x.shape[-1],plan['interface'],a.seed).to(device)
    optimizer=optimizer_for(model,plan);identity=training_identity(a,plan);step=0
    if a.resume:
        saved=load(a.resume)
        if saved['identity']!=identity:raise ValueError('exact resume identity mismatch')
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer']);step=saved['step']
        torch.set_rng_state(saved['rng']);random.setstate(saved['python_rng'])
        if device.type=='npu':torch.npu.set_rng_state_all(saved['accelerator_rng'])
    first=step;start=time.monotonic();step_sum=0.;work={};curve=[]
    total=a.limit if a.limit is not None else plan['training']['steps']
    initial=validate_branch(model,validation,a.batch)
    for f,v in initial.items():metrics.log(step,{f'validation/{f}/relative_mse':v})
    for index in range(step,total):
        if time.monotonic()-start>plan['training']['max_seconds_per_run']:
            raise TimeoutError('fixed per-run training budget exhausted')
        ids=index_batch(len(x),a.batch,a.seed,index).to(device)
        loss,norm,seconds=update(model,optimizer,x[ids],y[ids],plan,device)
        step=index+1;step_sum+=seconds
        for k,v in model.last_metrics.items():
            if k in ('work/candidates','work/full','work/messages','work/state_updates','work/read_events','work/source_rows','work/region_blocks','work/branch_samples'):
                work[k]=work.get(k,0)+int(v)
        if step==first+1 or step%plan['training']['log_every']==0 or step==total:
            grad={}
            for bucket in ('input','core','output'):
                grads=[p.grad.detach().float().square().sum() for n,p in model.named_parameters() if n.startswith(bucket) and p.grad is not None]
                grad['gradient/'+bucket+'_norm']=float(torch.stack(grads).sum().sqrt()) if grads else 0.
            metrics.log(step,{'train/loss':loss,'train/gradient_norm_before_clip':norm,'time/step_seconds':seconds,
                'time/total_step_seconds':step_sum,'work/examples':step*a.batch,
                'resource/cpu_maxrss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                **({'resource/npu_peak_allocated_mib':torch.npu.max_memory_allocated()/2**20} if device.type=='npu' else {}),
                **grad,**{'work_total/'+k.split('/',1)[1]:v for k,v in work.items()}})
        if step%plan['training']['eval_every']==0 or step==total:
            val=validate_branch(model,validation,a.batch);curve.append(dict(step=step,validation=val))
            metrics.log(step,{f'validation/{f}/relative_mse':v for f,v in val.items()})
        if step%plan['training']['checkpoint_every']==0 or step==total:
            checkpoint(a.output/'checkpoint.pt',model,optimizer,step,identity,cache,device)
            atomic_json(a.output/'state.json',dict(state='running',steps=step,updated_at=now()))
    final=curve[-1]['validation']
    gate=all(v<=plan['gates']['maximum_validation_relative_mse'] for v in final.values())
    return dict(state='completed',kind=a.kind,seed=a.seed,steps=step,batch=a.batch,trained_examples=step*a.batch,
        initial_validation=initial,validation=final,learning_gate_met=gate,curve=curve,work_totals=work,
        successful_step_seconds=step_sum,scope='shadow local MLP fitting; no baseline output modified during training',
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total_parameters=sum(p.numel() for p in model.parameters()),checkpoint_sha256=digest(a.output/'checkpoint.pt'),
        exact_resume='separate fresh-process gate',strict_numerics='not_checked_by_training')


def mechanisms(a,plan,device,metrics):
    cache=cache_data(a,plan);x,y,_=branch_data(cache,device);results={}
    for kind in plan['training']['candidates']:
        model=ShadowBranch(kind,x.shape[-1],plan['interface'],11).to(device)
        optimizer=optimizer_for(model,plan);checks={};before=cpu_tree(model.state_dict())
        xx=x[:2].clone().requires_grad_(True)
        pred=model(xx);loss=(pred-y[:2]).square().mean();loss.backward()
        checks['zero_initial_prediction']=bool(torch.equal(pred,torch.zeros_like(pred)))
        checks['output_gradient_nonzero']=float(model.output.weight.grad.norm())>0
        checks['inner_initial_zero']=all(p.grad is None or torch.count_nonzero(p.grad)==0 for p in model.core.parameters())
        # Independent derivative of the external MSE/readout, not an oracle schedule.
        with torch.no_grad():
            core_value=model.core(torch.tanh(model.input(xx.detach()))[:,None,:]) if kind=='tide' else model.core(torch.tanh(model.input(xx.detach())))
            expected=(2*(pred.detach()-y[:2])/pred.numel()).T@core_value
            checks['readout_gradient_strict']=bool(torch.allclose(expected,model.output.weight.grad,atol=plan['checks']['atol'],rtol=plan['checks']['rtol']))
            checks['readout_gradient_max_abs']=float((expected-model.output.weight.grad).abs().max())
        optimizer.step();optimizer.zero_grad(set_to_none=True);xx.grad=None
        loss=(model(xx)-y[:2]).square().mean();loss.backward()
        checks['inner_gradient_nonzero']=any(p.grad is not None and float(p.grad.norm())>0 for p in model.core.parameters())
        checks['input_root_gradient_nonzero']=xx.grad is not None and float(xx.grad.norm())>0
        checks['output_updated']=not torch.equal(before['output.weight'],model.output.weight.detach().cpu())
        parameters=[p for p in model.parameters() if p.requires_grad]
        checks['unique_optimizer_parameters']=len(parameters)==len({id(p) for p in parameters})
        checks['finite_gradients']=all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters)
        checks['finite_gradients']=bool(checks['finite_gradients'])
        if not all(v for k,v in checks.items() if k!='readout_gradient_max_abs'):
            raise AssertionError(f'new branch mechanism failed: {kind}: {checks}')
        results[kind]=checks;metrics.log(len(results),{f'mechanism/{kind}/passed':1})
    return dict(state='passed',checks=results,scope='external readout derivative, zero/nonzero gradient ingress, input root and unique optimizer; public Core5 historical strict failures retained')


@torch.no_grad()
def evaluate(a,plan,device,metrics):
    cache=cache_data(a,plan);fixtures=json.loads(Path(a.dataset).read_text())
    if digest(a.dataset)!=cache['dataset_sha256']:raise ValueError('dataset changed')
    lm,tokenizer,_=load_teacher(a.asset_path,device)
    stats={k:v.to(device) for k,v in cache['stats'].items()}
    model=None
    if a.checkpoint:
        saved=load(a.checkpoint)
        if saved['identity']['cache_sha256']!=digest(a.cache) or saved['identity']['plan_sha256']!=digest(a.plan):
            raise ValueError('evaluation checkpoint/cache identity mismatch')
        model=ShadowBranch(saved['identity']['kind'],896,plan['interface'],saved['identity']['seed']).to(device)
        model.load_state_dict(saved['model'],strict=True);model.eval()
    def predictor(x):
        if model is None:return stats['y_mean'].expand_as(x)
        return stats['y_mean']+stats['y_scale']*model((x-stats['x_mean'])/stats['x_scale'])
    all_results={};all_pass=True
    for family in cache['families']:
        rows=fixtures[family+'/'+a.split];observations=[];deltas=[];local_error=0.;local_denom=0.
        batch=plan['baseline']['batch_size']
        for offset in range(0,len(rows),batch):
            subset=rows[offset:offset+batch];inputs,answers,lengths=encode(subset,tokenizer,plan,device)
            capture={}
            with intervention(lm,capture=capture):baseline=last_logits(lm,inputs,use_cache=False)
            bp=baseline.log_softmax(-1);base_preds=bp.argmax(-1).cpu().tolist()
            if a.baseline_only:
                changed=baseline;delta=0.
            else:
                changed_capture={}
                with intervention(lm,predictor,plan['interface']['alpha'],changed_capture):
                    changed=last_logits(lm,inputs,use_cache=False)
                delta=changed_capture['delta_rms']
                xx,yy=capture['x'].to(device),capture['y'].to(device)
                local_error+=float((predictor(xx)-yy).square().sum())
                local_denom+=float((yy-stats['y_mean']).square().sum())
            cp=changed.log_softmax(-1);preds=cp.argmax(-1).cpu().tolist()
            kl=(bp.exp()*(bp-cp)).sum(-1).cpu().tolist();deltas.append(delta)
            for i,row in enumerate(subset):
                observations.append(dict(id=row['id'],answer_id=answers[i],base_prediction=base_preds[i],prediction=preds[i],
                    base_correct=base_preds[i]==answers[i],correct=preds[i]==answers[i],
                    base_nll=-float(bp[i,answers[i]]),nll=-float(cp[i,answers[i]]),kl=kl[i]))
        summary=paired_summary(observations);delta=sum(deltas)/len(deltas)
        passed=preservation_gate(summary,delta,plan['gates']) if not a.baseline_only else None
        # Paired bootstrap descriptive interval; small-sample screen, no population noninferiority claim.
        rng=random.Random(22011);differences=[int(r['correct'])-int(r['base_correct']) for r in observations]
        draws=sorted(sum(rng.choices(differences,k=len(differences)))/len(differences) for _ in range(2000))
        summary['paired_accuracy_difference_bootstrap_95']=[draws[49],draws[1949]]
        n=len(observations);rate=summary['newly_wrong']/n;z=1.959963984540054
        center=(rate+z*z/(2*n))/(1+z*z/n)
        radius=z*math.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/(1+z*z/n)
        summary['new_error_rate_wilson_95']=[max(0.,center-radius),min(1.,center+radius)]
        all_results[family]=dict(summary=summary,delta_rms=delta,preservation_screen_met=passed,
            relative_mse=local_error/local_denom if local_denom else None,per_example=observations)
        if not a.baseline_only:all_pass &= passed
        metrics.log(len(all_results),{f'quality/{family}/accuracy':summary['accuracy'],
            f'quality/{family}/baseline_accuracy':summary['baseline_accuracy'],
            f'quality/{family}/newly_wrong':summary['newly_wrong'],f'quality/{family}/answer_nll_increase':summary['answer_nll_increase'],
            f'quality/{family}/kl':summary['kl'],f'activity/{family}/delta_rms':delta})
    # Same active model, full prefix versus KV continuation. This is a strict
    # implementation check, independent of baseline-versus-new-model quality.
    continuation={'state':'not_checked'}
    if not a.baseline_only:
        sample=fixtures[cache['families'][0]+'/'+a.split][:1]
        inputs,_,_=encode(sample,tokenizer,plan,device)
        with intervention(lm,predictor,plan['interface']['alpha']):
            full=last_logits(lm,inputs,use_cache=False)
            prefix={k:v[:,:-1] for k,v in inputs.items()}
            past=lm(**prefix,use_cache=True,return_dict=True,logits_to_keep=1).past_key_values
            tail=dict(input_ids=inputs['input_ids'][:,-1:],attention_mask=inputs['attention_mask'],
                      position_ids=inputs['position_ids'][:,-1:])
            resumed=last_logits(lm,tail,use_cache=True,past_key_values=past)
        passed=torch.allclose(full,resumed,atol=plan['checks']['atol'],rtol=plan['checks']['rtol'])
        continuation=dict(state='passed' if passed else 'strict_failed_finite',
            atol=plan['checks']['atol'],rtol=plan['checks']['rtol'],
            max_abs=float((full-resumed).abs().max()),same_argmax=bool(torch.equal(full.argmax(-1),resumed.argmax(-1))))
        if not continuation['same_argmax']:
            raise AssertionError('active branch cache continuation changed discrete prediction')
    return dict(state='completed',split=a.split,alpha=0 if a.baseline_only else plan['interface']['alpha'],
                candidate='baseline' if a.baseline_only else ('train_mean' if model is None else a.kind),
                families=all_results,preservation_screen_met=all_pass if not a.baseline_only else None,
                checkpoint_sha256=digest(a.checkpoint) if a.checkpoint else None,
                continuation=continuation,
                scope='native next-token paired preservation; only final MLP final position; teacher computation retained; not full layer replacement, long generation or universal noninferiority')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','information','mechanisms','calibrate','train','evaluate'])
    for key in ('plan','output'):parser.add_argument('--'+key,required=True,type=Path)
    for key in ('cache','dataset','resume','checkpoint'):parser.add_argument('--'+key,type=Path)
    parser.add_argument('--asset-path');parser.add_argument('--recording-cli')
    parser.add_argument('--run-id',required=True);parser.add_argument('--kind',choices=['tide','mlp'],default='tide')
    parser.add_argument('--seed',type=int,default=11);parser.add_argument('--batch',type=int,default=2)
    parser.add_argument('--limit',type=int);parser.add_argument('--split',choices=['validation','test'],default='validation')
    parser.add_argument('--baseline-only',action='store_true')
    a=parser.parse_args();plan=json.loads(a.plan.read_text());start=time.monotonic()
    device=runtime('cpu' if a.stage=='information' else plan['device'],plan['threads'],a.seed)
    a.output.mkdir(parents=True,exist_ok=False)
    group={'prepare':'BASE','information':'INFO','mechanisms':'CHECK','calibrate':'CHECK','train':'SHADOW','evaluate':'PRESERVE'}[a.stage]
    manifest=dict(schema='tide-lab-run-v1',project='tide-learning-lab',group='TL-B4-'+group,study_id='TL-B4-'+group,
        run_id=a.run_id,name=a.run_id,created_at=now(),config=plan,config_sha256=digest(a.plan),
        identity=dict(source=source_identity(),installation=installed_identity(),device=str(device),
                      physical_visible=os.getenv('ASCEND_RT_VISIBLE_DEVICES'),
                      dtype='float64' if a.stage=='information' else 'float32',threads=plan['threads']),
        stage=a.stage,seed=a.seed,kind=a.kind,cache_sha256=digest(a.cache) if a.cache else None)
    atomic_json(a.output/'manifest.json',manifest);metrics=Metrics(a.output,a.run_id)
    result=dict(state='running',started_at=now());atomic_json(a.output/'state.json',result)
    code=0
    try:
        result.update(globals()[a.stage](a,plan,device,metrics))
    except BaseException as error:
        result.update(state='failed',error=repr(error),traceback=traceback.format_exc());code=1
        traceback.print_exc()
    finally:
        result.update(run_id=a.run_id,stage=a.stage,finished_at=now(),elapsed_seconds=time.monotonic()-start)
        atomic_json(a.output/'result.json',result);atomic_json(a.output/'state.json',result)
        metrics.close();atomic_json(a.output/'completed.json',dict(producer_finished=True,state=result['state']))
        if a.recording_cli:
            result['lens_projection_completed']=project_to_lens(a.output,a.recording_cli)
            atomic_json(a.output/'result.json',result)
    return code


if __name__=='__main__':
    raise SystemExit(main())

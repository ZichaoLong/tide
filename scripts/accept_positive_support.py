#!/usr/bin/env python3
"""Finite TL-REQ-003 consumer checks; installed public APIs own execution."""
import argparse
import copy
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
import torch
from tide_lab.records import Metrics, atomic_json, digest, now, project_to_lens
from tide_lab.train import runtime, installed_identity, source_identity, cpu_tree
from tide_lab.language_training import (make_a, optimizer_for, load_tokens, indices,
    batch_from, update, evaluate, checkpoint_state, restore, tensor_save)


POLICY = 'positive-floor-v1'


def exact(a, b, path='root'):
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a.cpu(), b.cpu()):
            raise AssertionError('exact resume tensor mismatch: '+path)
    elif isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise AssertionError('exact resume mapping mismatch: '+path)
        for k in a: exact(a[k], b[k], path+'/'+str(k))
    elif isinstance(a, (tuple, list)):
        if type(a) is not type(b) or len(a) != len(b):
            raise AssertionError('exact resume sequence mismatch: '+path)
        for i, (x, y) in enumerate(zip(a,b)): exact(x,y,path+'/'+str(i))
    elif a != b:
        raise AssertionError('exact resume scalar mismatch: '+path)


def load_json(path):
    return json.loads(Path(path).read_text())


def checked_material(cfg, key):
    item=cfg[key]
    if digest(item['checkpoint']) != item['sha256']:
        raise ValueError('original checkpoint hash changed: '+key)
    manifest=load_json(item['manifest'])
    state=torch.load(item['checkpoint'],map_location='cpu',weights_only=False)
    if state['schema']!='tide-language-checkpoint-v1' or state['identity']!=manifest['identity']:
        raise ValueError('original application identity mismatch: '+key)
    return manifest,state


def export_original(a,cfg,device,metrics):
    """Decode only with the original wheel; do not rewrite any old identity."""
    import tidegraph
    root=Path(tidegraph.__file__).parent
    installed=hashlib.sha256(''.join(str(p.relative_to(root))+digest(p) for p in sorted(root.rglob('*.py'))).encode()).hexdigest()
    outputs={}
    for key in ('hb','fc'):
        manifest,state=checked_material(cfg,key)
        if state['identity']['installation']['tidegraph_source_sha256'] != installed:
            raise ValueError('export must decode under the original installation')
        ident=state['identity'];model=make_a(manifest['config'],ident['kind'],ident['seed'],ident['depth'])
        model.load_state_dict(state['model'],strict=True)
        optimizer=optimizer_for(model,manifest['config'],'a',ident['kind'])
        optimizer.load_state_dict(state['optimizer'])
        if model.embedding.weight is not model.head.weight:raise AssertionError('tied weight lost')
        material=dict(schema='tide-explicit-policy-transfer-v1',original_identity=ident,
            original_checkpoint_sha256=cfg[key]['sha256'],original_model_id=model.encoder.model_id,
            original_graph_identity=model.encoder.binding.graph.identity,
            parameter_names=[n for n,p in model.named_parameters()],plan=manifest['config'],
            state=state,scope='parameter/optimizer/RNG/cursor material; graph state resets every sequence; not cross-policy exact resume')
        tensor_save(a.output/(key+'-material.pt'),material)
        outputs[key]=dict(step=state['step'],sha256=digest(a.output/(key+'-material.pt')),
            original_model_id=material['original_model_id'],parameters=len(material['parameter_names']))
    return dict(state='completed',exports=outputs,original_installed_source_sha256=installed)


def material(a,key,device):
    meta=load_json(a.material/'result.json')['exports'][key]
    path=a.material/(key+'-material.pt')
    if digest(path)!=meta['sha256']:raise ValueError('export changed')
    item=torch.load(path,map_location='cpu',weights_only=False)
    plan=copy.deepcopy(item['plan']);plan['core']['support_policy']=POLICY
    ident=item['original_identity']
    model=make_a(plan,ident['kind'],ident['seed'],ident['depth']).to(device)
    if [n for n,p in model.named_parameters()]!=item['parameter_names']:
        raise AssertionError('parameter order changed across policy conversion')
    model.load_state_dict(item['state']['model'],strict=True)
    optimizer=optimizer_for(model,plan,'a',ident['kind'])
    optimizer.load_state_dict(item['state']['optimizer'])
    for slots in optimizer.state.values():
        for k,v in slots.items():
            if isinstance(v,torch.Tensor) and k!='step':slots[k]=v.to(device)
    torch.set_rng_state(item['state']['rng']);random.setstate(item['state']['python_rng'])
    if model.encoder.model_id==item['original_model_id']:raise AssertionError('new policy retained old model identity')
    return item,plan,model,optimizer


def foundation(a,cfg,device,metrics):
    env=dict(os.environ,TIDE_POSITIVE_SUPPORT_DEVICE=str(device))
    paths=['tests/test_positive_support.py','tests/test_positive_support_config.py','tests/test_positive_support_graph.py']
    command=[sys.executable,'-m','pytest','-q','-p','no:cacheprovider','-o','pythonpath=',*paths,
             '--junitxml='+str(a.output/'support.xml')]
    with (a.output/'pytest.log').open('x') as log:
        proc=subprocess.run(command,cwd=a.foundation,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240)
    suites=ET.parse(a.output/'support.xml').getroot()
    counts={k:sum(int(s.get(k,0)) for s in suites.iter('testsuite')) for k in ('tests','failures','errors','skipped')}
    if proc.returncode or counts['failures'] or counts['errors'] or counts['skipped']:
        raise AssertionError('foundation directed checks failed: '+str(counts))
    return dict(state='completed',counts=counts,device_scope='13 NPU-target tests plus 17 CPU/config helpers' if device.type=='npu' else '30 CPU tests',
                command=command,foundation=cfg['foundation_revision'])


def task(a,cfg,device,metrics):
    from tide_lab.language_checks import a_checks
    data,_=load_tokens(cfg['data'],'a',32)
    reports={}
    for key in ('hb','fc'):
        m=load_json(cfg[key]['manifest']);plan=copy.deepcopy(m['config']);ident=m['identity']
        plan['core']['support_policy']=POLICY
        model=make_a(plan,'tide-dense',ident['seed'],ident['depth']).to(device)
        reports[key]=a_checks(model,data['train'],plan,'tide-dense',ident['seed'],device,ident['depth'])
        if not reports[key]['admitted_for_finite_exploration']:raise AssertionError('task admission failed')
    return dict(state='completed',checks=reports)


def epoch(model,optimizer,plan,ident,data,step,device,metrics):
    from tidegraph.weighted import observe_support
    ids=indices(len(data['train']['labels']),ident['batch'],ident['seed'],step)
    model.encoder.observe=(step+1)%4==0
    with observe_support() as obs:
        row=update(model,optimizer,batch_from(data['train'],ids,device),plan,'a','tide-dense',step,ident['schedule_steps'],device)
    row['support']=obs.metrics()
    for name,p in model.named_parameters():
        if not torch.isfinite(p).all() or (p.grad is not None and not torch.isfinite(p.grad).all()):
            raise AssertionError('nonfinite trained parameter/gradient: '+name)
    values={'train/loss':row['loss'],'train/gradient_norm':row['gradient_norm'],'train/seconds':row['seconds'],
            'support/repaired_coordinates':row['support']['repaired_coordinates'],'support/repaired_rows':row['support']['repaired_rows'],**row['work']}
    metrics.log(step+1,values)
    print(json.dumps(dict(step=step+1,loss=row['loss'],gradient_norm=row['gradient_norm'],repairs=row['support']['repaired_coordinates'])),flush=True)
    return row


def hb(a,cfg,device,metrics):
    from tidegraph.weighted import observe_support
    item,plan,model,optimizer=material(a,'hb',device)
    ident=item['original_identity'];data,_=load_tokens(cfg['data'],'a',ident['length'])
    # Keep the original full-model failure as an explicit negative control.
    strict=make_a(item['plan'],'tide-dense',ident['seed'],ident['depth']).to(device)
    strict.load_state_dict(item['state']['model'])
    ids=indices(len(data['train']['labels']),ident['batch'],ident['seed'],item['state']['step'])
    try:
        with torch.no_grad():strict(batch_from(data['train'],ids,device)['input_ids'])
    except ArithmeticError as error:
        if 'support underflowed' not in str(error):raise
        refusal=error.record();atomic_json(a.output/'strict-failure.json',refusal)
    else:raise AssertionError('original full-model failure did not reproduce')
    del strict
    with observe_support() as obs:
        initial=evaluate(model,data['validation'],'a',device,8,limit=8)
    initial['support']=obs.metrics();metrics.log(86,{'validation/nll':initial['nll']})
    identity=dict(schema='tide-policy-acceptance-identity-v1',source=source_identity(),installation=installed_identity(),
        policy=POLICY,model_id=model.encoder.model_id,graph_identity=model.encoder.binding.graph.identity,
        converted_from=item['original_checkpoint_sha256'],data_lock_sha256=digest(Path(cfg['data'])/'data-lock.json'),
        device=str(device),threads=1,seed=ident['seed'],batch=ident['batch'],length=ident['length'],schedule_steps=ident['schedule_steps'])
    step=item['state']['step'];tokens=item['state']['valid_tokens'];rows=[]
    while step<cfg['hb_final_step']:
        row=epoch(model,optimizer,plan,ident,data,step,device,metrics)
        step+=1;tokens+=row['valid_tokens'];rows.append(dict(step=step,**row))
        atomic_json(a.output/'progress.json',dict(step=step,target=cfg['hb_final_step']))
        if step==cfg['resume_split_step']:
            tensor_save(a.output/'split.pt',checkpoint_state(model,optimizer,step,tokens,identity,device,'a'))
    tensor_save(a.output/'final.pt',checkpoint_state(model,optimizer,step,tokens,identity,device,'a'))
    tensor_save(a.output/'final-gradients.pt',{n:p.grad for n,p in model.named_parameters()})
    with observe_support() as obs:
        final=evaluate(model,data['validation'],'a',device,8,limit=8)
    final['support']=obs.metrics();metrics.log(step,{'validation/nll':final['nll']})
    atomic_json(a.output/'resume-identity.json',identity)
    return dict(state='completed',strict_control='expected positive-support failure reproduced',start_step=item['state']['step'],end_step=step,
        training=rows,initial_validation=initial,final_validation=final,
        split_sha256=digest(a.output/'split.pt'),final_sha256=digest(a.output/'final.pt'),
        final_gradients_sha256=digest(a.output/'final-gradients.pt'),
        learning_scope='eight updates and fixed validation8 stability observation; no new quality or architecture claim')


def resume(a,cfg,device,metrics):
    item,plan,model,optimizer=material(a,'hb',device);ident=item['original_identity']
    identity=load_json(a.prior/'resume-identity.json');prior=load_json(a.prior/'result.json')
    if identity['installation']!=installed_identity() or identity['source']!=source_identity():
        raise AssertionError('fresh resume installation/source changed')
    for name,key in [('split.pt','split_sha256'),('final.pt','final_sha256'),('final-gradients.pt','final_gradients_sha256')]:
        if digest(a.prior/name)!=prior[key]:raise AssertionError('resume artifact changed')
    step,tokens=restore(a.prior/'split.pt',model,optimizer,identity,device,'a')
    data,_=load_tokens(cfg['data'],'a',ident['length']);rows=[]
    while step<cfg['hb_final_step']:
        row=epoch(model,optimizer,plan,ident,data,step,device,metrics)
        step+=1;tokens+=row['valid_tokens'];rows.append(dict(step=step,**row))
        expected=next(r for r in prior['training'] if r['step']==step)
        for key in ('loss','gradient_norm','lr','alpha','valid_tokens','work','support'):exact(row[key],expected[key],key)
    actual=checkpoint_state(model,optimizer,step,tokens,identity,device,'a')
    expected=torch.load(a.prior/'final.pt',map_location='cpu',weights_only=False)
    exact(actual,expected)
    exact(cpu_tree({n:p.grad for n,p in model.named_parameters()}),torch.load(a.prior/'final-gradients.pt',weights_only=False))
    return dict(state='completed',exact_new_process_resume='passed',atol=0,rtol=0,start_step=cfg['resume_split_step'],end_step=step,
        checked=['model','optimizer slots','RNG','data cursor','all final parameter gradients','loss/gradnorm/work/support every resumed update'],training=rows)


def fc(a,cfg,device,metrics):
    from tidegraph.weighted import observe_support
    item,plan,model,optimizer=material(a,'fc',device)
    data,_=load_tokens(cfg['data'],'a',32);reports={}
    for index,intervention in enumerate((None,'context_reset')):
        with observe_support() as obs:
            report=evaluate(model,data['test'],'a',device,8,intervention=intervention)
        report['support']=obs.metrics();reports['normal' if intervention is None else intervention]=report
        metrics.log(index,{'evaluation/nll':report['nll'],'evaluation/tokens':report['tokens'],
            'support/repaired_coordinates':report['support']['repaired_coordinates']})
    old=load_json(cfg['fc']['normal_result'])
    return dict(state='completed',evaluations=reports,original_normal_result_sha256=digest(cfg['fc']['normal_result']),
                original_normal_result=old,scope='same original trained weights, explicit policy conversion; full existing heldout set; no new training')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['export','foundation','task','hb','resume','fc'])
    for field in ('config','output'):p.add_argument('--'+field,type=Path,required=True)
    for field in ('material','prior','foundation'):p.add_argument('--'+field,type=Path)
    p.add_argument('--device',default='cpu');p.add_argument('--recording-cli')
    a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    cfg=load_json(a.config);start=time.monotonic();run_id='tl-support-v1-'+a.output.name
    device=runtime(a.device,1,11)
    manifest=dict(schema='tide-lab-run-v1',project='tide-learning-lab',run_id=run_id,name=run_id,group='TL-LM-EXEC',
        config=dict(study_id='TL-LM-EXEC',phase=a.phase,policy=POLICY,foundation=cfg['foundation_revision'],
                    device=str(device),seed=11,config_sha256=digest(a.config)),source=source_identity())
    atomic_json(a.output/'manifest.json',manifest)
    metrics=Metrics(a.output,run_id);result=dict(state='running')
    atomic_json(a.output/'result.json',result)
    try:
        if a.phase!='export':manifest['installation']=installed_identity();atomic_json(a.output/'manifest.json',manifest)
        result=globals()[a.phase if a.phase!='export' else 'export_original'](a,cfg,device,metrics)
    except BaseException as error:
        result=dict(state='failed',error=repr(error),traceback=traceback.format_exc())
        if callable(getattr(error,'record',None)):result['failure']=error.record()
    finally:
        metrics.close();result.update(finished_at=now(),elapsed_seconds=time.monotonic()-start,
            phase=a.phase,device=str(device),manifest_sha256=digest(a.output/'manifest.json'))
        atomic_json(a.output/'result.json',result);atomic_json(a.output/'completed.json',dict(state=result['state']))
        if a.recording_cli and a.phase in ('hb','resume','fc'):project_to_lens(a.output,a.recording_cli)
    print(json.dumps({k:v for k,v in result.items() if k in ('state','phase','error','elapsed_seconds')}),flush=True)
    return 0 if result['state']=='completed' else 1


if __name__=='__main__':raise SystemExit(main())

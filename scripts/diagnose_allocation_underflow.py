#!/usr/bin/env python3
"""Bounded application replay; installed public APIs own all graph execution."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
import torch
from tidegraph.weighted import allocate
from tide_lab.records import atomic_json, digest, now
from tide_lab.train import installed_identity, runtime, source_identity
from tide_lab.language_training import (batch_from, checkpoint_state, indices, load_tokens,
    make_a, optimizer_for, restore, tensor_save, update)


def analyze_failure(error):
    """Probe only the captured four-candidate row, never replace model execution."""
    captured = error.record()
    values = error.tensors
    probes = []
    for dtype, factor in [(torch.float32, 1.), (torch.float64, 1.),
                          (torch.float32, 2.), (torch.float32, 4.)]:
        fields = {k: v.to(device='cpu', dtype=dtype) for k, v in values.items()}
        row = dict(dtype=str(dtype), temperature_factor=factor,
                   scope='public scalar API on captured inputs; changed dtype/temperature is a separate probe')
        try:
            answer = allocate(fields['scores'], fields['caps'], fields['lower'], fields['upper'],
                fields['temperature']*factor, regularizer='ent', ent_alpha=1.,
                score_shift=fields.get('shift'))
            row.update(state='completed', controls=answer.controls.tolist(),
                       multiplier=float(answer.multiplier), residual=float(answer.residual),
                       constraint=answer.constraint,
                       all_positive_caps_have_positive_controls=bool(
                           ((fields['caps'] == 0) | (answer.controls > 0)).all()))
        except BaseException as exc:
            row.update(state='failed', error=repr(exc))
            if callable(getattr(exc, 'record', None)):row['failure'] = exc.record()
        probes.append(row)
    x=values['response'].double().cpu();tau=values['temperature'].double().cpu()
    log_response=x/tau-1
    caps=values['caps'].double().cpu();controls=values['controls'].double().cpu()
    return dict(captured=captured, probes=probes,
        normalized_response_log=log_response.tolist(),
        exp_of_captured_response_fp64=torch.minimum(log_response.exp(),caps).tolist(),
        underflow_coordinates=((caps>0)&(controls==0)).nonzero().flatten().tolist(),
        score_range=float(values['scores'].max()-values['scores'].min()),
        fp32_log_smallest_subnormal=math.log(2.)*(-149),
        fp64_log_smallest_subnormal=math.log(2.)*(-1074),
        limitation='The FP64 row probe does not certify the full FP64 trajectory, VJP, or all future inputs.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('config','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text());start=time.monotonic()
    a.output.mkdir(parents=True,exist_ok=False)
    stage=dict(schema='tide-language-chain-v1',state='running',started_at=now(),
        route='a',steps=[dict(name='checkpoint-replay',action='diagnose',state='running',output=str(a.output))],
        shared_deadline_epoch=cfg['deadline_epoch'],config_sha256=digest(a.config),
        scope='terminal HB16 failure localization, no training continuation or successful model claim')
    atomic_json(a.output/'stage.json',stage)
    result=dict(state='running',original_execution='failed',source=source_identity(),
                original_failure=cfg['original_result'])
    step=None;tokens=None;model=None;optimizer=None
    try:
        manifest=json.loads(Path(cfg['original_manifest']).read_text());ident=manifest['identity']
        plan=manifest['config'];device=runtime('cpu',1,ident['seed'])
        if (ident['device'],ident['dtype'],ident['threads'])!=('cpu','float32',1):
            raise ValueError('replay contract requires original CPU1 FP32')
        if installed_identity()!=ident['installation']:raise ValueError('installed dependency identity changed')
        verified={}
        for row in ident['source']['content']['files']:
            if row['path'].startswith('tide_lab/') and row['path'].endswith('.py'):
                if digest(Path(__file__).resolve().parents[1]/row['path'])!=row['sha256']:
                    raise ValueError('model/training source changed: '+row['path'])
                verified[row['path']]=row['sha256']
        if digest(cfg['checkpoint'])!=cfg['checkpoint_sha256']:raise ValueError('checkpoint changed')
        if digest(Path(cfg['data'])/'data-lock.json')!=ident['data_lock_sha256']:raise ValueError('data lock changed')
        data,_=load_tokens(cfg['data'],'a',ident['length'])
        model=make_a(plan,ident['kind'],ident['seed'],ident['depth']).to(device)
        optimizer=optimizer_for(model,plan,'a',ident['kind'])
        step,tokens=restore(cfg['checkpoint'],model,optimizer,ident,device,'a')
        original={}
        for line in Path(cfg['original_metrics']).read_text().splitlines():
            item=json.loads(line)
            if 'train/loss' in item['metrics']:original[item['step']]=item['metrics']
        result.update(restored_step=step,checkpoint_sha256=cfg['checkpoint_sha256'],
            unchanged_model_training_files=verified,identity_note='New diagnostic harness; original application identity is validated, preserved, and used only for state restore',
            observed_step_checks=[])
        with (a.output/'replay-metrics.jsonl').open('x') as log:
            while step<cfg['max_step']:
                if time.monotonic()-start>cfg['compute_seconds']-30 or time.time()>cfg['deadline_epoch']-30:
                    raise TimeoutError('bounded replay deadline')
                model.encoder.observe=(step+1)%plan.get('region_observe_every',32)==0
                ids=indices(len(data['train']['labels']),ident['batch'],ident['seed'],step)
                batch=batch_from(data['train'],ids,device)
                row=update(model,optimizer,batch,plan,'a',ident['kind'],step,ident['schedule_steps'],device)
                step+=1;tokens+=row['valid_tokens']
                log.write(json.dumps(dict(step=step,**row))+'\n');log.flush()
                if step in original:
                    expected=original[step]
                    check=dict(step=step,loss_matches=row['loss']==expected['train/loss'],
                               gradient_norm_matches=row['gradient_norm']==expected['train/gradient_norm'])
                    result['observed_step_checks'].append(check)
                    if not all(check[k] for k in ['loss_matches','gradient_norm_matches']):
                        raise RuntimeError('replay differs from recorded trajectory: '+str(check))
                atomic_json(a.output/'progress.json',dict(step=step,target=cfg['max_step'],valid_tokens=tokens))
                print('replayed',step,flush=True)
        result.update(state='completed',reproduction='not_observed_within_bound')
    except BaseException as error:
        if callable(getattr(error,'record',None)):
            result.update(state='failed',error=repr(error),successful_updates=step,
                          failed_update=None if step is None else step+1)
            diagnosis=analyze_failure(error);atomic_json(a.output/'allocation-failure.json',diagnosis)
            if model is not None:
                tensor_save(a.output/'failure-boundary.pt',checkpoint_state(model,optimizer,step,tokens,ident,device,'a'))
            result.update(state='completed',reproduction='captured_allocation_failure',
                successful_updates=step,failed_update=step+1,error=repr(error),
                capture_sha256=digest(a.output/'allocation-failure.json'),
                failure_boundary_sha256=digest(a.output/'failure-boundary.pt'),
                axes=dict(execution='diagnostic captured expected failure',original_numerical='failed',
                          original_discrete_support='failed',learning='not adjudicated',
                          recovery='checked against available scalar observations, not a full independent long trajectory'))
        else:
            result.update(state='failed',error=repr(error),traceback=traceback.format_exc())
    finally:
        result.update(finished_at=now(),elapsed_seconds=time.monotonic()-start)
        atomic_json(a.output/'result.json',result)
        stage.update(state=result['state'],finished_at=result['finished_at'],allocated_elapsed_seconds=result['elapsed_seconds'])
        stage['steps'][0].update(state=result['state']);atomic_json(a.output/'stage.json',stage)
    return 0 if result['state']=='completed' else 1


if __name__=='__main__':raise SystemExit(main())

#!/usr/bin/env python3
"""Read-only weights/data, one CPU VJP per checkpoint; no optimizer updates."""
import argparse
import gc
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tide_lab.records import Metrics, atomic_json, digest, now, project_to_lens


def main(a):
    import torch
    from torch.nn import functional as F
    from tide_lab.train import runtime, installed_identity, source_identity
    from tide_lab.campaign_data import Corpus
    from tide_lab.campaign_runtime import make_model
    device=runtime('cpu',1,11)
    root=a.root.resolve(); a.output.mkdir(parents=True,exist_ok=False)
    corpus=Corpus(root/'runs/v6-data-final',128,allow_test=False)
    ids=corpus.evaluation_ids('validation',32)[:2]
    batch=corpus.batch('validation',ids,device)
    batch={k:v[:,:16].clone() for k,v in batch.items()}
    identity=dict(source=source_identity(),installation=installed_identity(),
                  data_lock_sha256=digest(root/'runs/v6-data-final/data-lock.json'))
    result=dict(schema='tide-hb-gradient-probe-v1',state='running',started_at=now(),
        identity=identity,cases=[],validation_blocks=[corpus.metadata('validation',i) for i in ids],
        scope='B2/L16 validation prefix, initial or existing 262144-token weights; no optimizer step or continuation. A diagnostic directional VJP, not full-scale training/validation or Jacobian spectral norm.',
        acceptance=dict(execution='running',strict_numerical='not independently requalified',
            discrete_semantics='original public path; original acceptance retained',recovery='weight loading only; no exact-resume claim',learning='original curves reviewed separately'))
    atomic_json(a.output/'result.json',result)
    names=('fc16-kv-quad-retry1','fc32-kv-quad-retry1','hb40-kv-quad-retry1','hb88-kv-quad-memory2')
    started=time.monotonic()
    try:
        for name in names:
            original=root/('runs/v6-scale-'+name)/'train'
            manifest=json.loads((original/'manifest.json').read_text())
            for state in ('initial','262k'):
                if time.monotonic()-started>840:raise TimeoutError('bounded probe deadline')
                tag=name.split('-kv-')[0]+'-'+state
                path=a.output/tag;path.mkdir();run_id='tl-hb-grad-'+tag+'-20261111'
                config=manifest['config'];model=make_model(config,device)
                ancestry=dict(manifest=str(original/'manifest.json'),sha256=digest(original/'manifest.json'),stage=state)
                if state=='262k':
                    checkpoint=original/'tokens-262144.pt';saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
                    assert saved['progress']['tokens']==262144
                    model.load_state_dict(saved['model'],strict=True)
                    ancestry.update(checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),progress=saved['progress'])
                    del saved
                atomic_json(path/'manifest.json',dict(schema='tide-lab-run-v1',project='tide-learning-lab',
                    run_id=run_id,name=run_id,study_id='TL-HB-LEARNING-20261111',group='TL-HB-LEARNING-20261111',
                    stage='gradient-diagnostic',created_at=now(),config=dict(model=config['id'],weights=state,batch=len(ids),length=16),identity=identity))
                metrics=Metrics(path,run_id);model.eval();model.zero_grad(set_to_none=True)
                begin=time.monotonic()
                features=model.embedding(batch['input_ids']);features.retain_grad()
                hidden=model.encoder(features);hidden.retain_grad()
                logits=model.head(hidden)
                loss=F.cross_entropy(logits.flatten(0,1),batch['labels'].flatten(),ignore_index=-100)
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite probe loss')
                loss.backward()
                grads=[];disconnected=[]
                for n,p in model.named_parameters():
                    if not p.requires_grad:continue
                    if p.grad is None:disconnected.append(n);continue
                    if not torch.isfinite(p.grad).all():raise FloatingPointError('nonfinite parameter VJP: '+n)
                    grads.append(dict(name=n,norm=float(p.grad.detach().double().norm()),
                        parameter_norm=float(p.detach().double().norm()),elements=p.numel()))
                grads.sort(key=lambda r:r['norm'],reverse=True)
                norm=math.sqrt(sum(g['norm']**2 for g in grads))
                fg=float(features.grad.double().norm()) if features.grad is not None else None
                hg=float(hidden.grad.double().norm()) if hidden.grad is not None else None
                values=dict(loss=float(loss.detach()),preclip_gradient_norm=norm,
                    input_rms=float(features.detach().square().mean().sqrt()),
                    hidden_rms=float(hidden.detach().square().mean().sqrt()),
                    logit_rms=float(logits.detach().square().mean().sqrt()),
                    max_abs_logit=float(logits.detach().abs().max()),
                    mean_top1_probability=float(logits.detach().softmax(-1).max(-1).values.mean()),
                    input_feature_gradient_norm=fg,hidden_gradient_norm=hg,
                    directional_gradient_gain=fg/hg if fg is not None and hg else None)
                row=dict(id=tag,state='completed',ancestry=ancestry,values=values,
                    top_parameter_gradients=grads[:20],disconnected_parameter_names=disconnected,
                    parameter_group_note='Names identify unique registration owners; shared gradients are accumulated and cannot be assigned causally to only that node/region.',
                    work=model.encoder.last_metrics,seconds=time.monotonic()-begin)
                metrics.log(0,{'probe/'+k:v for k,v in values.items() if v is not None});metrics.close()
                atomic_json(path/'result.json',row);atomic_json(path/'completed.json',dict(producer_finished=True,state='completed'))
                if a.lens_cli:project_to_lens(path,a.lens_cli)
                result['cases'].append(row);atomic_json(a.output/'result.json',result)
                print(json.dumps(dict(id=tag,**values,top=grads[:2])),flush=True)
                del model,features,hidden,logits,loss
                gc.collect()
        result['state']=result['acceptance']['execution']='completed'
    except BaseException as error:
        result.update(state='failed',error=repr(error));result['acceptance']['execution']='failed'
        raise
    finally:
        result.update(finished_at=now(),elapsed_seconds=time.monotonic()-started)
        atomic_json(a.output/'result.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lens-cli',default='/mi/data2T/zlong/home/.local/bin/trackio-lens')
    main(p.parse_args())

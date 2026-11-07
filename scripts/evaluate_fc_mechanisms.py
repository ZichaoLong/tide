#!/usr/bin/env python3
"""Fixed-checkpoint validation interventions; never train or alter the old runs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import Metrics, atomic_json, digest, now, project_to_lens


def tensor_hash(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True);p.add_argument('--project-root',required=True)
    p.add_argument('--output',required=True);p.add_argument('--recording-cli',required=True)
    a=p.parse_args();plan=json.loads(Path(a.plan).read_text())
    root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=False)
    if plan['delays'] != [2,4,8] or plan['samples']!=128 or len(plan['checkpoints'])!=3:
        raise ValueError('outside fixed second-batch evaluation')
    from tide_lab.train import runtime,make_model,installed_identity,source_identity
    from tide_lab.tasks import RetrievalTask
    from tide_lab.diagnostics import swap_query,extend_delay
    import torch
    device=runtime('cpu',1,11);started=time.monotonic();run_id='tl-a1-mech-eval'
    identity=installed_identity()
    atomic_json(root/'manifest.json',dict(project='tide-learning-lab',name=run_id,run_id=run_id,
        study_id=plan['study_id'],group=plan['study_id'],config=plan,created_at=now(),
        source=source_identity(),installation=identity,plan_sha256=digest(a.plan)))
    metrics=Metrics(root,run_id)
    report=dict(state='running',study_id=plan['study_id'],started_at=now(),runs=[],
        axes=dict(execution='running',strict_numerical='not checked; inference uses original CPU64 path',
        discrete_semantics='task interventions checked',recovery='checkpoint decoding only; not optimizer resume',
        learning='post-training validation interventions; no new learning'))
    atomic_json(root/'result.json',report)
    try:
        for relative, expected in plan['model_source_sha256'].items():
            if digest(Path(__file__).resolve().parents[1]/relative)!=expected:
                raise ValueError('checkpoint model/task implementation changed')
        for item in plan['checkpoints']:
            if time.monotonic()-started > plan['max_seconds']:
                raise TimeoutError('mechanism evaluation budget exhausted')
            directory=Path(a.project_root)/item['run']
            config=json.loads((directory/'config.json').read_text())
            checkpoint=directory/'checkpoint.pt'
            if digest(checkpoint)!=item['checkpoint_sha256'] or digest(directory/'config.json')!=item['config_sha256']:
                raise ValueError('checkpoint or training config changed')
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if saved['step']!=400 or saved['config_sha256']!=item['config_sha256'] or saved['identity']['installation']!=identity:
                raise ValueError('checkpoint step/config/installation mismatch')
            task=RetrievalTask(config['task']);model=make_model(config,task,None).to(dtype=torch.float64)
            model.load_state_dict(saved['model'],strict=True);model.eval()
            for module in model.modules():
                if hasattr(module,'observe'): module.observe=False
            x,y=task.batch('val',plan['samples'],0,plan['data_seed'],'cpu')
            indices,swapped,swap_y=swap_query(task,x)
            cases=[('original',x,y),('query_swapped',swapped,swap_y)]
            cases += [(f'delay_{delay}',extend_delay(task,x,delay),y) for delay in plan['delays'] if delay!=task.config.delay]
            result=dict(seed=config['seed'],checkpoint_sha256=item['checkpoint_sha256'],
                config_sha256=item['config_sha256'],checkpoint_source=saved['identity']['source'].get('commit'),
                cases={},query_swap_indices=indices.tolist())
            for name,inputs,target in cases:
                begin=time.monotonic();predictions=[];losses=[]
                with torch.no_grad():
                    for offset in range(0,len(inputs),plan['batch_size']):
                        if time.monotonic()-started > plan['max_seconds']:
                            raise TimeoutError('mechanism evaluation budget exhausted')
                        logits=model(inputs[offset:offset+plan['batch_size']])
                        if not torch.isfinite(logits).all(): raise FloatingPointError('nonfinite evaluation')
                        predictions += logits.argmax(-1).tolist()
                        losses += torch.nn.functional.cross_entropy(logits,target[offset:offset+plan['batch_size']],reduction='none').tolist()
                labels=target.tolist();count=len(labels)
                row=dict(samples=count,accuracy=sum(a==b for a,b in zip(predictions,labels))/count,
                    loss=sum(losses)/count,seconds=time.monotonic()-begin,inputs_sha256=tensor_hash(inputs),
                    targets=labels,predictions=predictions,per_example_loss=losses)
                result['cases'][name]=row
                metrics.log(len(report['runs'])*4+len(result['cases']),
                    {f'seed{config["seed"]}/{name}/accuracy':row['accuracy'],
                     f'seed{config["seed"]}/{name}/loss':row['loss'],
                     f'seed{config["seed"]}/{name}/seconds':row['seconds']})
            old=[result['cases']['original']['predictions'][i] for i in indices.tolist()]
            new=result['cases']['query_swapped']['predictions'];old_y=y[indices].tolist()
            result['paired_query']=dict(samples=len(new),
                prediction_change_fraction=sum(a!=b for a,b in zip(old,new))/len(new),
                retains_old_answer_fraction=sum(a==b for a,b in zip(new,old_y))/len(new),
                original_accuracy_on_eligible=sum(a==b for a,b in zip(old,old_y))/len(new))
            report['runs'].append(result);atomic_json(root/'result.json',report)
            del model,saved
        report['state']='completed';report['axes']['execution']='completed'
    except BaseException as error:
        report.update(state='failed',error=repr(error));report['axes']['execution']='failed';raise
    finally:
        report.update(finished_at=now(),elapsed_seconds=time.monotonic()-started)
        atomic_json(root/'result.json',report);metrics.close()
        atomic_json(root/'completed.json',dict(producer_finished=True,state=report['state']))
    project_to_lens(root,a.recording_cli)
    print(json.dumps(dict(state=report['state'],runs=len(report['runs']))))


if __name__=='__main__':main()

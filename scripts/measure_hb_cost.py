#!/usr/bin/env python3
"""Finite complete-request/update costs: installed TIDE versus non-TIDE DAGs."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tide_lab.records import Metrics, atomic_json, digest, now, project_to_lens


def main(a):
    import torch
    from torch.nn import functional as F
    from tide_lab.train import runtime, synchronize, source_identity, installed_identity
    from tide_lab.campaign_runtime import make_model, optimizer_for, synthetic_batch, memory
    from tide_lab.direct_graph_cost import DirectGraphLM

    plan = json.loads(a.plan.read_text()); suite = plan['suites'][a.suite]
    a.output.mkdir(parents=True, exist_ok=False)
    device = runtime(suite['device'], suite['threads'], 11)
    identity = dict(source=source_identity(), installation=installed_identity(),
                    plan_sha256=digest(a.plan), physical_visibility=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'))
    report = dict(schema='tide-hb-cost-suite-v1', state='running', suite=a.suite,
                  started_at=now(), identity=identity, cases=[])
    atomic_json(a.output/'result.json', report)
    began = time.monotonic()
    try:
        for entry in suite['entries']:
            if time.monotonic()-began > suite['timeout_seconds']:
                raise TimeoutError('finite suite budget exhausted')
            case = copy.deepcopy(entry)
            directory = a.output/case['id']; directory.mkdir()
            run_id = 'tl-hb-cost-'+a.suite+'-'+case['id']
            result = dict(schema='tide-hb-cost-case-v1', state='running', run_id=run_id,
                case=case, observations=[], acceptance=dict(execution='running',
                strict_numerical='not_a_same_function_TIDE_comparison',
                discrete_semantics='direct dense DAG differs from TIDE routing',
                recovery='not_checked_here; no training continuation produced',
                learning='not_measured; repeated reset synthetic cost windows'),
                timer_scope=plan['timer_scope'], comparison_boundary=plan['comparison_boundary'])
            atomic_json(directory/'result.json', result)
            atomic_json(directory/'manifest.json', dict(schema='tide-lab-run-v1',
                project='tide-learning-lab', run_id=run_id, name=run_id,
                study_id='TL-HB-COST-20261111', group='TL-HB-COST-20261111',
                stage='cost-diagnostic', created_at=now(), config=dict(case, suite=a.suite,
                device=suite['device'], threads=suite['threads']), identity=identity))
            metrics = Metrics(directory, run_id)
            construction = time.monotonic()
            cfg = copy.deepcopy(plan['models'][case['topology']]['config'])
            cfg.update(device=suite['device'], threads=suite['threads'])
            if case['kind'] == 'tide':
                model = make_model(cfg, device)
            elif case['kind'] == 'transformer':
                model = make_model(cfg, device)
            else:
                model = DirectGraphLM(plan['models'][case['topology']]['graph'],
                                      width=128, attention=case['kind']=='direct-attention').to(device)
            model.train()
            initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            opt = optimizer_for(model, cfg)
            initial_opt = copy.deepcopy(opt.state_dict())
            batch = synthetic_batch(case['batch'], case['length'], 991, device)
            synchronize(device)
            result.update(construction_seconds=time.monotonic()-construction,
                parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                independent_batch=case['batch'], length=case['length'], accumulation=1,
                input_tokens=case['batch']*case['length'], input_seed=991,
                implementation='public native; complete returns' if case['kind']=='tide' else 'ordinary Torch, manual causal attention',
                cpu_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                affinity=sorted(os.sched_getaffinity(0)), memory_before=memory(device))
            if case['kind']=='tide':
                result['facts']=model.encoder.facts()
            try:
                for mode in case.get('modes', ['no_grad', 'grad_forward', 'train_step']):
                    for rep in range(plan['repetitions']+1):
                        reset_start=time.monotonic()
                        model.load_state_dict(initial, strict=True)
                        opt.load_state_dict(copy.deepcopy(initial_opt))
                        opt.zero_grad(set_to_none=True)
                        gc.collect()
                        if device.type=='npu': torch.npu.reset_peak_memory_stats()
                        synchronize(device)
                        reset_seconds=time.monotonic()-reset_start
                        start=time.monotonic()
                        opt.zero_grad(set_to_none=True)
                        with torch.set_grad_enabled(mode!='no_grad'):
                            logits=model(batch['input_ids'])
                            loss=F.cross_entropy(logits.flatten(0, 1), batch['labels'].flatten())
                            if not torch.isfinite(loss):raise FloatingPointError('nonfinite diagnostic loss')
                            norm=None
                            if mode=='train_step':
                                loss.backward()
                                norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                    cfg['training']['clip_norm'], error_if_nonfinite=True)
                                opt.step()
                            scalar=float(loss.detach())
                            gradient_norm=float(norm) if norm is not None else None
                            del loss, logits, norm
                        synchronize(device)
                        seconds=time.monotonic()-start
                        row=dict(mode=mode, repetition=rep, warmup=rep==0,
                            seconds=seconds, input_positions_per_second=result['input_tokens']/seconds,
                            loss=scalar, gradient_norm=gradient_norm, reset_seconds=reset_seconds,
                            work=dict(model.last_metrics), memory=memory(device))
                        result['observations'].append(row)
                        atomic_json(directory/'result.json', result)
                        metrics.log(len(result['observations']), {'timing/'+mode+'_seconds':seconds,
                            'throughput/'+mode+'_input_positions_per_second':row['input_positions_per_second'],
                            'diagnostic/loss':scalar, 'diagnostic/warmup':rep==0,
                            **({'diagnostic/gradient_norm':gradient_norm} if gradient_norm is not None else {})})
                        print(json.dumps(dict(case=case['id'], mode=mode, warmup=rep==0, seconds=seconds)), flush=True)
                result['summary']={mode: dict(median_seconds=statistics.median(
                    r['seconds'] for r in result['observations'] if r['mode']==mode and not r['warmup']),
                    repetitions=plan['repetitions']) for mode in {r['mode'] for r in result['observations']}}
                result['state']=result['acceptance']['execution']='completed'
            except BaseException as error:
                result.update(state='failed', error=repr(error));result['acceptance']['execution']='failed'
                raise
            finally:
                metrics.close();result['finished_at']=now()
                atomic_json(directory/'result.json', result)
                atomic_json(directory/'completed.json', dict(producer_finished=True, state=result['state']))
                if a.lens_cli:project_to_lens(directory, a.lens_cli)
                report['cases'].append(dict(id=case['id'], state=result['state'], path=str(directory),
                    summary=result.get('summary'), parameters=result['trainable_parameters']))
                atomic_json(a.output/'result.json', report)
            del model, opt, initial, initial_opt, batch
            gc.collect()
        report['state']='completed'
    except BaseException as error:
        report.update(state='failed', error=repr(error))
        raise
    finally:
        report.update(finished_at=now(), elapsed_seconds=time.monotonic()-began)
        atomic_json(a.output/'result.json', report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--suite',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lens-cli',default='trackio-lens')
    main(p.parse_args())

#!/usr/bin/env python3
"""Finite task-projection check for FC cost comparisons, not all-root qualification.

Each candidate consumes the same parameters/input and computes its own schedule.
Keep numerical, support, int64 state and None/present gradient differences separate.
No reference outputs/routes feed a candidate. No timers here are performance data.
"""
import argparse
import functools
import json
from pathlib import Path
import time
from durable_records import write_json
from benchmark_dense_query import task_source, geometry, DOWNSTREAM
from fc_comparison_diagnostics import boundary_scope


def difference(actual, expected, *, atol, rtol):
    import torch
    result = dict(tensors=0, strict_mismatches=[], none_mismatches=[], max_abs=0., nonfinite=[])
    if set(actual) != set(expected):
        raise ValueError("comparison field domains differ")
    for name in actual:
        a, e = actual[name], expected[name]
        if a is None or e is None:
            if a is not e:
                value = e if a is None else a
                result['none_mismatches'].append(dict(field=name, actual_none=a is None,
                    present_max_abs=float(value.abs().max()), numerical_zero=not bool(torch.count_nonzero(value))))
            continue
        result['tensors'] += 1
        if a.shape != e.shape or a.dtype != e.dtype:
            raise ValueError(f"comparison metadata mismatch: {name}")
        if not torch.isfinite(a).all() or not torch.isfinite(e).all():
            result['nonfinite'].append(name)
            continue
        floating = a.is_floating_point()
        error = (a-e).abs() if a.dtype != torch.bool else a != e
        largest = float(error.max()) if error.numel() else 0.
        result['max_abs'] = max(result['max_abs'], largest)
        close = error <= atol+rtol*e.abs() if floating else error == 0
        if not bool(close.all()):
            result['strict_mismatches'].append(dict(field=name, kind='numeric' if floating else 'discrete',
                max_abs=largest, reference_max_abs=float(e.abs().max()) if floating else None,
                coordinates=int((~close).sum()), elements=a.numel()))
    result['strict_passed'] = not any(result[k] for k in ('strict_mismatches', 'none_mismatches', 'nonfinite'))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--downstream-repo',type=Path,required=True)
    p.add_argument('--native-library')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','npu:0'),default='cpu')
    p.add_argument('--dtype',choices=('float32','float64'),default='float32')
    p.add_argument('--fc-shape',choices=('fc4','fc16'),default='fc16')
    p.add_argument('--batch',type=int,default=2)
    p.add_argument('--length',type=int,default=8)
    p.add_argument('--trace',action='store_true',help='also compare each candidate control/support; separate complete-return path')
    p.add_argument('--variants',nargs='+',choices=('native','python','query-direct','region-direct','both-direct','formula-sdpa'),
                   default=['native','python','query-direct','region-direct','both-direct','formula-sdpa'])
    a=p.parse_args()
    if 'native' in a.variants and not a.native_library:p.error('native needs its explicit matching build')
    if not 1<=a.batch<=8 or not 1<=a.length<=128:p.error('bounded B<=8/L<=128 required')
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    config,sha=task_source(a.downstream_repo,out/'downstream-source');geometry(config,a.fc_shape)
    import torch
    from tide_lab.campaign_runtime import make_model, synthetic_batch
    from tide_lab.language import sequence_readout
    from tide_lab.train import runtime
    import tide_lab.core5 as application
    from tidegraph.weighted.execution import eager_runner
    from fc_tensor_baseline import run
    application.eager_runner=functools.partial(eager_runner,query_execution='auto',prefill_schedule='auto',trace=a.trace)
    config.update(device=a.device,threads=1)
    config['core'].update(eager_implementation='python',dtype=a.dtype,
                         placement_preset='cpu' if a.device=='cpu' else 'mixed-c',native_library=a.native_library)
    config['training'].update(batch=a.batch,length=a.length)
    device=runtime(a.device,1,config['seed']);model=make_model(config,device).to(dtype=getattr(torch,a.dtype))
    inputs=synthetic_batch(a.batch,a.length,992,device)
    atol,rtol=(1e-10,1e-8) if a.dtype=='float64' else (1e-6,1e-5)
    record=dict(schema='tide-fc-projection-comparison-v1',execution='running',downstream=DOWNSTREAM,
                downstream_archive_sha256=sha,request=vars(a)|{'output_dir':str(out),'downstream_repo':str(a.downstream_repo)},
                scope='Same task loss/output/raw terminal state and numeric parameter/input derivatives. No complete-record/all-root or optimizer/restore claim.',
                tolerance=dict(atol=atol,rtol=rtol),variants={},comparisons={})
    write_json(out/'result.json',record)
    def evaluate(variant):
        model.zero_grad(set_to_none=True)
        features=model.embedding(inputs['input_ids']);features.retain_grad()
        boundary='public' if variant in ('native','python','formula','formula-sdpa') else variant
        with boundary_scope(boundary):
            if variant in ('formula','formula-sdpa'):
                result=run(model.encoder.binding,model.encoder.spec,features,
                           full_attention='sdpa' if variant=='formula-sdpa' else 'manual')
                hidden=result['hidden'];raw=result['states'];work=result['stats']
            else:
                model.encoder.eager_implementation='native' if variant=='native' else 'python'
                result=model.encoder.run_features(features,trace=a.trace)
                hidden=sequence_readout(model.encoder,result,features);work=dict(result.stats)
                raw={v:{name: value if name=='times' else torch.stack([result.continuation.states[b,v].slots[name]
                        for b in range(a.batch)]) for name,value in result.continuation.states[0,v].slots.items()}
                     for v in range(len(model.encoder.binding.graph.nodes))}
            logits=model.head(hidden)
            loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),inputs['labels'].flatten())
            loss.backward()
        def cpu(t):return None if t is None else t.detach().cpu().clone()
        values={'hidden':cpu(hidden),'logits':cpu(logits),'loss':cpu(loss)}
        values.update({f'state/{v}/{name}':cpu(t) for v,row in raw.items() for name,t in row.items()})
        if a.trace:
            spec=model.encoder.spec;g=model.encoder.binding.graph
            if variant in ('formula','formula-sdpa'):
                for r,region in enumerate(result['regions']):
                    for local in range(region['controls'].shape[-1]):
                        v=r*region['controls'].shape[-1]+local
                        for b in range(a.batch):
                            for pos in range(a.length):
                                values[f'control/{b}/{v}/{pos}']=cpu(region['controls'][b,pos,local])
                                values[f'active/{b}/{v}/{pos}']=cpu(region['active'][b,pos,local])
            else:
                for event in result.trace:
                    b,v=event['batch'],event['node'];pos=(event['time']-spec.rank(v))//spec.stride
                    values[f'control/{b}/{v}/{pos}']=cpu(event['control'])
                    values[f'active/{b}/{v}/{pos}']=torch.tensor(event['active'],dtype=torch.bool)
        gradients={'features':cpu(features.grad)}|{name:cpu(p.grad) for name,p in model.named_parameters() if p.requires_grad}
        return values,gradients,work
    try:
        expected,expected_grad,work=evaluate('formula');record['variants']['formula']=dict(work=work)
        public=None
        for variant in a.variants:
            begin=time.monotonic();values,gradients,work=evaluate(variant)
            record['variants'][variant]=dict(work=work,diagnostic_seconds=time.monotonic()-begin)
            record['comparisons'][variant+'-formula']=dict(values=difference(values,expected,atol=atol,rtol=rtol),
                gradients=difference(gradients,expected_grad,atol=atol,rtol=rtol))
            if variant=='python':public=(values,gradients)
            elif public is not None:
                record['comparisons'][variant+'-python']=dict(values=difference(values,public[0],atol=atol,rtol=rtol),
                    gradients=difference(gradients,public[1],atol=atol,rtol=rtol))
            write_json(out/'result.json',record)
            if any(section['nonfinite'] for comparison in record['comparisons'].values() for section in comparison.values()):
                raise FloatingPointError('nonfinite comparison result')
        record['execution']='completed'
    except BaseException as error:
        record.update(execution='failed',error=repr(error));raise
    finally:write_json(out/'result.json',record)
    print(json.dumps({key:{part:dict(strict_passed=v['strict_passed'],max_abs=v['max_abs'],
        mismatches=len(v['strict_mismatches']),none=len(v['none_mismatches'])) for part,v in value.items()}
        for key,value in record['comparisons'].items()}))


if __name__=='__main__':main()

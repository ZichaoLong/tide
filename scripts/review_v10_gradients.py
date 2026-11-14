#!/usr/bin/env python3
"""Read-only audit of the three bounded v10 gradient replays and candidates."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.review_io import write_review_json
from tide_lab.records import now

ROOT=Path(__file__).resolve().parents[1]
CASES=('hb-d128-r64-pool-s11','hb-d128-r64-pool-s22','hb-d128-r28-pool-s22')


def read(path):return json.loads(path.read_text()) if path.exists() else {}


def review_case(case):
    original=ROOT/'runs'/('v10-main-'+case)/'train'
    replay=ROOT/'runs'/('v10-diag-'+case);checks=ROOT/'runs'/('v10-fixcheck-'+case)
    failed=read(original/'result.json');progress=read(original/'progress.json')
    r=read(replay/'result.json');g=read(replay/'gradients.json');incident=read(replay/'operator-incident.json')
    full=read(checks/'full-update/result.json');common=r.get('common_step_comparisons',[])
    node={k:incident.get(k) for k in ('node','microbatch','upstream_all_finite','grad_inputs','grad_outputs','saved')}
    node['forward_trace']=[x.replace(str(ROOT)+'/', '<repo>/') for x in incident.get('forward_trace',[])
                           if '/tide_lab/' in x]
    references={}
    for backend in ('cpu','npu'):
        p=checks/('division-'+backend)/'result.json';x=read(p)
        references[backend]=dict(state=x.get('state','unchecked'),record=str(p.relative_to(ROOT)),
            broadcast_fp64_gradcheck=x.get('broadcast_fp64_gradcheck'),witness=x.get('witness'))
    variants={}
    for name,v in full.get('variants',{}).items():
        variants[name]={k:v.get(k) for k in ('loss','gradient_acceptance','gradient_norm','support_errors','floor_events','optimizer')}
        variants[name]['nonfinite_parameter_count']=len(v.get('nonfinite_parameters',[]))
    return dict(id=case,original_run='tl-v10-main-'+case,original_state=failed.get('state','unchecked'),
        original_last_step=progress.get('step'),original_last_tokens=progress.get('tokens'),
        original_error=failed.get('error'),original_source=failed.get('identity',{}).get('source_commit'),
        replay=dict(state=r.get('state','unchecked'),record=str((replay/'result.json').relative_to(ROOT)),
            training_source=r.get('training_source'),diagnostic_source=r.get('diagnostic_source'),
            checkpoint_sha256=r.get('checkpoint_sha256'),checkpoint_progress=r.get('checkpoint_progress'),
            failure_step=r.get('failure_step'),failure_progress=r.get('failure_progress'),seconds=r.get('seconds'),
            physical_visibility=r.get('physical_visibility'),common_steps=len(common),
            max_abs_loss_delta=max((abs(x['loss_delta']) for x in common),default=None),
            max_abs_gradient_norm_delta=max((abs(x['gradient_norm_delta']) for x in common),default=None),
            all_common_tokens_equal=all(x['tokens_equal'] for x in common) if common else None),
        gradients=dict(all_parameter_gradients_finite=g.get('all_parameter_gradients_finite'),
            nonfinite_parameters=g.get('nonfinite_parameters'),
            nonfinite_elements=sum(x['nan']+x['inf'] for x in g.get('parameters',{}).values()) if g else None,
            finite_elements_norm_fp64=g.get('finite_elements_norm_fp64')),incident=node,
        local_references=references,
        full_batch_candidate=dict(state=full.get('state','unchecked'),record=str((checks/'full-update/result.json').relative_to(ROOT)),
            source_commit=full.get('diagnostic_source'),checkpoint_sha256=full.get('checkpoint_sha256'),
            batches_sha256=full.get('batches_sha256'),
            forward_bitwise_equal_all_microbatches=full.get('forward_bitwise_equal_all_microbatches'),variants=variants),
        services={name:dict(state=(s:=read(ROOT/'artifacts/jobs'/name/'state.json')).get('state','unchecked'),
                           termination_verified_at_finalization=s.get('termination_verified'),
                           elapsed_seconds=s.get('elapsed_seconds'))
                  for name in ('v10-diag-'+case,'v10-fixcheck-'+case)})


def review():
    quick=read(ROOT/'runs/v10-diag-quick-r28-s22/result.json')
    scalar={}
    for backend in ('cpu','npu'):
        path=ROOT/'runs'/('v10-divprobe-'+backend+'-scalar')/'result.json';v=read(path)
        scalar[backend]={k:v.get(k) for k in ('state','device','cases','broadcast_fp64_gradcheck','seconds')}
        scalar[backend]['record']=str(path.relative_to(ROOT))
    return dict(schema='tide-v10-gradient-observations-v1',observed_at=now(),cases=[review_case(c) for c in CASES],
        scalar_probes=scalar,old_weights_failed_batch={k:quick.get(k) for k in ('state','mode','config_id','seconds','checkpoint_progress')},
        scope='Observed records only. The separately authored report reviews attribution; no automatic scientific support.',
        distinctions=dict(execution='Expected failure reproduction is completed diagnosis, not successful original training.',
            strict_numerical='Local reference checks retain 1e-6/1e-5; old full-model strict failures retained.',
            discrete_semantics='See unchanged support/floor observations; no general continuation claim.',
            recovery='Common logged values compared, not all historic parameter bytes. Candidate fresh-process resume unchecked.',
            learning='No new training trajectory or convergence result; main jobs and sealed test untouched.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);a=p.parse_args()
    value=review()
    if a.output:write_review_json(a.output,value)
    else:print(json.dumps(value,ensure_ascii=False,indent=2))

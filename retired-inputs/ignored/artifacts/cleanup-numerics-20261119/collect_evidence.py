"""Reviewable compact evidence; raw failed observations remain unchanged."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from job import atomic

identities = []
def read(name):
    path = ROOT/name
    data = path.read_bytes()
    identities.append(dict(path=name,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
    return json.loads(data)
def compact(check):
    c = check.get('comparison',check)
    return {k:v for k,v in c.items() if k not in ('largest_errors','screen_actual','screen_expected')}
def select(d,*keys):
    return {k:d[k] for k in keys if k in d}
def resource_update(d):
    return dict(checked=d['checked'],parameter_count=len(d['active_original_parameters']),
                before=sorted(set(d['before'].values())),after=sorted(set(d['after'].values())))

result = dict(schema='tide-cleanup-numerics-review-v1',state='completed_bounded_plan',
    created_utc=datetime.now(timezone.utc).isoformat(),base_commit='c5bfebaf219f4910965a3d5da0db483b686b25b2',
    fixed_math='4f5b9f20a4030d68ee66ae2d26e34659a5e37234',math_files=108,
    implementation_commit='97c52960358e6d5414fc987dc3861b1e85d997a9',
    implementation_tree=subprocess.check_output(['git','rev-parse','97c52960:src/tide'],cwd=ROOT,text=True).strip(),
    thresholds_unchanged=True, default_candidates_unchanged=True)
first = read('artifacts/cleanup-numerics-20261119/cleanup-execution-result.json')
updates = read('artifacts/cleanup-numerics-20261119/update2-retirement-result.json')
finish = read('artifacts/cleanup-numerics-20261119/finish-retirement-result.json')
plan = read('artifacts/cleanup-numerics-20261119/finish-plan.json')
groups = {}
for row in plan['files']:
    r=groups.setdefault(row['reason'],dict(files=0,allocated_bytes=0));r['files']+=1;r['allocated_bytes']+=row['guard']['allocated_bytes']
result['cleanup']=dict(old_payload_and_profiler=first,old_update2=updates,new_artifact_retirement=finish,
    new_artifact_groups=groups,verified_sources=plan['sources'],
    old_allocated_bytes_freed=43_633_938_432,
    preserved_old_payloads=dict(pt_count=290,f32_count=6734,allocated_gib=45.604084,
        grouping_gib={'ffn_medium':17.868,'memory_qualification':7.920,'training_paths_placement':7.759,
                      'execution_candidates':3.884,'reception':3.846,'native_f32':1.777,'others':2.551}),
    not_pending_jobs='retained failures/reproducers of completed finite experiments; not an instruction to restart matrices',
    preserve_new='FFN actual X/W/dY, original allocator full-shape witness, BF16 witness, training anchor and both checkpoints; all small results/identity/log/terminal records',
    prior_rounds_allocated_bytes_freed=[122_335_051_776,13_940_666_368])
saved=read('artifacts/numerics-saved-update-v1/diagnosis/result.json')
witness=read('artifacts/numerics-saved-update-v1/diagnosis/witness.json')
coordinates=[r for phase in witness['phases'] if phase['phase']=='update-2' for r in phase['adamw']
             if (r['parameter'],r['flat_index']) in [('regional.4.second.bias',47),('regional.26.second.bias',58)]]
result['saved_update']=dict(**select(saved,'state','original_state','original_source','tolerance','input_files',
    'all_original_threshold_field_sets_match','all_original_failure_field_sets_match'),
    original_failure_counts=[0,392,533],selected_adjacent_update_coordinates=coordinates,
    boundary='selected scalar equations explain optimizer amplification, not all upstream gradients or 533 failed fields; contribution order is not unique causality')
result['update2_replays']=[]
for name in ('update2-baseline','update2-weight','retired-update-baseline','retired-update-weight'):
    d=read(f'artifacts/numerics-{name}-v1/replay.json')
    result['update2_replays'].append(dict(**select(d,'state','owner','source','expected_observation','comparison_mode',
        'input_files','positions','rng_unchanged','rng_policy','retirement_policy'),
        comparison=compact(d['comparison']),optimizer_update=resource_update(d['optimizer_update'])))
ffn=read('artifacts/numerics-ffn-vjp-v1/diagnosis.json')
result['ffn_vjp']=select(ffn,'state','source','scope','tolerance','cpu_affinity','threads','capture',
    'original_selected_scope_exact','historical_selected_gradient_W_vs_baseline','local_against_fp64_formula',
    'local_grouped_vs_gather','retained_call','local_parameter_scope','replay_limit')
result['shared9b']=[]
for name in ('numerics-shared-routing-v1','numerics-quad-shared9b-npu-v1'):
    d=read(f'artifacts/{name}/routing.json'); c=d['comparison']
    grouped=Counter()
    for stage in c['stages']:
        grouped[stage['stage']]+=stage['difference']['threshold_failures']
    result['shared9b'].append(dict(run=name,**select(d,'state','outcome','source','scope','arguments',
        'parameters','input_sha256','thresholds','historical_witness_reproduced','allocation_witnesses'),
        comparison=select(c,'metadata_failures','support_differences','nonfinite_elements',
                          'threshold_failure_elements','first_changed_stage','witnesses'),
        threshold_failure_elements_by_stage=dict(grouped),paths=[dict(region_batch_size=p['region_batch_size'],
            routing=select(p['routing'],'region_calls','full','candidates','scalar_sha256'),output=p['output'],
            historical_output_identity=p.get('historical_output_identity')) for p in d['paths']]))
result['quad_face_npu']=read('artifacts/numerics-quad-face-npu-v1/qualification.json')
train=read('artifacts/numerics-quad-training-npu-v1/qualification/result.json')
r=train['result']
result['postfix_training']=dict(**select(train,'source','arguments','precision'),
    **select(r,'state','tolerance','restore_tolerance','resolved_candidates','boundary','anchor_optimizer'),candidates=[])
for c in r['candidates']:
    restore=c['restore']
    result['postfix_training']['candidates'].append(dict(name=c['name'],state=c['state'],config=c['config'],
        checks={k:compact(v) for k,v in c['checks'].items()},
        updates=[dict(index=u['index'],vs_baseline=compact(u['vs_baseline']) if 'vs_baseline' in u else None)
                 for u in c['updates']],
        restore=dict(**select(restore,'state','source','source_exact','config_exact','actual_update','pid','bundle_sha256'),
            checks={k:compact(v) for k,v in restore['checks'].items()},
            optimizer_update=resource_update(restore['optimizer_update']))))
bf=read('artifacts/numerics-bf16-training-routing-v1/routing.json')
result['bf16']=dict(**select(bf,'state','source','original_source','qualification','anchor','arguments','physical_devices',
    'thresholds','boundary','limitations','input_and_anchor_bytes_match_original','original_observations_reproduced','witness'),
    interpretation='third update own rounded scores cross true boundary; not the shared9B solver error; upstream full cause remains unlocalized')
result['cpu_suite_xml']=dict(ET.parse(ROOT/'artifacts/numerics-quad-cpu-suite-v1/cpu.xml').getroot().find('testsuite').attrib)
result['installed_smoke']=read('artifacts/cleanup-numerics-20261119/installed-smoke.json')
result['lifecycle']=read('artifacts/cleanup-numerics-20261119/final-terminal-audit.json')
result['smoke_retirement']=read('artifacts/cleanup-numerics-20261119/smoke-retirement.json')
result['final_storage_and_preservation']=read('artifacts/cleanup-numerics-20261119/final-inventory.json')
result['resource_scope']=dict(cpu='aarch64 Kunpeng shared host; original update/VJP replay pinned 32 threads on CPUs240–271; suite CPUs120–127; installed CPU1',
    accelerator='one cooperative-queue admitted idle Ascend910 64GiB at a time, observed physical11; no shared occupied NPU',
    software='ascend/dev-workspace-9.2.0; Torch2.12.0+cpu plus TorchNPU2.12/CANN9.2; project private overlay',
    production_arithmetic='FP32 first; old BF16 bounded witness; Decimal/CPU FP64 only local diagnosis',
    budget='CPU service4h, NPU service2 card-h, RSS128GiB, new artifacts4GiB; no9B training or performance matrix',
    new_artifacts_observed_pre_retirement_gib=result['final_storage_and_preservation']['prefinish_inventory']['new_artifacts_gib'])
result['limits']=['old numerical failures retained; fixed local solver defect does not certify all FP32 boundaries',
    'small post-fix cross-path updates retain1/1/8 failed fields; two local restored updates pass original tolerance',
    'post-fix shared9B support/time match, but298 internal scalar element comparisons fail original tolerance',
    'BF16 upstream score/gradient history cause not fully localized',
    'new production source has no new throughput qualification; old timing retains old identity',
    'no9B training, new multi-device/CUDA/compiled qualification or downstream learning-quality claims']
result['raw_record_identities']=identities
atomic(ROOT/'docs/evidence/cleanup-numerics-20261119.json',result)
print(json.dumps(dict(state='written',bytes=(ROOT/'docs/evidence/cleanup-numerics-20261119.json').stat().st_size,
    raw_records=len(identities))))

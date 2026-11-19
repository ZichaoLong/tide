import json
from pathlib import Path
from scripts.job import atomic

data=json.loads(Path('artifacts/agg-inference-preflight-20261118/training-summary.json').read_text())
names={'fc':'npu-fc-k-fp32','settle':'npu-hb-settle-fp32','timed':'npu-hb-timed-fp32',
       'half':'npu-hb-0p5b-fp32','light64':'npu-hb-light64-fp32'}
for row in data['continuous']:
    stem=Path(row['record']['path']).parts[1]
    tag=stem.split('agg-train-')[1].split('-fp32')[0]
    restored=next((r for r in data['restore'] if f'agg-restore-{tag}-fp32-' in r['record']['path'] and r['state']=='completed_finite'),None)
    if restored is None:
        continue
    a=row['arguments'];c=row['checkpoint'];r=restored
    description=f"FP32 {tag} training choice from 100 continuing updates; explicit device admission and downstream memory recheck required."
    value=dict(schema='tide-example-preset-1',description=description,
        model=dict(row['spec'], parameter_indexing=row['spec'].get('parameter_indexing', 'gather')),
        execution=dict(mode='train',device=None,precision='fp32',batch=a['batch'],length=a['prefill'],steps=100,
                       threads=4,learning_rate=1e-4,foreach=a['optimizer_foreach'],seed=751),
        recommendation=dict(status='selected_training_shape',scope=description,
            evidence=['docs/evidence/reception-training-20261118.md','docs/evidence/reception-numerics-20261118.md',
                      'docs/evidence/reception-performance-20261118.md','docs/evidence/reception-validation-20261118.md'],
            device='Explicit admitted NPU required; null device has no runtime default.',
            hardware=dict(device_capacity_gib=64,sampled_budget_gib=48,budget_is_hard_isolation=False,
                          sampled_budget_scope='allocated only; reserved memory and non-Torch allocations are separate',
                          resource_identity=row['resource_identity']),
            measurement=dict(source=row['source'],record=row['record']['path'],record_sha256=row['record']['sha256'],
                             parameters=row['parameters'],updates=row['updates'],steady_rule=row['steady_rule'],
                             steady_input_positions_per_second=row['steady_positions_per_second'],
                             peak_allocated_gib=row['peak_allocated_bytes']/2**30,
                             peak_reserved_gib=row['peak_reserved_bytes']/2**30,timing=row['timing']),
            qualification=dict(paired_restore=c['state'],saved_data_exact=c['saved_data']['exact'],
                               next_forward_exact=c['next_forward']['exact'],
                               cross_path='finite deviations retained; three incoming profiles and placed paths have separate evidence',
                               full_numerical_equivalence=False),
            fresh_process_restore=dict(state=r['state'],record=r['record']['path'],record_sha256=r['record']['sha256'],
                                       source=r['source'],checkpoint_producer_source=r['checkpoint_source'],
                                       checkpoint_sha256=r['checkpoint_file']['sha256'],
                                       optimizer_step_before=100,optimizer_step_after=101,
                                       optimizer_step_increment_errors=r['optimizer_step_increment_errors'],
                                       load_complete_seconds=r['load_complete_seconds'],
                                       next_update_complete_seconds=r['next_update_complete_seconds'],
                                       numerical_equivalence_qualification=False),
            entry_boundary='Preset carries model and runtime choices. The example has its own synthetic sequence, loss normalization and diagnostic checks; use the recorded benchmark command to reproduce timings.'))
    assert row['updates']==100 and row['all_finite'] and row['all_detached']
    assert r['optimizer_steps_before']==[100.0] and r['optimizer_steps_after']==[101.0] and not r['optimizer_step_increment_errors']
    atomic(Path('configs/recommended')/(names[tag]+'.json'),value)
    print(names[tag])

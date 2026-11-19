from pathlib import Path
import hashlib
import json
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.job import atomic

def read(path):
    return json.loads((ROOT/path).read_text())

def pointer(path):
    p = ROOT/path
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return dict(path=str(p.relative_to(ROOT)), sha256=h.hexdigest(), bytes=p.stat().st_size)

def run(name, result):
    base = Path('artifacts')/name
    d = read(base/result)
    status = read(base/'status.json')
    assert status['state'] in ('passed', 'failed'), (name, status['state'])
    return dict(name=name, source=d.get('source', status['source']), state=d.get('state'),
        result=pointer(base/result), launch=pointer(base/'launch.json'),
        status=pointer(base/'status.json'), queue=pointer(base/'queue.json') if (ROOT/base/'queue.json').exists() else None,
        command=status['command'], outer_state=status['state'], exit_code=status.get('exit_code'),
        physical_devices=d.get('physical_devices'), resource_identity=d.get('resource_identity'),
        sampled_peak_rss_bytes=status.get('peak_rss_bytes'), elapsed_seconds=status.get('elapsed'))

def selected(row, keys):
    return {k: row.get(k) for k in keys}

def write_json(name, data):
    atomic(ROOT/'docs/evidence'/name, data)

def searches():
    names=['training-budget-settle-base','training-budget-settle-batch',
        'training-budget-settle-recompute','training-budget-settle-both',
        'training-refine-settle-base','training-refine-settle-batch',
        'training-budget-fc-bf16','training-budget-timed-batch',
        'training-selected-settle-fp32','training-selected-settle-bf16','training-selected-timed-fp32']
    report=dict(schema='tide-training-search-evidence-v1', math_revision='4f5b9f20a4030d68ee66ae2d26e34659a5e37234',
        timing='fresh private state for each whole training iteration, continuing parameters/AdamW; two warmup updates and five normal timed updates; audit excluded',
        limitations=['finite tested candidates, not hardware limits or global optima',
            '48 GiB sampled allocator allocated budget, not hardware isolation or reserved-memory limit',
            'compatible execution paths can produce different activity after updates; not strict equal-trajectory speedups'], searches=[], variants=[])
    keys=('subject','batch','length','state','row_state','median_seconds','seconds','input_positions_per_second',
        'timed_peak_device_bytes','sample_range_over_median','initial_parameter_sha256','input_sha256',
        'result_file','device_memory_by_device','activity')
    for name in names:
        d=read(Path('artifacts')/name/'search/search.json')
        r=run(name,'search/search.json')
        r.update(arguments=d['arguments'], frozen_config=d['frozen_config'], axis_values=d['axis_values'],
            child_states=d['child_states'], audits=d['audits'],
            lines=[dict(subject=line['subject'], stop_reason=line['stop_reason'],
                observations=[selected(row, keys) for row in line['observations']]) for line in d['lines']])
        report['searches'].append(r)
    for name in ['training-options-settle-fp32','training-options-settle-bf16','training-options-timed-fp32','training-options-timed-bf16']:
        d=read(Path('artifacts')/name/'results/variants.json')
        r=run(name,'results/variants.json'); r.update(arguments=d['arguments'],summaries=d['summaries'],
            pairing_scope=d['pairing_scope'], children=d['children'])
        report['variants'].append(r)
    write_json('training-search-20261118.json',report)

def continuous():
    report=dict(schema='tide-training-continuous-evidence-v1', math_revision='4f5b9f20a4030d68ee66ae2d26e34659a5e37234',
        steady_definition='updates 3 through 20, sum(input_positions)/sum(complete_seconds); median separately',
        restore_scope='same-process paired checkpoint comparisons and separate new-process restore/update execution are distinct',
        rows=[])
    for name in ['fc-bf16','settle-fp32','settle-bf16','timed-fp32','scale-fp32','scale-bf16']:
        full='training-continuous-'+name
        d=read(Path('artifacts')/full/'continuous.json'); obs=d['observations']; steady=obs[2:]
        assert d['state']=='completed' and len(obs)==20
        assert all(all(row[k] for k in ('finite_output','finite_parameters','finite_gradients')) for row in obs)
        for i,row in enumerate(obs):
            c=row['continuation']
            assert all(c[k] for k in ('private_state_finite','private_state_detached','optimizer_finite'))
            assert c['optimizer_step_min']==c['optimizer_step_max']==i+1
            assert c['kv_retained_slot_max']<=max(c['kv_slot_limits'].values())
        r=run(full,'continuous.json')
        r.update(arguments=d['arguments'],parameters=d['parameters'],spec=d['spec'],
            precision=d['precision'],initial_parameters_sha256=d['initial_parameters_sha256'],input_sha256=d['input_sha256'],
            steady_median_seconds=statistics.median(x['complete_seconds'] for x in steady),
            steady_positions_per_second=sum(x['input_positions'] for x in steady)/sum(x['complete_seconds'] for x in steady),
            peak_allocated_bytes=max(x['device_memory']['peak_allocated_bytes'] for x in obs),
            peak_reserved_bytes=max(x['device_memory']['peak_reserved_bytes'] for x in obs),
            observations=[selected(x,('index','complete_seconds','backward_seconds','optimizer_seconds','input_positions','state_position','state_memory','device_memory','continuation')) for x in obs],
            checkpoint=d['checkpoint'],checkpoint_device_memory=d.get('checkpoint_device_memory'))
        restored='training-restore-'+name
        t=read(Path('artifacts')/restored/'restore.json')
        assert t['state']=='completed_finite' and not t['optimizer_step_increment_errors']
        assert all(v['finite'] for v in t['updated_finite'].values())
        r['fresh_process']=run(restored,'restore.json')
        r['fresh_process'].update(selected(t,('checkpoint_source','checkpoint_source_scope','checkpoint_file','input_sha256','input_shape',
            'load_complete_seconds','next_update_complete_seconds','next_update_peak_device_bytes','loaded_state_position','restored_state_position',
            'loaded_state_memory','restored_state_memory','updated_finite','private_state_detached','optimizer_step_increment_errors',
            'parameter_tensors_with_gradients','comparison_qualification','numerical_equivalence_qualification','phases','optimizer_before','optimizer_after')))
        r['fresh_process']['checkpoint_producer_source_verified']=d['source']
        r['fresh_process']['checkpoint']=pointer(Path('artifacts')/full/'checkpoint.pt')
        if not d['source'].startswith(t['checkpoint_source']):
            r['fresh_process']['source_label_correction']='CLI checkpoint-source was supplied as ae247dee for this restore; actual producer is the linked continuous record source. Raw result retained. These revisions have the same src/tide tree.'
        assert t['input_sha256']==d['input_sha256']
        report['rows'].append(r)
    report['production_trees']={rev:subprocess.check_output(['git','rev-parse',rev+':src/tide'],cwd=ROOT,text=True).strip() for rev in ['ae247dee','22527cf4']}
    write_json('training-continuous-20261118.json',report)

if __name__=='__main__':
    searches(); continuous()

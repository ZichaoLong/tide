"""Read original result files; independently check counts and summarize samples."""
import hashlib
import json
import math
from pathlib import Path
import statistics
from datetime import datetime, timezone

from scripts.job import atomic

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent


def identity(path):
    return dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def row(name, result, subject=None):
    current = result if subject is None else result['subjects'][subject]
    observations = current.get('observations', [])
    if not observations:
        return dict(name=name, state=result['state'], last_stage=result['last_stage'], samples=0)
    batch = result['arguments']['batch']
    schedule = result['schedule']
    total = schedule['prefill'] + schedule['decode_steps'] + schedule['stream_positions']
    assert schedule['stream_positions'] == 0
    for observed in observations:
        previous = 0
        for call in observed['calls']:
            assert call['start'] == previous == call['state_before']
            assert call['state_after'] == call['stop']
            assert call['input_positions'] == batch*(call['stop']-call['start'])
            assert call['seconds'] > 0
            previous = call['stop']
        assert previous == total == observed['final_state_position']
        assert observed['input_positions'] == batch*total
        stable = [c for c in observed['calls'] if c['phase'] == 'decode'][1:]
        assert len(stable) == schedule['decode_steps']-1 == observed['stable_decode']['calls']
        seconds = sum(c['seconds'] for c in stable)
        positions = sum(c['input_positions'] for c in stable)
        assert math.isclose(observed['stable_decode']['positions_per_second'], positions/seconds, rel_tol=1e-12)
        assert observed['drain_seconds'] == 0
    rates = [o['stable_decode']['positions_per_second'] for o in observations]
    devices = set(d for o in observations for d in o['allocator'])
    median = statistics.median(rates)
    memory = {kind: max((m[f'peak_{kind}_bytes'] for o in observations for m in o['allocator'].values()), default=0)
              for kind in ('allocated','reserved')}
    diagnostics = current['diagnostics']
    assert diagnostics['finite_outputs'] and diagnostics['finite_private_state'] and diagnostics['private_state_detached']
    assert current['initial_parameters']['finite']
    audit = result.get('matching_reference', {}).get('audit') if subject == 'tide' else None
    transport = current['placement'].get('predecessor_transport', [])
    remote = [item for item in transport if item['remote']]
    if subject != 'tide':
        spec = current['spec']
        expected = 2*batch*spec['layers']*min(total,spec['window']-1)*spec['d']*4
        for observed in observations:
            assert observed['state_memory']['payload_bytes'] == expected
    return dict(name=name,state=current['state'] if subject else result['state'],samples=len(observations),
        source=result['source'], arguments=result['arguments'], schedule=schedule,
        config=result.get('config'),spec=current['spec'],parameters=current['parameters'],
        precision=result['precision'],physical_devices=result['physical_devices'],
        resource_identity=result['resource_identity'],cpu_affinity=result['cpu_affinity'],
        torch_threads=result['torch_threads'],placement=current['placement'],
        static_transport_plan=dict(entries=len(transport),remote_entries=len(remote),
            remote_source_node_memberships=sum(x['unique_source_nodes'] for x in remote),
            scope='static predecessor/destination plan, not observed transport time, bytes or runtime copy calls'),
        comparison=current.get('comparison', result.get('comparison')),
        initial_parameters=current['initial_parameters'],input_sha256=current['input_sha256'],
        shared_input_prefix=current.get('shared_input_prefix'),diagnostics=diagnostics,
        audit=audit,observations=observations,
        metrics=dict(stable_decode_positions_per_second=median,sample_rates=rates,
                     range_over_median=(max(rates)-min(rates))/median,
                     mean_decode_call_seconds=statistics.median(o['stable_decode']['seconds']/o['stable_decode']['calls'] for o in observations),
                     median_decode_call_seconds=statistics.median(o['stable_decode']['median_call_seconds'] for o in observations),
                     p95_decode_call_seconds=statistics.median(o['stable_decode']['p95_call_seconds'] for o in observations),
                     per_card_decode_positions_per_second=median/len(devices),
                     first_output_seconds=statistics.median(o['first_any_backbone_output_seconds'] for o in observations),
                     prompt_seconds=statistics.median(o['prompt_output_ready_seconds'] for o in observations),
                     peak_bytes=memory,participating_devices=len(devices)),
        independent_checks=dict(calls_and_state_progression=True,stable_input_denominator=True,
                                no_intermediate_reset_or_drain=True,
                                fp32_transformer_kv_payload=subject!='tide'))


submissions = json.loads((BASE/'submissions.json').read_text())
rows=[]
for submitted in submissions:
    run=ROOT/'artifacts'/submitted['name']
    status=json.loads((run/'status.json').read_text())
    entry=dict(name=submitted['name'],outer_state=status['state'])
    if (run/'inference.json').exists():
        result=json.loads((run/'inference.json').read_text())
        entry['result']=row(submitted['name'],result,None if submitted['name'].startswith('depth-') else 'tide')
    entry['files']={f:identity(run/f) for f in ('launch.json','status.json','queue.json','environment.json','inference.json','reference.json') if (run/f).exists()}
    entry['launch']=json.loads((run/'launch.json').read_text())
    rows.append(entry)
old=[]
for scale in ('9b','17b'):
    path=ROOT/f'artifacts/agg-inference-hb{scale}-fp32-v1/inference.json'
    result=json.loads(path.read_text())
    for subject in ('tide','C'):
        old.append(dict(file=identity(path),result=row(f'original-{scale}-{subject}',result,subject)))
checks={}
for entry in rows:
    r=entry.get('result',{})
    if 'initial_parameters' in r and r['name'].startswith('batch-'):
        scale='17b' if 'hb17b' in r['name'] else '9b'
        original=next(x['result'] for x in old if x['result']['name']==f'original-{scale}-tide')
        checks[r['name']]=dict(same_initial_parameter_hash=r['initial_parameters']['sha256']==original['initial_parameters']['sha256'])
        assert checks[r['name']]['same_initial_parameter_hash']
byname={x['name']:x.get('result') for x in rows}
for batch in (4,8):
    left=byname.get(f'batch-hb17b-b{batch}-fp32-v1')
    right=byname.get(f'batch-hb17b-b{batch}-four-fp32-v1')
    if not (left and right and left.get('audit') and right.get('audit')):
        continue
    checks[f'17b_b{batch}_two_vs_four_devices']={
        'same_initial_parameter_hash':left['initial_parameters']['sha256']==right['initial_parameters']['sha256'],
        'same_input_hash':left['input_sha256']==right['input_sha256'],
        'same_work':left['audit']['work']==right['audit']['work'],
        'same_query_packing':left['audit']['query_packing']==right['audit']['query_packing'],
        'same_effective_touched':left['audit']['effective_touched']==right['audit']['effective_touched'],
        'same_output_support':left['diagnostics']['output_support_sha256']==right['diagnostics']['output_support_sha256'],
        'same_output_times':left['diagnostics']['original_output_times_sha256']==right['diagnostics']['original_output_times_sha256']}
record=dict(schema='tide-depth-batch-followup-1',recorded_at_utc=datetime.now(timezone.utc).isoformat(),
            runs=rows,original_comparisons=old,pairwise_checks=checks,
            all_complete=all(x['outer_state']=='passed' and x.get('result',{}).get('state')=='completed' and x['result']['samples']==5 for x in rows),
            scope='supplementary fixed-depth Transformer and Tide batch/placement measurements; original A-D retained; host shared, cooperative exclusive device admission')
atomic(BASE/'summary.json',record)
for entry in rows:
    r=entry.get('result',{}); m=r.get('metrics',{})
    print(entry['name'],entry['outer_state'],r.get('state'),r.get('samples'),m.get('stable_decode_positions_per_second'),m.get('range_over_median'))
print('all_complete',record['all_complete'])

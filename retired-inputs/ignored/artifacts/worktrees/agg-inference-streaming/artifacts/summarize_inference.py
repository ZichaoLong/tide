"""Read-only finite snapshot of parent-owned inference records."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path('/mi/data2T/zlong/home/llm/graph-execution-foundation')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from scripts.job import atomic


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read(path):
    if not path.exists():
        return None, None
    raw = path.read_bytes()
    return json.loads(raw), dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def stats(values):
    numbers = [x for x in values if x is not None]
    assert all(math.isfinite(x) for x in numbers)
    return dict(samples=values, median=statistics.median(numbers) if numbers else None,
                min=min(numbers) if numbers else None, max=max(numbers) if numbers else None)


def path_get(value, path):
    for key in path.split('.'):
        value = value.get(key) if isinstance(value, dict) else None
    return value


METRICS = {
    'first_any_seconds': 'first_any_backbone_output_seconds',
    'prompt_ingested_seconds': 'prompt_ingested_seconds',
    'prompt_ready_seconds': 'prompt_output_ready_seconds',
    'first_incremental_seconds': 'first_incremental_call_seconds',
    'prefill_positions_per_second': 'prefill.positions_per_second',
    'stable_decode_positions_per_second': 'stable_decode.positions_per_second',
    'stable_decode_median_call_seconds': 'stable_decode.median_call_seconds',
    'stable_decode_p95_call_seconds': 'stable_decode.p95_call_seconds',
    'chunked_stream_positions_per_second': 'chunked_stream.positions_per_second',
    'complete_positions_per_second': 'complete_positions_per_second',
    'complete_seconds': 'complete_seconds',
    'drain_seconds': 'drain_seconds',
}


def placement_summary(placement):
    result = {k: v for k, v in placement.items() if k in (
        'kind', 'devices', 'requested_transport', 'transport', 'region_concurrency',
        'actual_devices', 'total_unique_parameters', 'unused_admitted_devices', 'communication')}
    result['full_record_sha256'] = digest(placement)
    for key in ('region_partitions', 'layer_partitions'):
        if key in placement:
            result[key+'_counts'] = dict(Counter(placement[key]))
            result[key+'_sha256'] = digest(placement[key])
    return result


def audit_summary(audit):
    result = {k: v for k, v in audit.items() if k in (
        'parameters', 'effective_touched', 'stages', 'input_positions', 'sample_time_slots',
        'logical_axis', 'work', 'work_per_input_position', 'query_packing', 'retention')}
    matrices = audit.get('node_matrices', {})
    totals = defaultdict(lambda: defaultdict(int))
    for name, counts in matrices.items():
        group = name.split('.shards.')[0]
        if '.heads.' in group:
            group = group.split('.heads.')[0]+'.heads'
        for key, value in counts.items():
            totals[group][key] += value
    result['node_matrices_grouped'] = dict(totals)
    result['node_matrix_cost_definition'] = audit.get('matrix_cost_definition')
    return result


def verify_call_boundaries(run, name, subject):
    schedule, batch = run['schedule'], run['arguments']['batch']
    positions = schedule['prefill']+schedule['decode_steps']+schedule['stream_positions']
    observations = subject.get('observations', [])
    if not observations:
        return dict(checked_samples=0)
    depths = run.get('matching_reference', {}).get('matches', {}).get('C', {}).get('targets', {}).get('blocks')
    topology = run['config']['topology']
    timed = topology['timing'] == 'timed'
    period = topology.get('period') or (4 if timed else depths+(2 if topology['kind'] in ('fc', 'lattice') else 1))
    signatures = []
    for observation in observations:
        calls = observation['calls']
        previous = consumed = 0
        for call in calls:
            assert call['state_before'] == previous
            assert call['state_after'] >= previous
            assert call['input_positions'] == batch*(call['stop']-call['start'])
            consumed += call['input_positions']
            previous = call['state_after']
        assert consumed == batch*positions
        assert observation['input_positions'] == batch*positions
        assert calls[-1]['state_after'] == observation['final_state_position']
        assert sum(c['phase'] == 'drain' for c in calls) == int(name == 'tide' and timed)
        assert observation['diagnostic_checks_in_timing'] is False
        stable = observation['stable_decode']
        if stable['calls']:
            assert stable['calls'] == schedule['decode_steps']-1
            assert stable['input_positions'] == batch*(schedule['decode_steps']-1)
            assert math.isclose(stable['positions_per_second'], stable['input_positions']/stable['seconds'])
        signature = [{k:c[k] for k in ('phase','start','stop','input_positions','state_before','state_after','output_slots')} for c in calls]
        signatures.append(digest(signature))
    assert len(set(signatures)) == 1
    output = dict(checked_samples=len(observations), call_layout_sha256=signatures[0],
                  initial_position=0, final_state_position=observations[0]['final_state_position'],
                  calls=len(calls), input_positions_per_sample=positions,
                  sole_final_drain=name == 'tide' and timed)
    diagnosis = subject.get('diagnostics', {})
    if name == 'tide':
        bounds = diagnosis['original_time_bounds']
        assert len(bounds) == len(calls)
        for call, bound in zip(calls, bounds):
            before, after = call['state_before'], call['state_after']
            expected = (before*period+1, after*period, (after-before)*period) if timed else (
                before*period+depths, (after-1)*period+depths, after-before)
            assert (bound['first_original_time'],bound['last_original_time'],bound['returned_time_slots']) == expected
        output.update(original_output_time_bounds_checked=len(bounds), graph_period=period, graph_depth=depths,
            cache_metadata_status='not captured by timed benchmark; separate cache diagnostic required')
    else:
        spec = subject['spec']
        expected_bytes = 2*batch*spec['layers']*min(positions,spec['window']-1)*spec['d']*4
        assert diagnosis['state_memory']['payload_bytes'] == expected_bytes
        output['fp32_final_kv_payload_matches_declared_window'] = True
        output['expected_final_kv_payload_bytes'] = expected_bytes
    return output


def subject_summary(run, name, value):
    observations = value.get('observations', [])
    result = {k:v for k,v in value.items() if k in ('state','comparison','initial_parameters','parameters','input_sha256','input_shape','shared_input_prefix')}
    result['spec'] = {k:v for k,v in value.get('spec', {}).items() if k != 'placement'}
    result['placement'] = placement_summary(value.get('placement', {}))
    result['samples'] = len(observations)
    result['metrics'] = {key:stats([path_get(obs,path) for obs in observations]) for key,path in METRICS.items()}
    memory = defaultdict(lambda: defaultdict(list))
    for observation in observations:
        for device, counters in observation.get('allocator', {}).items():
            for key, count in counters.items():
                memory[device][key].append(count)
    result['allocator_per_device_max_bytes'] = {device:{key:max(counts) for key,counts in counters.items()} for device,counters in memory.items()}
    result['max_per_device_peak_allocated_gib'] = max((m['peak_allocated_bytes']/2**30 for m in result['allocator_per_device_max_bytes'].values()), default=None)
    result['max_per_device_peak_reserved_gib'] = max((m['peak_reserved_bytes']/2**30 for m in result['allocator_per_device_max_bytes'].values()), default=None)
    diagnosis = value.get('diagnostics', {})
    result['diagnostics'] = {k:v for k,v in diagnosis.items() if k not in ('output_available_by_call','original_time_bounds')}
    result['record_boundary_checks'] = verify_call_boundaries(run,name,value)
    if value.get('cold'):
        result['cold'] = {key:path_get(value['cold'],path) for key,path in METRICS.items()}
    return result


def summarize(directory):
    raw, fingerprints = {}, {}
    for filename in ('launch.json','status.json','queue.json','environment.json','inference.json','reference.json'):
        value, identity = read(directory/filename)
        if value is not None:
            raw[filename] = value; fingerprints[filename] = identity
    if not raw:
        return None
    result = dict(run=directory.name, files=fingerprints,
                  job={k:v for k,v in raw.get('status.json',{}).items() if k in ('state','source','started','finished','elapsed','peak_rss_bytes','max_rss_bytes','exit_code','error')},
                  queue=raw.get('queue.json'),
                  launch_arguments=raw.get('launch.json',{}).get('arguments'),
                  environment=raw.get('environment.json',{}).get('environment',{}))
    run=raw.get('inference.json')
    if not run:
        result['result_state']='not_started_or_no_record'
        return result
    result['result_state']=run['state']
    for key in ('source','arguments','config','schedule','precision','resource_identity','physical_devices','torch','cpu_affinity','torch_threads','host_load_before','host_load_after','last_stage'):
        result[key]=run.get(key)
    ref = run.get('matching_reference', {})
    result['reference']={k:ref[k] for k in ('reference_sha256','tide_parameters','tide_input_sha256','matches','matching_boundary') if k in ref}
    if ref:
        assert digest({k:v for k,v in ref.items() if k!='reference_sha256'}) == ref['reference_sha256']
        matches, audit=ref['matches'],ref['audit']
        for key, match in matches.items():
            c=match['constructor']; width, layers, ffn=c['d'],c['layers'],c['ffn_width']
            counted=layers*(4*width*width+3*width*ffn+2*width)+width
            assert counted==match['actual']['parameters']
            target=audit['parameters']['unique_parameters'] if key in ('A','C') else audit['effective_touched']['mean']
            assert match['targets']['parameters']==target
            assert math.isclose((counted-target)/target,match['relative_deviation']['parameters'],abs_tol=1e-15)
        assert matches['A']['targets']['blocks']==matches['B']['targets']['blocks']
        assert matches['C']['targets']['blocks']==matches['D']['targets']['blocks']
        result['reference_checks']=dict(canonical_reference_digest=True, joint_targets_match_actual_audit=True,
            transformer_parameter_formula='layers*(4*d*d+3*d*ffn_width+2*d)+d', constructor_counts_verified=4)
    result['audit']=audit_summary(ref.get('audit',{}))
    result['subjects']={name:subject_summary(run,name,value) for name,value in run['subjects'].items()}
    result['all_requested_subjects_complete'] = all(result['subjects'].get(name,{}).get('state')=='completed' and result['subjects'][name]['samples']==run['arguments']['samples'] for name in run['arguments']['subjects'])
    return result


def summarize_cache(directory):
    raw, fingerprints = {}, {}
    for filename in ('launch.json','status.json','queue.json','cache.json'):
        value, identity = read(directory/filename)
        if value is not None:
            raw[filename] = value; fingerprints[filename] = identity
    result=dict(run=directory.name, files=fingerprints, job=raw.get('status.json',{}).get('state'),
                queue=raw.get('queue.json'), result_state=raw.get('cache.json',{}).get('state','not_started_or_no_record'))
    value=raw.get('cache.json')
    if not value:
        return result
    for key in ('source','arguments','config','precision','resource_identity','physical_devices','parameters',
                'placement','input_sha256','input_shape','device_memory','error'):
        if key in value: result[key]=value[key]
    diagnostic=value.get('diagnosis',{})
    result['diagnosis']={k:v for k,v in diagnostic.items() if k not in ('snapshots','output')}
    result['output']={k:v for k,v in diagnostic.get('output',{}).items() if k not in ('output_available_by_call','original_time_bounds')}
    snapshots=[]
    observed,full,evicted=set(),set(),set()
    for snapshot in diagnostic.get('snapshots',[]):
        kv_rows=[r for bank in snapshot['kv'] for r in bank['rows']]
        ssd_rows=[r for bank in snapshot['ssd'] for r in bank['rows']]
        positive=[r for r in kv_rows if not r['empty']]
        if value['state']=='passed':
            assert not snapshot.get('failures',[])
            assert not snapshot['pending']['failures'] and all(not r['failures'] for r in snapshot['pending']['banks'])
            for bank in snapshot['kv']:
                assert not bank['failures'] and bank['dtypes']['times']=='torch.int64'
                assert bank['observed_slots']<=bank['capacity']
            for row in kv_rows:
                key=row['owner'],row['sample'];observed.add(key)
                if row['full']:full.add(key)
                assert not row['failures'] and row['retained_times_sha256']==row['expected_times_sha256']
                assert row['positive_entries']==row['expected_positive_entries']
                assert row['full']==(row['positive_entries']==row['configured_capacity'])
                if not row['empty']:
                    assert type(row['first_positive_time']) is int and type(row['last_positive_time']) is int
                    assert row['maximum_age_from_last_write']==row['last_write_time']-row['first_positive_time']
            for bank in snapshot['ssd']:
                assert not bank['failures'] and bank['times_dtype']==bank['convolution_times_dtype']=='torch.int64'
                for row in bank['rows']:
                    assert not row['failures'] and row['retained_times_sha256']==row['expected_times_sha256']
                    assert row['raw_last_write_time']==row['expected_last_write_time']
            for witness in snapshot['evictions']:
                assert witness['retained_write_set_verified']
                assert witness['current_first_time']>witness['previous_first_time']
                assert witness['old_first_age_now']==witness['current_last_write_time']-witness['previous_first_time']
                assert witness['old_first_age_now']>=witness['window']
                evicted.add((witness['owner'],witness['sample']))
        bounds=lambda key:[min(r[key] for r in positive),max(r[key] for r in positive)] if positive else None
        snapshots.append(dict(call=snapshot['call'],phase=snapshot['phase'],state_position=snapshot['state_position'],
            kv_banks=len(snapshot['kv']),kv_owner_samples=len(kv_rows),kv_full=sum(r['full'] for r in kv_rows),
            kv_empty=sum(r['empty'] for r in kv_rows),kv_positive_entries=sum(r['positive_entries'] for r in kv_rows),
            capacities=sorted({bank['capacity'] for bank in snapshot['kv']}),windows=sorted({bank['window'] for bank in snapshot['kv']}),
            first_time_range=bounds('first_positive_time'),last_time_range=bounds('last_positive_time'),
            last_write_range=bounds('last_write_time'),max_age_range=bounds('maximum_age_from_last_write'),
            ssd_owner_samples=len(ssd_rows),ssd_conv_entries=sum(r['positive_entries'] for r in ssd_rows),
            pending_positive_nonterminal_senders=sum(r['outgoing_positive_pending'] or 0 for r in snapshot['pending']['banks']),
            verified_evictions=sum(w['retained_write_set_verified'] for w in snapshot['evictions']),
            example_eviction=snapshot['evictions'][0] if snapshot['evictions'] else None))
    result['snapshots']=snapshots
    if value['state']=='passed':
        assert diagnostic['passed'] and not diagnostic['structural_failures']
        assert diagnostic['write_ledger_failure_count']==diagnostic['output_time_failure_count']==0
        assert len(observed)==diagnostic['kv_owner_samples_observed']
        assert len(full)==diagnostic['kv_owner_samples_ever_full']
        assert len(evicted)==diagnostic['kv_owner_samples_with_verified_eviction']
        assert diagnostic['output_time_checks']==len(value['actual_calls'])
        assert diagnostic['output']['finite_outputs'] and diagnostic['output']['finite_private_state']
        if value['config']['topology']['timing']=='timed':
            assert snapshots[-1]['phase']=='drain' and snapshots[-1]['pending_positive_nonterminal_senders']==0
        result['record_checks']=dict(original_int64_metadata=True, positive_write_digests_equal=True,
            eviction_counts_and_expired_old_first=True, declared_owner_counts=True, final_pending_rule=True)
    return result


def pairwise_checks(runs):
    by_name={r['run']:r for r in runs}
    pairs=[('region_batch','agg-inference-hb17b-serial-b1-fp32-v1','agg-inference-hb17b-serial-b4-fp32-v1'),
           ('region_concurrency','agg-inference-hb17b-serial-b4-fp32-v1','agg-inference-hb17b-fp32-v1'),
           ('parameter_indexing','agg-inference-hb9b-sustain-fp32-v1','agg-inference-hb9b-static-fp32-v1')]
    output={}
    for label, left, right in pairs:
        a,b=by_name[left],by_name[right]
        x,y=a.get('subjects',{}).get('tide',{}),b.get('subjects',{}).get('tide',{})
        if x.get('state')!='completed' or y.get('state')!='completed':
            output[label]=dict(left=left,right=right,state='pending')
            continue
        identities=dict(source_commit=a['source']['source_commit']==b['source']['source_commit'],
            tide_content=a['source']['tide_content_sha256']==b['source']['tide_content_sha256'],
            harness_content=a['source']['harness_content_sha256']==b['source']['harness_content_sha256'],
            initial_parameters=x['initial_parameters']['sha256']==y['initial_parameters']['sha256'],
            input=x['input_sha256']==y['input_sha256'],input_shape=x['input_shape']==y['input_shape'],
            schedule=a['schedule']==b['schedule'],precision=a['precision']==b['precision'],
            region_partitions=x['placement'].get('region_partitions_sha256')==y['placement'].get('region_partitions_sha256'))
        assert all(identities[k] for k in ('initial_parameters','input','input_shape','schedule','precision','region_partitions'))
        config_diff={k:[a['config'].get(k),b['config'].get(k)] for k in set(a['config'])|set(b['config']) if a['config'].get(k)!=b['config'].get(k)}
        if label=='region_concurrency':assert not config_diff
        if label=='region_batch':assert set(config_diff)=={'region_batch_size'}
        if label=='parameter_indexing':assert set(config_diff)=={'parameter_indexing'}
        equal={k:a['audit'][k]==b['audit'][k] for k in ('effective_touched','work','query_packing')}
        for key in ('output_support_sha256','original_output_times_sha256'):
            equal[key]=x['diagnostics'][key]==y['diagnostics'][key]
        equal['per_time_parameter_counts_sha256']=a['audit']['retention']['per_time_parameter_counts_sha256']==b['audit']['retention']['per_time_parameter_counts_sha256']
        ratios={k:y['metrics'][k]['median']/x['metrics'][k]['median'] for k in METRICS
                if x['metrics'][k]['median'] not in (None,0) and y['metrics'][k]['median'] is not None}
        output[label]=dict(left=left,right=right,state='completed',equal_identities=identities,
            config_difference=config_diff,physical_devices=[a['physical_devices'],b['physical_devices']],
            requested_region_concurrency=[a['arguments']['region_concurrency'],b['arguments']['region_concurrency']],
            equal_aggregated_diagnostics=equal,right_divided_by_left=ratios,
            limitation='diagnostic count/support agreement does not compare raw values, gradients, or every internal action identity; physical devices differ')
    return output


def main():
    rows=[summarize(p) for p in sorted((ROOT/'artifacts').glob('agg-inference-*')) if p.is_dir()]
    rows=[row for row in rows if row]
    output=dict(schema='tide-inference-evidence-snapshot-1', captured_utc=datetime.now(timezone.utc).isoformat(),
                scope='read-only snapshot of independently atomic files; running records are provisional',
                units='seconds, B*input_positions/second; memory=max per-device peak, GiB=2**30 bytes',
                aggregation='per-metric median/min/max of complete normal observations; no inferred samples',
                runs=rows,pairwise_checks=pairwise_checks(rows),
                cache_diagnostics=[summarize_cache(p) for p in sorted((ROOT/'artifacts').glob('agg-cache-*-fp32-v1')) if p.is_dir()])
    witness,identity=read(HERE/'parameter-view-copy-witness.json')
    if witness:
        for row in witness['rows']:
            assert row['gather']['output_bytes']-row['static_view']['output_bytes']==row['saved_parameter_index_output_bytes']
            assert all(row[k] for k in ('raw_value_exact','strength_exact','state_exact','work_exact'))
        output['small_cpu_parameter_materialization_witness']=dict(file=identity,payload=witness,
            limit='logical cumulative parameter advanced-index outputs; not NPU bandwidth, physical traffic, throughput or peak memory')
    output['math_revision']='4f5b9f20a4030d68ee66ae2d26e34659a5e37234'
    output['all_requested_records_complete']=(all(row.get('all_requested_subjects_complete',False) and row['job'].get('state')=='passed' for row in rows)
        and all(row['result_state']=='passed' and row['job']=='passed' for row in output['cache_diagnostics']))
    if output['all_requested_records_complete']:
        output['schema']='tide-inference-evidence-1'
        output['scope']='terminal read-only evidence of declared finite inference schedules and separate cache metadata diagnostics'
    atomic(HERE/'inference-summary.json',output)
    print(json.dumps({'snapshot':str(HERE/'inference-summary.json'),'runs':[(r['run'],r['result_state']) for r in output['runs']]}))


if __name__=='__main__':
    main()

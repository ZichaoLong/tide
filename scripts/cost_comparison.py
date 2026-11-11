#!/usr/bin/env python3
"""Collect explicitly paired cost records; no fixed job names or run directories.

Adapters normalize existing dense-query and layer-chain results. Comparison
roles remain declarations; costs never promote numerical or restore evidence.
A new workload/backend normally only needs a record adapter, not a new runner.
"""
import argparse
import json
import math
from pathlib import Path
from durable_records import write_json
from source_identity import digest


def measured(rows):
    actual = [r for r in rows if not r['warmup']]
    if len(actual) != 1:
        raise ValueError('one independent measured interval required by these adapters')
    return actual[0]


def dense(d):
    c = d['resolved']; row = measured(d['phases'])
    return dict(seconds=row['complete_wall_seconds'], batch=c['training']['batch'],
        input_positions=row['tokens'], first_token_seconds=row.get('first_token_seconds'),
        first_outputs=len(row['first_tokens']) if 'first_tokens' in row else None,
        first_token_scope=row.get('ttft_scope'),
        length=c['training']['length'], mode=d['mode'], device=c['device'],
        threads=d['effective_threads'], dtype=c['core'].get('dtype', 'float32'),
        phases=row, work=row['work'], case=d['fc_shape'], client=c['core']['eager_implementation'],
        policy=d['query_execution'], baseline=d['baseline'], timing=d['phase_fences'],
        scope=d['scope'], parameters=d['trainable_parameters'],
        source=d.get('foundation_source'), dirty=d.get('foundation_dirty'),
        precision=d['precision'], initial_parameters_sha256=d['initial_parameters_sha256'],
        measured_parameters_sha256=d['measured_parameters_sha256'],
        placement=c['core']['placement_preset'], schedule=d['prefill_schedule'],
        observations=d['trace_records'], diagnostic_boundary=d['diagnostic_boundary'],
        downstream=d['downstream'], downstream_archive_sha256=d['downstream_archive_sha256'],
        resolved=c, affinity=d['affinity'], visibility=d['visibility'])


def layer(d):
    c=d['config'];row=measured(d['observations'])
    return dict(seconds=row['seconds'],batch=c['batch'],length=c['length'],mode=c['mode'],
        input_positions=c['batch']*c['length'],
        device=c['device'],threads=c['threads'],dtype=c['dtype'],phases=row['phases'],work=row['actual_work'],
        case=f"{c['family']}-chain-d{c['width']}-layers{c['depth']}-heads{c['heads']}",
        client=c['implementation'],baseline=c['implementation'],policy='auto',timing='phase',
        scope=d['timing'],parameters=d['parameters'],source=d['source'],dirty=None,
        precision=d['numerical_policy'],initial_parameters_sha256=d['initial_parameters_sha256'],
        measured_parameters_sha256=d['initial_parameters_sha256'])


ADAPTERS={'tide-dense-query-cost-v1':dense, 'tide-layer-chain-cost-v1':layer}
RELATIONS={'same-function','task-projection','different-function'}


def collect(manifest, directory):
    if manifest.get('schema') != 'tide-cost-comparison-v1':
        raise ValueError('unknown comparison manifest')
    result={}
    for item in manifest['records']:
        name=item['name']
        if name in result:raise ValueError('duplicate record name')
        path=Path(item['path']);path=path if path.is_absolute() else directory/path
        d=json.loads(path.read_text());adapter=ADAPTERS[d['schema']]
        row=dict(name=name,record=str(path.resolve()),record_sha256=digest(path),
                 execution=d['execution'],strict_numerical=d['strict_numerical'],
                 recovery=d['recovery'],native_build=d.get('native_build'))
        if d['execution']=='completed':
            row.update(adapter(d));seconds=row['seconds']
            if not math.isfinite(seconds) or seconds<=0:raise ValueError('invalid measured duration')
            row['input_positions_per_second']=row['input_positions']/seconds
            if row.get('first_token_seconds') is not None:
                # Complete cleanup/sync interval is not necessarily TTFT.
                ttft=row['first_token_seconds']
                if not math.isfinite(ttft) or ttft<=0:raise ValueError('invalid first-token duration')
                if row.get('first_outputs') is not None:
                    row['first_outputs_per_second']=row['first_outputs']/ttft
            # Throughput counts input positions, not every node/edge update.
        else:
            row['error']=d.get('error','incomplete execution')
        result[name]=row
    pairs=[]
    for request in manifest['comparisons']:
        relation=request['relation']
        if relation not in RELATIONS:raise ValueError('comparison relation must be declared')
        a,e=(result[request[k]] for k in ('candidate','reference'))
        pair=dict(request)
        if a['execution']!='completed' or e['execution']!='completed':
            pair['state']='unavailable'
        else:
            axes=('case','batch','length','mode','device','dtype','threads','timing')
            differences={k:[a[k],e[k]] for k in axes if a[k]!=e[k]}
            if differences:raise ValueError(f'incompatible measured axes: {differences}')
            if relation in ('same-function','task-projection') and a['measured_parameters_sha256']!=e['measured_parameters_sha256']:
                raise ValueError('same-function/task-projection comparison requires identical starting parameters')
            pair.update(state='cost_observed',seconds_delta=a['seconds']-e['seconds'],
                        candidate_over_reference=a['seconds']/e['seconds'],
                        equivalence='not_inferred_from_costs')
            work=('candidate_events','selected_events','visited_edges','block_positions')
            pair['work_count_differences']={k:[a['work'].get(k),e['work'].get(k)]
                for k in work if a['work'].get(k)!=e['work'].get(k)}
            pair['work_count_scope']='Equal counts do not establish event/support identity.'
            ap=a['phases'].get('phases_seconds',a['phases'])
            ep=e['phases'].get('phases_seconds',e['phases'])
            if a['timing']=='phase' and e['timing']=='phase':
                phases={k:ap[k]-ep[k] for k in ap.keys() & ep.keys()
                        if isinstance(ap[k],(int,float)) and isinstance(ep[k],(int,float))}
                pair['phase_seconds_delta']=phases
                pair['unpaired_seconds_delta']=pair['seconds_delta']-sum(phases.values())
            else:
                pair['phase_delta_scope']='No device-phase subtraction from host-enqueue intervals.'
        pairs.append(pair)
    from execution_diagnostics import attachments
    return dict(schema='tide-cost-review-v1',question=manifest['question'],records=list(result.values()),
                comparisons=pairs,collector_sha256=digest(__file__),
                diagnostics=attachments(manifest.get('diagnostics',[]),directory),
                scope='Explicit finite comparisons. Timing, numerical, independent roots, hardware and recovery remain separate facts.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=collect(json.loads(a.manifest.read_text()),a.manifest.resolve().parent)
    write_json(a.output,result)
    for row in result['records']:
        print(row['name'], row['execution'], f"{row['seconds']:.6g}s {row['input_positions_per_second']:.6g} positions/s" if 'seconds' in row else row['error'])


if __name__=='__main__':main()

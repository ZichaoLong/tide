#!/usr/bin/env python3
"""Read-only run audit; this extraction does not decide scientific acceptance."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.job import inspect
from scripts.review_io import write_review_json
from tide_lab.records import digest,now
from tide_lab.supervisor import verify_source


ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(path.read_text())


def main():
    result=dict(schema='tide-positive-support-review-v1',at=now(),
        scope='mechanical extraction; final acceptance and transfer conditions require authored review',
        foundation=read(ROOT/'dependencies/positive-floor-v1/foundation.lock.json'),jobs={},task_checks={},lens=[],inputs=[])
    total=0.
    for backend in ('cpu','npu'):
        name='support-v1-'+backend;directory=ROOT/'runs'/name
        job=ROOT/'artifacts/jobs'/name/'job.json';record=read(job)
        status=inspect(job);stage=read(directory/'stage.json')
        source=verify_source(Path(record['source']['path']))
        inputs=[dict(path=p['path'],expected_sha256=p['sha256'],actual_sha256=digest(p['path'])) for p in record['inputs']]
        if any(r['expected_sha256']!=r['actual_sha256'] for r in inputs):raise AssertionError('fixed input changed')
        result['inputs'].extend(inputs)
        elapsed=status['worker'].get('elapsed_seconds',0.) if status.get('worker') else 0.
        total+=elapsed
        result['jobs'][name]=dict(lifecycle=status,stage=stage,source=record['source'],
            verified_source_sha256=source['content']['tree_sha256'],worker_seconds=elapsed)
        for phase in ('foundation','task','export','hb','resume','fc'):
            path=directory/phase/'result.json'
            if not path.exists():continue
            item=read(path)
            result['jobs'][name][phase]=dict(path=str(path.relative_to(ROOT)),sha256=digest(path),state=item['state'],error=item.get('error'),elapsed_seconds=item.get('elapsed_seconds'))
            if phase=='foundation':result['jobs'][name][phase]['counts']=item.get('counts')
            if phase=='task':
                for topology,report in item.get('checks',{}).items():
                    core=report['checks']['core'];strict=core['strict']
                    result['task_checks'][backend+'-'+topology]=dict(numerical=strict['numerical'],
                        discrete=strict['discrete'],nonfinite=strict['nonfinite'],roots=core['roots'],
                        causal=core['causal'],causal_mechanisms=core['causal_mechanisms'],
                        chunked=core['chunked'],streaming=core['streaming'],depth_facts=core['depth_facts'])
            if phase in ('hb','resume','fc'):result[phase]=item
            projection=directory/phase/'projection.json'
            if projection.exists():
                data=read(projection);manifest=read(directory/phase/'manifest.json')
                result['lens'].append(dict(run_id=manifest['run_id'],project=manifest['project'],phase=phase,
                    state=data.get('status'),readback_matches=data.get('readback_matches'),
                    lens_run_name=data.get('destination_name'),event_count=data.get('event_count'),
                    projection_path=str(projection.relative_to(ROOT)),projection_sha256=digest(projection)))
    lock=read(ROOT/'dependencies/positive-floor-v1/installation.lock.json')
    package=lock['installed_identity']['package'];root=Path(package['path'])
    mismatch=[name for name,sha in package['files_sha256'].items() if digest(root/name)!=sha]
    if mismatch:raise AssertionError('installed content changed: '+str(mismatch))
    result['installation_verified_files']=len(package['files_sha256'])
    result['compute_worker_seconds']=total
    result['development_check_seconds']=10.79+49.12
    result['acceptance_measured_seconds']=total+result['development_check_seconds']
    result['prior_v5_compute_worker_seconds']=22931.15
    result['v5_including_acceptance_worker_seconds']=22931.15+total
    result['v5_including_acceptance_and_development_seconds']=22931.15+result['acceptance_measured_seconds']
    result['budget_note']='also retain development checks: 10.79s rejected old-lock attempt, 49.12s explicit-lock cadence checks; one-hour acceptance cap not expanded'
    write_review_json(ROOT/'docs/evidence/positive-support-acceptance-v1.json',result)
    print(json.dumps(dict(jobs={k:v['stage']['state'] for k,v in result['jobs'].items()},worker_seconds=total,lens=result['lens']),ensure_ascii=False))


if __name__=='__main__':main()

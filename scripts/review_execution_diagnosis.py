#!/usr/bin/env python3
"""Read diagnosis artifacts into review evidence; never modify workload records."""
import argparse
import collections
import csv
import datetime
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.review_io import write_review_json
from tide_lab.records import digest, now

ROOT = Path(__file__).resolve().parents[1]


def artifact(path):
    relative = path.relative_to(ROOT)
    parts = list(relative.parts)
    if 'npu-profile' in parts:
        # Vendor directory names embed the private host name; hash pins content.
        parts[parts.index('npu-profile')+1] = '*'
    return dict(path=str(Path(*parts)), bytes=path.stat().st_size, sha256=digest(path))


def device_profile(directory):
    found = list(directory.glob('npu-profile/*/ASCEND_PROFILER_OUTPUT'))
    if not found:
        return None
    base = found[0]
    operators = collections.defaultdict(lambda: [0, 0., 0.])
    shapes = collections.Counter()
    with (base/'operator_details.csv').open() as stream:
        for row in csv.DictReader(stream):
            item = operators[row['Name']]
            item[0] += 1
            item[1] += float(row['Host Self Duration(us)'])
            item[2] += float(row['Device Self Duration(us)'])
            if row['Name'] in ('aten::matmul', 'aten::mm', 'aten::mv', 'aten::bmm'):
                shapes[(row['Name'], row['Input Shapes'])] += 1
    tasks = collections.defaultdict(lambda: [0, 0.])
    aicpu = collections.defaultdict(lambda: [0, 0.])
    with (base/'task_time.csv').open() as stream:
        for row in csv.DictReader(stream):
            item = tasks[row['kernel_type']]
            item[0] += 1
            item[1] += float(row['task_time(us)'])
            if row['kernel_type'] == 'KERNEL_AICPU':
                item = aicpu[row['kernel_name']]
                item[0] += 1
                item[1] += float(row['task_time(us)'])
    kernels = collections.Counter()
    kernel_us = 0.
    with (base/'kernel_details.csv').open() as stream:
        for row in csv.DictReader(stream):
            kernels[row['Name']] += 1
            kernel_us += float(row['Duration(us)'])
    with (base/'step_trace_time.csv').open() as stream:
        timing = list(csv.DictReader(stream))
    return dict(scope='Instrumented CPU/NPU trace; operator counts include nested calls. Device Computing/Free are profiler categories, not uninstrumented utilization. Kernel duration sums are not critical-path wall time.',
        operator_rows=sum(v[0] for v in operators.values()),
        operators={k:dict(calls=v[0], host_self_ms=v[1]/1000, device_self_ms=v[2]/1000)
                   for k,v in sorted(operators.items(), key=lambda kv:kv[1][1], reverse=True)},
        matrix_shapes=[dict(operator=k[0], input_shapes=k[1], calls=v) for k,v in shapes.most_common(40)],
        task_types={k:dict(count=v[0], duration_ms=v[1]/1000) for k,v in tasks.items()},
        aicpu_kernels={k:dict(count=v[0], duration_ms=v[1]/1000) for k,v in aicpu.items()},
        kernel_count=sum(kernels.values()), kernel_duration_ms=kernel_us/1000,
        kernel_names=dict(kernels.most_common(30)), profiler_step_time=timing,
        artifacts=[artifact(base/name) for name in ('operator_details.csv','kernel_details.csv','task_time.csv','step_trace_time.csv')])


def review():
    records = []
    paths = sorted((ROOT/'runs').glob('v6-diag-*/result.json'))
    paths += sorted((ROOT/'runs/v6-diag-variants').glob('*/result.json'))
    for path in paths:
        value = json.loads(path.read_text())
        config = value['resolved']
        row = dict(run_id='tl-'+path.parent.name, result=artifact(path), state=value['state'],
            source_commit=value['source']['commit'], installation=value.get('installation'),
            device=config['device'], physical_visibility=value.get('physical_visibility'),
            implementation=config['core']['eager_implementation'], placement=config['core']['placement_preset'],
            width=config['core']['width'], batch=config['training']['batch'], length=config['training']['length'],
            native_options=config.get('diagnostic_native_options'), trainable_parameters=value.get('trainable_parameters'),
            phases=value['phases'], profile=value['profile'], acceptance=value['acceptance'], error=value.get('error'))
        projection=path.parent/'projection.json'
        if projection.exists():
            p=json.loads(projection.read_text())
            row['lens_projection']={k:p.get(k) for k in ('status','project','destination_name','destination_run_id','readback_matches')}
        ops=path.parent/'operators.json'
        if ops.exists():
            agg=collections.defaultdict(lambda:[0,0.])
            matrix=[]
            for item in json.loads(ops.read_text()):
                agg[item['operator']][0]+=item['count']
                agg[item['operator']][1]+=item['self_cpu_ms']
                if item['operator']=='aten::matmul':matrix.append(item)
            row['cpu_profiler']=dict(artifact=artifact(ops),
                operators={k:dict(calls=v[0],self_cpu_ms=v[1]) for k,v in sorted(agg.items(),key=lambda kv:kv[1][1],reverse=True)},
                matrix_shapes=sorted(matrix,key=lambda r:r['count'],reverse=True)[:25])
        row['device_profile']=device_profile(path.parent)
        records.append(row)
    jobs=[]
    cpu_seconds = npu_seconds = 0.
    for path in sorted((ROOT/'artifacts/jobs').glob('v6-diag-*/job.json')):
        j=json.loads(path.read_text());s=json.loads((path.parent/'state.json').read_text())
        parse = datetime.datetime.fromisoformat
        end = s.get('ended_at', s.get('finalized_at'))
        seconds = ((parse(end) if end else datetime.datetime.now(datetime.timezone.utc)) - parse(s['started_at'])).total_seconds()
        cpu_seconds += seconds*j['resources']['threads']
        npu_seconds += seconds*(j['device']=='npu')
        admitted = json.loads((path.parent/'submission-observation.json').read_text())
        jobs.append(dict(job_id=j['job_id'],unit=j['unit'],state=s['state'],
            initial_slice=admitted['systemd'].get('Slice'), termination_verified=s.get('termination_verified'),
            allocated_seconds=seconds, resources=j['resources'],record=str(path.relative_to(ROOT)),source_commit=j['source']['commit']))
    return dict(schema='tide-execution-diagnosis-review-v1',updated_at=now(),study_id='TL-EXEC-v6-PROFILE',
        review_status='measurement_digest_only; interpretation in linked human-authored report',
        runs=records,jobs=jobs, service_allocation=dict(cpu_core_hours=cpu_seconds/3600,
            npu_device_hours=npu_seconds/3600, scope='Whole service reservation including startup, profiling, analysis and recording; development/read-only review separate'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('docs/evidence/language-execution-diagnosis-v6.json'))
    a=p.parse_args();write_review_json(a.output,review())

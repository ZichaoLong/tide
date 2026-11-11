#!/usr/bin/env python3
"""Collect a completed run_fc_comparison investigation without rerunning models.

Reads the finite suite/job records and optional actual profiler exports. Numeric
failures stay failures; process completion is never promoted to equivalence.
"""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from durable_records import write_json
from source_identity import digest
from summarize_query_profile import summarize


def read(path):
    return json.loads(path.read_text())


def cpu_profile(path):
    rows = read(path / 'operators.json')
    flops, calls = 0, 0
    attention = Counter()
    for row in rows:
        name, shapes, count = row['operator'], row['input_shapes'], row['count']
        if name == 'aten::mm':
            flops += 2 * shapes[0][0] * shapes[0][1] * shapes[1][1] * count
            calls += count
        elif name == 'aten::bmm':
            flops += 2 * math.prod(shapes[0]) * shapes[1][2] * count
            calls += count
        elif name == 'aten::addmm':
            flops += 2 * shapes[1][0] * shapes[1][1] * shapes[2][1] * count
            calls += count
        if any(key in name.lower() for key in ('scaled_dot', 'flash', 'attention')):
            attention[name] += count
    stage_times, scalar_calls = Counter(), Counter()
    stages = path / 'stage-operators.json'
    if stages.exists():
        for row in read(stages)['operators']:
            stage_times[row['stage']] += row['self_cpu_ms']
            if row['operator'] == 'aten::_local_scalar_dense':
                scalar_calls[row['stage']] += row['count']
    return dict(aten_records=sum(r['count'] for r in rows if r['operator'].startswith('aten::')),
                observed_matrix_flops=flops, observed_matrix_calls=calls,
                matrix_scope='mm/bmm/addmm only; excludes fused Attention and nonmatrix work',
                stage_dispatcher_self_ms=dict(stage_times), scalar_calls=dict(scalar_calls),
                attention_dispatch=dict(attention),
                operators_sha256=digest(path / 'operators.json'),
                stage_operators_sha256=digest(stages) if stages.exists() else None)


def npu_profile(path, record, exports=None):
    summary = path / 'profile-summary.json'
    if exports is None:
        exports = next(Path(record['profile_output']).glob('*/ASCEND_PROFILER_OUTPUT'), None)
    if not summary.exists() and exports is None:
        raise FileNotFoundError('NPU profile exports or preserved profile-summary.json required')
    d = read(summary) if summary.exists() else summarize(
        exports)
    dispatch, scope = Counter(), Counter()
    if exports is not None:
        with (exports / 'operator_details.csv').open(newline='') as stream:
            for row in csv.DictReader(stream):
                name = row['Name']
                if any(key in name.lower() for key in ('scaled_dot', 'flash', 'attention')):
                    dispatch[name] += 1
                if name in ('task::backward', 'task::clip', 'task::optimizer', 'Optimizer.step#AdamW.step'):
                    scope[name] += 1
    return {k: d[k] for k in ('csv_sha256', 'device_kernel_records', 'step_trace_time',
                              'top_host_self', 'task_types', 'aicpu_kernels', 'scope')} | dict(
        attention_dispatch=dict(dispatch) if exports is not None else None,
        completion_events=dict(scope) if exports is not None else None,
        attention_kernels=[r for r in d['kernels'] if any(
            word in r['name'].lower() for word in ('attention', 'flash'))])


def numeric_section(row):
    failures = row['strict_mismatches']
    return {k: row[k] for k in ('strict_passed', 'max_abs', 'tensors', 'none_mismatches', 'nonfinite')} | dict(
        failed_fields=len(failures),
        failed_coordinates=sum(r['coordinates'] for r in failures),
        discrete_failed_fields=sum(r['kind'] == 'discrete' for r in failures),
        max_failure=max(failures, key=lambda r: r['max_abs'], default=None),
        first_failed_fields=failures[:4],
        field_families=dict(Counter(r['field'].split('/')[0].split('.')[0] for r in failures)))


def work_summary(row):
    names = ('candidate_events', 'selected_events', 'visited_edges', 'greedy_stages',
             'window_components', 'window_cyclic_components', 'read_calls', 'full_calls',
             'region_batch_calls', 'max_state_sequence', 'max_region_batch', 'max_full_batch',
             'state_scalar_sequence_steps', 'state_sequence_calls', 'block_positions',
             'query_packed-v2_nodes', 'query_legacy_nodes')
    return {k: row[k] for k in names if k in row}


def collect(root, jobs):
    job_records = {p.parent.name: read(p) for p in sorted(jobs.glob('fccompare-*/status.json'))}
    if not job_records or any(d['state'] in ('starting', 'running') for d in job_records.values()):
        raise ValueError('completed job identities required')
    launch = read(root / 'launch.json')
    for name, record in job_records.items():
        log = jobs / name / 'task.log'
        record['task_log_sha256'] = digest(log)
        record['profiler_notices'] = [line for line in log.read_text().splitlines() if any(
            text in line for text in ('Incorrect schedule', 'aic metrics', 'profiling data parsed'))]

    def source(path):
        name = path.relative_to(root).parts[0]
        suffix = 'sdpa' if name.startswith('sdpa4d') else 'triage' if name.startswith('triage') else name.split('-')[0]
        return job_records['fccompare-' + suffix]['source']

    costs, checks, profiles, configurations = [], [], [], {}
    native_build = None
    for path in sorted(root.glob('**/result.json')):
        d = read(path)
        identity = dict(record=str(path.relative_to(root)), sha256=digest(path), source=source(path))
        if d['schema'] == 'tide-fc-projection-comparison-v1':
            checks.append(identity | dict(execution=d['execution'], request=d['request'],
                tolerance=d['tolerance'], variants={name: v | dict(work=work_summary(v['work']))
                    for name, v in d['variants'].items()}, scope=d['scope'],
                comparisons={name: {part: numeric_section(v) for part, v in comparison.items()}
                             for name, comparison in d['comparisons'].items()}))
            continue
        if d['schema'] != 'tide-dense-query-cost-v1':
            raise ValueError('unexpected result schema')
        if len(d['phases']) != 2 or d['phases'][1]['warmup']:
            raise ValueError('expected one warmup and one measured interval')
        m, config = d['phases'][1], d['resolved']
        if d.get('native_build'):
            native_build = d['native_build']
        if (path.relative_to(root).parts[0] == 'cpu-main' and d['baseline'] == 'graph'
                and d['mode'] == 'training'):
            configurations[d['fc_shape']] = config
        if not math.isclose(m['input_positions_per_second'], m['tokens'] / m['complete_wall_seconds']):
            raise ValueError('throughput numerator mismatch')
        costs.append(identity | {k: d[k] for k in ('execution', 'baseline', 'mode', 'fc_shape',
            'phase_fences', 'diagnostic_boundary', 'formula_attention', 'effective_threads',
            'effective_interop_threads', 'precision', 'trainable_parameters', 'initial_parameters_sha256',
            'measured_parameters_sha256', 'profile', 'construction_seconds')} | dict(
                device=config['device'], client=config['core']['eager_implementation'],
                placement=config['core']['placement_preset'] if d['baseline'] in {'graph', 'columns', 'window'} else 'direct-device',
                window_execution=d.get('effective_window_execution'),
                batch=config['training']['batch'], length=config['training']['length'],
                trace=d['trace_records'], affinity=d['affinity'], visibility=d['visibility'],
                measurement=m | dict(work=work_summary(m['work']))))
        if d['profile'] != 'none':
            profile = identity | dict(profile=d['profile'], instrumented_step=d['instrumented_step'],
                                       instrumented_stages=d.get('instrumented_stages'))
            profile['packing_totals'] = {k: v for k, v in d.get('instrumented_packing', {}).items()
                                         if v and k.count('/') <= 3}
            if d['profile'] == 'cpu':
                profile['cpu'] = cpu_profile(path.parent)
            elif d['profile'] == 'npu':
                profile['npu'] = npu_profile(path.parent, d)
            profiles.append(profile)
    qualification = []
    for path in sorted(root.glob('*qualification.xml')):
        tree = ET.parse(path)
        qualification.append(dict(record=path.name, sha256=digest(path),
            source=job_records['fccompare-sdpa']['source'] if path.name.startswith('final-') else launch['source'],
            suites=[dict(e.attrib) for e in tree.iter('testsuite')]))
    commands = {str(p.relative_to(root)): read(p) for p in sorted(root.glob('*/commands.json'))}
    return dict(schema='tide-fc-three-layer-review-v1', collector_sha256=digest(__file__), launch=launch, jobs=job_records,
                source_identity_note='Job source/build identities override the historical foundation pin embedded in the downstream config.',
                native_build=native_build, configurations=configurations,
                commands=commands, standalone_launchers={p.name: p.read_text() for p in sorted(root.glob('*.sh'))},
                queues={p.name: read(p) for p in sorted(root.glob('*queue.json'))},
                qualification=qualification, costs=costs, checks=checks, profiles=profiles,
                scope='Finite FC4/FC16 CPU/NPU task-projection investigation. Not complete-root/optimizer/restore equivalence, not a production optimization. Numeric strict failures are independent of job completion.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--jobs-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    record = collect(args.run_root, args.jobs_root)
    write_json(args.output, record)
    print(json.dumps({k: len(record[k]) for k in ('costs', 'checks', 'profiles', 'qualification')}))


if __name__ == '__main__':
    main()

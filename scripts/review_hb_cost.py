#!/usr/bin/env python3
"""Read terminal HB measurements; never modify runs, jobs, or frozen inputs."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import statistics
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.job import inspect
from scripts.review_io import write_review_json
from tide_lab.records import digest, now

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def identity(value):
    source = value['source']
    return dict(source_commit=source['commit'],
                source_tree_sha256=source['content']['tree_sha256'],
                installation=value['installation'],
                plan_sha256=value.get('plan_sha256'),
                physical_visibility=value.get('physical_visibility'))


def file_record(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)), sha256=digest(path))


def job_record(name):
    path = ROOT/'artifacts/jobs'/name/'job.json'
    spec = read(path)
    observation = inspect(path)
    if (observation['state']['state'] not in ('completed', 'failed', 'stopped', 'timed_out')
            or not observation['termination_verified_now']):
        raise ValueError('terminal review requires no surviving descendants: '+name)
    state = observation['state']
    return dict(id=name, record=file_record(path), state_file=file_record(path.parent/'state.json'),
        state=state['state'], termination_verified_now=observation['termination_verified_now'],
        unit=spec['unit'], control_group=state.get('control_group'),
        source=spec['source'], resources=spec['resources'], inputs=spec['inputs'],
        command=spec['command'], started_at=state['started_at'], ended_at=state['ended_at'],
        elapsed_seconds=state['elapsed_seconds'], workload_exit_code=state.get('workload_exit_code'),
        cpu_core_hours_conservative=state['elapsed_seconds']*spec['resources']['threads']/3600,
        npu_card_hours_conservative=state['elapsed_seconds']*spec['resources']['npu_count']/3600,
        admission=observation.get('queue'), inspection_systemd=observation['systemd'])


def projection(path):
    p = path/'projection.json'
    if not p.exists():
        return dict(status='missing', verified=False)
    value = read(p)
    manifest = read(path/'manifest.json')
    events = sum(1 for _ in (path/'metrics.jsonl').open())
    return dict(record=file_record(p), **value,
                verified=(value.get('status') == 'verified' and value.get('readback_matches') is True
                          and value.get('project') == 'tide-learning-lab'
                          and value.get('source_run_id') == manifest['run_id']
                          and value.get('event_count') == events))


def main():
    plan_path = ROOT/'configs/checks/hb-cost-20261111.json'
    plan = read(plan_path)
    jobs, suites = [], []
    for name, config in plan['suites'].items():
        job = job_record('hb-cost-20261111-'+name); jobs.append(job)
        directory = ROOT/'runs'/job['id']; result = read(directory/'result.json')
        cases = []
        for entry in config['entries']:
            path = directory/entry['id']; case_path = path/'result.json'
            if not case_path.exists():
                cases.append(dict(id=entry['id'], state='not_run', case=entry))
                continue
            case = read(case_path)
            modes = {}
            for mode in ('no_grad', 'grad_forward', 'train_step'):
                observations = [r for r in case['observations'] if r['mode'] == mode and not r['warmup']]
                times = [r['seconds'] for r in observations]
                if not times:
                    continue
                modes[mode] = dict(seconds=times, median_seconds=statistics.median(times),
                    min_seconds=min(times), max_seconds=max(times),
                    input_positions_per_second=case['input_tokens']/statistics.median(times),
                    work=[r['work'] for r in observations],
                    loss=[r['loss'] for r in observations],
                    gradient_norm=[r['gradient_norm'] for r in observations])
            cases.append(dict(id=entry['id'], state=case['state'], case=case['case'],
                record=file_record(case_path), run_id=case['run_id'],
                trainable_parameters=case['trainable_parameters'], modes=modes,
                observations=case['observations'],
                cpu_threads=case['cpu_threads'], interop_threads=case['interop_threads'],
                affinity=case['affinity'], acceptance=case['acceptance'],
                construction_seconds=case['construction_seconds'], projection=projection(path)))
        suites.append(dict(id=name, state=result['state'], record=file_record(directory/'result.json'),
            identity=identity(result['identity']), elapsed_seconds=result.get('elapsed_seconds'),
            device=config['device'], threads=config['threads'], cases=cases,
            process_peak_rss_mib=max((r['memory'].get('resources/peak_rss_mib',0)
                for c in cases for r in c.get('observations',[])),default=None),
            npu_peak_allocator_mib=max((r['memory'].get('resources/npu_peak_allocated_mib',0)
                for c in cases for r in c.get('observations',[])),default=None)))

    probe_job = job_record('hb-gradient-20261111'); jobs.append(probe_job)
    probe_path = ROOT/'runs/hb-gradient-20261111/result.json'; probe = read(probe_path)
    probe_review = dict(schema='tide-hb-gradient-review-v1', at=now(), state='reviewed_with_limits',
        review_script_sha256=digest(Path(__file__)),
        record=file_record(probe_path), identity=identity(probe['identity']), job=probe_job,
        validation_blocks=probe['validation_blocks'], scope=probe['scope'], acceptance=probe['acceptance'],
        cases=[dict(**case, projection=projection(probe_path.parent/case['id'])) for case in probe['cases']],
        interpretation='Large directional gradients and initial output scales motivate recipe interventions; this probe does not establish their causal role, certify native VJP correctness, or measure learning.',
        caveats=['The directional gain is not a Jacobian spectral norm.',
                 'B2/L16 prefix is not full L128 validation.',
                 'Unique registration owners do not identify exclusive per-node gradient contributions.',
                 'No optimizer update, changed checkpoint, new exact resume, or sealed-test access.'])
    write_review_json(ROOT/'docs/evidence/hb-gradient-review-20261111.json', probe_review)

    by_suite = {s['id']: {c['id']: c for c in s['cases']} for s in suites}
    comparisons = []
    for topology in ('fc16', 'hb40', 'hb88'):
        for suite in ('cpu-main', 'npu-main'):
            direct = by_suite[suite][topology+'-direct-attention-b4-l32']
            tide = by_suite[suite][topology+'-tide-b4-l32']
            if all('train_step' in c.get('modes', {}) for c in (direct, tide)):
                comparisons.append(dict(topology=topology, suite=suite,
                    tide_over_direct_different_function_cost_ratio=tide['modes']['train_step']['median_seconds']/direct['modes']['train_step']['median_seconds'],
                    scope='Different functions and duties, not removable overhead or equivalent speedup.'))

    all_cases = [c for s in suites for c in s['cases']]
    projections = [c.get('projection', {}) for c in all_cases]+[c['projection'] for c in probe_review['cases']]
    starts = [datetime.fromisoformat(j['started_at']) for j in jobs]
    ends = [datetime.fromisoformat(j['ended_at']) for j in jobs]
    paths = [ROOT/'runs'/j['id'] for j in jobs]+[ROOT/'artifacts/jobs'/j['id'] for j in jobs]
    paths += [ROOT/'.local/source-hb-cost-20261111', ROOT/'.local/source-hb-grad-20261111']
    disk = subprocess.run(['du', '-s', '-B1', *map(str, paths)], check=True, capture_output=True, text=True)
    allocated_disk_bytes = sum(int(line.split()[0]) for line in disk.stdout.splitlines())
    report = dict(schema='tide-hb-cost-review-v1', at=now(), state='reviewed_with_limits',
        review_script_sha256=digest(Path(__file__)),
        plan=file_record(plan_path), budgets=plan['budgets'], jobs=jobs, suites=suites,
        comparison_boundary=plan['comparison_boundary'], timer_scope=plan['timer_scope'],
        cost_ratios=comparisons,
        counts=dict(planned_cost_cases=sum(len(s['entries']) for s in plan['suites'].values()),
                    completed_cost_cases=sum(c['state']=='completed' for c in all_cases),
                    gradient_cases=len(probe['cases']), lens_verified=sum(p.get('verified',False) for p in projections)),
        resources=dict(wall_span_seconds=(max(ends)-min(starts)).total_seconds(),
            cpu_core_hours_conservative=sum(j['cpu_core_hours_conservative'] for j in jobs),
            npu_card_hours_conservative=sum(j['npu_card_hours_conservative'] for j in jobs),
            run_job_and_frozen_source_allocated_bytes=allocated_disk_bytes,
            charge_scope='Reserved threads times complete service lifetime, including startup/recording and NPU host threads; not sampled CPU utilization.',
            memory_scope='ru_maxrss is process-lifetime high-water, not per-case incremental RSS. NPU allocator peaks are reset per repetition; they exclude other processes and some runtime/driver memory.'),
        caveats=['All windows reset weights and empty AdamW state: train timing includes first-step optimizer-state allocation; not a steady-state optimizer benchmark.',
                 'Synthetic all-position head/loss workload; input-position throughput is not generated-token throughput or measured language learning speed.',
                 'No-grad also computes all-position loss, so it is not a production inference/TTFT contract.',
                 'Shared NPU and broad host affinity; external CPU/host-memory contention not eliminated, three repeats are descriptive rather than confidence bounds.',
                 'Direct fan-in initialization and uniform merges differ from TIDE default initialization, weighted transport and sparse execution.',
                 'FP32 dtype is fixed, but backend HF32 toggles were not serialized separately; no new strict cross-backend numerical qualification.',
                 'Larger direct B16/L128 changes batch and length together; no pure batch-scaling attribution.',
                 'Installed 74e7bbd cost is not the current upstream working-tree performance.',
                 'Original strict finite numerical failures and limited recovery qualification remain; different-function direct checks do not repair them.'],
        learning_review='docs/evidence/hb-learning-review-20261111.json',
        gradient_review='docs/evidence/hb-gradient-review-20261111.json')
    write_review_json(ROOT/'docs/evidence/hb-cost-review-20261111.json', report)
    print(json.dumps(dict(counts=report['counts'], resources=report['resources']), indent=2))


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()

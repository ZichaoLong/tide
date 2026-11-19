#!/usr/bin/env python3
"""Synthesize retained records only; does not execute any model or device work."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEST = ROOT / 'docs/evidence/ffn-execution-devices-20261119.json'


def atomic(path, value):
    data = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    fd, staged = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, path)
        dfd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(staged):
            os.unlink(staged)


def fingerprint(path):
    data = path.read_bytes()
    return dict(path=str(path.relative_to(ROOT)), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def read(name):
    return json.loads((HERE / (name + '.json')).read_text())


def subset(value, keys):
    return {key: value[key] for key in keys if key in value}


def normal_suite(s):
    out = subset(s, ['name', 'state', 'source_commit', 'arguments', 'worker_requested_affinity',
                     'physical_devices', 'launch', 'status', 'suite', 'outer_state', 'suite_state',
                     'expected_children', 'expected_repeats', 'actual_child_order', 'normal_summary',
                     'same_repeat_pairs', 'calibration_only', 'aggregation_scope', 'parent_service_sampled_peak_rss_bytes'])
    out['normal_repeats'] = []
    for r in s['normal_repeats']:
        row = {k: v for k, v in r.items() if k not in {
            'recorded_normal_summary', 'recorded_boundary', 'raw_normal_samples',
            'host_load_before', 'host_load_after'}}
        row['normal_samples'] = [{k: v for k, v in x.items() if k != 'nonadditive_phase_subitems'}
                                 for x in r['raw_normal_samples']]
        out['normal_repeats'].append(row)
    out['r0_diagnostics'] = {label: subset(value, ['file', 'scope', 'work_summary', 'ledger_phase_summary', 'availability'])
                             for label, value in s['r0_diagnostics'].items()}
    return out


def phase_means(suite, label):
    rows = [r for r in suite['normal_repeats'] if r['label'] == label]
    samples = [x for row in rows for x in row['raw_normal_samples']]
    assert samples
    values = {key: statistics.mean(x['exclusive_phase_seconds'][key] for x in samples)
              for key in samples[0]['exclusive_phase_seconds']}
    values['complete_seconds'] = statistics.mean(x['complete_seconds'] for x in samples)
    assert abs(values['complete_seconds'] - sum(v for k, v in values.items() if k != 'complete_seconds')) < 1e-9
    return dict(label=label, samples=len(samples), arithmetic_means_seconds=values)


def main():
    normal = read('normal-review')
    infer = read('inference-stage-review')
    npu_stage = read('medium-stage-review')
    cpu_stage = read('cpu-stage-review')
    qual = read('medium-qualification-review')
    audit = read('resource-audit')
    plan = read('plan')
    assert normal['state'] == 'completed_read_review', normal['state']
    assert infer['state'] == 'completed_review', infer['state']
    assert audit['state'] == 'completed_read_audit', audit['state']
    assert not audit['skipped_active_jobs'] and not audit['audit_findings_jobs']
    expected = read('progress')['jobs']
    assert len(audit['jobs']) == len(expected)
    assert {x['name'] for x in audit['jobs']} == set(expected)
    assert all(x['terminal_audit']['terminal'] and x['terminal_audit']['audit_passed'] for x in audit['jobs'])
    for key in ['cpu_elapsed_within_budget', 'npu_queue_elapsed_within_budget',
                'all_jobs_sampled_rss_within_recorded_bound', 'all_jobs_requested_threads_within_plan',
                'all_jobs_recorded_rss_bound_within_plan',
                'all_observed_npu_allocator_values_within_recorded_limits']:
        assert audit['totals'][key] is True, key
    assert audit['disk']['within_64_gib_budget'] is True
    for key in ['max_observed_npu_peak_allocated_bytes', 'max_observed_npu_peak_reserved_bytes']:
        assert 0 < audit['totals'][key] <= plan['budget']['max_hbm_gib'] * 2**30, key
    for job in audit['jobs']:
        bound = plan['budget']['large_cpu_seconds' if job['name'] == 'ffn-large-cpu-timing-v1' else 'ordinary_seconds']
        assert 0 <= job['service_elapsed_seconds_including_waits'] <= bound, job['name']
    assert qual['locked']
    protected = []
    for rel, digest in plan['protected_sha256'].items():
        item = fingerprint(ROOT / rel)
        assert item['sha256'] == digest, rel
        item['unchanged'] = True
        protected.append(item)
    tree = subprocess.check_output(['git', 'rev-parse', 'HEAD:src/tide'], cwd=ROOT, text=True).strip()
    assert tree == '89f2707e71d989340c12de310797a4c7f4c99f64'
    upstream = json.loads(subprocess.check_output(['python', 'scripts/check_upstream.py'], cwd=ROOT, text=True))
    assert upstream['state'] == 'passed' and upstream['revision'] == plan['math']
    report_names = ['normal-review', 'medium-npu-timing-review', 'medium-stage-review', 'cpu-stage-review',
                    'inference-stage-review', 'medium-qualification-review', 'd768-diagnosis-review',
                    'small-npu-review', 'cpu-combined-review', 'cpu-locality-observation', 'resource-audit']
    reports = [fingerprint(HERE / (n + '.json')) for n in report_names]
    scripts = [fingerprint(HERE / n) for n in ['collect_normal.py', 'collect_inference_stages.py',
               'audit_jobs.py', 'compose_final.py', 'qualification-final-review.md',
               'inference-final-review.md', 'plan.json']]
    payload = dict(
        schema='tide.ffn-execution-devices.1', state='completed_finite_plan',
        observed_at=datetime.now(timezone.utc).isoformat(),
        implementation_commit='05e8338bf62569100d194848689d545f1140afbc', production_tree=tree,
        measurement_tools=['2b8851a7', 'a5853660', '46317a8d'], upstream=upstream,
        protected_user_files=protected, report_inputs=reports, synthesis_inputs=scripts,
        recommendation='Keep gather/rectangular default; W/P/WP have retained cross-path numerical limitations and did not accelerate measured medium training. No W/P/WP 9B measurement.',
        attribution='No identifiable additive split into unstructured organization versus complex arithmetic. Intrusive exclusive stages, separate operator evidence and finite W/P interventions are reported; these are not recoverable normal-wall percentages.',
        qualifications=dict(
            cpu_suite='979 passed, 1 strict xfailed, 1 warning on 05e8338b; 19 tool-only follow-up checks passed on 46317a8d',
            small_cpu='One d64/KV32/F256 independent-FFN fixture: baseline/W/P within original tolerance. WP same parameter field fails in two update comparisons, three coordinate occurrences/two distinct coordinates. Four own fresh-process updates exact.',
            small_npu='Independent-FFN fixture: baseline/W within threshold; P/WP each 11 failed field comparisons in continuing updates, including repeated state representations. Four own restored updates within threshold; only W exact.',
            local_d768='Four original CPU-FP32-literal output comparisons failed at 2e-6/2e-6; original input and parameter VJPs within 6e-6/6e-6. Supplemental diagnosis only reran input VJP; same-NPU paths and FP64 reference within bounds, original failures retained, root cause unresolved.',
            medium_npu='Original baseline inference-grad 14 failed field comparisons; repeated VJP within threshold; steps1-4 completed and checkpoint position512 saved. 20GiB artifact refusal prevented restoration package; W not started.',
            medium_cpu='Baseline/W cross-mode each124 failed field comparisons; W later updates392/533 failed fields. Own fresh-process updates exact. P initial inference/VJP55/62 failures; first-update computation followed by32GiB artifact refusal, no complete durable update result or recovery.',
            standalone_npu_recovery='Existing checkpoint reused with new empty-gradient/seed2718 reference, not unsaved original gradients/RNG. Saved data/source/config/RNG/next forward exact; actual425-Parameter-tensor AdamW step4-to5 within original threshold, nonexact.',
            original_thresholds_unchanged=True,
            detailed_review=fingerprint(HERE / 'medium-qualification-review.json')),
        normal_policies=normal['policies'], memory_boundary_correction=normal['memory_boundary_correction'],
        normal_suites=[normal_suite(s) for s in normal['suites']],
        normal_arithmetic_phase_means=[dict(suite=s['name'], rows=[phase_means(s, row['label']) for row in s['normal_summary']])
                                       for s in normal['suites']],
        cross_cpu_npu_identity=normal['cross_cpu_npu_identity'],
        calibration=normal['cpu16_cpu32_calibration'], cpu_locality=normal['cpu_locality_saved_observation'],
        training_stages=dict(npu=dict(tables=npu_stage['tables'], scope=npu_stage['scope'], findings=npu_stage['findings'], checks=npu_stage['review_checks']),
                             cpu=dict(tables=cpu_stage['tables'], scope=cpu_stage['scope'], findings=cpu_stage['findings'], checks=cpu_stage['checks'])),
        inference_stages=dict(tables=infer['stage_tables'], boundaries=infer['boundaries'], cases=[]),
        resources=subset(audit, ['state', 'budgets', 'totals', 'skipped_active_jobs', 'audit_findings_jobs', 'jobs', 'disk', 'scope']),
        limits=['FP32 only in this finite plan; CPU FP64 only bounded local diagnosis.',
                'One exclusively admitted idle Ascend910; no shared-NPU, CUDA, multi-NPU or 9B training qualification.',
                'B16 medium training and B8 large inference; no new batch search or global optimum.',
                'Fixed32 large Transformer is8.778371328B versus Tide7.268408784B; auxiliary reference does not replace joint A-D.',
                'Normal/stage/operator passes and CPU NUMA placements remain distinct; finite differences and host contention are retained.',
                'Throughput, mechanism checks and restoration do not establish downstream task quality.'])
    recovery_path = ROOT / 'artifacts/ffn-medium-npu-recovery-v1/recovery/result.json'
    recovery = json.loads(recovery_path.read_text())
    update = recovery['restore']['optimizer_update']
    payload['standalone_recovery_details'] = dict(
        input=fingerprint(recovery_path), state=recovery['state'], qualified=recovery['qualified'],
        source=recovery['source'], tolerance=recovery['tolerance'], boundary=recovery['boundary'],
        reference_anchor=subset(recovery['reference_anchor'], ['gradients', 'seed', 'meaning', 'state_position', 'deterministic_algorithms']),
        restore=subset(recovery['restore'], ['state', 'summary', 'pid', 'source_exact', 'config_exact', 'saved_data_exact', 'actual_update']),
        comparisons={k: v['comparison'] for k, v in recovery['restore']['checks'].items()},
        optimizer=dict(checked=update['checked'], active_parameter_tensors=len(update['active_original_parameters']),
                       before_step_range=[min(update['before'].values()), max(update['before'].values())],
                       after_step_range=[min(update['after'].values()), max(update['after'].values())]),
        memory_budget=recovery['memory_budget'], artifact_retention_scope=recovery['artifacts'])
    for c in infer['cases']:
        case = subset(c, ['name', 'state', 'path', 'issues', 'raw_identity', 'source', 'arguments', 'schedule',
                          'parameters', 'initial_parameters', 'input_sha256', 'input_shape', 'shared_input_prefix',
                          'hardware', 'normal', 'boundary', 'launch_identity', 'environment_identity'])
        case['operator_observations'] = subset(c['operators'], ['ffn_weight_bank_gathers', 'packing_from_operands',
                                                               'allocation', 'scalar_reads', 'matrix_totals'])
        case['ffn_matrix_shape_witnesses'] = [row for phase in ('prefill', 'decode')
            for row in [x for x in c['operators']['actual_matrix_shapes']
                        if x['phase'] == phase and 'ffn' in x['path']][:2]]
        q = c['qualification']
        case['qualification'] = {k: v for k, v in q.items() if k != 'diagnostics'}
        payload['inference_stages']['cases'].append(case)
    atomic(DEST, payload)
    print(json.dumps(dict(output=str(DEST.relative_to(ROOT)), **fingerprint(DEST)), ensure_ascii=False))


if __name__ == '__main__':
    main()

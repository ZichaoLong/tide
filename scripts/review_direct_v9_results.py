#!/usr/bin/env python3
"""Audit terminal v9 records and paired validation; never execute a model."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.review_direct_v9 import read, review
from scripts.review_hb_learning import paired
from scripts.review_io import write_review_json
from tide_lab.records import digest, now


def numerical_failures(value):
    if isinstance(value, dict):
        return int(value.get('state') == 'failed') + sum(numerical_failures(v) for v in value.values())
    if isinstance(value, list):
        return sum(numerical_failures(v) for v in value)
    return 0


def audit(root):
    observation = review(root)
    original = observation['campaign']
    recovery = read(root/'runs/v9-recovery-20261112/state.json')
    if recovery.get('state') != 'completed':
        raise ValueError('recovery controller is not terminal/completed')
    cases = {c['id']: c for c in original['phases']['main']['cases']}
    cases.update({c['id']: c for c in recovery['resolved_main_cases']})
    extension = recovery['phases']['extend']['cases']
    if len(cases) != 58 or len(extension) != 12:
        raise ValueError('unexpected main/extension table')

    def relative(path):
        return str(Path(path).resolve().relative_to(root))

    rows, validations = [], {}
    validation_signature = None
    checkpoint_hashes = {}
    data_sha256 = digest(root/'runs/v9-data-r3/data-lock.json')
    for phase, case in [('main', c) for c in cases.values()] + [('extend', c) for c in extension]:
        output = Path(case['output']); train = output/'train'; ident = case['id']
        config = read(Path(case['config'])); result = read(train/'result.json')
        progress = read(train/'progress.json'); val = read(train/'validation-final.json')
        manifest = read(train/'manifest.json'); lens = read(train/'projection.json')
        inference = read(train/'inference.json'); identity = result['identity']
        target = int(case['train_arguments'][case['train_arguments'].index('--until-tokens')+1])
        if (digest(case['config']) != identity['config_sha256'] or manifest['identity'] != identity
                or manifest['config'] != config or identity['data_lock_sha256'] != data_sha256):
            raise ValueError('config/manifest identity mismatch: '+ident)
        if not (result['state'] == 'completed' and progress['tokens'] == target
                and val['state'] == 'completed' and val['tokens'] == 131072
                and math.isfinite(val['nll']) and not val['test_access']
                and val['nll'] == result['final_validation_nll']
                and inference['state'] == 'completed'
                and lens['status'] == 'verified' and lens['readback_matches'] is True):
            raise ValueError('incomplete endpoint: '+ident)
        signature = sorted((r['block_id'], r['document_id'], r['tokens']) for r in val['records'])
        if validation_signature is None:
            validation_signature = signature
        if signature != validation_signature:
            raise ValueError('endpoints use different validation targets: '+ident)
        if identity['source_commit'] != '6e7bd49eb5e35c15445826cac3a9b6ddda75aff4':
            raise ValueError('training source changed: '+ident)
        admission_path = Path(case.get('reuse_admission') or output/'check/result.json')
        recovery_path = Path(case.get('reuse_recovery') or output/'recovery.json')
        admission, probe = read(admission_path), read(recovery_path)
        if not (admission.get('admitted_for_finite_exploration') and probe.get('state') == 'passed'
                and probe['config_sha256'] == identity['config_sha256']):
            raise ValueError('missing admitted mechanism/recovery record: '+ident)
        ancestry = result.get('ancestry')
        if ancestry:
            path = ancestry['checkpoint']
            if path not in checkpoint_hashes:
                checkpoint_hashes[path] = digest(path)
            if checkpoint_hashes[path] != ancestry['sha256']:
                raise ValueError('parent checkpoint changed: '+ident)
        work = val['work']
        ratio = lambda a, b: work.get(a, 0)/work[b] if work.get(b) else None
        structure = config.get('direct', {}).get('plan', {})
        graph = structure.get('graph', {})
        row = dict(id=ident, phase=phase, study=config['study_id'], seed=config['seed'],
            kind=config['kind'], config=relative(case['config']), config_sha256=identity['config_sha256'],
            output=relative(output), run_id=manifest['run_id'], state=result['state'], identity=identity,
            result_sha256=digest(train/'result.json'),
            parameters=result['inventory']['trainable_parameters'], candidate=config.get('candidate'),
            topology_counts=graph.get('counts'), transformer=config.get('transformer'),
            training=config['training'], target_tokens=target, tokens=progress['tokens'], steps=progress['step'],
            dropped_tail_tokens=progress.get('dropped_tail_tokens'),
            training_seconds=result['training_seconds'], elapsed_seconds=result['elapsed_seconds'],
            ancestry=ancestry, parent_hash_verified=bool(ancestry),
            admission=dict(record=relative(admission_path), sha256=digest(admission_path),
                state=admission['state'], axes=admission['axes'],
                strict_failed_checks=numerical_failures(admission['checks'])),
            recovery=dict(record=relative(recovery_path), sha256=digest(recovery_path),
                state=probe['state'], scope=probe['scope']),
            axes=result['axes'], inference=inference, lens=lens,
            validation=dict(record=relative(train/'validation-final.json'), sha256=digest(train/'validation-final.json'),
                nll=val['nll'], ppl=math.exp(val['nll']), tokens=val['tokens'], seconds=val['seconds'],
                body_full=ratio('work/body_full','work/body_node_slots'),
                body_receipt=ratio('work/body_candidates','work/body_node_slots'),
                body_full_per_candidate=ratio('work/body_full','work/body_candidates'),
                whole_full=ratio('work/full','work/node_slots'),
                physical_full=ratio('work/physical_full_rows','work/node_slots'), work=work))
        rows.append(row); validations[(phase, ident)] = train/'validation-final.json'

    groups = defaultdict(list)
    for row in rows:
        groups[(row['phase'], row['id'].rsplit('-s',1)[0])].append(row)
    means = []
    for (phase, name), items in sorted(groups.items()):
        mean = lambda k: statistics.mean(r['validation'][k] for r in items) if items[0]['validation'][k] is not None else None
        means.append(dict(phase=phase, name=name, seeds=[r['seed'] for r in items],
            parameters=items[0]['parameters'], target_tokens=items[0]['tokens'],
            seed_nll=[r['validation']['nll'] for r in items],
            **{k:mean(k) for k in ('nll','body_full','body_receipt','body_full_per_candidate','whole_full','physical_full')}))

    comparisons = []
    def compare(left_phase, left, right_phase, right, seed):
        a, b = f'{left}-s{seed}', f'{right}-s{seed}'
        value = paired(validations[(left_phase,a)], validations[(right_phase,b)])
        if not value['complete_same_blocks'] or value['tokens'] != 131072:
            raise ValueError('paired comparison lacks identical targets')
        value.update(left_phase=left_phase, right_phase=right_phase, left_id=a, right_id=b)
        value['left'] = relative(value['left']); value['right'] = relative(value['right'])
        comparisons.append(value)

    for seed in (11,22):
        for left,right in (
            ('hb88-pool','hb40-pool'),('hb184-pool','hb88-pool'),
            ('hb88-quad','hb40-quad'),('hb184-quad','hb88-quad'),
            ('hb40-wide192-pool','hb88-pool'),('hb40-wide256-pool','hb184-pool'),
            ('hb40-wide256-pool','tf-hb184'),('r64-h2-q3','r64-h2-pool'),
            ('r64-h2-q3','r64-h2-rotate3'),('r32-h3-q10','r32-h2-q10'),
            ('r32-h3-bern','r32-h2-bern'),('tf-hb184','tf-hb88')):
            compare('main',left,'main',right,seed)
        for name in ('r32-h2-q10','r64-h2-q3','r32-h2-bern','r64-h2-bern'):
            compare('main','tree-'+name,'main',name,seed)
        for name in ('hb88-pool','hb88-quad','tf-hb88','r64-h2-q3','r64-h2-bern','r64-h2-pool'):
            compare('extend',name,'main',name,seed)
        for left,right in (('hb88-pool','tf-hb88'),('hb88-quad','tf-hb88'),
                           ('r64-h2-q3','r64-h2-pool'),('r64-h2-q3','r64-h2-bern')):
            compare('extend',left,'extend',right,seed)

    jobs = observation['jobs']; costs = []
    for job in jobs:
        folder = root/'artifacts/jobs'/job['id']; spec = read(folder/'job.json')
        state = read(folder/'state.json'); queue = read(folder/'queue-state.json')
        seconds = state.get('elapsed_seconds',0)
        costs.append(dict(id=job['id'], state=job['state'], service_seconds=seconds,
            allocated_cpu_thread_seconds=seconds*spec['resources']['threads'],
            allocated_npu_seconds=queue.get('elapsed_seconds',0)*queue.get('npu_count',0),
            state_record=relative(folder/'state.json'), state_sha256=digest(folder/'state.json'),
            queue_record=relative(folder/'queue-state.json') if queue else None))
    budget = read(root/'configs/studies/language/direct-v9/budget.json')
    stopped = [r for r in observation['training_stages'] if r['execution'] != 'completed']
    all_terminated = all(j['termination_verified'] for j in jobs)
    return dict(schema='tide-direct-v9-terminal-review-v1', observed_at=now(),
        reviewer_sha256=digest(Path(__file__)), jobs=jobs, all_jobs_terminated=all_terminated,
        job_states=dict(Counter(j['state'] for j in jobs)),
        attempt_states=dict(Counter(s['execution'] for s in observation['training_stages'])),
        resolved_main_states=dict(Counter(r['state'] for r in rows if r['phase']=='main')),
        extension_states=dict(Counter(r['state'] for r in rows if r['phase']=='extend')),
        completed_attempt_lens_verified=sum(s['execution']=='completed' and s['lens_status']=='verified'
            and s['lens_readback_matches'] is True for s in observation['training_stages']),
        original_controller={k:original.get(k) for k in ('state','error','finished_at','deadline_epoch')},
        recovery_controller={k:recovery.get(k) for k in ('state','finished_at','controller_source','training_source','deadline_epoch')},
        extension_decision=recovery['extension_decision'], original_interruptions=stopped,
        old_projection_finalization='docs/evidence/direct-v9-recording-finalization-20261112.json',
        budget=budget, resources=recovery['resources'], allocation_costs=costs,
        allocation_hours=dict(npu=sum(c['allocated_npu_seconds'] for c in costs)/3600,
            cpu_thread=sum(c['allocated_cpu_thread_seconds'] for c in costs)/3600,
            scope='Recorded full service/thread and cooperative NPU lease durations, including startup/checks/logging/wait; not measured busy compute or exclusive device-hours. Review overhead excluded.'),
        data=dict(lock='runs/v9-data-r3/data-lock.json',sha256=data_sha256,
            validation_tokens=131072, validation_documents=len({r[1] for r in validation_signature}),
            confirmation='prepared 262144 targets; not evaluated; pending frozen-finalist review',test='sealed'),
        cases=rows, group_means=means, paired_comparisons=comparisons,
        scope='Read-only terminal training/validation audit. No fresh evaluation, training, test access or scientific-support automation. Selection validation, two seeds except FC16/32; paired document intervals exclude seed and selection uncertainty. Physical Full remains dense masked. Independent confirmation and general foundation continuation remain unchecked.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--write',type=Path)
    args = parser.parse_args(); result = audit(args.root.resolve())
    if args.write:
        write_review_json(args.write,result,root=args.root.resolve())
    print(json.dumps({k:result[k] for k in ('observed_at','job_states','all_jobs_terminated','attempt_states',
        'resolved_main_states','extension_states','completed_attempt_lens_verified','allocation_hours')},ensure_ascii=False,indent=2))

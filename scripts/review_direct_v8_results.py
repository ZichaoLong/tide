#!/usr/bin/env python3
"""Read existing v8 records and summarize terminal results; never run a model."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.review_hb_learning import paired
from scripts.review_io import write_review_json
from tide_lab.records import digest, now


def read(path):
    return json.loads(path.read_text())


def review(root, observation):
    source = read(observation)
    rows, validations = [], {}
    for case in source['cases']:
        output = root / Path(case['record']).parent
        result = read(output / 'result.json')
        config = read(root / case['config'])
        if digest(root / case['config']) != result['identity']['config_sha256']:
            raise ValueError('configuration identity mismatch: ' + case['id'])
        row = dict(case)
        row.update(topology=config.get('candidate', {}).get('recipe', {}).get('topology'),
                   memory=config.get('candidate', {}).get('recipe', {}).get('family'),
                   execution=result['state'],
                   recorded_axes=result.get('axes'),
                   learning_review='incomplete' if result['state'] != 'completed' else 'finite_complete_validation',
                   failed_before_update=result['state'] == 'failed' and case['tokens'] == 0)
        if result['state'] == 'completed':
            path = output / 'validation-final.json'
            val = read(path)
            if not (case['tokens'] == 1048576 and val['tokens'] == 32768 and
                    val['state'] == 'completed' and math.isfinite(val['nll']) and
                    case['recovery'] == 'passed' and case['inference'] == 'completed' and
                    not val['test_access']):
                raise ValueError('incomplete accepted result: ' + case['id'])
            validations[case['id']] = path
            work = val['work']
            candidate_rows, full_rows = work.get('work/candidates', 0), work.get('work/full', 0)
            physical = work.get('work/physical_full_rows', 0)
            row.update(steps=result['progress']['step'],
                       training_seconds=result['training_seconds'], elapsed_seconds=result['elapsed_seconds'],
                       validation_path=str(path.relative_to(root)), validation_sha256=digest(path),
                       final_work=work,
                       logical_full_fraction=full_rows / candidate_rows if candidate_rows else None,
                       logical_full_over_physical=full_rows / physical if physical else None)
            initial = read(output / 'validation-initial.json')
            initial_rows = {(r['block_id'], r['tokens']): r for r in initial['records']}
            final_rows = {(r['block_id'], r['tokens']): r for r in val['records']}
            common = initial_rows.keys() & final_rows.keys()
            tokens = sum(initial_rows[k]['tokens'] for k in common)
            if not tokens:
                raise ValueError('no matched initial/final targets')
            row['matched_initial_final'] = dict(tokens=tokens,
                initial_nll=sum(initial_rows[k]['nll_sum'] for k in common) / tokens,
                final_nll=sum(final_rows[k]['nll_sum'] for k in common) / tokens,
                scope='Identical whole block/target-count intersection; not the full validation set or an uncertainty estimate.')
        if result['state'] == 'failed':
            row['learning_review'] = 'execution_interrupted; no complete final quality comparison'
            row['residual'] = float(result['error'].split('residual ')[1].split(' > ')[0])
            row['original_threshold'] = 2e-5
        rows.append(row)

    lookup = {r['id']: r for r in rows}
    coverage = [r for r in rows if r['kind'] == 'direct_semantic' and r['seed'] == 11 and not r['id'].endswith('-r2')]
    confirmations = [r for r in rows if r['kind'] == 'direct_semantic' and r['seed'] == 22]
    families = defaultdict(list)
    for row in coverage:
        families[row['family']].append(row)
    family_table = []
    for name, candidates in sorted(families.items()):
        item = dict(family=name)
        for topology in ('fc', 'hb'):
            selected = [r for r in candidates if r['topology'] == topology]
            completed = [r for r in selected if r['state'] == 'completed']
            best = min(completed, key=lambda r: r['validation_nll']) if completed else None
            item[topology] = dict(count=len(selected), states=dict(Counter(r['state'] for r in selected)),
                best_id=best['id'] if best else None, best_nll=best['validation_nll'] if best else None)
        family_table.append(item)

    pairs = []
    for second in confirmations:
        first = lookup[second['id'][:-3] + 's11']
        values = [first['validation_nll'], second['validation_nll']]
        pairs.append(dict(id=second['id'][:-4], first_id=first['id'], second_id=second['id'],
            parameters=first['parameters'], states=[first['state'], second['state']],
            validation_nll=values, mean_nll=sum(values) / 2 if all(v is not None for v in values) else None,
            logical_full_fraction=[first.get('logical_full_fraction'), second.get('logical_full_fraction')]))
    comparisons = []
    for seed in (11, 22):
        for left, right in (
            ('fc-send-direct_attention', 'fc-anchor'), ('hb-send-pooled_attention', 'hb-anchor'),
            ('fc-temperature1', 'fc-anchor'), ('hb-temperature1', 'hb-anchor'),
            ('hb-temperature1', 'hb-send-pooled_attention'),
            ('fc-send-direct_attention', 'tf6-gpt2-lr0.001'),
            ('hb-send-pooled_attention', 'tf12-gpt2-lr0.001')):
            a, b = f'{left}-s{seed}', f'{right}-s{seed}'
            value = paired(validations[a], validations[b])
            if not (value['complete_same_blocks'] and value['tokens'] == 32768):
                raise ValueError('comparison changed validation targets')
            value.update(left_id=a, right_id=b)
            for key in ('left', 'right'):
                value[key] = str(Path(value[key]).relative_to(root))
            comparisons.append(value)

    # Equal recorded losses are observations, not proof of general function equivalence.
    aliases = ['fc-unit-mu-s11', 'fc-send-direct_attention-s11', 'fc-send-unit_attention-s11',
               'fc-send-pooled_attention-s11', 'fc-direct-received-s11']
    first_records = read(validations[aliases[0]])['records']
    alias_equal = all(read(validations[c])['records'] == first_records for c in aliases[1:])
    budget = source['resources']
    duration = (datetime.fromisoformat(budget['finished_at']) - datetime.fromisoformat(budget['started_at'])).total_seconds()
    failures = [r for r in rows if r['state'] == 'failed']
    return dict(schema='tide-direct-v8-terminal-review-v1', observed_at=now(),
        observation=str(observation.relative_to(root)), observation_sha256=digest(observation),
        script_sha256=digest(Path(__file__)), all_jobs_terminated=all(j['terminated'] for j in source['jobs']),
        job_count=len(source['jobs']), total_states=dict(Counter(r['state'] for r in rows)),
        coverage_states=dict(Counter(r['state'] for r in coverage)),
        confirmation_states=dict(Counter(r['state'] for r in confirmations)),
        family_table=family_table, seed_pairs=pairs, comparisons=comparisons, cases=rows,
        failures=[{k: r.get(k) for k in ('id', 'tokens', 'error', 'residual', 'failed_before_update', 'memory', 'config', 'record')} for r in failures],
        checks=dict(strict_failure_cases=sum(r['strict_failure_count'] > 0 for r in rows),
            recovery_states=dict(Counter(r['recovery'] for r in rows)),
            lens_verified=sum(r['lens']['status'] == 'verified' and r['lens']['readback_matches'] is True for r in rows),
            completed_final_validation_and_inference=sum(r['state'] == 'completed' for r in rows)),
        coincident_fc_validation=dict(configs=aliases, exactly_equal_final_block_records=alias_equal,
            scope='This evaluated seed/checkpoint/fixture only; not independent evidence or universal equivalence.'),
        costs=dict(supervisor_wall_seconds=duration, finished_at=budget['finished_at'],
            npu_device_hours=budget['usage']['npu_device_seconds'] / 3600,
            cpu_core_hours=budget['usage']['cpu_core_seconds'] / 3600,
            new_disk_gib=budget['headroom']['new_disk_bytes'] / 2**30,
            quota_headroom_gib=budget['headroom']['quota_headroom_bytes'] / 2**30,
            scope='Conservative full service allocation including waits/startup; CPU includes host and 2 core-hour development reserve.'),
        scope='Validation screening and conditional seed replication; no test access, new evaluation, general execution equivalence, real sparse Full execution or scaling certification.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, default=Path('docs/evidence/direct-v8-results.json'))
    args = parser.parse_args()
    root = args.root.resolve()
    result = review(root, root / 'docs/evidence/direct-v8-observations.json')
    write_review_json(args.output, result, root=root)
    print(json.dumps({k: result[k] for k in ('coverage_states', 'confirmation_states', 'checks', 'costs')}, indent=2))

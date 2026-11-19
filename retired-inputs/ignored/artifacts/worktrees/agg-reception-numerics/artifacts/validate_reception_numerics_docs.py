from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT = Path('/mi/data2T/zlong/home/llm/graph-execution-foundation')
summary = json.loads(Path('docs/evidence/reception-numerics-20261118.json').read_text())
checks = []
states = Counter()
restored_states = Counter()
witnesses = Counter()
mode_work = []
update_work = []


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify_comparison(raw, reduced):
    comparison = raw['comparison']
    failed = comparison['threshold_failures']
    assert comparison['exact'] == reduced['exact']
    assert len(failed) == reduced['threshold_failed_fields']
    assert comparison['tolerance'] == summary['tolerance']
    if 'threshold_failure_fields' in reduced:
        assert failed == reduced['threshold_failure_fields']
    if 'threshold_field_list_sha256' in reduced:
        assert sha(json.dumps(failed, separators=(',', ':')).encode()) == reduced['threshold_field_list_sha256']
    if not comparison['exact']:
        rows = comparison['largest_errors']
        assert max([r['scaled_error'] for r in rows] or [0]) == reduced['max_scaled_error']
        if rows:
            assert rows[0] == reduced['largest_scaled_row']
        assert {key: value.get('field') for key, value in raw['first_differences'].items()} == reduced['first_returned_fields']
        assert sum(reduced['failure_families'].values()) == len(failed)
    checks.append(raw)


for run in summary['runs']:
    for record in run['originals'].values():
        content = (ROOT / record['path']).read_bytes()
        assert len(content) == record['bytes'] and sha(content) == record['sha256']
    original = json.loads((ROOT / run['originals']['result']['path']).read_text())
    result = original['result']
    assert original['source'] == run['source']
    assert original['source_dirty'] == run['source_dirty']
    assert original['package_identity']['sha256'] == run['package_sha256']
    assert original['harness_sha256'] == run['harness_sha256']
    assert original['policy'] == run['precision_policy']
    assert result['state'] == run['state']
    assert result['tolerance'] == summary['tolerance']
    assert result['restore_tolerance'] == summary['restore_tolerance']
    assert result['resolved_candidates'] == run['resolved_candidates']
    assert result['candidates'][0]['config'] == run['baseline_config']
    assert result['input_hashes'] == run['workload']['input_hashes']
    assert result['anchor_optimizer'] == run['workload']['anchor_optimizer']
    assert result['fixed_start']['sha256'] == run['workload']['fixed_start_sha256']
    base = result['candidates'][0]
    assert len(result['candidates']) == len(run['candidates'])
    for raw, reduced in zip(result['candidates'], run['candidates']):
        assert raw['name'] == reduced['name'] and raw['state'] == reduced['state']
        states[raw['state']] += 1
        assert set(raw['checks']) == set(reduced['checks'])
        for name, check in raw['checks'].items():
            verify_comparison(check, reduced['checks'][name])
        for name, obs in raw['observations'].items():
            equal = obs['work'] == base['observations'][name]['work']
            assert equal == reduced['baseline_work_equal_by_mode'][name]
            mode_work.append(equal)
            if name.startswith('vjp'):
                assert len(obs['actual_none_parameters']) == reduced['actual_none_parameter_counts'][name]
        assert len(raw['updates']) == len(reduced['updates'])
        for ru, su in zip(raw['updates'], reduced['updates']):
            assert ru['index'] + 1 == su['index']
            equal = ru['work'] == base['updates'][ru['index']]['work']
            assert equal == reduced['baseline_work_equal_by_update'][ru['index']]
            update_work.append(equal)
            if 'vs_baseline' in ru:
                verify_comparison(ru['vs_baseline'], su['comparison'])
            else:
                assert su['comparison'] is None
        witness = raw.get('first_divergence_witness')
        if witness:
            assert witness == reduced['first_internal_witness']
        else:
            assert reduced['first_internal_witness']['state'] == 'not_collected'
        witnesses[reduced['first_internal_witness']['state']] += 1
        rr, sr = raw['restore'], reduced['restore']
        for key in ('state', 'actual_update', 'new_process', 'config_exact', 'source_exact', 'exit_code', 'checkpoint_sha256', 'bundle_sha256'):
            assert rr[key] == sr[key]
        restored_states[rr['state']] += 1
        assert rr['work'] == rr['original_update_work']
        assert sr['original_update_work_equal']
        optimizer = rr['optimizer_update']
        assert len(optimizer['active_original_parameters']) == sr['optimizer_update']['active_parameter_count'] == 364
        assert sorted(set(optimizer['before'].values())) == sr['optimizer_update']['before_steps'] == [4]
        assert sorted(set(optimizer['after'].values())) == sr['optimizer_update']['after_steps'] == [5]
        for name, check in rr['checks'].items():
            verify_comparison(check, sr['checks'][name])

aggregate = summary['aggregate']
assert aggregate['candidate_states'] == dict(states)
assert aggregate['restored_states'] == dict(restored_states)
assert aggregate['internal_witness_states'] == dict(witnesses)
assert aggregate['comparison_count'] == len(checks)
assert aggregate['exact_comparison_count'] == sum(c['comparison']['exact'] for c in checks)
assert aggregate['threshold_failed_comparison_count'] == sum(bool(c['comparison']['threshold_failures']) for c in checks)
assert aggregate['threshold_failed_field_occurrences'] == sum(len(c['comparison']['threshold_failures']) for c in checks)
for flag, expected in aggregate['structural_flag_occurrences'].items():
    assert expected == sum(len(c['comparison'][flag]) for c in checks)
assert aggregate['none_mismatch_checks'] == sum(set(c['screen_actual']['none_fields']) != set(c['screen_expected']['none_fields']) for c in checks)
assert aggregate['unequal_discrete_hash_checks'] == sum(c['screen_actual']['discrete_sha256'] != c['screen_expected']['discrete_sha256'] for c in checks)
assert aggregate['unequal_support_hash_checks'] == sum(c['screen_actual']['support_sha256'] != c['screen_expected']['support_sha256'] for c in checks)
assert aggregate['all_recorded_mode_work_equal'] == all(mode_work)
assert aggregate['all_recorded_update_work_equal'] == all(update_work)


def verify_file_records(value):
    if isinstance(value, dict):
        if {'path', 'sha256', 'bytes'} <= value.keys():
            content = (ROOT / value['path']).read_bytes()
            assert sha(content) == value['sha256'] and len(content) == value['bytes']
        for item in value.values():
            verify_file_records(item)
    elif isinstance(value, list):
        for item in value:
            verify_file_records(item)


verify_file_records(summary)
links_checked = 0
for name in ('docs/evidence/reception-numerics-20261118.md', 'docs/reception-execution.md', 'docs/instances.md'):
    path = Path(name)
    for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)', path.read_text()):
        if re.match(r'[a-zA-Z]+:', target):
            continue
        target = target.split('#', 1)[0]
        if target:
            assert (path.parent / target).exists(), (name, target)
            links_checked += 1
print(json.dumps({'state': 'passed', 'comparisons_verified_directly_against_originals': len(checks),
                  'original_checksums': 'all recorded originals/config/CPU records matched',
                  'local_links_checked': links_checked, 'candidate_states': dict(states)}))

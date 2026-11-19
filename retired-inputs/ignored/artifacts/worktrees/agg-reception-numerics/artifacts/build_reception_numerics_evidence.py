"""Reduce existing records and verify their frozen bytes; no model execution."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path('/mi/data2T/zlong/home/llm/graph-execution-foundation')
DEST = Path(__file__).resolve().parents[1]/'docs/evidence/reception-numerics-20261118.json'
MEMO = ROOT/'artifacts/worktrees/agg-reception-restore/artifacts/npu-reception-paths-memo.json'


def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def file_record(path):
    content = path.read_bytes()
    return dict(path=str(path.relative_to(ROOT)), sha256=hash_bytes(content), bytes=len(content))


def atomic(path, value):
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2); stream.write('\n')
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(parent)
        finally: os.close(parent)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def compact(check, *, full_failure_list=False):
    result = dict(exact=check['exact'], threshold_failed_fields=check['threshold_failure_count'])
    if check['exact']:
        return result
    result.update(max_scaled_error=check['largest_scaled_error'],
                  failure_families=check['threshold_failure_families'],
                  first_returned_fields={key:value.get('field') for key,value in check['first_returned_differences'].items()})
    if check['largest_scaled_rows']:
        result['largest_scaled_row'] = check['largest_scaled_rows'][0]
    if full_failure_list:
        result['threshold_failure_fields'] = check['threshold_failure_fields']
    elif check['threshold_failure_fields']:
        result['threshold_field_list_sha256'] = hash_bytes(json.dumps(check['threshold_failure_fields'],
                                                                    separators=(',', ':')).encode())
    return result


memo = json.loads(MEMO.read_text())
report = dict(schema='tide-reception-numerics-20261118-v1',
    math='4f5b9f20a4030d68ee66ae2d26e34659a5e37234',
    scope='FP32 NPU compact-9 Settle incoming_light_64/128/wide, fixed nonempty optimizer/private-state anchor, three continued updates and candidate-local fresh-process restored update',
    method='read-only reduction of the original JSON; full field lists and coordinate witnesses remain in the hashed originals',
    field_count_definition='tensor-field occurrences, not scalar errors or independent bugs; segment and continued-state fields can repeat related quantities',
    largest_scaled_row_definition='one tensor ranked by max(abs(actual-reference)/(atol+rtol*abs(reference))); its max_abs is not asserted to be the global maximum absolute error',
    tolerance=dict(atol=1e-6,rtol=2e-3), restore_tolerance=dict(atol=1e-6,rtol=2e-3),
    execution_state_is_not_numerical_state='outer status passed/exit 0 accepts finite_deviation; candidate and comparison states below retain it',
    aggregate=memo['aggregate'], runs=[],
    bounded_explanations=memo['interpretations'])

for p in memo['profiles']:
    root = ROOT/Path(p['input_file']).relative_to(ROOT).parent.parent
    launch = json.loads((root/'launch.json').read_text())
    status = json.loads((root/'status.json').read_text())
    queue = json.loads((root/'queue.json').read_text())
    original_bytes = (root/'paths/result.json').read_bytes()
    assert hash_bytes(original_bytes) == p['input_sha256']
    original = json.loads(original_bytes)
    mismatches = []
    for name, expected in p['package_identity']['files'].items():
        frozen = (root/'source/src/tide'/name).read_bytes()
        committed = subprocess.check_output(['git','show',f"{p['source']}:src/tide/{name}"])
        if hash_bytes(frozen) != expected or hash_bytes(committed) != expected:
            mismatches.append(name)
    harness = subprocess.check_output(['git','show',f"{p['source']}:scripts/check_training_paths.py"])
    config_path = root/'source'/p['arguments']['config']
    committed_config = subprocess.check_output(['git','show',f"{p['source']}:{p['arguments']['config']}"])
    assert not mismatches
    assert hash_bytes(harness) == p['harness_sha256']
    assert hash_bytes((root/'source/scripts/check_training_paths.py').read_bytes()) == p['harness_sha256']
    assert config_path.read_bytes() == committed_config
    assert status['state'] == 'passed' and status['exit_code'] == 0
    run = dict(profile=p['profile'], name=root.name, source=p['source'], source_dirty=p['source_dirty'],
        package_sha256=p['package_identity']['sha256'], harness_sha256=p['harness_sha256'],
        byte_verification=dict(package_python_files=len(p['package_identity']['files']),
            package_record_and_frozen_files_match_commit=True, main_harness_matches_commit=True,
            config_matches_commit=True, limitation='Tide Python files, main checker and selected config; not all auxiliary harness files or external toolchains'),
        originals={name:file_record(root/path) for name,path in
            [('result','paths/result.json'),('launch','launch.json'),('status','status.json'),
             ('environment','environment.json'),('queue','queue.json')]},
        config=file_record(config_path), command=launch['arguments']['command'],
        lifecycle=dict(state=status['state'],exit_code=status['exit_code'],started=status['started'],
            finished=status['finished'],elapsed_seconds=status['elapsed'],sampled_peak_rss_bytes=status['peak_rss_bytes']),
        resources=dict(physical_ids=queue.get('physical_ids'),logical_device=p['arguments']['device'],
            threads=p['arguments']['threads'],cpu_affinity=original['cpu_affinity'],identity=p['resource_identity'],
            admission='exclusive cooperative queue; external uncoordinated users not controlled; shared host'),
        precision_policy=p['policy'],torch=p['torch'], state=p['state'],
        baseline_config=p['candidates'][0]['config'],resolved_candidates=p['resolved_candidates'],
        workload=dict(batch=p['arguments']['batch'],length=p['arguments']['length'],chunk=p['arguments']['chunk'],
            steps_after_anchor=p['arguments']['steps'],input_hashes=p['input_hashes'],
            anchor_optimizer=p['anchor_optimizer'],fixed_start_sha256=p['fixed_start'],boundary=p['boundary']),
        candidates=[])
    for c in p['candidates']:
        row=dict(name=c['name'],state=c['state'],
            checks={check['phase']:compact(check,full_failure_list=True) for check in c['checks']},
            updates=[dict(index=u['index'],comparison=None if u['comparison'] is None else compact(u['comparison'])) for u in c['updates']],
            actual_none_parameter_counts={name:len(values) for name,values in c['actual_none_parameters'].items()},
            baseline_work_equal_by_mode=c['baseline_work_equal_by_mode'],
            baseline_work_equal_by_update=[u['baseline_work_equal'] for u in c['updates']],
            first_internal_witness=c['first_internal_witness'],
            restore={name:value for name,value in c['restore'].items() if name!='checks'})
        row['restore']['checks']={v['phase'].removeprefix('restore.'):compact(v,full_failure_list=True) for v in c['restore']['checks']}
        run['candidates'].append(row)
    report['runs'].append(run)

decimal_path=ROOT/'artifacts/worktrees/agg-inference-reception/artifacts/reception-fp32-adamw-witness.json'
decimal=json.loads(decimal_path.read_text())
threads_path=ROOT/'artifacts/worktrees/agg-reception-restore/artifacts/restore-threads-b4/report.json'
threads=json.loads(threads_path.read_text())
report['cpu_context'] = dict(
    independent_formula=dict(source='6907f3d5',tests='tests/test_reception.py',passed=40,dtype='CPU FP64',
        scope='all four profiles, separate/fused QKV, repeated/shared owners, values/all-root VJPs, structural deletion, real zero-mass right derivatives; small explicit fixtures',
        limitation='not an integrated NPU trajectory proof; CPU FP64 is diagnostic'),
    decimal_adamw=dict(original=file_record(decimal_path),source=decimal['source'],production_tree=decimal['production_tree'],
        harness_sha256=decimal['harness_sha256'],configs=decimal['configs'],target_parameter=decimal['target_parameter'],
        threshold=decimal['threshold'],updates=[dict(step=u['step'],coordinate=u['coordinate'],
            original_within_threshold=u['comparison']['within_threshold'],
            selected_coordinate_gradients=[side['gradient'][u['coordinate']] for side in u['after']],actual_delta=u['actual_delta'],
            decimal_predicted_delta=u['decimal_predicted_delta'],delta_absolute_error=u['delta_absolute_error']) for u in decimal['updates']],
        limitation='explains only the selected CPU regional bias coordinates; none of the NPU receive.q/k maximum-failure coordinates was replayed'),
    thread_restore=dict(original=file_record(threads_path),source=threads['source'],production_tree=threads['production_tree'],
        harness_sha256=threads['harness_sha256'],cpu_affinity=threads['cpu_affinity'],seed=threads['seed'],
        original_suite=file_record(ROOT/'artifacts/agg-integration-cpu-v1/pytest.xml'),
        original_suite_result='715 passed / 11 failed; ten diagnostic-alias failures handled separately; one Reception restore exact failure retained',
        minimal_reproduction=file_record(ROOT/'artifacts/worktrees/agg-reception-restore/artifacts/thread-leak-original.xml'),
        fixed_directed_tests=file_record(ROOT/'artifacts/worktrees/agg-reception-restore/artifacts/thread-isolation-fixed.xml'),
        test_fix_commit='1553b301',test_fix_scope='restore Torch thread count between tests; exact assertion and production behavior unchanged; 14 directed tests passed',
        observations=[dict(threads=o['threads'],saved_exact=o['saved']['exact'],
            restored_update_exact=o['restored_update_exact']['exact'],
            restored_update_within_threshold=o['restored_update_tolerance']['within_threshold'],
            restored_update_max_abs=o['restored_update_tolerance']['max_abs'],
            restored_update_differing_fields=[v['field'] for v in o['restored_update_differences']],
            same_owner_repeat_exact=o['same_owner_repeat_exact']['exact'],
            same_owner_repeat_within_threshold=o['same_owner_repeat_tolerance']['within_threshold']) for o in threads['observations']],
        limitation='CPU Timed restore/repeated-backward witness; does not identify the NPU embedding-gradient repeat mechanism'))
report['unresolved'] = [
    'All three first cross-candidate internal witnesses hit 262144 captured elements; no internal NPU causal first divergence was established.',
    'The first returned field in sorted tree order is not the earliest execution boundary.',
    'No component-isolating or scalar NPU replay was run to assign the qkv_mask or combined optimized differences to one operator.',
    'Nine candidate-local restored updates within tolerance do not make cross-candidate continued training equivalent.',
    'No additional BF16, Timed NPU, large-batch, long-sequence, CUDA or task-quality qualification follows from this page.'
]
atomic(DEST, report)
print(json.dumps(dict(output=str(DEST),bytes=DEST.stat().st_size,aggregate=report['aggregate'])))

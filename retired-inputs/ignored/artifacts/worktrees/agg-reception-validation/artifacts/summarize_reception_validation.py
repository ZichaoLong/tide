"""Read existing qualification records; never import Torch or start a workload."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path('/mi/data2T/zlong/home/llm/graph-execution-foundation')
DEST = Path(__file__).resolve().parent / 'reception-validation-summary.json'
NAMES = ('agg-integration-cpu-final-v1', 'agg-installed-final-v1',
         'agg-placement-light64-fp32-v1', 'agg-paths-light64-bf16-v1', 'agg-paths-static-fp32-v1')
FLAGS = ('metadata_failures', 'dtype_differences', 'nonfinite', 'support_differences', 'work_differences')


def digest(content):
    return hashlib.sha256(content).hexdigest()


def original(path):
    content = path.read_bytes()
    return dict(path=str(path.relative_to(ROOT)), sha256=digest(content), bytes=len(content))


def atomic(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(content); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(parent)
        finally: os.close(parent)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def family(name):
    if name.startswith('root.forward.') or name.startswith('root.segments.'):
        return 'forward_state' if '.state.' in name else 'forward_output'
    if name.startswith('root.gradients.'): return 'parameter_gradient'
    if name == 'root.input_gradient': return 'input_gradient'
    if name.startswith('root.parameters.'): return 'updated_parameter'
    if name.startswith('root.optimizer.'): return 'optimizer_state'
    if name.startswith('root.state.'): return 'continued_state'
    return 'other'


def failures(fields):
    return dict(threshold_failed_fields=len(fields), failure_families=dict(Counter(family(v) for v in fields)),
                threshold_field_list_sha256=digest(json.dumps(fields, separators=(',', ':')).encode()))


def path_comparison(check):
    c = check['comparison']
    a, b = check['screen_actual'], check['screen_expected']
    rows = c['largest_errors']
    return dict(exact=c['exact'], tolerance=c['tolerance'], **failures(c['threshold_failures']),
        **{flag: c[flag] for flag in FLAGS},
        discrete_hash_equal=a['discrete_sha256'] == b['discrete_sha256'],
        support_hash_equal=a['support_sha256'] == b['support_sha256'],
        none_fields_only_actual=sorted(set(a['none_fields'])-set(b['none_fields'])),
        none_fields_only_expected=sorted(set(b['none_fields'])-set(a['none_fields'])),
        largest_scaled_row=rows[0] if rows else None,
        first_returned_fields={k:v.get('field') for k,v in check['first_differences'].items()},
        first_support_witness=check['first_differences'].get('support'))


def compact_placement(value):
    if value is None: return None
    result={k:v for k,v in value.items() if k!='predecessor_transport'}
    if 'predecessor_transport' in value:
        rows=value['predecessor_transport']
        result['predecessor_transport_summary']=dict(recorded_entries=len(rows),remote_entries=sum(r['remote'] for r in rows),
            detail='per-layer/source/destination deduplicated node inventory remains in original')
    return result


def installed_comparison(c):
    return dict(**{k:c[k] for k in ('exact','within_threshold','finite','atol','rtol','tensors',
                                   'metadata_failures','nonfinite','support_differences')},
                **failures(c['threshold_failures']), largest_absolute_row=c['largest'][0] if c['largest'] else None)


def placed_comparison(c):
    fields = [r['path'] for r in c['numerical_failures']]
    top = max(c['numerical_failures'], key=lambda r: r.get('max_abs', 0), default=None)
    return dict(**{k:v for k,v in c.items() if k != 'numerical_failures'}, **failures(fields),
                largest_absolute_failure=top)


archives = {}

def verify_source(path, source):
    if source not in archives:
        archive = tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git', 'archive', source])))
        archives[source] = {m.name: archive.extractfile(m).read() for m in archive if m.isfile()}
    files = archives[source]
    mismatches = [name for name, content in files.items() if not (path/name).is_file() or (path/name).read_bytes() != content]
    assert not mismatches, (source, mismatches)
    fp = hashlib.sha256()
    for name, content in sorted(files.items()):
        fp.update(name.encode()); fp.update(b'\0'); fp.update(hashlib.sha256(content).digest())
    return dict(tracked_files=len(files), all_tracked_file_bytes_match_git_archive=True,
                aggregate_name_sha256_digest=fp.hexdigest(),
                production_tree=subprocess.check_output(['git', 'rev-parse', source + ':src/tide'], text=True).strip(),
                scope='read-only verification of Git archive regular files after the run/snapshot; generated and untracked files excluded')


def live_descendants(status):
    alive=[]
    for item in status.get('descendant_identities',[]):
        p=Path('/proc')/str(item['pid'])/'stat'
        try:
            fields=p.read_text().rsplit(')',1)[1].split()
            if int(fields[19]) == item['start_ticks'] and fields[0] != 'Z': alive.append(item)
        except (FileNotFoundError, ProcessLookupError): pass
    return alive


def lifecycle(root):
    launch=json.loads((root/'launch.json').read_text())
    status=json.loads((root/'status.json').read_text())
    terminal=status['state'] not in ('running','submitted','queued')
    result=dict(name=root.name, source=launch['source'], terminal=terminal,
        status={k:v for k,v in status.items() if k!='command'},
        source_verification=verify_source(root/'source', launch['source']),
        originals={name:original(root/name) for name in ('launch.json','status.json','environment.json','stdout.log','stderr.log')},
        requested=launch['arguments'],resource_enforcement=launch['resource_enforcement'],
        resource_identity=launch['environment'].get('TIDE_RESOURCE_IDENTITY'),
        unit=launch['unit'])
    if (root/'queue.json').exists():
        result['queue']=json.loads((root/'queue.json').read_text())
        result['originals']['queue.json']=original(root/'queue.json')
    if terminal:
        result['terminal_stderr']=(root/'stderr.log').read_text()
        proc=subprocess.run(['systemctl','--user','show',launch['unit'],
            '-p','ActiveState','-p','SubState','-p','Result','-p','MainPID','-p','ExecMainStatus'], capture_output=True,text=True)
        result['terminal_unit_observation']=dict(return_code=proc.returncode,output=proc.stdout.strip(),error=proc.stderr.strip())
        result['recorded_descendants_still_alive']=live_descendants(status)
    return result


def paths(root):
    path=root/'paths/result.json'
    if not path.exists(): return None
    raw=json.loads(path.read_text()); r=raw['result']; base=r['candidates'][0]; checks=[]; cs=[]
    for c in r['candidates']:
        row=dict(name=c['name'],state=c['state'],config=c['config'],
            checks={n:path_comparison(v) for n,v in c['checks'].items()},
            actual_none_parameter_counts={n:len(v['actual_none_parameters']) for n,v in c['observations'].items() if n.startswith('vjp')},
            baseline_work_equal_by_mode={n:v['work']==base['observations'][n]['work'] for n,v in c['observations'].items()},
            work_by_mode={n:v['work'] for n,v in c['observations'].items()},updates=[],
            first_internal_witness=c.get('first_divergence_witness',dict(state='not_collected')))
        checks.extend(row['checks'].values())
        for u in c['updates']:
            compare=None if 'vs_baseline' not in u else path_comparison(u['vs_baseline'])
            row['updates'].append(dict(index=u['index']+1,work=u['work'],
                baseline_work_equal=u['work']==base['updates'][u['index']]['work'],comparison=compare))
            if compare is not None: checks.append(compare)
        rr=c.get('restore',{}); o=rr.get('optimizer_update',{})
        row['restore']=dict(**{k:rr.get(k) for k in ('state','actual_update','new_process','config_exact','source_exact','exit_code','checkpoint_sha256','bundle_sha256')},
            checks={n:path_comparison(v) for n,v in rr.get('checks',{}).items()},
            optimizer_update=dict(checked=o.get('checked'),active_parameter_tensors=len(o.get('active_original_parameters',[])),
                before_steps=sorted(set(o.get('before',{}).values())),after_steps=sorted(set(o.get('after',{}).values()))),
            original_update_work_equal=rr.get('work')==rr.get('original_update_work'))
        checks.extend(row['restore']['checks'].values());cs.append(row)
    return dict(original=original(path), source=raw['source'],source_dirty=raw['source_dirty'],
        package_sha256=raw['package_identity']['sha256'],harness_sha256=raw['harness_sha256'],
        torch=raw['torch'],policy=raw['policy'],state=r['state'],arguments=raw['arguments'],
        tolerance=r['tolerance'],restore_tolerance=r['restore_tolerance'],boundary=r['boundary'],
        anchor_optimizer=r['anchor_optimizer'],fixed_start_sha256=r['fixed_start']['sha256'],
        input_hashes=r['input_hashes'],resolved_candidates=r['resolved_candidates'],candidates=cs,
        aggregate=dict(candidate_states=dict(Counter(c['state'] for c in cs)),comparison_count=len(checks),
            exact_comparison_count=sum(v['exact'] for v in checks),
            threshold_failed_comparison_count=sum(bool(v['threshold_failed_fields']) for v in checks),
            threshold_failed_field_occurrences=sum(v['threshold_failed_fields'] for v in checks),
            flags={flag:sum(len(v[flag]) for v in checks) for flag in FLAGS},
            none_mismatch_checks=sum(bool(v['none_fields_only_actual'] or v['none_fields_only_expected']) for v in checks),
            unequal_discrete_hash_checks=sum(not v['discrete_hash_equal'] for v in checks),
            unequal_support_hash_checks=sum(not v['support_hash_equal'] for v in checks),
            all_mode_work_equal=all(all(c['baseline_work_equal_by_mode'].values()) for c in cs),
            all_update_work_equal=all(u['baseline_work_equal'] for c in cs for u in c['updates'])))


def installation_summary(root):
    path=root/'installed/result.json'
    if not path.exists(): return None
    r=json.loads(path.read_text()); cs=[]
    for c in r['cases']:
        rr=c['resume']; ins=rr['installation']
        cs.append(dict(**{k:c[k] for k in ('path','precision','execution_variant','config','parameter_count','state',
                'input_sha256','initial_state_dict_sha256','grad_forward_connected','initial_vjp','checkpoint_config_exact',
                'resume_command','resume_exit_code','checkpoint_sha256','expected_sha256','projection_dtype_passed')},
            placement=compact_placement(c['placement']),initial_region_batch_sizes=c.get('initial_region_batch_sizes'),
            inference_grad_forward=installed_comparison(c['inference_grad_forward']),
            saved_checkpoint_data=installed_comparison(c['saved_checkpoint_data']),updates=c['updates'],
            original_fourth_update=c['original_fourth_update'],
            projection_observation_summary=dict(dtypes=sorted(set(v['output_dtype'] for v in c['actual_dtype_observations'])),
                phases=sorted(set(v['phase'] for v in c['actual_dtype_observations'])),
                worker_observed=any(v['worker_thread'] for v in c['actual_dtype_observations'])),
            resume=dict(**{k:rr[k] for k in ('state','fresh_process','pid','config_exact','actual_fourth_update','none_gradients_preserved','projection_dtype_passed')},
                checks={n:installed_comparison(rr[n]) for n in ('checkpoint_data','next_forward','gradients','updated')},
                installation=dict(state=ins['state'],wheel_sha256=ins['wheel_sha256'],isolated=ins['isolated'],cwd=ins['cwd'],
                    pythonpath_present=ins['pythonpath_present'],installed_python_files=len(ins['installed_python_files']),loaded_modules=len(ins['loaded_modules']),
                    all_installed_match=all(v['matches_wheel_and_source'] for v in ins['installed_python_files']),
                    all_loaded_match=all(v['matches_wheel'] for v in ins['loaded_modules'])))))
    ins=r['final_installation']; complete=root/'installation-complete.json'
    wheel=Path(ins['wheel'])
    assert digest(wheel.read_bytes()) == ins['wheel_sha256']
    with zipfile.ZipFile(wheel) as archive:
        for row in ins['installed_python_files']:
            data=archive.read(row['name'])
            assert digest(data) == row['sha256']
            assert data == Path(row['path']).read_bytes() == (root/'source/src'/row['name']).read_bytes()
    for row in ins['loaded_modules']:
        assert digest(Path(row['path']).read_bytes()) == row['sha256']
    postflight=dict(wheel_sha256_matches_record=True,installed_python_files_matching_wheel_and_frozen_source=len(ins['installed_python_files']),
                   loaded_module_file_bytes_matching_record=len(ins['loaded_modules']))
    stdout=[]
    for line in (root/'stdout.log').read_text().splitlines():
        if line.startswith('{'):
            obj=json.loads(line)
            if 'kind' in obj:
                if obj['kind']=='resolved_settings':
                    obj={k:v for k,v in obj.items() if k not in ('source','recommendation','purpose')}
                stdout.append(obj)
    return dict(original=original(path),state=r['state'],schema_revision=r['schema_revision'],harness_sha256=r['harness_sha256'],
        source_manifest=original(root/'source-manifest.json'),mode_comparison_tolerance=r['mode_comparison_tolerance'],
        completion=None if not complete.exists() else dict(original=original(complete),**json.loads(complete.read_text())),
        installation={k:v for k,v in ins.items() if k not in ('wheel_files','installed_python_files','loaded_modules')},
        postflight_byte_verification=postflight,
        installed_python_file_count=len(ins['installed_python_files']),loaded_module_count=len(ins['loaded_modules']),
        all_installed_match=all(v['matches_wheel_and_source'] for v in ins['installed_python_files']),
        all_loaded_match=all(v['matches_wheel'] for v in ins['loaded_modules']),cases=cs,example_stdout_records=stdout)


def placement_summary(root):
    path=root/'placement.json'
    if not path.exists(): return None
    r=json.loads(path.read_text()); result={k:v for k,v in r.items() if k!='observations'}
    result['original']=original(path); result['observations']=[]
    for c in r['observations']:
        row={k:v for k,v in c.items() if k not in ('comparisons','updates','operator_witness')}
        if 'placement' in row: row['placement']=compact_placement(row['placement'])
        row['comparisons']={n:placed_comparison(v) for n,v in c.get('comparisons',{}).items()}
        row['updates']=[]
        for u in c.get('updates',[]):
            ur={k:v for k,v in u.items() if k!='parameters'}
            if 'parameters' in u: ur['parameters']=placed_comparison(u['parameters'])
            row['updates'].append(ur)
        if 'operator_witness' in c:
            row['operator_witness']=c['operator_witness']
        result['observations'].append(row)
    return result


out=dict(schema='tide-reception-validation-20261118-v1',generated_at_utc=datetime.now(timezone.utc).isoformat(),
    method='read-only reduction of existing artifacts and frozen source byte verification; no model or device execution',
    math='4f5b9f20a4030d68ee66ae2d26e34659a5e37234',
    first_returned_field_scope='lexicographic tree order, not earliest chronological/causal divergence',
    counts='failed tensor-field occurrences can include related quantities repeated across comparisons; not scalar-error or independent-bug counts',
    jobs=[])
for name in NAMES:
    root=ROOT/'artifacts'/name
    row=lifecycle(root)
    if name=='agg-integration-cpu-final-v1':
        path=root/'tests.xml'
        if path.exists():
            xml=ET.parse(path).getroot(); suites=[xml] if xml.tag=='testsuite' else list(xml.iter('testsuite'))
            cases=list(xml.iter('testcase'))
            row['tests']=dict(original=original(path),suites=[s.attrib for s in suites],testcase_count=len(cases),
                failures=[dict(c.attrib,message=f.get('message'),detail=f.text) for c in cases for f in c if f.tag in ('failure','error')],
                skipped=[c.attrib for c in cases if c.find('skipped') is not None],stdout_tail=(root/'stdout.log').read_text()[-5000:])
    elif name=='agg-installed-final-v1': row['result']=installation_summary(root)
    elif name=='agg-placement-light64-fp32-v1': row['result']=placement_summary(root)
    else: row['result']=paths(root)
    out['jobs'].append(row)
out['all_terminal']=all(r['terminal'] for r in out['jobs'])
atomic(DEST,json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(output=str(DEST),bytes=DEST.stat().st_size,all_terminal=out['all_terminal'],
    jobs=[dict(name=r['name'],state=r['status']['state'],inner_state=None if r.get('result') is None else r['result'].get('state'),
               source_files=r['source_verification']['tracked_files']) for r in out['jobs']]),ensure_ascii=False))

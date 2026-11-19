"""Post-retirement byte/availability checks; no model or device use."""
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
def module(name, filename):
    spec=importlib.util.spec_from_file_location(name,HERE/filename)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
cleanup=module('cleanup_verify','cleanup.py')
materializer=module('materializer','frozen-materialize-record.py')
plan=json.loads((HERE/'plan.json').read_text())
result=json.loads((HERE/'result.json').read_text())
assert result['state']=='completed'
terminal=[]
for suffix,base in [('v1',HERE/'attempt-v1'),('v2',HERE)]:
    unit=f'tide-storage-finalization-cleanup-{suffix}.service'
    properties=subprocess.check_output(['systemctl','--user','show',unit,'-p','MainPID','-p','ControlGroup',
        '-p','ActiveState','-p','SubState','-p','Result','-p','ExecMainStatus'],text=True)
    state=json.loads((base/'cleanup-status.json').read_text())
    assert 'MainPID=0\n' in properties and 'ControlGroup=\n' in properties
    assert state['state']==('terminated' if suffix=='v1' else 'passed')
    for row in state['descendant_identities']:
        proc=Path('/proc')/str(row['pid'])/'stat'
        assert not proc.exists() or int(proc.read_text().rsplit(')',1)[1].split()[19]) != row['start_ticks']
    terminal.append(dict(unit=unit,properties=properties,status=state,descendants_terminal=True))
for row in plan['retire']+plan['archives']:
    assert not os.path.lexists(ROOT/row['path'])
for row in plan['protected']:
    cleanup.same(row);assert cleanup.sha(ROOT/row['path'])==row['sha256']
assert {p:cleanup.sha(ROOT/p) for p in plan['user_docs']}==plan['user_docs']
index=json.loads((HERE/'archive-index.json').read_text())
for row in index['records']:
    assert cleanup.sha(ROOT/row['archive_path'])==row['archive_sha256']
scratch=HERE/'record-smoke-v2';scratch.mkdir(exist_ok=False)
choices=[max(index['records'],key=lambda r:r['original_bytes']),
         next(r for r in index['records'] if r['path']=='artifacts/ffn-medium-npu-qualification-v1/qualification/result.json')]
checks=[]
for i,row in enumerate(choices):
    out=scratch/f'{i}.json'
    check=materializer.materialize(HERE/'archive-index.json',row['path'],out,root=ROOT)
    assert cleanup.sha(out)==row['original_sha256']
    if i==1:
        record=json.loads(out.read_text())
        check['original_state']=record.get('state')
        assert check['original_state']=='execution_failed', check['original_state']
    checks.append(check)
# The interrupted attempt created only these exact partial files. Its originals
# and completed archives were verified before the second attempt started.
partial=json.loads((HERE/'interrupted-output-inventory.json').read_text())['partial']
removed=[]
for row in partial:
    p=ROOT/row['path'];s=p.lstat()
    assert p.name.startswith('.archive-') and not p.is_symlink() and s.st_nlink==1
    assert s.st_size==row['bytes'] and s.st_mtime_ns==row['mtime_ns']
    removed.append(dict(**row,sha256=cleanup.sha(p)));p.unlink();cleanup.sync_directory(p.parent)
tracked,trees=cleanup.git_inventory()
assert trees==plan['worktrees'], 'tracked/untracked worktree state changed'
upstream=subprocess.check_output(['python','scripts/check_upstream.py'],cwd=ROOT,text=True)
payload=dict(state='passed',archives_verified=len(index['records']),retired_files_absent=len(plan['retire']),
    original_json_absent=len(plan['archives']),protected_files_verified=len(plan['protected']),
    user_docs_unchanged=True,worktrees_unchanged=True,upstream=upstream,
    real_materialization_checks=checks,removed_interrupted_partials=removed,services=terminal,
    source=plan['source'],scope='Storage identities, exact restoration of JSON, and terminal ownership; no numerical requalification.')
cleanup.atomic(HERE/'verification.json',payload)
print(json.dumps({k:v for k,v in payload.items() if k not in {'services','removed_interrupted_partials'}},indent=2))

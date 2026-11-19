"""Apply the reviewed project-only inventory after all workloads have ended."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

root=Path('/mi/data2T/zlong/home/llm/graph-execution-foundation')
sys.path.insert(0,str(root/'scripts'))
from job import atomic

directory=root/'artifacts/retirement-roadmap-20261118'
inventory=json.loads((directory/'inventory.json').read_text())
assert inventory['state']=='dry_run_only'
assert '--apply' in sys.argv,'explicit reviewed apply flag required'
for name,expected in inventory['user_files'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected['sha256'],name
for run in (root/'artifacts').iterdir():
    if not (run/'launch.json').exists():continue
    status=json.loads((run/'status.json').read_text())
    assert status['state']!='running',run
    launch=json.loads((run/'launch.json').read_text())
    observed=subprocess.check_output(['systemctl','--user','show',launch['unit'],'-p','MainPID','-p','ControlGroup'],text=True)
    assert 'MainPID=0' in observed and '\nControlGroup=\n' in '\n'+observed,(run,observed)
    for identity in status.get('descendant_identities',[]):
        stat=Path('/proc')/str(identity['pid'])/'stat'
        if not stat.exists():continue
        try:fields=stat.read_text().rsplit(')',1)[1].split()
        except FileNotFoundError:continue
        assert int(fields[19])!=identity['start_ticks'] or fields[0]=='Z',(run,identity)

active=[]
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():continue
    try:cwd=(proc/'cwd').resolve()
    except (OSError,RuntimeError):continue
    if str(cwd).startswith(str(root/'artifacts')+'/'):active.append(dict(pid=int(proc.name),cwd=str(cwd)))
assert not active,active

result=dict(state='removing',started=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    inventory_sha256=hashlib.sha256((directory/'inventory.json').read_bytes()).hexdigest(),
    reviewed_candidate_paths=len(inventory['candidates']),removed=[],preserved_user_files=inventory['user_files'],
    retained_witnesses=inventory['retained_witnesses'],
    protected='main private environment, wheelhouse, all small results/configs/commands/lifecycle/logs, Git worktrees/history, user documents and all external/public/consumer paths',
    no_active_artifact_cwd=active)
atomic(directory/'retirement.json',result)
paths=[root/entry['path'] for entry in inventory['candidates']]
assert len(paths)==len(set(paths))
for path in paths:
    assert path.resolve().is_relative_to(root/'artifacts') and not path.is_symlink(),path
    assert not any(path!=other and path.is_relative_to(other) for other in paths),path
for entry,path in zip(inventory['candidates'],paths):
    assert path.exists(),path
    if 'sha256' in entry:
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(2**20),b''):digest.update(chunk)
        assert digest.hexdigest()==entry['sha256'],path
    if path.is_dir():shutil.rmtree(path)
    else:path.unlink()
    result['removed'].append(dict(entry,state='removed'))
    atomic(directory/'retirement.json',result)
for name,expected in inventory['user_files'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected['sha256'],name
result.update(state='completed',finished=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    removed_allocated_bytes=sum(x['allocated_bytes'] for x in result['removed']),
    quota_after=subprocess.check_output(['quota','-s'],text=True),
    artifact_du_after=subprocess.check_output(['du','-x','-B1','-s',str(root/'artifacts')],text=True),
    git_status_after=subprocess.check_output(['git','status','--short','--branch'],cwd=root,text=True))
atomic(directory/'retirement.json',result)
print(json.dumps(dict(state=result['state'],removed_paths=len(result['removed']),removed_allocated_bytes=result['removed_allocated_bytes'])))

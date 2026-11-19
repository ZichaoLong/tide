import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

root=Path('/mi/data2T/zlong/home/llm/graph-execution-foundation')
sys.path.insert(0,str(root/'scripts'))
from job import atomic

def allocated(path):
    seen=set()
    total=0
    paths=[path]+list(path.rglob('*')) if path.is_dir() and not path.is_symlink() else [path]
    for child in paths:
        info=child.lstat()
        identity=(info.st_dev,info.st_ino)
        if identity not in seen:
            seen.add(identity)
            total+=info.st_blocks*512
    return total

def fingerprint(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''):digest.update(chunk)
    return digest.hexdigest()

record=dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),root=str(root),
    state='dry_run_only',quota=subprocess.check_output(['quota','-s'],text=True),
    disk=subprocess.check_output(['df','-h',str(root)],text=True),
    worktrees=subprocess.check_output(['git','worktree','list','--porcelain'],cwd=root,text=True),
    git_status=subprocess.check_output(['git','status','--short','--branch'],cwd=root,text=True),
    rules='Only current-thread project-owned artifacts; preserve all root/user/downstream changes, launch/input identities, small results/logs, original terminal failures and small unresolved witnesses. No public/reference/consumer mutation.',
    user_files={},runs=[],candidates=[],retained_witnesses=[])
for name in ['tide-learning-lab-TL-REQ-004-handoff.md','tide-learning-lab-TL-REQ-005-handoff.md','tide-learning-lab-TL-REQ-006-handoff.md','upstream-lh-semantics-discussion-20261113.md']:
    path=root/'docs'/name
    record['user_files'][str(path.relative_to(root))]=dict(bytes=path.stat().st_size,sha256=fingerprint(path))
archive_cache={}
retained={root/'artifacts/qualification-cpu-v8/tensors'/name for name in ['attention.pt','fc-k.pt','fc-s.pt']}
for directory in sorted((root/'artifacts').iterdir()):
    if not directory.is_dir() or not (directory/'launch.json').exists():continue
    launch=json.loads((directory/'launch.json').read_text())
    status=json.loads((directory/'status.json').read_text()) if (directory/'status.json').exists() else {}
    unit=launch['unit']
    terminal=subprocess.check_output(['systemctl','--user','show',unit,'-p','ActiveState','-p','SubState','-p','MainPID','-p','ControlGroup','-p','Result','-p','ExecMainStatus'],text=True)
    properties=dict(line.split('=',1) for line in terminal.splitlines() if '=' in line)
    completed=properties.get('MainPID')=='0' and properties.get('ActiveState') in ('inactive','failed') and status.get('state') not in ('running',None)
    row=dict(name=directory.name,source=launch['source'],unit=unit,status_state=status.get('state'),terminal=properties,terminal_and_stopped=completed,allocated_bytes=allocated(directory))
    record['runs'].append(row)
    if not completed:continue
    frozen=directory/'source'
    if frozen.is_dir():
        revision=launch['source']
        if revision not in archive_cache:
            payload=subprocess.check_output(['git','archive',revision],cwd=root)
            expected={}
            with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
                for member in archive:
                    if member.isfile():expected[member.name]=hashlib.sha256(archive.extractfile(member).read()).hexdigest()
            archive_cache[revision]=expected
        expected=archive_cache[revision]
        mismatched=[]
        for name,digest in expected.items():
            path=frozen/name
            if not path.is_file() or path.is_symlink() or fingerprint(path)!=digest:mismatched.append(name)
        extras=[]
        for path in frozen.rglob('*'):
            if path.is_dir():continue
            relative=str(path.relative_to(frozen))
            generated=any(part in ('__pycache__','.pytest_cache') or part.endswith('.egg-info') for part in path.relative_to(frozen).parts)
            if relative not in expected and not generated:extras.append(relative)
        row['frozen_archive_check']=dict(tracked_files=len(expected),mismatched=mismatched,unrecognized_extras=extras)
        if not mismatched and not extras:
            record['candidates'].append(dict(path=str(frozen.relative_to(root)),kind='rebuildable_git_archive',allocated_bytes=allocated(frozen),source=revision))
    for path in directory.rglob('*.pt'):
        if 'source' in path.relative_to(directory).parts:continue
        if path in retained or directory.name=='routing-npu-v3':
            record['retained_witnesses'].append(dict(path=str(path.relative_to(root)),allocated_bytes=allocated(path),reason='small bounded unresolved numerical/support witness'))
        else:
            record['candidates'].append(dict(path=str(path.relative_to(root)),kind='rebuildable_tensor_or_checkpoint',allocated_bytes=allocated(path),sha256=fingerprint(path)))
    if directory.name=='wheel-installed-cpu-v15':
        for name in ['installed-venv','build-copy']:
            path=directory/name
            if path.exists():record['candidates'].append(dict(path=str(path.relative_to(root)),kind='project_only_installed_smoke_or_build',allocated_bytes=allocated(path)))
record['candidate_allocated_bytes']=sum(x['allocated_bytes'] for x in record['candidates'])
record['artifacts_allocated_bytes']=allocated(root/'artifacts')
destination=root/'artifacts/retirement-roadmap-20261118'
destination.mkdir(exist_ok=True)
atomic(destination/'inventory.json',record)
print(json.dumps(dict(state=record['state'],runs=len(record['runs']),candidate_paths=len(record['candidates']),candidate_allocated_bytes=record['candidate_allocated_bytes'],artifacts_allocated_bytes=record['artifacts_allocated_bytes'],source_review_exceptions=[x['name'] for x in record['runs'] if x.get('frozen_archive_check',{}).get('mismatched') or x.get('frozen_archive_check',{}).get('unrecognized_extras')]),indent=2))

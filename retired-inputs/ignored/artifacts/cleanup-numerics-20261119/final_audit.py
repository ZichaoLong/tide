"""Final terminal and preservation audit, without running model work."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from job import atomic
from prepare_cleanup import same_guard
from finish_cleanup import terminal

def read(path):
    return json.loads(path.read_text())
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

jobs=[terminal(p.parent) for p in sorted((ROOT/'artifacts').glob('numerics-*/launch.json'))]
for kind in ('retirement','finish','smoke'):
    launch=read(HERE/f'{kind}-launch.json');s=read(HERE/f'{kind}-status.json')
    assert s['state']=='passed', (kind,s['state'])
    props=subprocess.check_output(['systemctl','--user','show',launch['unit'],'-p','MainPID','-p','Result',
        '-p','ExecMainStatus','-p','ControlGroup','-p','ActiveState','-p','SubState'],text=True)
    parsed=dict(row.split('=',1) for row in props.splitlines())
    assert parsed['MainPID']=='0' and not parsed['ControlGroup'],(kind,props)
    for identity in s['descendant_identities']:
        try:
            fields=Path('/proc',str(identity['pid']),'stat').read_text().rsplit(')',1)[1].split()
            assert int(fields[19])!=identity['start_ticks'],(kind,identity)
        except FileNotFoundError:
            pass
    jobs.append(dict(run=str(HERE.relative_to(ROOT))+':'+kind,unit=launch['unit'],source=s['source'],
        state=s['state'],elapsed=s['elapsed'],peak_rss_bytes=s['peak_rss_bytes'],
        descendant_count=len(s['descendant_identities']),live_descendants=[],unit_properties=props,npu_count=0))
terminal_record=dict(state='passed',jobs=jobs,services=len(jobs),descendants=sum(j['descendant_count'] for j in jobs),
    cpu_service_hours=sum(j['elapsed'] for j in jobs if not j['npu_count'])/3600,
    npu_service_card_hours=sum(j['elapsed']*j['npu_count'] for j in jobs)/3600,
    peak_sampled_process_tree_rss_bytes=max(j['peak_rss_bytes'] for j in jobs),
    scope='source-scoped workload lifecycle including queue/dependency waits; not model timing',
    finished_utc=datetime.now(timezone.utc).isoformat())
atomic(HERE/'final-terminal-audit.json',terminal_record)

def retire_smoke():
    if (HERE/'smoke-retirement.json').exists():
        assert read(HERE/'smoke-retirement.json')['state']=='completed'
        assert not (HERE/'smoke-scratch').exists()
        return
    # Retire only this successful installation smoke's own temporary products.
    # Symlink targets (the public interpreter) are never followed or removed.
    smoke=read(HERE/'installed-smoke.json');assert smoke['state']=='passed'
    scratch=HERE/'smoke-scratch'
    rows=[]
    for p in sorted(scratch.rglob('*')):
        if p.is_symlink() or p.is_file():
            s=p.lstat();row=dict(path=str(p.relative_to(ROOT)),allocated_bytes=s.st_blocks*512,bytes=s.st_size,
                                type='symlink' if p.is_symlink() else 'file')
            if p.is_symlink():row['target']=os.readlink(p)
            else:row['sha256']=sha(p)
            rows.append(row)
    atomic(HERE/'smoke-retirement-plan.json',dict(state='reviewed_successful_temporary_only',files=rows,
        scope='exact project-owned smoke-scratch; no external symlink targets'))
    for row in rows:(ROOT/row['path']).unlink()
    for p in sorted(scratch.rglob('*'),key=lambda p:len(p.parts),reverse=True):
        if p.is_dir():p.rmdir()
    scratch.rmdir()
    fd=os.open(HERE,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
    atomic(HERE/'smoke-retirement.json',dict(state='completed',files=len(rows),
        allocated_bytes=sum(r['allocated_bytes'] for r in rows),plan_sha256=sha(HERE/'smoke-retirement-plan.json')))
    

retire_smoke()

plan=read(HERE/'cleanup-plan.json');deleted=read(HERE/'update2-retirement-result.json')['removed']
count,allocated=0,0
for row in plan['protected']['retained_tensors']:
    if row['path'] in deleted:
        assert not (ROOT/row['path']).exists()
    else:
        same_guard(row['path'],row['stat_guard']);count+=1;allocated+=row['stat_guard']['allocated_bytes']
for row in plan['protected']['failed_ffn_profiler']:
    same_guard(row['path'],row['stat_guard'])
for row in plan['retained_records']:
    same_guard(row['path'],row['stat_guard']);assert sha(ROOT/row['path'])==row['sha256']
for row in plan['preserved_records']:
    assert sha(ROOT/row['preserved_path'])==row['sha256']
for row in plan['candidates']:
    assert not os.path.lexists(ROOT/row['path'])
users=read(HERE/'plan.json')['preserve']['user_files_sha256']
assert all(sha(ROOT/p)==h for p,h in users.items())
fixed=subprocess.check_output([sys.executable,str(ROOT/'scripts/check_upstream.py')],cwd=ROOT,text=True)
assert json.loads(fixed)['state']=='passed'
repo=int(subprocess.check_output(['du','-sx','-B1',str(ROOT)],text=True).split()[0])
account_scan=subprocess.run(['du','-sx','-B1','/mi/data2T/zlong/home'],text=True,capture_output=True)
account=int(account_scan.stdout.split()[0])
quota_exact=subprocess.check_output(['quota','-w','-v'],text=True)
quota_kib=int(next(line for line in quota_exact.splitlines() if line.startswith('/dev/')).split()[1])
quota=subprocess.check_output(['quota','-s'],text=True)
new=sum(p.lstat().st_blocks*512 for root in [HERE]+list((ROOT/'artifacts').glob('numerics-*'))
        for p in root.rglob('*') if p.is_file() or p.is_symlink())
record=dict(state='passed',repository_allocated_bytes=repo,account_home_readable_allocated_bytes=account,account_home_scan_exit=account_scan.returncode,
    account_home_scan_stderr=account_scan.stderr,account_quota_used_bytes=quota_kib*1024,quota_exact=quota_exact,quota=quota,
    preserved_old_tensor_files=count,preserved_old_tensor_allocated_bytes=allocated,
    preserved_failed_profiler_files=len(plan['protected']['failed_ffn_profiler']),
    preserved_original_small_records=len(plan['retained_records']),preserved_small_copies=len(plan['preserved_records']),
    user_files_sha256=users,fixed_math_check=json.loads(fixed),new_artifacts_remaining_allocated_bytes=new,
    prefinish_inventory=read(HERE/'prefinish-inventory.json'),
    observed_utc=datetime.now(timezone.utc).isoformat(),
    external_boundaries='main .git/math, registered/downstream worktrees, environments and unique untracked source retained')
atomic(HERE/'final-inventory.json',record)
print(json.dumps(dict(services=terminal_record['services'],descendants=terminal_record['descendants'],
                     repository_gib=repo/2**30,account_quota_gib=quota_kib/2**20,preserved_tensors=count)))

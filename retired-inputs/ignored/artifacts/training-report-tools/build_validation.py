from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.job import atomic

def read(path): return json.loads((ROOT/path).read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def pointer(path):
    path=ROOT/path
    return dict(path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size)
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
def inspect_archive(source, path):
    rows=subprocess.check_output(['git','ls-tree','-r','-z',source],cwd=ROOT).split(b'\0')
    mismatches=[];count=0
    for row in rows:
        if not row:continue
        meta,name=row.split(b'\t',1); mode,kind,digest=meta.split();name=name.decode()
        if kind!=b'blob':continue
        count+=1;p=path/name
        if not p.exists():mismatches.append(name+':missing');continue
        data=p.read_bytes()
        actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest().encode()
        if actual!=digest:mismatches.append(name)
    return dict(source=source,path=str(path.relative_to(ROOT)),tracked_blobs=count,mismatches=mismatches)
def process_live(identity):
    p=Path('/proc')/str(identity['pid'])/'stat'
    try:
        rest=p.read_text().rsplit(')',1)[1].split()
        return int(rest[19])==identity['start_ticks']
    except (OSError,ValueError,IndexError):return False

record=dict(schema='tide-training-validation-v1',recorded_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    review_source=git('rev-parse','HEAD'),math=json.loads(subprocess.check_output([sys.executable,'scripts/check_upstream.py'],cwd=ROOT,text=True)),
    production_tree=git('rev-parse','HEAD:src/tide'),source_scope='production package, full suite, later directed checks and installed checks retain distinct frozen identities',
    tests=[],user_files=[],lifecycle=[],frozen_archives=[],installed={})
for name,count in [('training-cpu-suite',625),('training-final-targeted',83)]:
    base=Path('artifacts')/name
    state=read(base/'status.json')
    assert state['state']=='passed'
    assert f'{count} passed' in (ROOT/base/'stdout.log').read_text()
    record['tests'].append(dict(name=name,source=state['source'],count=count,status=pointer(base/'status.json'),
        stdout=pointer(base/'stdout.log'),stderr=pointer(base/'stderr.log'),launch=pointer(base/'launch.json'),command=state['command']))
record['completion_sync_regression']=read(Path('artifacts/training-report-tools/completion-sync-check.json'))
record['completion_sync_regression']['source']='f8235fe48ff794f0e5ac60bf7565d7f918479e6c'
record['support_catalog']=dict(identity=pointer(Path('artifacts/training-report-tools/support-catalog.json')),data=read(Path('artifacts/training-report-tools/support-catalog.json')))
old=read(Path('docs/evidence/capacity-validation-20261118.json'))
for item in old['user_files']:
    actual=pointer(Path(item['path']));actual['matches_prior_sha256']=actual['sha256']==item['sha256']
    assert actual['matches_prior_sha256'];record['user_files'].append(actual)
for status_path in sorted((ROOT/'artifacts').glob('training-*/status.json')):
    base=status_path.parent;status=json.loads(status_path.read_text());launch=read(base/'launch.json')
    assert status['state'] in ('passed','failed'),(base.name,status['state'])
    live=[x for x in status.get('descendant_identities',[]) if process_live(x)]
    assert not live,(base.name,live)
    observed=subprocess.run(['systemctl','--user','show',launch['unit']+'.service','-p','LoadState','-p','ActiveState','-p','SubState','-p','MainPID','-p','ControlGroup'],capture_output=True,text=True)
    props=dict(line.split('=',1) for line in observed.stdout.splitlines() if '=' in line)
    assert props.get('MainPID','0')=='0' and props.get('ActiveState','inactive')!='active',props
    record['lifecycle'].append(dict(name=base.name,status=pointer(status_path),state=status['state'],exit_code=status.get('exit_code'),
        source=status['source'],unit=launch['unit'],unit_observation=props,tracked_descendants=len(status.get('descendant_identities',[])),live_descendants=live))
    archive=inspect_archive(status['source'],Path(launch['source_path']))
    assert not archive['mismatches'],archive
    record['frozen_archives'].append(archive)
base=Path('artifacts/training-installed')
installed=read(base/'installed.json'); assert installed['state']=='completed'
checks=read(base/'results/result.json')
record['installed']=dict(identity=pointer(base/'installed.json'),commands=installed['commands'],source=installed['source'],
    production_tree=git('rev-parse',installed['source']+':src/tide'),result=pointer(base/'results/result.json'),
    state=checks['state'],wheel=installed['wheel'],source_manifest=pointer(base/'source-manifest.json'),
    driver=pointer(Path('artifacts/training-report-tools/install_validate.py')),
    driver_source=(ROOT/'artifacts/training-report-tools/install_validate.py').read_text(),
    source_unchanged=installed['source_unchanged'],source_files=installed['source_files'],
    initial_installation=checks['initial_installation'],final_installation=checks['final_installation'],cases=checks['cases'],
    entry_logs=[pointer(base/name) for name in ['entry-default.stdout','entry-training.stdout','entry-resume.stdout']])
assert record['production_tree']==record['installed']['production_tree']==git('rev-parse','ae247dee:src/tide')
record['preflights']=[pointer(Path('artifacts/training-preflight-20261118')/name) for name in ['preflight.json','resumed-preflight.json']]
record['artifact_policy']='No deletions; original failures, user files, frozen archives, checkpoints, diagnostic payloads, wheel and private environment retained. No downstream/public environment mutation.'
atomic(ROOT/'docs/evidence/training-validation-20261118.json',record)
print(json.dumps(dict(jobs=len(record['lifecycle']),archives=len(record['frozen_archives']),tests=[r['count'] for r in record['tests']],installed=checks['state'],user_files=len(record['user_files']))))

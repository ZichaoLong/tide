"""One-shot, explicit project retirement inventory/preservation. No model imports."""
import base64
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path('/mi/data2T/zlong/home/llm/graph-execution-foundation')
CONTROL=Path(__file__).resolve().parent
BASE='728d19a470d16d392d5dd6a2452161d1a3e9959e'
ENV={**os.environ,'GIT_OPTIONAL_LOCKS':'0'}
SHM=['gef-aggregate-tensor-development-20261112','gef-column-scale-20261111',
 'gef-fc-batch-full-20261112','gef-kv-prepare-development-20261112',
 'gef-kv-tensor-development-20261112','gef-p1p6-20261112',
 'gef-read-region-owner-20261112','gef-region-tensor-20261112',
 'gef-region-transport-development-20261112','gef-resident-kv-owner-20261112',
 'gef-resident-resources-20261112','gef-resident-services-20261112',
 'gef-resident-services-context-20261112','gef-resident-tile-20261112',
 'gef-rewrite-20261112','gef-source-bank-20261112','gef-source-planning-20261112',
 'gef-standalone-block-adapter-20261112','gef-tensor-rewrite-20261112',
 'zlong-gef-block-206bc9b','zlong-gef-block-26ab27d','zlong-gef-block-5ff3ce2',
 'zlong-gef-block-dev-20261111','zlong-gef-block-profiles',
 'zlong-gef-fc-20261111-15d391f']
OUTSIDE=ROOT.parent/'gef-lm-experiments/20261102'
DOWNSTREAM=ROOT.parent/'tide-learning-lab'
ROOTS=[ROOT/n for n in ('artifacts','.local','.venv-instances','.pytest_cache',
 'python','cpp','cmake','tools','templates','.torch-portability')]
ROOTS += [Path('/dev/shm')/n for n in SHM]
ROOTS += [OUTSIDE/n for n in ('source','engines','env','install','jobs','recovery','runs','site')]
USER_FILES=[ROOT/'docs'/n for n in ('tide-learning-lab-TL-REQ-004-handoff.md',
 'tide-learning-lab-TL-REQ-005-handoff.md','tide-learning-lab-TL-REQ-006-handoff.md',
 'upstream-lh-semantics-discussion-20261113.md')]

def command(args,cwd=ROOT):
    return subprocess.check_output(args,cwd=cwd,env=ENV)

def save(name,value):
    p=CONTROL/name
    temporary=p.with_suffix(p.suffix+'.tmp')
    with temporary.open('w') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temporary,p)

def inside(p,roots=ROOTS):
    return any(p==r or p.is_relative_to(r) for r in roots)

def worktrees():
    out=[]
    for block in command(['git','worktree','list','--porcelain']).decode().strip().split('\n\n'):
        row=dict(line.split(' ',1) if ' ' in line else (line,True) for line in block.splitlines())
        out.append(row)
    return out

def dependencies():
    hits=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            targets=[('cwd',proc/'cwd'),('exe',proc/'exe')]
            targets += [('fd',p) for p in (proc/'fd').iterdir()]
            for kind,p in targets:
                try:
                    value=os.readlink(p).removesuffix(' (deleted)')
                    if value.startswith('/') and inside(Path(value)):
                        hits.append({'pid':int(proc.name),'kind':kind,'path':value})
                except OSError:pass
            for line in (proc/'maps').read_text().splitlines():
                fields=line.split(None,5)
                if len(fields)==6 and fields[5].startswith('/') and inside(Path(fields[5])):
                    hits.append({'pid':int(proc.name),'kind':'map','path':fields[5]})
        except (PermissionError,FileNotFoundError,ProcessLookupError):pass
    return hits

def units():
    rows=json.loads(command(['systemctl','--user','list-units','--all','--output=json']).decode())
    selected=[]
    for row in rows:
        name=row['unit'];description=row.get('description','')
        if name.startswith('gef-governance-'):continue
        if str(DOWNSTREAM) in description or name.startswith('tide-lab-'):continue
        if (str(ROOT) in description or str(OUTSIDE) in description or name.startswith('gef-')):
            props=command(['systemctl','--user','show',name,'-p','Id','-p','ActiveState',
                '-p','SubState','-p','Result','-p','ExecMainStatus','-p','MainPID',
                '-p','WorkingDirectory','-p','FragmentPath','-p','Transient']).decode()
            data=dict(line.split('=',1) for line in props.splitlines() if '=' in line)
            if str(DOWNSTREAM) in data.get('WorkingDirectory',''):continue
            data['description']=description
            selected.append(data)
    return selected

def scan():
    assert not (CONTROL/'inventory.json').exists(),'new observation required'
    tracked=set(command(['git','ls-files','-z']).decode().split('\0'))
    for r in ROOTS:
        if r.is_relative_to(ROOT):
            assert not any((ROOT/p).is_relative_to(r) for p in tracked if p),r
    info={'base':BASE,'started':time.time(),'protected':[str(ROOT),str(DOWNSTREAM),str(OUTSIDE/'models')],
          'roots':[],'worktrees':[],'user_files':{},'units':units(),'dependencies':dependencies()}
    assert not info['dependencies'],info['dependencies']
    assert not any(int(u['MainPID']) or u['ActiveState'] not in ('failed','inactive') for u in info['units'])
    for p in USER_FILES:info['user_files'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    info['downstream_head']=command(['git','rev-parse','HEAD'],DOWNSTREAM).decode().strip()
    info['downstream_status']=command(['git','status','--porcelain=v1','-uall'],DOWNSTREAM).decode()
    info['models_links']={str(p):os.readlink(p) for p in (OUTSIDE/'models').iterdir() if p.is_symlink()}
    for r in ROOTS:
        if not r.exists():continue
        assert not r.is_symlink(),r
        s=r.stat()
        size=int(command(['du','-s','-x','-B1',str(r)]).split()[0])
        info['roots'].append({'path':str(r),'device':s.st_dev,'inode':s.st_ino,'allocated_bytes':size,
                              'decision':'retire after preservation; project-owned terminal material'})
    with gzip.open(CONTROL/'dirty-worktrees.jsonl.gz','wt') as out:
        for w in worktrees():
            p=Path(w['worktree'])
            if not inside(p):continue
            assert subprocess.run(['git','merge-base','--is-ancestor',w['HEAD'],BASE],cwd=ROOT).returncode==0
            status=command(['git','status','--porcelain=v1','-uall'],p).decode()
            row={**w,'status':status};info['worktrees'].append(row)
            if not status:continue
            patch=command(['git','diff','HEAD','--binary'],p)
            untracked=command(['git','ls-files','--others','--exclude-standard','-z'],p).decode().split('\0')
            files=[]
            for rel in filter(None,untracked):
                f=p/rel
                if f.is_symlink():files.append({'path':rel,'symlink':os.readlink(f)});continue
                data=f.read_bytes();files.append({'path':rel,'sha256':hashlib.sha256(data).hexdigest(),
                    'data_base64':base64.b64encode(data).decode(),'mode':f.stat().st_mode})
            out.write(json.dumps({**row,'patch_base64':base64.b64encode(patch).decode(),'untracked':files})+'\n')
    reachable={line.split(b' ',1)[0].decode() for line in command(['git','rev-list','--objects',BASE]).splitlines()}
    seen=set();counts=Counter();large=[]
    skip={'.git','site-packages','dist-packages','__pycache__','node_modules','CMakeFiles','_deps','kernel_meta'}
    extensions={'.py','.cpp','.cc','.c','.h','.hpp','.cu','.cuh','.sh','.cmake','.md','.toml','.yaml','.yml','.json','.jsonl'}
    with gzip.open(CONTROL/'small-records.jsonl.gz','wt',compresslevel=6) as out:
        for r in ROOTS:
            if not r.exists():continue
            for directory,dirs,files in os.walk(r,followlinks=False):
                dirs[:]=[d for d in dirs if d not in skip and not (Path(directory)/d).is_symlink()]
                for name in files:
                    p=Path(directory)/name
                    if p.is_symlink():
                        out.write(json.dumps({'path':str(p),'symlink':os.readlink(p)})+'\n');counts['symlinks']+=1;continue
                    suffix=p.suffix.lower()
                    if suffix not in extensions and name not in ('CMakeLists.txt','run.log','launch.json','status.json'):continue
                    s=p.stat()
                    if s.st_size>4*2**20:
                        large.append({'path':str(p),'bytes':s.st_size,'reason':'large generated/raw record; not byte-retained'})
                        counts['large_record_skipped']+=1;continue
                    data=p.read_bytes()
                    if name=='run.log' and len(data)>16384:data=data[-16384:]
                    blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
                    row={'path':str(p),'bytes':s.st_size,'mode':s.st_mode,'git_blob':blob}
                    if blob in reachable:row['retained']='reachable_git_blob';counts['in_git']+=1
                    elif blob in seen:row['retained']='bundle_duplicate';counts['duplicate']+=1
                    else:
                        row['retained']='bundle_bytes';row['data_base64']=base64.b64encode(data).decode()
                        seen.add(blob);counts['unique_bytes']+=len(data);counts['unique_files']+=1
                    if name=='run.log':row['log_tail_only']=s.st_size>len(data)
                    out.write(json.dumps(row,ensure_ascii=False)+'\n');counts['records']+=1
    info['preservation_counts']=dict(counts);info['large_not_byte_retained']=large
    info['finished']=time.time()
    info['preservation_files']={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in (CONTROL/'dirty-worktrees.jsonl.gz',CONTROL/'small-records.jsonl.gz')}
    save('inventory.json',info)
    print(json.dumps({'roots':len(info['roots']),'worktrees':len(info['worktrees']),
                     'bytes':sum(r['allocated_bytes'] for r in info['roots']),
                     'preservation':info['preservation_files'],'counts':dict(counts)},indent=2),flush=True)

def remove():
    info=json.loads((CONTROL/'inventory.json').read_text())
    assert (CONTROL/'preservation-commit.txt').exists(),'preservation must be committed and verified first'
    commit=(CONTROL/'preservation-commit.txt').read_text().strip()
    assert command(['git','cat-file','-t',commit]).strip()==b'commit'
    for name,identity in info['preservation_files'].items():
        data=command(['git','show',f'{commit}:retirement-preservation/{name}'])
        assert hashlib.sha256(data).hexdigest()==identity['sha256']
    hits=dependencies();save('predelete-dependencies.json',hits);assert not hits,hits
    for u in units():assert int(u['MainPID'])==0 and u['ActiveState'] in ('failed','inactive'),u
    for r in info['roots']:
        p=Path(r['path']);s=p.stat();assert (s.st_dev,s.st_ino)==(r['device'],r['inode']),p
    current={w['worktree']:w for w in worktrees()}
    for w in info['worktrees']:
        p=Path(w['worktree']);assert current[str(p)]['HEAD']==w['HEAD']
        assert command(['git','status','--porcelain=v1','-uall'],p).decode()==w['status'],p
    record={'preservation_commit':commit,'removed':[],'removed_worktrees':[],'started':time.time()};save('deletion.json',record)
    for w in info['worktrees']:
        p=Path(w['worktree']);assert inside(p)
        subprocess.run(['git','worktree','remove','--force',str(p)],cwd=ROOT,check=True)
        record['removed_worktrees'].append(str(p));save('deletion.json',record)
    for r in info['roots']:
        p=Path(r['path']);assert inside(p) and p!=ROOT and not p.is_symlink()
        if p.exists():shutil.rmtree(p)
        record['removed'].append(r);save('deletion.json',record)
        print('retired',p,flush=True)
    # Terminal transient units are released only after their failure identities are preserved.
    released=[]
    for u in info['units']:
        if u.get('Transient')=='yes' and u['ActiveState']=='failed':
            subprocess.run(['systemctl','--user','reset-failed',u['Id']],check=True)
            released.append(u['Id'])
    for p,h in info['user_files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    assert command(['git','rev-parse','HEAD'],DOWNSTREAM).decode().strip()==info['downstream_head']
    assert command(['git','status','--porcelain=v1','-uall'],DOWNSTREAM).decode()==info['downstream_status']
    assert {str(p):os.readlink(p) for p in (OUTSIDE/'models').iterdir() if p.is_symlink()}==info['models_links']
    record.update(finished=time.time(),released_terminal_units=released,remaining_worktrees=worktrees())
    save('deletion.json',record)

if __name__=='__main__':
    {'scan':scan,'remove':remove}[sys.argv[1]]()

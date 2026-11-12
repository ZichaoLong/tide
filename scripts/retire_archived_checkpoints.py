#!/usr/bin/env python3
"""Finite, manifest-driven retirement of completed v6-v8 and old GEF checkpoints.

Never visits v9 for mutation. All byte identities are durable before mutation;
retained aliases are hardlinked only after a full SHA256 equality check.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.review_io import write_review_json
from scripts.retire_priority_artifacts import digest
from tide_lab.records import now
from tide_lab.supervisor import identity_alive

SCHEMA = 'tide-historical-checkpoint-retirement-v1'
PROTECTED_CASES = {
    'v6-cover-hb-direct-received', 'v6-cover-hb-send-constrained-bern',
    'v6-cover-hb-send-regional_total-bern',
}


def read(path):
    return json.loads(Path(path).read_text())


def file_row(path, base):
    s = path.lstat()
    return dict(path=str(path.relative_to(base)), logical_bytes=s.st_size,
                allocated_bytes=s.st_blocks * 512, mtime_ns=s.st_mtime_ns,
                inode=s.st_ino, device=s.st_dev, mode=s.st_mode)


def checked(base, row, *, single_link=True):
    base = Path(base).absolute()
    rel = Path(row['path'])
    if rel.is_absolute() or '..' in rel.parts or not rel.parts:
        raise ValueError('unsafe relative path')
    p = base / rel
    if p.resolve() != p or base.resolve() != base:
        raise ValueError('symlink in artifact path')
    s = p.lstat()
    if not stat.S_ISREG(s.st_mode) or (single_link and s.st_nlink != 1):
        raise ValueError('expected a regular artifact with the declared link policy')
    actual = (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    expected = tuple(row[k] for k in ('device', 'inode', 'logical_bytes', 'mtime_ns'))
    if actual != expected:
        raise ValueError(f'artifact changed: {p}')
    return p


def scope_check(row):
    parts = Path(row['path']).parts
    if row['root_id'] == 'tide':
        if (len(parts) != 4 or parts[0] != 'runs' or parts[2] != 'train'
                or not re.fullmatch(r'v[678]-.+', parts[1])
                or parts[1] in PROTECTED_CASES
                or not re.fullmatch(r'(latest|previous|tokens-\d+)\.pt', parts[3])):
            raise ValueError('outside completed historical training checkpoint scope')
    elif row['root_id'] == 'gef':
        if (len(parts) != 6 or parts[:2] != ('20261102', 'runs') or parts[4] != 'run'
                or parts[5] not in {'checkpoint.pt', 'state.bin', 'expected-state.bin', 'restored-state.bin'}
                or row['action'] != 'delete'):
            raise ValueError('outside historical GEF benchmark scope')
    else:
        raise ValueError('unapproved root')


def retained_selection(root):
    # Named scientific retention, not an automatic winner-only policy.
    keep = {}
    def add(case, reason, names=('latest.pt',)):
        for name in names:
            p = root/'runs'/case/'train'/name
            if not p.is_file():
                raise FileNotFoundError(p)
            keep[str(p.relative_to(root))] = reason
    for topo in ('fc', 'hb'):
        for recipe in ('anchor', 'send-proportional', 'send-pooled_attention'):
            add(f'v6-cover-{topo}-{recipe}', 'original foundation small-width anchor and sending counterexample')
    for size in ('5m', '10m', '19m'):
        add(f'v6-scale-transformer-{size}-retry1', 'original foundation scaling Transformer controls')
    add('v6-scale-transformer-10m-retry1', 'matched 262K control for the original HB diagnosis', ('tokens-262144.pt',))
    for p in sorted((root/'runs').glob('v7-main-*/train/latest.pt')):
        add(p.parent.parent.name, 'all ten paired-seed direct graph / Transformer 4M endpoints')
    for topo in ('fc', 'hb'):
        add(f'v8-anchor-{topo}-anchor-s11', 'v8 anchor paired with selected semantic candidates')
        add(f'v8-confirm-{topo}-anchor-s22', 'v8 anchor paired with selected semantic candidates')
        for prefix, seed in (('cover', 11), ('confirm', 22)):
            add(f'v8-{prefix}-{topo}-temperature1-s{seed}', 'paired-seed quadratic quality/activity candidate')
    for prefix, seed in (('cover', 11), ('confirm', 22)):
        add(f'v8-{prefix}-fc-send-direct_attention-s{seed}', 'paired-seed FC quality reference')
        add(f'v8-{prefix}-hb-send-pooled_attention-s{seed}', 'paired-seed HB quality reference')
        add(f'v8-{prefix}-fc-ssd-mem-off-s{seed}', 'paired-seed negative memory ablation')
    for size in ('tf6', 'tf12'):
        add(f'v8-tf-{size}-gpt2-lr0.001-s11', 'calibrated Transformer control')
        add(f'v8-tf-confirm-{size}-gpt2-lr0.001-s22', 'calibrated Transformer control')
    add('v8-repair-fc-mem-gated_delta-state-s11-r2', 'named numerical repair; original failed attempt retained separately')
    return keep


def retention_inventory(roots, excluded):
    """Metadata fingerprint of every untouched file in old runs and the GEF tree."""
    h = hashlib.sha256(); count = 0; size = 0
    for key, base in sorted(roots.items()):
        starts = [base/'20261102'] if key == 'gef' else sorted(
            p for p in (base/'runs').iterdir() if p.is_dir() and not p.is_symlink() and not p.name.startswith('v9-'))
        for start in starts:
            for directory, dirs, files in os.walk(start, followlinks=False):
                dirs.sort(); files.sort()
                links = [n for n in dirs if (Path(directory)/n).is_symlink()]
                dirs[:] = [n for n in dirs if n not in links]
                for name in sorted(files + links):
                    p = Path(directory)/name; rel = str(p.relative_to(base))
                    if (key, rel) in excluded:
                        continue
                    row = file_row(p, base)
                    if p.is_symlink():row['link_target'] = os.readlink(p)
                    # Link count/ctime are intentionally absent: retained hardlink targets change them.
                    h.update((json.dumps([key, row], sort_keys=True)+'\n').encode())
                    count += 1; size += row['allocated_bytes']
    return dict(files=count, allocated_bytes=size, metadata_sha256=h.hexdigest())


def live_audit(root, candidates):
    own = {p for line in Path('/proc/self/cgroup').read_text().splitlines()
           for p in line.split(':', 2)[-1].split('/') if p.endswith('.service')}
    output = subprocess.run(['systemctl', '--user', 'list-units', '--type=service',
        '--state=active,activating,deactivating,reloading', '--plain', '--no-legend', '--no-pager'],
        check=True, text=True, capture_output=True).stdout
    active = [line.split()[0] for line in output.splitlines() if line.strip()]
    relevant = [u for u in active if u.startswith(('tide-lab-', 'gef-lm-')) and u not in own]
    if any(not u.startswith('tide-lab-v9-') for u in relevant):
        raise RuntimeError(f'non-v9 experiment service active: {relevant}')
    pid_checks = 0; protected_active_inputs = []
    records = [(p, False) for p in (root/'artifacts/jobs').glob('*/job.json')]
    records += [(p, True) for p in (root.parent/'gef-lm-experiments/20261102/jobs').glob('*/status.json')]
    for p, gef in records:
        j = read(p);unit = j.get('metadata', {}).get('unit') if gef else j.get('unit')
        if unit in own:continue
        if unit in relevant:
            for item in j.get('inputs', []):
                path = Path(item['path']).resolve()
                if any(c == path or path in c.parents for c in candidates):
                    raise RuntimeError(f'active input overlaps retirement: {path}')
                protected_active_inputs.append(str(path))
            continue
        if not gef and p.parent.name.startswith('v9-'):continue
        state = j if gef else read(p.with_name('state.json'))
        if state.get('state') not in {'completed','failed','timed_out','stopped','passed','cancelled'}:
            raise RuntimeError(f'historical job lacks terminal state: {p}')
        rows = [state.get(k, {}) for k in ('supervisor','launcher')] + state.get('descendants', [])
        rows += [dict(pid=state.get(k+'_pid'), start_ticks=state.get(k+'_start_ticks')) for k in ('supervisor','workload')]
        for item in rows:
            if item.get('pid') and item.get('start_ticks') is not None:
                pid_checks += 1
                if identity_alive(item):raise RuntimeError(f'historical process still alive: {p}')
    denied = 0; scanned = 0
    target_inodes = {(p.stat().st_dev,p.stat().st_ino) for p in candidates}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            if proc.stat().st_uid != os.getuid():continue
            scanned += 1
            for fd in (proc/'fd').iterdir():
                try:s=fd.stat()
                except OSError:continue
                if (s.st_dev,s.st_ino) in target_inodes:
                    raise RuntimeError(f'candidate is open in PID {proc.name}')
        except (PermissionError,FileNotFoundError):denied += 1
    return dict(at=now(), active_v9_units=relevant, checked_pid_identities=pid_checks,
                active_input_paths=sorted(set(protected_active_inputs)), same_uid_processes=scanned,
                unavailable_fd_directories=denied)


def make_plan(root, output):
    if output.exists():raise FileExistsError(output)
    roots = dict(tide=root, gef=root.parent/'gef-lm-experiments')
    keep = retained_selection(root);actions=[];proofs={}
    for result in sorted((root/'runs').glob('v[678]-*/train/result.json')):
        train=result.parent;case=train.parent.name;r=read(result)
        if r.get('state')!='completed' or case in PROTECTED_CASES:continue
        complete=train/'completed.json'
        if read(complete).get('state')!='completed':raise ValueError(f'not terminal: {train}')
        for p in sorted(train.glob('*.pt')):
            rel=str(p.relative_to(root))
            if rel in keep:continue
            row=dict(file_row(p,root),root_id='tide',action='delete',case=case)
            endpoint=f"tokens-{r['progress']['tokens']}.pt"
            target=str((train/'latest.pt').relative_to(root))
            if target in keep and p.name in ('previous.pt',endpoint):
                row.update(action='deduplicate',keeper=target)
            scope_check(row);actions.append(row)
        for p in (result,complete):proofs[str(p)]=digest(p)
    for train in sorted((roots['gef']/'20261102/runs').glob('*/*/run')):
        candidates=[p for p in train.iterdir() if p.name in {'checkpoint.pt','state.bin','expected-state.bin','restored-state.bin'}]
        if not candidates:continue
        if read(train/'result.json').get('state')!='completed' or read(train/'resume.json').get('state')!='passed':
            raise ValueError(f'GEF result/recovery not passed: {train}')
        for p in candidates:
            row=dict(file_row(p,roots['gef']),root_id='gef',action='delete',case=train.parent.name)
            scope_check(row);actions.append(row)
        for name in ('result.json','resume.json'):proofs[str(train/name)]=digest(train/name)
    excluded={(r['root_id'],r['path']) for r in actions}
    selected=[dict(file_row(root/p,root),root_id='tide',reason=reason) for p,reason in sorted(keep.items())]
    summary=Counter()
    for row in actions:summary[row['root_id']+'_'+row['action']]+=row['allocated_bytes']
    plan=dict(schema=SCHEMA,mode='fixed-plan',created_at=now(),authorization='user-approved-2026-11-12-expanded-retirement',
        roots={k:str(v) for k,v in roots.items()},actions=actions,selected_checkpoints=selected,proofs=proofs,
        retention_policy='All v9, every non-completed trajectory, all other historical files and GEF model links retained. Only named completed training/benchmark checkpoint files can change.',
        expected_reclaimed_bytes=sum(summary.values()),groups_bytes=dict(summary),
        preserved_metadata=retention_inventory(roots,excluded))
    plan['activity']=live_audit(root,{roots[r['root_id']]/r['path'] for r in actions})
    for row in actions:checked(roots[row['root_id']],row)
    write_review_json(output,plan,root=root)
    print(json.dumps(dict(actions=len(actions),retained_selected=len(selected),gib=plan['expected_reclaimed_bytes']/2**30,groups_gib={k:v/2**30 for k,v in summary.items()})),flush=True)


def mutate(base, row):
    p=checked(base,row)
    if row['action']=='delete':p.unlink()
    elif row['action']=='deduplicate':
        keeper=Path(base)/row['keeper']
        if row['sha256']!=row['keeper_sha256']:raise ValueError('nonidentical alias')
        tmp=p.with_name(p.name+'.retirement-link')
        if tmp.exists() or tmp.is_symlink():raise FileExistsError(tmp)
        os.link(keeper,tmp,follow_symlinks=False)
        os.replace(tmp,p)
    else:raise ValueError('unknown action')
    fd=os.open(p.parent,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def apply_plan(root, plan_path, output):
    if output.exists():raise FileExistsError('do not rerun retirement over an existing receipt')
    plan=read(plan_path)
    if plan['schema']!=SCHEMA or plan['authorization']!='user-approved-2026-11-12-expanded-retirement':raise ValueError('wrong plan')
    roots={k:Path(v) for k,v in plan['roots'].items()}
    if roots!={'tide':root,'gef':root.parent/'gef-lm-experiments'}:raise ValueError('wrong roots')
    actions=[dict(r,state='pending') for r in plan['actions']]
    excluded={(r['root_id'],r['path']) for r in actions}
    paths={roots[r['root_id']]/r['path'] for r in actions}
    report=dict(schema=SCHEMA,mode='execution',state='preflight',started_at=now(),plan_sha256=digest(plan_path),
                files=actions,reclaimed_bytes=0,completed_actions=0)
    def interrupted(sig,frame):raise InterruptedError(f'signal {sig}; inspect prepared receipt before any continuation')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        for row in actions:
            scope_check(row);checked(roots[row['root_id']],row)
        for p,h in plan['proofs'].items():
            if digest(p)!=h:raise ValueError(f'terminal evidence changed: {p}')
        report['preflight_activity']=live_audit(root,paths)
        before=retention_inventory(roots,excluded)
        if before!=plan['preserved_metadata']:raise ValueError('retained historical inventory changed')
        report['retained_before']=before
        write_review_json(output,report,root=root)
        print(f"preflight passed; hashing {len(actions)} exact artifacts",flush=True)
        def hash_row(row):
            p=checked(roots[row['root_id']],row);h=digest(p);checked(roots[row['root_id']],row)
            return h
        with ThreadPoolExecutor(max_workers=2) as pool:
            for i,h in enumerate(pool.map(hash_row,actions)):
                actions[i].update(sha256=h,state='hashed')
                if (i+1)%50==0:print(f'hashed {i+1}/{len(actions)}',flush=True)
        keepers={}
        for row in plan['selected_checkpoints']:
            p=checked(root,row);keepers[row['path']]=dict(row,sha256=digest(p))
        report['retained_checkpoints']=list(keepers.values())
        for row in actions:
            if row['action']=='deduplicate':
                row['keeper_sha256']=keepers[row['keeper']]['sha256']
                if row['sha256']!=row['keeper_sha256']:raise ValueError('duplicate hypothesis rejected; retain both')
        report['predelete_activity']=live_audit(root,paths)
        report.update(state='prepared',prepared_at=now())
        write_review_json(output,report,root=root)
        print('all identities durable; applying fixed manifest',flush=True)
        for i,row in enumerate(actions):
            if row['action']=='deduplicate':checked(root,keepers[row['keeper']],single_link=False)
            mutate(roots[row['root_id']],row)
            row.update(state='deleted' if row['action']=='delete' else 'deduplicated',finished_at=now())
            report['completed_actions']+=1;report['reclaimed_bytes']+=row['allocated_bytes']
            if (i+1)%50==0:write_review_json(output,report,root=root)
        after=retention_inventory(roots,excluded)
        if after!=before:raise ValueError('retained metadata mismatch after retirement')
        for row in actions:
            p=roots[row['root_id']]/row['path']
            if row['action']=='delete':
                if p.exists() or p.is_symlink():raise ValueError('deleted artifact still exists')
            elif not p.samefile(roots[row['root_id']]/row['keeper']):raise ValueError('alias does not share retained file')
        for row in keepers.values():
            if digest(root/row['path'])!=row['sha256']:raise ValueError('retained weight bytes changed')
        report.update(state='completed',finished_at=now(),retained_after=after,
            verification='all retired paths/aliases verified; all selected checkpoint SHA256 and untouched historical metadata unchanged')
    except BaseException as error:
        report.update(state='failed',error=repr(error),finished_at=now());raise
    finally:write_review_json(output,report,root=root)
    print(json.dumps({k:report[k] for k in ('state','completed_actions','reclaimed_bytes')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('plan','apply'));p.add_argument('--root',type=Path,required=True)
    p.add_argument('--plan',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve()
    if a.mode=='plan':make_plan(root,a.output)
    else:apply_plan(root,a.plan,a.output)

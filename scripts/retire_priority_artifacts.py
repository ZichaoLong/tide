#!/usr/bin/env python3
"""Retire only the user's approved four groups from the fixed disk audit."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.review_io import write_review_json
from tide_lab.records import now

GROUPS = {
    'tide-completed-recovery-probes': 603,
    'tide-rebuildable-b-features': 3,
    'gef-terminal-qwen05-benchmark-checkpoints': 2,
    'gef-rebuildable-bf16-gguf': 2,
}


def checked_file(base, row):
    base = Path(base).absolute()
    rel = Path(row['path'])
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('unsafe artifact path')
    path = base / rel
    if path.resolve() != path or base.resolve() != base:
        raise ValueError('artifact or ancestor is a symlink')
    s = path.lstat()
    if not stat.S_ISREG(s.st_mode) or s.st_nlink != 1:
        raise ValueError('expected a single-link regular artifact')
    if (s.st_size, s.st_mtime_ns) != (row['logical_bytes'], row['mtime_ns']):
        raise ValueError(f'artifact changed since audit: {path}')
    return path, s


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def inspect_jobs(audit):
    # Also reject new project services that were absent from the fixed audit.
    # The current housekeeping service is the sole exception.
    own_units = {part for line in Path('/proc/self/cgroup').read_text().splitlines()
                 for part in line.split(':', 2)[-1].split('/') if part.endswith('.service')}
    running = subprocess.run(['systemctl', '--user', 'list-units', '--type=service',
                              '--state=active,activating,deactivating,reloading',
                              '--no-legend', '--no-pager', '--plain'],
                             check=True, capture_output=True, text=True)
    new_live = [line.split()[0] for line in running.stdout.splitlines() if line.strip()
                and line.split()[0].startswith(('tide-lab-', 'gef-lm-'))
                and line.split()[0] not in own_units]
    if new_live:
        raise RuntimeError(f'project services are live: {new_live}')
    units = sorted({r['unit'] for group in audit['job_status']
                    for r in group['records'] if r.get('unit')})
    result = subprocess.run(['systemctl', '--user', 'show', *units,
                             '--property=Id,ActiveState,MainPID'],
                            check=True, capture_output=True, text=True)
    live = []
    for block in result.stdout.split('\n\n'):
        d = dict(line.split('=', 1) for line in block.splitlines() if '=' in line)
        if d.get('MainPID', '0') != '0' or d.get('ActiveState') in {
                'active', 'activating', 'deactivating', 'reloading'}:
            live.append(d)
    identities = 0
    for group in audit['job_status']:
        for record in group['records']:
            d = json.loads((Path(group['root'])/record['path']).read_text())
            rows = [v for k in ('supervisor', 'launcher')
                    if isinstance((v := d.get(k)), dict)] + d.get('descendants', [])
            rows += [dict(pid=d.get(k+'_pid'), start_ticks=d.get(k+'_start_ticks'))
                     for k in ('supervisor', 'workload')]
            for item in rows:
                pid, ticks = item.get('pid'), item.get('start_ticks')
                if not pid or ticks is None:
                    continue
                identities += 1
                try:
                    raw = Path(f'/proc/{pid}/stat').read_text()
                    fields = raw[raw.rfind(')')+2:].split()
                except FileNotFoundError:
                    continue
                if int(fields[19]) == ticks and fields[0] != 'Z':
                    live.append(dict(pid=pid, record=record['path']))
    if live:
        raise RuntimeError(f'referenced jobs/processes are live: {live}')
    return dict(at=now(), units=len(units), process_records=identities, live=[])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--apply-approved-first-tier', action='store_true')
    a = p.parse_args()
    if not a.apply_approved_first_tier:
        raise ValueError('explicit approved-first-tier action is required')
    root = a.root.resolve()
    allowed = {root, root.parent/'gef-lm-experiments'}
    audit = json.loads(a.audit.read_text())
    if a.output.exists():
        raise FileExistsError('choose a new retirement report; never overwrite one')
    groups = [g for g in audit['candidate_groups'] if g['tier'] == 'first']
    if {g['id']: len(g['files']) for g in groups} != GROUPS:
        raise ValueError('approved group/count mismatch')
    report = dict(schema='tide-artifact-retirement-v2', state='preflight',
                  authorized='user-2026-11-11-clean-priority-candidates-only',
                  started_at=now(), audit_sha256=digest(a.audit), deleted_bytes=0,
                  groups=[], files=[], scope='exact 610 files; public original weights retained')
    for group in groups:
        if Path(group['root']) not in allowed:
            raise ValueError('unapproved root')
        report['groups'].append(dict(id=group['id'], count=len(group['files']),
                                    allocated_bytes=group['allocated_bytes']))
        for row in group['files']:
            path, s = checked_file(group['root'], row)
            if any(name in path.parts for name in (
                    'v6-cover-hb-direct-received', 'v6-cover-hb-send-constrained-bern',
                    'v6-cover-hb-send-regional_total-bern')):
                raise ValueError('protected unresolved failure')
            if group['id'] == 'tide-completed-recovery-probes':
                recovery = root/row['recovery_record']
                if digest(recovery) != row['recovery_record_sha256']:
                    raise ValueError('recovery record changed')
                if json.loads(recovery.read_text())['state'] != 'passed':
                    raise ValueError('recovery did not pass')
            report['files'].append(dict(row, root=group['root'], group=group['id'],
                                        inode=s.st_ino, device=s.st_dev, state='pending'))
    report['preflight_jobs'] = inspect_jobs(audit)
    write_review_json(a.output, report, root=root)
    print('preflight passed; hashing 610 exact artifacts before any deletion', flush=True)

    def hash_one(row):
        path, _ = checked_file(row['root'], row)
        sha = digest(path)
        if row.get('recorded_checkpoint_sha256') and sha != row['recorded_checkpoint_sha256']:
            raise ValueError(f'checkpoint hash mismatch: {path}')
        if row['group'] == 'gef-terminal-qwen05-benchmark-checkpoints':
            expected = json.loads(path.with_name('checkpoint-request.json').read_text())['sha256']
            if sha != expected:
                raise ValueError('old benchmark checkpoint hash mismatch')
        checked_file(row['root'], row)
        return sha

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for i, sha in enumerate(pool.map(hash_one, report['files'])):
                report['files'][i].update(sha256=sha, hash_verified_at=now(), state='hashed')
                if (i+1) % 100 == 0:
                    print(f'hashed {i+1}/610', flush=True)
        report['predelete_jobs'] = inspect_jobs(audit)
        report['state'] = 'prepared'
        report['prepared_at'] = now()
        # All byte identities are durable before the first unlink.
        write_review_json(a.output, report, root=root)
        for row in report['files']:
            path, s = checked_file(row['root'], row)
            if (s.st_dev, s.st_ino) != (row['device'], row['inode']):
                raise ValueError('artifact inode changed')
            path.unlink()
            fd = os.open(path.parent, os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            row.update(state='deleted', deleted_at=now())
            report['deleted_bytes'] += row['allocated_bytes']
        report.update(state='completed', finished_at=now(), deleted_files=len(report['files']),
                      retention='original results/configs/metrics/failures/lifecycle retained; exact retired bytes unavailable')
    except BaseException as error:
        report.update(state='failed', error=repr(error), finished_at=now())
        raise
    finally:
        write_review_json(a.output, report, root=root)
    print(json.dumps({k: report[k] for k in ('state', 'deleted_files', 'deleted_bytes')}), flush=True)


if __name__ == '__main__':
    main()

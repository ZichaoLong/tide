#!/usr/bin/env python3
"""One authorized project cleanup; fixed targets, no symlink traversal or retries."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path('/mi/data2T/zlong/home/llm/tide-learning-lab')
AUTHORIZATION = '2026-11-18-user-approved-project-history-retirement'
UNIT = 'tide-lab-retire-project-20261118-r2.service'
KEEP = {
    'runs': {'v9-data-r3', 'v11-confirmation-data', 'v12-data', 'v13-data'},
    'artifacts': {'wheels', 'setup'},
    '.local': {'language-data-venv', 'LENS.md', 'cleanup-20261118'},
}


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def target(root, name):
    p = Path(name)
    if p.is_absolute() or len(p.parts) != 2 or any(v in ('.', '..') for v in p.parts):
        raise ValueError('target must be an immediate child of an allowed directory')
    base, child = p.parts
    if base not in KEEP or child in KEEP[base]:
        raise ValueError('protected or outside cleanup scope: ' + name)
    parent = root / base
    if parent.is_symlink() or parent.resolve() != root / base:
        raise ValueError('cleanup parent is redirected')
    return parent / child


def inventory(path):
    entries, inodes, paths = [], {}, []
    pending = [path]
    base_device = path.lstat().st_dev
    while pending:
        p = pending.pop()
        s = p.lstat()
        if s.st_dev != base_device or s.st_uid != os.getuid():
            raise ValueError('different filesystem or owner: ' + str(p))
        relative = '.' if p == path else str(p.relative_to(path))
        link = os.readlink(p) if stat.S_ISLNK(s.st_mode) else None
        # Unlinking an earlier authorized alias changes nlink, not the bytes.
        # Count links across the whole plan below, outside this drift digest.
        entries.append((relative, s.st_dev, s.st_ino, s.st_mode, s.st_size,
                        s.st_mtime_ns, link))
        paths.append(p)
        item = inodes.setdefault((s.st_dev, s.st_ino), [s.st_blocks*512, 0, s.st_nlink, stat.S_ISREG(s.st_mode)])
        item[1] += 1
        if stat.S_ISDIR(s.st_mode):
            pending.extend(p.iterdir())
        elif not (stat.S_ISREG(s.st_mode) or stat.S_ISLNK(s.st_mode)):
            raise ValueError('special file in deletion tree: ' + str(p))
    entries.sort()
    return dict(metadata_sha256=hashlib.sha256(json.dumps(entries, separators=(',', ':')).encode()).hexdigest(),
                paths=len(entries), files=sum(stat.S_ISREG(v[3]) for v in entries),
                symlinks=sum(stat.S_ISLNK(v[3]) for v in entries),
                allocated_bytes=sum(v[0] for v in inodes.values())), inodes, paths


def retained(root):
    paths = []
    for base, names in KEEP.items():
        for name in names:
            if name == 'cleanup-20261118':
                continue
            p = root / base / name
            if p.exists() or p.is_symlink():
                paths.append(p)
    paths += [root / v for v in ('.venv', '.venv-positive-floor-v1', 'dependencies', 'configs',
                                 'scripts', 'tide_lab', 'tests', 'README.md', 'AGENTS.md')]
    rows = []
    for p in paths:
        _, _, contents = inventory(p)
        for f in contents:
            if f.is_symlink():
                rows.append((str(f.relative_to(root)), 'link', os.readlink(f)))
            elif f.is_file():
                rows.append((str(f.relative_to(root)), 'file', f.stat().st_size, digest(f)))
    rows.sort()
    return dict(files=len(rows), content_sha256=hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest())


def no_writers(root, deleted_paths):
    raw = subprocess.check_output(['systemctl', '--user', 'list-units', '--type=service',
                                   '--state=running,activating,deactivating', '--no-pager', '--no-legend', '--plain'], text=True)
    for line in raw.splitlines():
        unit = line.split()[0]
        if unit == UNIT:
            continue
        props = subprocess.check_output(['systemctl', '--user', 'show', unit, '-p', 'ExecStart'], text=True)
        if str(root) in props:
            raise RuntimeError('project service is still active: ' + unit)
    prefixes = tuple(str(p) + '/' for p in deleted_paths)
    exact = {str(p) for p in deleted_paths}
    inaccessible = 0
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            for f in (proc / 'fd').iterdir():
                try:
                    dest = os.readlink(f)
                except FileNotFoundError:
                    continue
                if dest in exact or dest.startswith(prefixes):
                    raise RuntimeError('open deletion input: PID ' + proc.name + ' ' + dest)
        except (FileNotFoundError, ProcessLookupError):
            pass
        except PermissionError:
            inaccessible += 1
    return dict(project_services='none_except_this_cleanup', same_account_fd_inaccessible=inaccessible)


def make_directories_removable(path):
    """Only grant owner write on owned directories already approved for deletion."""
    changed = []
    for parent, _, _ in os.walk(path, followlinks=False):
        fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            s = os.fstat(fd)
            if s.st_uid != os.getuid():
                raise ValueError('directory owner changed: ' + parent)
            if not s.st_mode & stat.S_IWUSR:
                os.fchmod(fd, stat.S_IMODE(s.st_mode) | stat.S_IWUSR)
                changed.append(parent)
        finally:
            os.close(fd)
    return changed


def prepare(root, plan_path):
    if plan_path.exists():
        raise FileExistsError('do not overwrite a prepared manifest')
    stop = json.loads((root / 'docs/evidence/project-retirement-stop-20261118.json').read_text())
    if stop['state'] != 'stopped_verified':
        raise RuntimeError('project stop is not verified')
    paths = [target(root, str(p.relative_to(root))) for base, keep in KEEP.items()
             for p in sorted((root / base).iterdir()) if p.name not in keep]
    guard = no_writers(root, paths)
    rows, unique = [], {}
    for path in paths:
        row, inodes, _ = inventory(path)
        row['path'] = str(path.relative_to(root))
        rows.append(row)
        for key, value in inodes.items():
            if key in unique:
                unique[key][1] += value[1]
            else:
                unique[key] = value[:]
    external = [v for v in unique.values() if v[3] and v[1] != v[2]]
    if external:
        raise RuntimeError('hard links cross deletion boundary; reassess exact scope')
    plan = dict(schema='tide-project-history-retirement-plan-v1', authorization=AUTHORIZATION,
                root=str(root), prepared_at=now(), state='prepared', stop_record='docs/evidence/project-retirement-stop-20261118.json',
                guard=guard, preserved=retained(root), keep={k: sorted(v) for k,v in KEEP.items()}, targets=rows,
                unique_allocated_bytes=sum(v[0] for v in unique.values()),
                target_paths=sum(v['paths'] for v in rows), target_files=sum(v['files'] for v in rows),
                script_sha256=digest(__file__), notes='Metadata/inode manifest; no full hashing or backup of retired model bytes. Small historical reports remain in Git.')
    atomic(plan_path, plan)
    print(json.dumps({k: plan[k] for k in ('state', 'unique_allocated_bytes', 'target_files', 'target_paths', 'preserved')}, indent=2))


def execute(root, plan_path, result_path):
    if result_path.exists():
        raise FileExistsError('cleanup receipt already exists; no automatic rerun')
    plan = json.loads(plan_path.read_text())
    if plan['authorization'] != AUTHORIZATION or plan['root'] != str(root) or plan['script_sha256'] != digest(__file__):
        raise ValueError('plan/source/authorization mismatch')
    paths = [target(root, r['path']) for r in plan['targets']]
    result = dict(schema='tide-project-history-retirement-execution-v1', state='preflight',
                  started_at=now(), plan_sha256=digest(plan_path), script_sha256=digest(__file__),
                  unit=UNIT, cgroup=Path('/proc/self/cgroup').read_text(), completed_targets=[],
                  directory_permission_adjustments=[])
    atomic(result_path, result)
    try:
        result['guard'] = no_writers(root, paths)
        if retained(root) != plan['preserved']:
            raise RuntimeError('preserved inputs changed before cleanup')
        for row, path in zip(plan['targets'], paths):
            actual, _, _ = inventory(path)
            if actual != {k:v for k,v in row.items() if k != 'path'}:
                raise RuntimeError('target changed: ' + row['path'])
        result['state'] = 'deleting'
        atomic(result_path, result)
        for index, (row, path) in enumerate(zip(plan['targets'], paths)):
            actual, _, _ = inventory(path)
            if actual['metadata_sha256'] != row['metadata_sha256']:
                raise RuntimeError('target changed immediately before deletion: ' + row['path'])
            if path.is_symlink() or path.is_file():
                path.unlink()
            else:
                result['directory_permission_adjustments'].extend(
                    str(Path(p).relative_to(root)) for p in make_directories_removable(path))
                shutil.rmtree(path)
            result['completed_targets'].append(row['path'])
            if index % 100 == 0:
                atomic(result_path, result)
        if any(p.exists() or p.is_symlink() for p in paths):
            raise RuntimeError('a retired target remains')
        if retained(root) != plan['preserved']:
            raise RuntimeError('retained content changed')
        result.update(state='completed', finished_at=now(), deleted_unique_allocated_bytes=plan['unique_allocated_bytes'],
                      deleted_files=plan['target_files'], retained_content_unchanged=True,
                      final_guard=no_writers(root, paths))
    except BaseException as error:
        result.update(state='failed', error=repr(error), finished_at=now())
        atomic(result_path, result)
        raise
    atomic(result_path, result)
    print(json.dumps({k: result[k] for k in ('state','deleted_unique_allocated_bytes','deleted_files','retained_content_unchanged')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'execute'])
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--result', type=Path)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT or not ROOT.is_dir():
        raise ValueError('must run in the authorized project directory')
    if args.action == 'prepare':
        prepare(ROOT, args.plan)
    else:
        if args.result is None:
            raise ValueError('--result is required')
        execute(ROOT, args.plan, args.result)

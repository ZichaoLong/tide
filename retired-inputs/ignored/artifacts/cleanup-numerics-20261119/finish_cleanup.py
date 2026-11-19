"""Finite retirement of this round's verified, completed-use artifacts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from job import atomic
from prepare_cleanup import guard, same_guard


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def link_guard(path):
    info=path.lstat()
    assert path.is_symlink()
    return dict(device=info.st_dev,inode=info.st_ino,mtime_ns=info.st_mtime_ns,
        ctime_ns=info.st_ctime_ns,allocated_bytes=info.st_blocks*512,bytes=info.st_size,
        mode=info.st_mode,links=info.st_nlink,target=os.readlink(path))


def verify_row(row):
    if row.get('type')=='symlink':
        assert link_guard(ROOT/row['path'])==row['guard']
    else:
        same_guard(row['path'],row['guard'])


def terminal(run):
    launch = json.loads((run / 'launch.json').read_text())
    status = json.loads((run / 'status.json').read_text())
    assert status['state'] in ('passed', 'failed', 'terminated'), run
    props = subprocess.check_output(['systemctl', '--user', 'show', launch['unit'],
        '-p', 'MainPID', '-p', 'Result', '-p', 'ExecMainStatus', '-p', 'ControlGroup',
        '-p', 'ActiveState', '-p', 'SubState'], text=True)
    parsed = dict(x.split('=', 1) for x in props.splitlines())
    assert parsed['MainPID'] == '0' and not parsed['ControlGroup'], (run, props)
    alive = []
    for identity in status.get('descendant_identities', []):
        try:
            fields = Path('/proc', str(identity['pid']), 'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[19]) == identity['start_ticks']:
                alive.append(dict(**identity, proc_state=fields[0]))
        except FileNotFoundError:
            pass
    assert not alive, (run, alive)
    return dict(run=str(run.relative_to(ROOT)), unit=launch['unit'], source=status['source'],
        state=status['state'], elapsed=status.get('elapsed'), peak_rss_bytes=status.get('peak_rss_bytes'),
        descendant_count=len(status.get('descendant_identities', [])), live_descendants=alive,
        unit_properties=props, npu_count=launch['arguments']['npu_count'],
        physical_devices=json.loads((run/'queue.json').read_text()).get('physical_ids') if (run/'queue.json').exists() else None)


def build_plan():
    jobs = [p.parent for p in sorted((ROOT/'artifacts').glob('numerics-*/launch.json'))]
    records = [terminal(run) for run in jobs]
    rows, sources, cached = [], [], {}
    def add(path, reason, **identity):
        name = str(path.relative_to(ROOT))
        rows.append(dict(path=name, reason=reason, guard=guard(name), **identity))
    for run, record in zip(jobs, records):
        source, revision = run/'source', record['source']
        subprocess.run(['git', 'merge-base', '--is-ancestor', revision,
                        'origin/graph-execution-foundation'], cwd=ROOT, check=True)
        if revision not in cached:
            tree = subprocess.check_output(['git', 'ls-tree', '-rz', revision], cwd=ROOT)
            cached[revision] = {entry.split(b'\t', 1)[1].decode(): entry.split(b'\t', 1)[0].decode().split()
                                for entry in tree.split(b'\0') if entry}
        tracked = cached[revision]
        verified = 0
        for path in sorted(source.rglob('*')):
            assert not path.is_symlink(), path
            if not path.is_file():
                continue
            name = str(path.relative_to(source))
            if name in tracked:
                mode, kind, oid = tracked[name]
                data = path.read_bytes()
                actual = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
                assert kind == 'blob' and actual == oid, path
                add(path, 'byte_verified_frozen_source', commit=revision, git_blob=oid)
                verified += 1
            elif ('__pycache__' in path.parts and path.suffix == '.pyc') or '.pytest_cache' in path.parts:
                add(path, 'generated_python_or_pytest_cache', sha256=sha(path))
            else:
                raise RuntimeError('Unique/unexpected frozen-source file: '+str(path))
        assert verified == len(tracked), (source, verified, len(tracked))
        sources.append(dict(path=str(source.relative_to(ROOT)), commit=revision, verified_files=verified))
    suite = ROOT/'artifacts/numerics-quad-cpu-suite-v1'
    assert json.loads((suite/'status.json').read_text())['state'] == 'passed'
    for path in sorted((suite/'pytest').rglob('*')):
        if path.is_symlink():
            assert path.resolve().is_relative_to(suite/'pytest'), path
            rows.append(dict(path=str(path.relative_to(ROOT)),type='symlink',
                reason='generated_pytest_current_link_only; target_not_followed',guard=link_guard(path)))
        elif path.is_file():
            add(path, 'successful_cpu_suite_generated_temporary_data')
    training = ROOT/'artifacts/numerics-quad-training-npu-v1/qualification'
    for candidate in ('baseline', 'combined'):
        record = json.loads((training/candidate/'restore.json').read_text())
        assert record['state'] == 'within_threshold' and record['actual_update']
        for name, result in record['checks'].items():
            check = result['comparison']
            for field in ('metadata_failures','dtype_differences','nonfinite','support_differences','threshold_failures','work_differences'):
                assert not check[field], (candidate, name, field)
            if name in ('saved_data', 'next_forward'):
                assert check['exact']
        for name in ('restore-input.pt',):
            path = training/candidate/name
            add(path, 'successful_candidate_local_restore_payload; retain anchor and failed cross-path small result', sha256=sha(path))
    assert len({r['path'] for r in rows}) == len(rows)
    plan = dict(state='review_required', created_utc=datetime.now(timezone.utc).isoformat(),
        jobs=records, sources=sources, files=rows,
        allocated_bytes=sum(r['guard']['allocated_bytes'] for r in rows),
        preserve=['all original unresolved failure inputs except already verified update2 retirements',
                  'new local VJP and allocation witnesses', 'new training anchor/checkpoints and all small results',
                  'user files, main math/git, old unique worktree source and downstream dependencies'])
    atomic(HERE/'finish-plan.json', plan)
    print(json.dumps(dict(files=len(rows), sources=len(sources), allocated_bytes=plan['allocated_bytes'],
                         plan_sha256=sha(HERE/'finish-plan.json'))))


def execute(expected):
    plan_path = HERE/'finish-plan.json'
    assert sha(plan_path) == expected
    plan = json.loads(plan_path.read_text())
    for record in plan['jobs']:
        terminal(ROOT/record['run'])
    for row in plan['files']:
        verify_row(row)
    # No source copy is a consumer installation. Verify no current same-user
    # process has a cwd/open file/mapping under the exact candidate set.
    candidates = {str(ROOT/r['path']) for r in plan['files']}
    source_roots = [str(ROOT/s['path'])+'/' for s in plan['sources']]
    live_refs, restricted = [], []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            values = [os.readlink(proc/'cwd')]
            values += [os.readlink(p) for p in (proc/'fd').iterdir()]
            values += [line.rsplit(' ', 1)[-1] for line in (proc/'maps').read_text().splitlines()]
            matches = [v for v in values if v in candidates or any(v == s[:-1] or v.startswith(s) for s in source_roots)]
            if matches:
                live_refs.append(dict(pid=int(proc.name), paths=sorted(set(matches))))
        except FileNotFoundError:
            pass
        except PermissionError:
            restricted.append(int(proc.name))
    assert not live_refs, live_refs
    removed, touched = [], set()
    with (HERE/'finish-retirement.jsonl').open('x') as receipt:
        for row in plan['files']:
            verify_row(row)
            path = ROOT/row['path']; path.unlink(); touched.add(path.parent)
            removed.append(row['path'])
            receipt.write(json.dumps(dict(path=row['path'], allocated_bytes=row['guard']['allocated_bytes']))+'\n')
        receipt.flush(); os.fsync(receipt.fileno())
    for directory in touched:
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    # Only empty, explicitly retired source/temp directories are removed.
    for root in [ROOT/s['path'] for s in plan['sources']] + [ROOT/'artifacts/numerics-quad-cpu-suite-v1/pytest']:
        for path in sorted(root.rglob('*'), key=lambda p:len(p.parts), reverse=True):
            if path.is_dir():
                path.rmdir()
        root.rmdir()
    assert all(not (ROOT/name).exists() for name in removed)
    atomic(HERE/'finish-retirement-result.json', dict(state='completed', plan_sha256=expected,
        files=len(removed), allocated_bytes=plan['allocated_bytes'], live_refs=live_refs,
        restricted_system_pids=restricted, finished_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(state='completed', files=len(removed), allocated_bytes=plan['allocated_bytes'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute-sha')
    args = parser.parse_args()
    execute(args.execute_sha) if args.execute_sha else build_plan()

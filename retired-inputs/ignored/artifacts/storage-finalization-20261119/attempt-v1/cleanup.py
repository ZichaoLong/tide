"""Fixed-list artifact retirement and verified lossless JSON archiving.

No broad deletion, directory removal, Git changes, or model computation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('frozen_job', HERE/'frozen-job.py')
job = importlib.util.module_from_spec(spec); spec.loader.exec_module(job)
atomic = job.atomic
PLAN = HERE/'plan.json'
INDEX = HERE/'archive-index.json'
KEEP = [
    'artifacts/qualification-cpu-v8/tensors/attention.pt',
    'artifacts/training-attention-mode-witness/results/witness.pt',
    'artifacts/regional-rounding-npu-v1/settle/witness.pt',
    'artifacts/regional-rounding-npu-v1/timed/witness.pt',
    'artifacts/worktrees/region-diagnosis/artifacts/region-diagnosis-cpu-a4509760/witness.pt',
    'artifacts/ffn-local-diagnosis-v1/diagnosis.tensors.pt',
]
NATIVE = 'artifacts/native-regional-npu-same-v1/checks/native-bias-rounding'
RECENT = [
    'artifacts/numerics-quad-training-npu-v1/qualification/anchor.pt',
    'artifacts/numerics-quad-training-npu-v1/qualification/combined/checkpoint.pt',
    'artifacts/numerics-quad-training-npu-v1/qualification/baseline/checkpoint.pt',
    'artifacts/numerics-ffn-vjp-v1/diagnosis.tensors.pt',
    'artifacts/numerics-bf16-training-routing-v1/routing.witness.pt',
    'artifacts/numerics-shared-routing-v1/routing.data/allocation-witnesses.pt',
]
USER_DOCS = {
    'docs/tide-learning-lab-TL-REQ-004-handoff.md': '69607cba0a7e87a260929ed5741d996103cef804c51f3fa2c02b3bd7a454a220',
    'docs/tide-learning-lab-TL-REQ-005-handoff.md': '83bec6f6eaa476ae9d7c859fe168333ac2ee564433297f36ab9aa1ef33c38732',
    'docs/tide-learning-lab-TL-REQ-006-handoff.md': '8706f0477c8f31d0c8337e55ba7f8f5b28bb9a455656586fa1585c82ec58b4df',
    'docs/upstream-lh-semantics-discussion-20261113.md': '3118419cb014570395323baf887ffe5a7e797f97049e85eacef26ecb1fd81c07',
}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def guard(name):
    rel = PurePosixPath(name)
    assert not rel.is_absolute() and '..' not in rel.parts and str(rel) == name
    assert rel.parts[0] == 'artifacts'
    path = ROOT
    for part in rel.parts:
        path = path/part
        info = path.lstat()
        assert not stat.S_ISLNK(info.st_mode), name
    assert stat.S_ISREG(info.st_mode) and info.st_nlink == 1, name
    return dict(bytes=info.st_size, allocated_bytes=info.st_blocks*512, device=info.st_dev,
                inode=info.st_ino, mtime_ns=info.st_mtime_ns, ctime_ns=info.st_ctime_ns,
                mode=info.st_mode, links=info.st_nlink)


def same(row):
    assert guard(row['path']) == row['stat_guard'], row['path']


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def git_inventory():
    text = subprocess.check_output(['git', 'worktree', 'list', '--porcelain'], cwd=ROOT, text=True)
    tracked, worktrees = set(), []
    for line in text.splitlines():
        if not line.startswith('worktree '):
            continue
        root = Path(line[9:])
        names = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z']).split(b'\0')
        tracked.update(str(root/os.fsdecode(n)) for n in names if n)
        status = subprocess.check_output(['git', '-C', str(root), 'status', '--short'])
        head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        worktrees.append(dict(path=str(root), head=head, status=status.decode(), status_sha256=hashlib.sha256(status).hexdigest()))
    return tracked, worktrees


def live_references(rows):
    names = {str(ROOT/r['path']) for r in rows}
    identities = {(r['stat_guard']['device'], r['stat_guard']['inode']) for r in rows}
    hits, restricted, inspected = [], [], 0
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            info = dict(line.split(':', 1) for line in (proc/'status').read_text().splitlines() if ':' in line)
            if info['State'].lstrip().startswith('Z'):
                continue
            argv = (proc/'cmdline').read_bytes().split(b'\0')
            for arg in argv:
                value = os.fsdecode(arg)
                if value in names or (value.startswith(str(ROOT/'artifacts')+'/') and
                        any(n.startswith(value.rstrip('/')+'/') for n in names)):
                    hits.append(dict(pid=int(proc.name), kind='argv', path=value))
            for fd in (proc/'fd').iterdir():
                try:
                    s = fd.stat()
                    if (s.st_dev, s.st_ino) in identities:
                        hits.append(dict(pid=int(proc.name), kind='fd', path=os.readlink(fd)))
                except (FileNotFoundError, ProcessLookupError):
                    pass
            for line in (proc/'maps').read_text().splitlines():
                parts = line.split(None, 5)
                if len(parts) == 6 and parts[-1] in names:
                    hits.append(dict(pid=int(proc.name), kind='mmap', path=parts[-1]))
            inspected += 1
        except PermissionError:
            restricted.append(dict(pid=int(proc.name), name=(proc/'comm').read_text().strip()))
        except (FileNotFoundError, ProcessLookupError):
            pass
    return dict(observed_utc=now(), hits=hits, restricted_processes=restricted, inspected=inspected)


def prepare():
    assert not PLAN.exists()
    tracked, worktrees = git_inventory()
    old_path = ROOT/'artifacts/cleanup-numerics-20261119/cleanup-plan.json'
    old = json.loads(old_path.read_text())
    removed = set(json.loads((old_path.parent/'update2-retirement-result.json').read_text())['removed'])
    native = sorted(str(p.relative_to(ROOT)) for p in (ROOT/NATIVE).rglob('*') if p.is_file())
    keep = set(KEEP + native)
    retire = []
    for row in old['protected']['retained_tensors']:
        name = row['path']
        if name in removed:
            assert not (ROOT/name).exists()
            continue
        same(row)
        assert str(ROOT/name) not in tracked
        if name not in keep:
            retire.append(dict(path=name, stat_guard=row['stat_guard'], disposition='explicit_retirement_not_regeneration'))
    assert len(retire) + sum(r['path'] in keep for r in old['protected']['retained_tensors']) == 7024
    protected = []
    for name in sorted(keep | set(RECENT)):
        row = dict(path=name, stat_guard=guard(name), sha256=sha(ROOT/name))
        same(row); protected.append(row)
    excluded_prefixes = ['artifacts/numerics-', 'artifacts/cleanup-numerics-',
                         'artifacts/disk-cleanup-', 'artifacts/storage-finalization-']
    excluded_dirs = {str(Path(n).parent) for n in KEEP} | {NATIVE}
    archives, excluded = [], []
    for parent, directories, files in os.walk(ROOT/'artifacts', followlinks=False):
        directories[:] = [d for d in directories if not (Path(parent)/d).is_symlink() and d != '.git']
        for filename in files:
            p = Path(parent)/filename
            if p.suffix != '.json' or p.is_symlink() or p.stat().st_size < 1 << 20:
                continue
            name = str(p.relative_to(ROOT))
            reason = None
            if str(p) in tracked:
                reason = 'tracked_file_in_registered_worktree'
            elif any(name.startswith(prefix) for prefix in excluded_prefixes):
                reason = 'recent_diagnostic_or_cleanup_identity'
            elif set(p.relative_to(ROOT).parts) & {'source', 'build-source', 'frozen-source', 'build'}:
                reason = 'source_or_build_snapshot'
            elif any(name.startswith(d+'/') for d in excluded_dirs):
                reason = 'retained_local_witness_package'
            elif name in {'artifacts/ffn-medium-cpu-qualification-v1/qualification/result.json',
                          'artifacts/memory-npu-qual-bf16-v1/qualification.json'}:
                reason = 'readable_diagnostic_context'
            job_root = p.parent
            while job_root != ROOT:
                if (job_root/'status.json').is_file() and (job_root/'launch.json').is_file():
                    break
                job_root = job_root.parent
            if reason is None and job_root == ROOT:
                reason = 'readable_aggregate_without_own_terminal_job'
            if reason:
                excluded.append(dict(path=name, reason=reason, bytes=p.stat().st_size))
                continue
            assert not p.with_suffix('.json.gz').exists(), name
            terminal_path = job_root/'status.json'
            terminal = json.loads(terminal_path.read_text())
            assert terminal['state'] in {'passed', 'failed', 'terminated'}, str(terminal_path)
            archives.append(dict(path=name, stat_guard=guard(name), archive_path=name+'.gz',
                terminal_status=str(terminal_path.relative_to(ROOT)), terminal_state=terminal['state'],
                terminal_status_sha256=sha(terminal_path)))
    rows = retire + archives
    live = live_references(rows)
    assert not live['hits'], live['hits']
    user = {p:sha(ROOT/p) for p in USER_DOCS}
    assert user == USER_DOCS
    total = int(subprocess.check_output(['du', '-s', '-B1', str(ROOT)]).split()[0])
    plan = dict(schema='tide-storage-finalization-plan-v1', state='prepared', created_utc=now(),
        source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        original_plan_sha256=sha(old_path), repository_allocated_bytes_before=total,
        retire=retire, archives=sorted(archives, key=lambda r:r['path']), excluded_large_json=excluded,
        protected=protected, user_docs=user, worktrees=worktrees, live_observation=live,
        limits=dict(workers=8, cpu_affinity=list(range(176,184)), max_rss_gib=16, wall_seconds=2700),
        qualification='Retirement does not turn historical failure into qualification; no promise to regenerate retired full dumps.',
        sums=dict(retire_files=len(retire), retire_allocated_bytes=sum(r['stat_guard']['allocated_bytes'] for r in retire),
                  archive_files=len(archives), archive_allocated_bytes=sum(r['stat_guard']['allocated_bytes'] for r in archives)))
    atomic(PLAN, plan)
    print(json.dumps(dict(plan_sha256=sha(PLAN), **plan['sums']), indent=2))


def archive_one(row):
    same(row)
    path, output = ROOT/row['path'], ROOT/row['archive_path']
    fd, temporary = tempfile.mkstemp(prefix='.archive-', dir=output.parent)
    stage = Path(temporary)
    h, size = hashlib.sha256(), 0
    try:
        with os.fdopen(fd, 'wb') as target, path.open('rb') as source:
            with gzip.GzipFile(filename='', mode='wb', compresslevel=6, fileobj=target, mtime=0) as zipped:
                for block in iter(lambda: source.read(4 << 20), b''):
                    h.update(block); size += len(block); zipped.write(block)
            target.flush(); os.fsync(target.fileno())
        same(row)
        check, length = hashlib.sha256(), 0
        with gzip.open(stage, 'rb') as stream:
            for block in iter(lambda: stream.read(4 << 20), b''):
                check.update(block); length += len(block)
        assert length == size == row['stat_guard']['bytes'] and check.digest() == h.digest()
        archive_sha = sha(stage)
        os.link(stage, output); stage.unlink(); sync_directory(output.parent)
        # Original remains until the complete archive index is fsynced.
        return dict(path=row['path'], archive_path=row['archive_path'], original_bytes=size,
                    original_sha256=h.hexdigest(), archive_sha256=archive_sha,
                    archive_bytes=output.stat().st_size, archive_allocated_bytes=output.stat().st_blocks*512,
                    original_stat_guard=row['stat_guard'], verified_full_decompression=True)
    finally:
        if stage.exists():
            stage.unlink()


def execute(expected):
    assert sha(PLAN) == expected
    plan = json.loads(PLAN.read_text())
    assert not (HERE/'result.json').exists() and not INDEX.exists()
    tracked, worktrees = git_inventory()
    assert worktrees == plan['worktrees'], 'worktree state changed'
    rows = plan['retire'] + plan['archives']
    for row in rows:
        same(row)
        assert str(ROOT/row['path']) not in tracked
        if 'terminal_status' in row:
            assert sha(ROOT/row['terminal_status']) == row['terminal_status_sha256']
    for row in plan['protected']:
        same(row); assert sha(ROOT/row['path']) == row['sha256']
    assert {p:sha(ROOT/p) for p in USER_DOCS} == USER_DOCS
    live = live_references(rows)
    atomic(HERE/'live-pre-execution.json', live)
    assert not live['hits'], live['hits']
    receipt = []
    print('Hashing fixed retirement inventory', flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(sha, ROOT/r['path']):r for r in plan['retire']}
        for f in as_completed(futures):
            row = futures[f]; same(row)
            receipt.append(dict(**row, sha256=f.result()))
    receipt.sort(key=lambda r:r['path'])
    atomic(HERE/'retirement-identities.json', dict(schema='tide-retired-payloads-v1', plan_sha256=expected,
        qualification=plan['qualification'], files=receipt))
    print('Compressing and full-byte verifying JSON records', flush=True)
    records = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(archive_one, r) for r in plan['archives']]
        for f in as_completed(futures):
            records.append(f.result())
            if len(records) % 50 == 0:
                print(f'verified archives {len(records)}/{len(futures)}', flush=True)
    records.sort(key=lambda r:r['path'])
    atomic(INDEX, dict(schema='tide-json-archive-index-v1', created_utc=now(), source=plan['source'],
        plan_sha256=expected, note='Exact raw bytes; qualifications unchanged. Retired tensors are not reconstructed.', records=records))
    # Mutation happens only after every original has a durable identity/receipt.
    live = live_references(rows)
    atomic(HERE/'live-pre-unlink.json', live)
    assert not live['hits'], live['hits']
    removed, directories = [], set()
    with (HERE/'unlink-receipts.jsonl').open('x') as log:
        for row in rows:
            same(row)
            path = ROOT/row['path']
            log.write(json.dumps(dict(path=row['path'], action='unlink_intent', utc=now()))+'\n')
            log.flush(); os.fsync(log.fileno())
            path.unlink(); directories.add(path.parent); removed.append(row['path'])
        for directory in directories:
            sync_directory(directory)
        log.write(json.dumps(dict(action='complete', files=len(removed), utc=now()))+'\n')
        log.flush(); os.fsync(log.fileno())
    for row in plan['protected']:
        same(row); assert sha(ROOT/row['path']) == row['sha256']
    assert {p:sha(ROOT/p) for p in USER_DOCS} == USER_DOCS
    savings = sum(r['original_stat_guard']['allocated_bytes']-r['archive_allocated_bytes'] for r in records)
    result = dict(state='completed', finished_utc=now(), source=plan['source'], plan_sha256=expected,
        retired_files=len(receipt), retired_allocated_bytes=plan['sums']['retire_allocated_bytes'],
        archives=len(records), archive_original_allocated_bytes=plan['sums']['archive_allocated_bytes'],
        archive_allocated_bytes=sum(r['archive_allocated_bytes'] for r in records),
        archive_savings_allocated_bytes=savings, total_gross_freed_bytes=plan['sums']['retire_allocated_bytes']+savings,
        archive_index_sha256=sha(INDEX), retirement_identities_sha256=sha(HERE/'retirement-identities.json'),
        protected_verified=len(plan['protected']), user_docs_verified=True,
        live_inputs=live, qualification=plan['qualification'])
    atomic(HERE/'result.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--execute-sha')
    args = parser.parse_args()
    assert args.prepare != bool(args.execute_sha)
    prepare() if args.prepare else execute(args.execute_sha)

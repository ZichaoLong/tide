#!/usr/bin/env python3
"""Verify six pinned public checkpoints, then stage by repo/revision with aliases."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.review_io import write_review_json
from scripts.retire_priority_artifacts import inspect_jobs
from tide_lab.records import atomic_json, now


def inspect_file(path, expected=None):
    s = path.lstat()
    if not stat.S_ISREG(s.st_mode) or s.st_nlink != 1 or s.st_uid != os.getuid():
        raise ValueError(f'only owned ordinary files are admitted: {path}')
    if expected and s.st_size != expected['bytes']:
        raise ValueError(f'catalog byte length mismatch: {path}')
    sha = hashlib.sha256()
    blob = hashlib.sha1(f'blob {s.st_size}\0'.encode())
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''):
            sha.update(b)
            if expected and not expected.get('sha256'):
                blob.update(b)
    if expected:
        if expected.get('sha256') and sha.hexdigest() != expected['sha256']:
            raise ValueError(f'upstream LFS SHA256 mismatch: {path}')
        if not expected.get('sha256') and expected.get('git_blob') != blob.hexdigest():
            raise ValueError(f'upstream Git blob mismatch: {path}')
    after = path.stat()
    if (s.st_ino, s.st_size, s.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError(f'file changed while hashing: {path}')
    return dict(bytes=s.st_size, allocated_bytes=s.st_blocks*512, sha256=sha.hexdigest(),
                inode=s.st_ino, device=s.st_dev, mtime_ns=s.st_mtime_ns,
                upstream_verified=bool(expected), upstream_sha256=expected.get('sha256') if expected else None)


def relocate(source, destination):
    if source.is_symlink() or source.resolve() != source or not source.is_dir():
        raise ValueError('source must be an ordinary absolute directory')
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('destination already exists')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.stat().st_dev != destination.parent.stat().st_dev:
        raise ValueError('staging must use a same-filesystem rename')
    os.rename(source, destination)
    try:
        source.symlink_to(destination, target_is_directory=True)
    except BaseException:
        os.rename(destination, source)
        raise
    for p in (source.parent, destination.parent):
        fd = os.open(p, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def checkpoint_files(source):
    paths = sorted(source.rglob('*'))
    if any(p.is_symlink() for p in paths):
        raise ValueError('checkpoint tree must not contain symlinks')
    return [p for p in paths if not p.is_dir()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--models', type=Path, required=True)
    p.add_argument('--catalog', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--staging', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root, models, stage = a.root.resolve(), a.models.absolute(), a.staging.absolute()
    if models.resolve() != models or stage.parent.resolve() != stage.parent:
        raise ValueError('source and staging ancestors must not be symlinks')
    if a.output.exists() or stage.exists() or stage.is_symlink():
        raise FileExistsError('fresh staging/report required')
    expected_models = json.loads(a.catalog.read_text())['models']
    if set(expected_models) != {'qwen05', 'qwen15', 'qwen3', 'qwen7', 'qwenmoe', 'olmoe'}:
        raise ValueError('expected the six audited public checkpoints')
    report = dict(schema='tide-public-model-staging-v1', state='verifying', started_at=now(),
                  catalog_sha256=hashlib.sha256(a.catalog.read_bytes()).hexdigest(),
                  source_catalog=str(a.catalog), staging=str(stage),
                  proposed_destination='/mi/data2T/models/public-checkpoints-20261111',
                  sudo_publication='not_executed', models=[], files=[])
    audit = json.loads(a.audit.read_text())
    report['preflight_jobs'] = inspect_jobs(audit)
    tasks = []
    for alias, model in sorted(expected_models.items()):
        if not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+', model['id']) or not re.fullmatch(r'[a-f0-9]{40}', model['revision']):
            raise ValueError('unsafe model identity')
        source = models/(alias+'-'+model['revision'])
        if source.is_symlink() or not source.is_dir():
            raise ValueError('source checkpoint is absent/already moved')
        rel = Path('huggingface')/model['id']/model['revision']
        if '..' in rel.parts:
            raise ValueError('unsafe model identity')
        expected = {f['path']: f for f in model['files']}
        files = checkpoint_files(source)
        if not set(expected).issubset({str(f.relative_to(source)) for f in files}):
            raise ValueError('missing upstream checkpoint files')
        report['models'].append(dict(alias=alias, repository=model['id'], revision=model['revision'],
                                     original_path=str(source), relative_path=str(rel), state='pending'))
        tasks += [(f, expected.get(str(f.relative_to(source))), str(rel/f.relative_to(source))) for f in sorted(files)]
    write_review_json(a.output, report, root=root)
    try:
        def worker(task):
            path, expected, rel = task
            row = inspect_file(path, expected)
            return dict(row, path=rel, original_path=str(path))
        with ThreadPoolExecutor(max_workers=2) as pool:
            for row in pool.map(worker, tasks):
                report['files'].append(row)
                if row['path'].endswith('.safetensors'):
                    print('verified', row['path'], flush=True)
        report['state'] = 'verified_before_move'
        report['premove_jobs'] = inspect_jobs(audit)
        write_review_json(a.output, report, root=root)
        stage.mkdir()
        for m in report['models']:
            source, dest = Path(m['original_path']), stage/m['relative_path']
            selected = [r for r in report['files'] if r['original_path'].startswith(str(source)+'/')]
            if {str(p) for p in checkpoint_files(source)} != {r['original_path'] for r in selected}:
                raise ValueError('checkpoint file set changed after verification')
            # Recheck every inode and timestamp immediately before rename.
            for row in selected:
                s = Path(row['original_path']).lstat()
                if (s.st_ino, s.st_dev, s.st_size, s.st_mtime_ns) != (
                        row['inode'], row['device'], row['bytes'], row['mtime_ns']):
                    raise ValueError('model changed after verification')
            relocate(source, dest)
            m.update(state='staged_with_original_alias', staged_path=str(dest))
            print('staged', m['repository'], flush=True)
            write_review_json(a.output, report, root=root)
        for row in report['files']:
            a1, b1 = Path(row['original_path']).stat(), (stage/row['path']).stat()
            if (a1.st_ino, a1.st_dev, a1.st_size) != (b1.st_ino, b1.st_dev, row['bytes']):
                raise ValueError('old-path alias does not preserve file identity')
        (stage/'UPSTREAM-CATALOG.json').write_bytes(a.catalog.read_bytes())
        (stage/'SHA256SUMS').write_text(''.join(f"{r['sha256']}  {r['path']}\n" for r in report['files']))
        report.update(state='completed', finished_at=now(),
                      allocated_model_bytes=sum(r['allocated_bytes'] for r in report['files']),
                      weights=len([r for r in report['files'] if r['path'].endswith('.safetensors')]),
                      original_aliases_verified=True, copied_weight_bytes=0)
        atomic_json(stage/'CATALOG.json', report)
    except BaseException as error:
        report.update(state='failed', error=repr(error), finished_at=now())
        raise
    finally:
        write_review_json(a.output, report, root=root)
    print(json.dumps({k: report[k] for k in ('state', 'weights', 'allocated_model_bytes')}), flush=True)


if __name__ == '__main__':
    main()

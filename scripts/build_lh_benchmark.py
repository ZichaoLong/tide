#!/usr/bin/env python3
"""Build the standalone benchmark against an immutable original-LH snapshot."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from durable_records import write_json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources(root):
    paths = sorted((root/'cpp/lh_bench').glob('*')) + [root/'cpp/bench/metrics_jsonl_writer.h']
    return {str(p.relative_to(root)): digest(p) for p in paths if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--build-dir', required=True)
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 4:
        parser.error('build jobs must be in [1,4]')
    root = Path(__file__).resolve().parents[1]
    snapshot, build = Path(args.snapshot).resolve(), Path(args.build_dir).resolve()
    if snapshot == build or snapshot in build.parents or build in snapshot.parents:
        parser.error('build and immutable snapshot must be separate')
    manifest = json.loads((snapshot/'manifest.json').read_text())
    if manifest.get('schema') != 'lh-cpp-snapshot-v1':
        parser.error('LH source snapshot required')
    def verify():
        actual = {str(p.relative_to(snapshot)): digest(p) for p in snapshot.rglob('*')
                  if p.is_file() and p.name != 'manifest.json'}
        if actual != manifest['files']:
            raise ValueError('LH snapshot changed')
        if hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest() != manifest['identity']:
            raise ValueError('LH snapshot identity mismatch')
    verify()
    before = sources(root)
    os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
    import torch
    subprocess.run(['cmake', '-S', str(root/'cpp/lh_bench'), '-B', str(build), '-G', 'Ninja',
                    '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_PREFIX_PATH='+torch.utils.cmake_prefix_path,
                    '-DTIDE_LH_SNAPSHOT='+str(snapshot)], check=True)
    subprocess.run(['cmake', '--build', str(build), '--parallel', str(args.jobs)], check=True)
    verify()
    if sources(root) != before:
        raise ValueError('benchmark source changed during build')
    write_json(build/'build-manifest.json', dict(schema='lh-bench-build-v1', sources=before,
                binary_sha256=digest(build/'tide-lh-bench'), lh_snapshot=manifest, torch=torch.__version__,
                cxx11_abi=torch.compiled_with_cxx11_abi(), assertion_policy='Release; runtime assertions off',
                parallel_policy='original ENABLE_PARALLEL_FOR with OpenMP'))


if __name__ == '__main__':
    main()

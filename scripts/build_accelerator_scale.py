#!/usr/bin/env python3
"""Build only the benchmark consumer against a verified, installed native core."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from accelerator_scale_identity import client_hash
from build_identity import source_hash
from build_environment import cache_values
from durable_records import write_json
from source_identity import source_state, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-build', type=Path, required=True)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--jobs', type=int, choices=(1, 2, 4, 8), default=2)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    core, build = args.core_build.resolve(), args.build_dir.resolve()
    manifest = json.loads((core / 'build-manifest.json').read_text())
    if manifest['cpp_source_sha256'] != source_hash(root):
        parser.error('core source hash differs; rebuild the core first')
    if manifest['backend'] == 'npu' and manifest['npu_runtime'] != 'standalone':
        parser.error('standalone benchmark needs the standalone NPU SDK build')
    for name, expected in manifest['binary_sha256'].items():
        if digest(core / name) != expected:
            parser.error('core artifact changed: ' + name)
    identity, (commit, dirty) = client_hash(root), source_state(root)
    build.mkdir(parents=True, exist_ok=False)
    prefix = build / 'core-prefix'
    subprocess.run(['cmake', '--install', str(core), '--prefix', str(prefix)], check=True)
    package, = prefix.glob('lib*/cmake/TideGraph')
    prefixes = [str(prefix), *os.environ.get('CMAKE_PREFIX_PATH', '').split(os.pathsep)]
    torch_dir = cache_values(core)['Torch_DIR']
    subprocess.run(['cmake', '-S', str(root / 'tools/accelerator_scale'), '-B', str(build),
                    '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DTideGraph_DIR=' + str(package),
                    '-DTorch_DIR=' + torch_dir,
                    '-DCMAKE_PREFIX_PATH=' + ';'.join(p for p in prefixes if p)], check=True)
    subprocess.run(['cmake', '--build', str(build), '--parallel', str(args.jobs)], check=True)
    subprocess.run(['ctest', '--test-dir', str(build), '--output-on-failure'], check=True, timeout=120)
    binary = build / 'tide-accelerator-scale'
    closure = subprocess.check_output(['ldd', str(binary)], text=True)
    if any(item in closure.lower() for item in ('not found', 'libtorch_python', 'libpython', '/stubs/', '/stub/')):
        raise RuntimeError('standalone benchmark loader has unresolved, Python or stub dependencies')
    (build / 'loader.txt').write_text(closure)
    if client_hash(root) != identity or source_state(root) != (commit, dirty):
        raise RuntimeError('benchmark source changed during build')
    write_json(build / 'client-manifest.json', dict(schema='tide-accelerator-scale-build-v1',
        source=commit, dirty=dirty, client_source_sha256=identity, core=manifest,
        binary_sha256=digest(binary), loader_sha256=digest(build / 'loader.txt'),
        additional_binaries={name: digest(build / name) for name in ('tide-bounded-schedule-check','tide-bounded-scale')
                             if (build / name).is_file()}))
    print('Built standalone benchmark:', binary)


if __name__ == '__main__':
    main()

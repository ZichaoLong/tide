#!/usr/bin/env python3
"""Build installed eager/optional resident consumers with exact core identities."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from build_environment import cache_values
from build_identity import source_hash
from durable_records import write_json, replace_text
from source_identity import source_state, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--core-build', type=Path, required=True)
    p.add_argument('--resident-build', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--jobs', type=int, choices=(1, 2, 4, 8), default=2)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    core, out = a.core_build.resolve(), a.output_dir.resolve()
    record = json.loads((core / 'build-manifest.json').read_text())
    if record['cpp_source_sha256'] != source_hash(root):
        p.error('core source differs; rebuild first')
    if record['backend'] == 'npu' and record.get('npu_runtime') != 'standalone':
        p.error('standalone consumer requires the standalone NPU core')
    builds = [(core, record)]
    if a.resident_build:
        from build_device_control import component_hash
        backend = a.resident_build.resolve()
        resident = json.loads((backend / 'control-build.json').read_text())
        if (resident['core'] != record or resident.get('runtime', resident.get('npu_runtime')) != 'standalone'
                or resident['component_sha256'] != component_hash(root)):
            p.error('resident build differs from core/source/runtime')
        builds.append((backend, resident))
    for directory, manifest in builds:
        for name, expected in manifest['binary_sha256'].items():
            if digest(directory / name) != expected:
                p.error('changed build artifact: ' + name)
    source, dirty = source_state(root)
    out.mkdir(parents=True, exist_ok=False)
    prefix, build = out / 'prefix', out / 'consumer'
    result = dict(schema='tide-online-consumer-build-v1', source=source, dirty=dirty,
                  state='running', core=record, commands=[],
                  resident=builds[1][1] if len(builds) == 2 else None,
                  consumer_sources={f.name:digest(f) for f in (root / 'tools/online_bench').iterdir() if f.is_file()})
    def run(command):
        command = list(map(str, command));result['commands'].append(command)
        write_json(out / 'result.json', result);subprocess.run(command, check=True)
    try:
        run(['cmake', '--install', core, '--prefix', prefix])
        if a.resident_build:
            # Only public resident package; vendor kernel archives are embedded.
            run(['cmake', '-DCMAKE_INSTALL_PREFIX='+str(prefix), '-DCMAKE_INSTALL_LOCAL_ONLY=TRUE',
                 '-P', a.resident_build.resolve() / 'cmake_install.cmake'])
        prefixes = [str(prefix), *os.environ.get('CMAKE_PREFIX_PATH', '').split(os.pathsep)]
        run(['cmake', '-S', root / 'tools/online_bench', '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
             '-DTIDE_ONLINE_RESIDENT='+('ON' if a.resident_build else 'OFF'), '-DBUILD_TESTING=ON',
             '-DTorch_DIR='+cache_values(core)['Torch_DIR'], '-DCMAKE_PREFIX_PATH='+';'.join(x for x in prefixes if x)])
        run(['cmake', '--build', build, '--parallel', a.jobs])
        names = ['tidegraph-online-bench', 'tidegraph-eager-capacity-probe', 'tidegraph-eager-precision-probe']
        result['binary_sha256'], result['loader_sha256'] = {}, {}
        for name in names:
            loader = subprocess.check_output(['ldd', str(build / name)], text=True)
            if any(x in loader.lower() for x in ('not found', 'libtorch_python', 'libpython', '/stub/', '/stubs/', '/simulator/')):
                raise RuntimeError('standalone consumer loader closure failed: ' + name)
            replace_text(out / (name+'-loader.txt'), loader)
            result['binary_sha256'][name] = digest(build / name)
            result['loader_sha256'][name] = digest(out / (name+'-loader.txt'))
        if source_state(root) != (source, dirty):
            raise RuntimeError('source changed during consumer build')
        result['state'] = 'passed'
    except BaseException as error:
        result.update(state='failed', error=repr(error));raise
    finally:
        write_json(out / 'result.json', result)


if __name__ == '__main__':
    main()

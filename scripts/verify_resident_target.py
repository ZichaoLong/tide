#!/usr/bin/env python3
"""Complete resident target gate: real devices, both runtime owners, no skips.

Run on a clean fixed source under a bounded durable job. GPU compilation and
CPU contracts do not substitute for this gate. Profiling/performance are separate.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

from build_device_control import component_hash
from build_identity import source_hash
from device_component_checks import CHECKS
from durable_records import write_json
from source_identity import source_state, digest


def check_build(root, build, *, backend, owner):
    record = json.loads((build / 'control-build.json').read_text())
    if (record.get('backend', 'npu') != backend or
            record.get('runtime', record.get('npu_runtime')) != owner or
            record['component_sha256'] != component_hash(root) or
            record['core']['cpp_source_sha256'] != source_hash(root)):
        raise ValueError('resident build does not match source/backend/runtime owner')
    for name, expected in record['binary_sha256'].items():
        if digest(build / name) != expected:
            raise ValueError('resident binary changed: ' + name)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', required=True, help='explicit indexed cuda:N or npu:N')
    p.add_argument('--python-core-build', type=Path, required=True)
    p.add_argument('--python-resident-build', type=Path, required=True)
    p.add_argument('--standalone-resident-build', type=Path, required=True)
    p.add_argument('--online-build', type=Path, required=True,
                   help='build_online_consumer.py output, with resident enabled')
    p.add_argument('--installed-consumer-build', type=Path, required=True,
                   help='build_resident_consumer.py output')
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    if not re.fullmatch(r'(?:cuda|npu):(0|[1-9][0-9]*)', a.device):
        p.error('an explicit indexed CUDA/NPU target is required')
    root = Path(__file__).resolve().parents[1]
    source, dirty = source_state(root)
    if dirty:
        p.error('qualification requires a clean immutable source')
    backend = a.device.split(':')[0]
    python_build = a.python_resident_build.resolve()
    standalone = a.standalone_resident_build.resolve()
    core = a.python_core_build.resolve()
    py_record = check_build(root, python_build, backend=backend, owner='python')
    cpp_record = check_build(root, standalone, backend=backend, owner='standalone')
    core_record = json.loads((core/'build-manifest.json').read_text())
    if core_record != py_record['core']:
        p.error('Python core is not the one recorded by its resident build')
    for name, expected in core_record['binary_sha256'].items():
        if digest(core/name) != expected:
            p.error('Python core binary changed: ' + name)
    online = a.online_build.resolve()
    online_record = json.loads((online/'result.json').read_text())
    if online_record.get('state') != 'passed' or online_record.get('resident') != cpp_record:
        p.error('online consumer requires the exact standalone resident build')
    binary = online/'consumer/tidegraph-online-bench'
    if digest(binary) != online_record['binary_sha256'].get(binary.name):
        p.error('online consumer binary changed')
    installed = a.installed_consumer_build.resolve()
    installed_record = json.loads((installed/'result.json').read_text())
    client = installed/'consumer/resident-consumer'
    if (installed_record.get('state') != 'passed' or installed_record.get('backend') != cpp_record
            or digest(client) != installed_record['binary_sha256']):
        p.error('installed consumer differs from its resident build')
    out = a.output_dir.resolve();out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='tide-resident-target-v1', source=source, dirty=dirty,
                  device=a.device, state='running', stages=[],
                  python_build=py_record, standalone_build=cpp_record,
                  scope='complete declared resident correctness; independent CPU oracles, FP32/FP16, three families, both schedules, VJPs, optimizers, windows, checkpoints, capacity, 1/2/3 owners; no performance or profiling claim')
    write_json(out/'result.json', report)
    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD='0', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONPATH=os.pathsep.join((str(root/'python'), str(core))),
               TIDE_BUILD_DIR=str(core), TIDE_RESIDENT_DEVICE=a.device,
               TIDE_RESIDENT_LIBRARY=str(python_build), TIDE_ONLINE_DEVICE=a.device,
               TIDE_ONLINE_BINARY=str(binary))

    def run(name, command, timeout):
        row = dict(name=name, command=list(map(str, command)), state='running', timeout_seconds=timeout)
        report['stages'].append(row);write_json(out/'result.json', report)
        with (out/(name+'.log')).open('x') as log:
            result = subprocess.run(row['command'], cwd=root, env=env,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        row.update(exit_code=result.returncode, state='passed' if result.returncode == 0 else 'failed',
                   log_sha256=digest(out/(name+'.log')))
        write_json(out/'result.json', report)
        if result.returncode:
            raise RuntimeError(name + ' failed; inspect its retained log')

    try:
        # A requested target never succeeds by skipping unavailable hardware.
        preflight = ('import torch;from tidegraph.runtime import resolve_device;'
                     f'd,_=resolve_device({a.device!r});api=getattr(torch,d.type);'
                     'assert api.device_count()>=d.index+3,"three consecutive devices required";'
                     'x=torch.ones(3,device=d);assert (x+x).cpu().tolist()==[2,2,2]')
        run('preflight', [sys.executable, '-c', preflight], 60)
        run('components', [sys.executable, root/'scripts/verify_device_control.py',
            '--device', a.device, '--build-dir', standalone, '--output-dir', out/'components',
            '--checks', *CHECKS], 7200)
        run('installed-client', [client, '--device='+a.device, '--dtype=float32',
                                 '--training-devices=2'], 180)
        tests = sorted(str(f.relative_to(root)) for f in (root/'tests').glob('test_resident_*.py'))
        tests += ['tests/test_online_resident_consumer.py', 'tests/test_online_resident_chunking.py',
                  'tests/test_online_consumer_memory.py']
        run('public-consumers', [sys.executable, '-m', 'pytest', *tests, '-q', '--dtype', 'both',
            '--junitxml', out/'public.xml', '--basetemp', out/'test-tmp'], 7200)
        cases = ET.parse(out/'public.xml').findall('.//testcase')
        if not cases or any(c.find('skipped') is not None for c in cases):
            raise RuntimeError('complete target gate requires nonempty public tests with zero skips')
        if source_state(root) != (source, dirty):
            raise RuntimeError('source changed during qualification')
        report.update(state='passed', public_cases=len(cases), skipped=0)
    except BaseException as error:
        report.update(state='failed', error=repr(error));raise
    finally:
        write_json(out/'result.json', report)


if __name__ == '__main__':
    main()

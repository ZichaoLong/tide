#!/usr/bin/env python3
"""Real CANN programs with deterministic API failures; no stuck kernels."""
import argparse
import json
from pathlib import Path
import subprocess

from build_device_control import component_hash
from build_identity import source_hash
from durable_records import write_json
from source_identity import digest, source_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', required=True)
    args = parser.parse_args()
    if args.device != 'npu' and not args.device.startswith('npu:'):
        parser.error('failure gate requires explicit NPU')
    root = Path(__file__).resolve().parents[1]
    build = args.build_dir.resolve()
    manifest = json.loads((build/'control-build.json').read_text())
    binary = build/'tide-device-failure-check'
    if (manifest['component_sha256'] != component_hash(root)
            or manifest['core']['cpp_source_sha256'] != source_hash(root)
            or manifest['binary_sha256'].get(binary.name) != digest(binary)):
        parser.error('failure gate build/source fingerprint mismatch')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source, dirty = source_state(root)
    report = dict(schema='tide-device-failures-v1', source=source, dirty=dirty,
                  build=manifest, state='running', cases=[],
                  scope='injected API failures on real finite CANN programs; not a physical hang/reset qualification')
    write_json(out/'result.json', report)
    try:
        for name, extra, code, marker in (
            ('recoverable', [], 0, 'device-failure: passed recoverable_cases=5'),
            ('quarantine', ['--quarantine'], 86, 'device-quarantine: checked retained_owners=true finalize_refused=true worker_exit=86'),
        ):
            command = [str(binary), '--device='+args.device, '--dtype=float32', *extra]
            log_path = out/(name+'.log')
            with log_path.open('w') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=60)
            text = log_path.read_text()
            passed = result.returncode == code and marker in text
            if 'fall back to run on the CPU' in text or 'npu_cpu_fallback' in text:
                passed = False
            report['cases'].append(dict(name=name, command=command, expected_exit=code,
                exit_code=result.returncode, state='passed' if passed else 'failed',
                log=log_path.name, log_sha256=digest(log_path)))
            write_json(out/'result.json', report)
            if not passed:
                raise RuntimeError('device failure gate failed: '+str(log_path))
            print(name, 'passed', flush=True)
        if source_state(root) != (source, dirty):
            raise RuntimeError('source changed during failure gate')
        report['state'] = 'passed'
    except BaseException as error:
        report.update(state='failed', error=repr(error))
        raise
    finally:
        write_json(out/'result.json', report)


if __name__ == '__main__':
    main()

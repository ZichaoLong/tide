#!/usr/bin/env python3
"""Finite original-LH Attention bring-up and two-scale execution pipeline."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
from durable_records import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--wide-graph-input', required=True)
    p.add_argument('--narrow-graph-input', required=True)
    p.add_argument('--json-include', required=True)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    stages = []
    def run(name, command, timeout):
        start = time.monotonic()
        print('START', name, flush=True)
        result = dict(name=name, command=command, state='running')
        stages.append(result)
        write_json(out / 'pipeline.json', dict(stages=stages, complete=False))
        try:
            code = subprocess.run(command, timeout=timeout).returncode
            result.update(state='passed' if code == 0 else 'failed', exit_code=code)
        except subprocess.TimeoutExpired:
            result.update(state='failed', exit_code=124, error='stage timeout')
        result['elapsed_seconds'] = time.monotonic() - start
        write_json(out / 'pipeline.json', dict(stages=stages, complete=False))
        print('DONE', result, flush=True)
        return result['exit_code'] == 0
    def build(label, profile, extra=()):
        return run('build-' + label, [sys.executable, 'scripts/build_lh_original.py',
            '--source', args.source, '--revision', 'a10fdb1', '--profile', profile,
            '--graph-input', args.wide_graph_input if profile == 'wide' else args.narrow_graph_input,
            '--json-include', args.json_include, '--output-dir', str(out / (label + '-prepared')),
            '--jobs', '2', *extra], 1200)
    def measure(label, mode, small=False):
        return run(label + '-' + mode, [sys.executable, 'scripts/benchmark_lh_original.py',
            '--device', 'cpu', '--dtype', 'float32', '--prepared', str(out / (label + '-prepared')),
            '--output-dir', str(out / (label + '-' + mode)), '--mode', mode,
            '--threads', '8' if small else '160', '--blas-threads', '1',
            '--memory-gib', '32' if small else '1280', '--timeout-seconds', '120' if small else '1800'],
            180 if small else 1920)
    if not build('smoke', 'wide', ['--width', '64', '--batch', '4', '--steps', '4']):
        return 1
    for mode in ('nograd', 'grad-forward'):
        if not measure('smoke', mode, small=True):
            return 1
    for profile in ('wide', 'narrow'):
        if not build(profile, profile):
            return 1
        if measure(profile, 'nograd'):
            measure(profile, 'grad-forward')
    write_json(out / 'pipeline.json', dict(stages=stages, complete=True))
    return int(any(stage['state'] != 'passed' for stage in stages))


if __name__ == '__main__':
    sys.exit(main())

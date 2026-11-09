#!/usr/bin/env python3
"""Finite sequential variants in one resource-owning diagnostic service."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json, now


def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    state = dict(schema='tide-diagnosis-variants-v1', state='running', started_at=now(), cases=[])
    plan = json.loads(a.plan.read_text())
    if not 1 <= len(plan['cases']) <= 5:
        raise ValueError('bounded diagnostic requires one to five declared variants')
    for case in plan['cases']:
        path = a.output / case['name']
        command = [sys.executable, 'scripts/diagnose_campaign_execution.py',
            '--config', str(a.config), '--output', str(path), '--device', 'npu:0',
            '--implementation', 'native', '--placement', case.get('placement', 'mixed-c'),
            '--native-library', str(a.native_library), '--batch', '8', '--length', '16',
            '--threads', '2', '--updates', '3', '--lens-cli', a.lens_cli]
        if case.get('width'):
            command += ['--width', str(case['width'])]
        if case.get('native_transport'):
            command += ['--native-transport']
        row = dict(name=case['name'], state='running', started_at=now(), command=command)
        state['cases'].append(row)
        atomic_json(a.output / 'state.json', state)
        try:
            with (a.output / (case['name'] + '.log')).open('x') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=case['timeout_seconds'])
            row.update(state='completed' if result.returncode == 0 else 'failed', returncode=result.returncode)
        except subprocess.TimeoutExpired:
            row.update(state='timed_out')
        row['finished_at'] = now()
        atomic_json(a.output / 'state.json', state)
    state.update(state='completed_with_failures' if any(r['state'] != 'completed' for r in state['cases']) else 'completed', finished_at=now())
    atomic_json(a.output / 'state.json', state)
    return 0 if state['state'] == 'completed' else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['plan', 'config', 'output', 'native-library']:
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--lens-cli', required=True)
    raise SystemExit(main(p.parse_args()))

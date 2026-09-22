#!/usr/bin/env python3
"""Fixed graph-only PDG ramp; no weights imported, no adaptive parameter search."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from build_identity import revision
from durable_records import write_json
from experiment_record import utc_now


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', required=True, choices=['cpu'])
    p.add_argument('--wide-topology', required=True)
    p.add_argument('--narrow-topology', required=True)
    p.add_argument('--build-dir', required=True)
    p.add_argument('--output-dir', required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]; out = Path(a.output_dir).resolve()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=root).strip():
        p.error('clean source required')
    out.mkdir(parents=True, exist_ok=False)
    record = dict(source=revision(root), state='running', started=utc_now(), cases=[],
                  scope='graph-only comparable-scale Attention; fixed seeded inputs; no LH numerical equivalence')
    def save(): write_json(out/'pilot.json', record)
    save(); exit_code = 0
    def run(name, overrides, expected_timeout=False):
        config = dict(dtype='float32', topology=a.wide_topology, width=64, batch=4, steps=4,
                      warmup=1, workers=3, threads=1, packed=1, grad=0, check=0, vocab=50304,
                      emission='row', seed=7, timeout_seconds=180, memory_gib=16)
        config.update(overrides)
        command = [sys.executable, str(root/'scripts/benchmark_pdg_scale.py'), '--device', a.device,
                   '--build-dir', str(Path(a.build_dir).resolve()), '--output-dir', str(out/name)]
        for k, v in config.items(): command += ['--'+k.replace('_', '-'), str(v)]
        print('START', name, flush=True)
        code = subprocess.run(command, cwd=root).returncode
        summary_path = out/name/'summary.json'
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        passed = code == 0
        if expected_timeout:
            passed = (code != 0 and summary.get('status') == 'failed'
                      and 'TimeoutExpired' in (summary.get('error') or '')
                      and summary.get('native_exit_code') is not None and not summary.get('unreaped_child_pid'))
        item = dict(name=name, command=command, exit_code=code, expectation='timeout' if expected_timeout else 'complete',
                    accepted=passed, finished=utc_now())
        record['cases'].append(item); save(); print('DONE', item, flush=True)
        if summary.get('unreaped_child_pid') or not summary:
            raise RuntimeError('missing terminal child evidence; do not overlap another case')
        return passed
    try:
        prerequisites = [
            ('timeout-finalization', dict(width=16, vocab=257, steps=1000, warmup=0, timeout_seconds=1), True),
            ('smoke-fp64', dict(dtype='float64', width=16, vocab=257, check=1), False),
            ('smoke-fp32', dict(width=16, vocab=257, check=1), False),
            ('smoke-grad-forward', dict(width=16, vocab=257, grad=1), False),
            ('wide-ramp-256-b32', dict(width=256, batch=32, steps=6, warmup=2, workers=16, memory_gib=64, timeout_seconds=300), False),
            ('wide-ramp-2048-b64', dict(width=2048, batch=64, steps=8, warmup=2, workers=160, memory_gib=1280, timeout_seconds=1200), False)]
        for name, config, timeout in prerequisites:
            if not run(name, config, timeout): raise RuntimeError('prerequisite failed: '+name)
        targets = [
            ('wide-2048-b512-parallel', dict(width=2048, batch=512, steps=12, warmup=4, workers=160, memory_gib=1280, timeout_seconds=1800)),
            ('wide-2048-b512-serial', dict(width=2048, batch=512, steps=4, warmup=1, workers=1, memory_gib=1280, timeout_seconds=600)),
            ('narrow-128-b512-parallel', dict(topology=a.narrow_topology, width=128, batch=512, steps=8, warmup=2,
                                           workers=160, memory_gib=1280, timeout_seconds=1800))]
        for name, config in targets:
            if not run(name, config): exit_code = 1
    except BaseException as exc:
        exit_code = 1; record['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        record.update(state='passed' if exit_code == 0 else 'failed', exit_code=exit_code, finished=utc_now()); save()
    return exit_code


if __name__ == '__main__': sys.exit(main())

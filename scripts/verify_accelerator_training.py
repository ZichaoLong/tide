#!/usr/bin/env python3
"""Complete-window training and optimizer parity for the standalone consumer."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from durable_records import write_json
from source_identity import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-dir', type=Path, required=True)
    p.add_argument('--check-atol', type=float)
    p.add_argument('--check-rtol', type=float)
    p.add_argument('--dtype', choices=('float32', 'float16'), default='float32')
    p.add_argument('--device', choices=('cpu', 'npu'), required=True)
    p.add_argument('--devices', type=int, default=1)
    p.add_argument('--ranking-device', choices=('cpu', 'model'), default='cpu')
    p.add_argument('--event-device', choices=('cpu', 'model'), default='cpu')
    p.add_argument('--control-device', choices=('cpu', 'model'), default='model')
    p.add_argument('--read-device', choices=('cpu', 'model'), default='model')
    p.add_argument('--read-dtype', choices=('float32', 'float64'), default='float32')
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    binary = a.build_dir.resolve() / 'tide-accelerator-scale'
    edges = []
    for src, dst in [(0, 0), (4, 4), (0, 4), (4, 0)]:
        edges.extend((src, dst+i) for i in range(4))
        edges.extend((src+i, dst) for i in range(1, 4))
    edges.append((0, 3))
    topology = out / 'topology.txt'
    topology.write_text('TIDE_PDG_SCALE_1\n4 2 2 1 1 2 '+str(len(edges))+'\n'
                        +''.join(f'{s} {t}\n' for s, t in edges))
    record = dict(schema='tide-accelerator-training-gates-v1', state='running',
                  binary_sha256=digest(binary), device=a.device, devices=a.devices, dtype=a.dtype, check_atol=a.check_atol, check_rtol=a.check_rtol,
                  ranking_device=a.ranking_device, event_device=a.event_device,
                  control_device=a.control_device, read_device=a.read_device,
                  read_dtype=a.read_dtype, npu_task_queue=0, cases=[])
    write_json(out / 'gates.json', record)
    env = dict(os.environ, TASK_QUEUE_ENABLE='0')
    try:
        for memory in ('add', 'attention'):
            for optimizer in ('sgd', 'adamw'):
                name = memory+'-'+optimizer
                cmd = [str(binary), '--device', a.device, '--dtype', a.dtype, '--devices', str(a.devices),
                       '--topology', str(topology), '--output-dir', str(out / name), '--run-id', name,
                       '--width', '8', '--batch', '2', '--vocab', '17', '--steps', '3', '--warmup', '0',
                       '--workers', '3', '--threads', '1', '--memory', memory, '--grad', '1', '--check', '1',
                       '--placement', 'locality', '--transport', 'resident', '--read-device', a.read_device,
                       '--read-dtype', a.read_dtype, '--control-device', a.control_device,
                       '--ranking-device', a.ranking_device, '--event-device', a.event_device,
                       '--full-autograd', 'batched', '--aggregate-autograd', 'batched',
                       '--packed-sources', '1', '--batch-next', '1', '--parallel-regions', '1',
                       '--training-steps', '3', '--training-warmup', '1', '--optimizer', optimizer]
                for key in ('check_atol', 'check_rtol'):
                    if getattr(a,key) is not None: cmd += ['--'+key.replace('_','-'),str(getattr(a,key))]
                item = dict(name=name, command=cmd, state='running'); record['cases'].append(item)
                write_json(out / 'gates.json', record)
                with (out / (name+'.log')).open('x') as log:
                    run = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=900)
                item['exit_code'] = run.returncode
                if run.returncode: raise RuntimeError(f'{name}: exit {run.returncode}')
                if 'CHECK training three complete windows' not in (out / (name+'.log')).read_text():
                    raise ValueError('missing independent training acceptance')
                events = [json.loads(line) for line in (out / name / 'metrics.jsonl').read_text().splitlines()]
                assert len(events) == 3
                assert all(e['metrics']['train/gradient_owners'] > 0 and e['metrics']['perf/backward_seconds'] > 0
                           and e['metrics']['train/optimizer_state_owners'] > 0 for e in events)
                item['state'] = 'passed'; write_json(out / 'gates.json', record)
                print(name, 'passed', flush=True)
        record['state'] = 'passed'
    except BaseException as exc:
        record.update(state='failed', error=f'{type(exc).__name__}: {exc}')
        if record['cases'] and record['cases'][-1]['state'] == 'running': record['cases'][-1]['state'] = 'failed'
        raise
    finally:
        write_json(out / 'gates.json', record)


if __name__ == '__main__': main()

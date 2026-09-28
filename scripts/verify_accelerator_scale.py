#!/usr/bin/env python3
"""Bounded consumer parity, residency, placement and rejection gates."""
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
    p.add_argument('--device', choices=('cpu', 'npu'), required=True)
    p.add_argument('--devices', type=int, default=1)
    p.add_argument('--npu-task-queue', type=int, choices=(0, 1, 2), default=0)
    p.add_argument('--vjp-policy', choices=('strict', 'basis-conditioned'), default='strict')
    p.add_argument('--read-device', choices=('cpu', 'model'), default='cpu')
    p.add_argument('--read-dtype', choices=('float64', 'float32'), default='float64')
    p.add_argument('--control-device', choices=('cpu', 'model'), default='cpu')
    p.add_argument('--ranking-device', choices=('cpu', 'model'), default='cpu')
    p.add_argument('--event-device', choices=('cpu', 'model'), default='cpu')
    p.add_argument('--reference-read-dtype', choices=('matched', 'float64'), default='matched')
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args(); out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    binary = a.build_dir.resolve()/'tide-accelerator-scale'
    edges = []
    for src, dst in [(0, 0), (4, 4), (0, 4), (4, 0)]:
        edges.extend((src, dst+i) for i in range(4))
        edges.extend((src+i, dst) for i in range(1, 4))
    edges.append((0, 3))
    topology = out/'topology.txt'
    topology.write_text('TIDE_PDG_SCALE_1\n4 2 2 1 1 2 '+str(len(edges))+'\n'
                        +''.join(f'{src} {dst}\n' for src, dst in edges))
    base = [str(binary), '--device', a.device, '--devices', str(a.devices), '--dtype', 'float32',
            '--topology', str(topology), '--width', '8', '--batch', '2', '--vocab', '17',
            '--steps', '3', '--warmup', '1', '--workers', '3', '--check', '1', '--vjp-policy', a.vjp_policy,
            '--full-autograd', 'batched', '--aggregate-autograd', 'batched',
            '--packed-sources', '1', '--batch-next', '1']
    for key in ('read_device', 'read_dtype', 'control_device', 'reference_read_dtype', 'ranking_device', 'event_device'):
        base += ['--'+key.replace('_', '-'), getattr(a, key)]
    results = []; mappings = {}; env = dict(os.environ)
    if a.device == 'npu': env['TASK_QUEUE_ENABLE'] = str(a.npu_task_queue)
    record = dict(schema='tide-accelerator-scale-gates-v1', binary_sha256=digest(binary),
                  device=a.device, devices=a.devices, npu_task_queue=a.npu_task_queue,
                  vjp_policy=a.vjp_policy, read_device=a.read_device, read_dtype=a.read_dtype,
                  control_device=a.control_device, reference_read_dtype=a.reference_read_dtype,
                  ranking_device=a.ranking_device,event_device=a.event_device,
                  state='running', cases=results)
    write_json(out/'gates.json', record)
    try:
        transports = ('resident',) if 'model' in (a.read_device, a.control_device,a.ranking_device,a.event_device) else ('resident', 'host')
        for transport in transports:
            for memory in ('add', 'attention'):
                for policy in ('memory', 'locality'):
                    name = f'{transport}-{memory}-{policy}'
                    command = [*base, '--transport', transport, '--memory', memory, '--placement', policy,
                               '--grad', '1', '--run-id', name, '--output-dir', str(out/name)]
                    with (out/(name+'.log')).open('x') as log:
                        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600, env=env)
                    assert 'CHECK complete' in (out/(name+'.log')).read_text()
                    events = [json.loads(line) for line in (out/name/'metrics.jsonl').read_text().splitlines()]
                    assert len(events) == 3 and all(e['metrics']['check/logits_requires_grad'] == 1 for e in events)
                    expected_parameters = ((4*9 if memory == 'attention' else 0)+len(edges))*64+64+len(edges)+3+2*17*8
                    assert all(e['metrics']['model/parameters'] == expected_parameters for e in events)
                    mappings[transport, memory, policy] = json.loads((out/name/'placement.json').read_text())
                    assert all(mappings[transport,memory,policy][key] == getattr(a,key)
                               for key in ('read_device','read_dtype','control_device'))
                    if policy == 'locality':
                        assert mappings[transport,memory,policy]['cut_edges'] <= mappings[transport,memory,'memory']['cut_edges']
                    if a.device == 'npu' and transport == 'resident' and a.devices > 1:
                        assert sum(e['metrics']['transfer/device_to_device_bytes'] for e in events) > 0
                    results.append(dict(name=name, command=command, state='passed'))
                    write_json(out/'gates.json', record); print(name, 'passed', flush=True)
        for key, value in [('--transport', 'invalid'), ('--placement', 'invalid'), ('--devices', '0'), ('--vjp-policy', 'invalid'),
                           ('--ranking-device','invalid'),('--event-device','invalid'),('--read-device','invalid'), ('--control-device','invalid'), ('--read-dtype','float16')]:
            bad = out/('rejected-'+key[2:]); command = [*base, key, value, '--run-id', 'bad', '--output-dir', str(bad)]
            run = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30, env=env)
            assert run.returncode != 0 and not bad.exists()
        record['state'] = 'passed'
    except BaseException as exc:
        record.update(state='failed', error=f'{type(exc).__name__}: {exc}'); raise
    finally:
        write_json(out/'gates.json', record)


if __name__ == '__main__': main()

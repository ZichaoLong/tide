#!/usr/bin/env python3
"""Independent finite-window gates for the optional bounded consumer."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
from accelerator_scale_identity import client_hash
from build_identity import source_hash
from durable_records import write_json
from experiment_record import utc_now
from foundation_lifecycle import run_child
from source_identity import digest, source_state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'npu'), required=True)
    p.add_argument('--dtype', choices=('float32', 'float16'), default='float32')
    p.add_argument('--devices', type=int, default=1)
    p.add_argument('--topology', type=Path)
    p.add_argument('--width', type=int, default=8)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--steps', type=int, default=3)
    p.add_argument('--timeout-seconds', type=int, default=600)
    p.add_argument('--check-atol', type=float)
    p.add_argument('--check-rtol', type=float)
    a = p.parse_args()
    if (not 1 <= a.devices <= 16 or a.device == 'cpu' and a.devices != 1
            or not 4 <= a.width <= 64 or a.width % 4 or not 1 <= a.batch <= 8
            or not 1 <= a.steps <= 6 or not 1 <= a.timeout_seconds <= 1800):
        p.error('invalid bounded parity dimensions/device/timeout')
    if any(v is not None for v in (a.check_atol, a.check_rtol)):
        if a.dtype != 'float16' or any(v is not None and (not math.isfinite(v) or v <= 0)
                                     for v in (a.check_atol, a.check_rtol)):
            p.error('positive finite custom tolerances require FP16')
    root = Path(__file__).resolve().parents[1]; build = a.build_dir.resolve(); out = a.output_dir.resolve()
    binary = build/'tide-bounded-schedule-check'; manifest = json.loads((build/'client-manifest.json').read_text())
    identity, binary_hash = client_hash(root), digest(binary); source, dirty = source_state(root)
    if dirty: raise ValueError('qualification requires a clean frozen commit')
    if (manifest['client_source_sha256'] != identity
            or manifest['core']['cpp_source_sha256'] != source_hash(root)
            or manifest.get('additional_binaries', {}).get(binary.name) != binary_hash
            or a.device == 'npu' and manifest['core']['backend'] != 'npu'):
        raise ValueError('bounded source/build/backend mismatch')
    out.mkdir(parents=True, exist_ok=False); topology = out/'topology.txt'
    if a.topology: shutil.copyfile(a.topology, topology)
    else:
        edges = []
        for src, dst in [(0, 0), (4, 4), (0, 4), (4, 0)]:
            edges.extend((src, dst+i) for i in range(4)); edges.extend((src+i, dst) for i in range(1, 4))
        edges.append((0, 3))  # physical parallel-edge identity
        topology.write_text('TIDE_PDG_SCALE_1\n4 2 2 1 1 2 '+str(len(edges))+'\n'
                            +''.join(f'{s} {t}\n' for s, t in edges))
    record = dict(schema='tide-bounded-scheduler-gates-v2', state='running', source=source, dirty=dirty,
                  client_source_sha256=identity, binary_sha256=binary_hash, topology_sha256=digest(topology),
                  device=a.device, dtype=a.dtype, devices=a.devices, width=a.width, batch=a.batch, steps=a.steps,
                  isolated_vjp_policy='two shared output-independent binary-fraction cotangents plus connected zero',
                  check_atol=a.check_atol if a.check_atol is not None else 1e-3 if a.dtype == 'float16' else 1e-6,
                  check_rtol=a.check_rtol if a.check_rtol is not None else 2e-2 if a.dtype == 'float16' else 1e-5,
                  started=utc_now(), cases=[])
    save = lambda: write_json(out/'gates.json', record)
    save(); cases = []
    if a.device == 'npu' and a.devices == 2: cases.append(('add', 0, 1, 1))
    cases += [(m, t, r, 0) for m in ('add', 'attention') for t in (0, 1)
              for r in ((0, 1) if a.device == 'npu' else (0,))]
    env = dict(os.environ, TASK_QUEUE_ENABLE='0', TORCH_DEVICE_BACKEND_AUTOLOAD='0',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    try:
        for memory, training, replay, peer in cases:
            name = f'{memory}-train{training}-replay{replay}-peer{peer}'
            config = dict(device=a.device, dtype=a.dtype, devices=a.devices, topology=str(topology), memory=memory,
                          width=a.width, batch=a.batch, vocab=17, steps=a.steps, warmup=0, seed=7,
                          output_dir=str(out/name), run_id=name, training=training, replay=replay, peer_check=peer)
            if a.dtype == 'float16': config.update(check_atol=record['check_atol'], check_rtol=record['check_rtol'])
            cmd = [str(binary)]
            for key, value in config.items(): cmd += ['--'+key.replace('_', '-'), str(value)]
            cell = dict(name=name, command=cmd, state='running', started=utc_now()); record['cases'].append(cell); save()
            audit = {}
            with (out/(name+'.log')).open('x') as log:
                code = run_child(cmd, cwd=root, env=env, log=log, affinity=sorted(os.sched_getaffinity(0)),
                                 timeout=a.timeout_seconds, memory_budget=16*2**30, audit=audit)
            write_json(out/(name+'-lifecycle.json'), audit)
            cell.update(state='passed' if code == 0 else 'failed', exit_code=code, finished=utc_now()); save()
            if code: raise RuntimeError(name+': native exit '+str(code))
            if 'PASS ' not in (out/(name+'.log')).read_text(): raise ValueError('missing native gate acceptance')
            print(name, 'passed', flush=True)
        if client_hash(root) != identity or digest(binary) != binary_hash or source_state(root) != (source, dirty):
            raise ValueError('source/build changed during qualification')
        record['state'] = 'passed'
    except BaseException as exc:
        record.update(state='failed', error=f'{type(exc).__name__}: {exc}')
        if record['cases'] and record['cases'][-1]['state'] == 'running': record['cases'][-1]['state'] = 'failed'
        raise
    finally:
        record['finished'] = utc_now(); save()


if __name__ == '__main__': main()

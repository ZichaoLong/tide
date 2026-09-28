#!/usr/bin/env python3
"""Record a bounded standalone CPU/NPU node-sharding benchmark."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import statistics
import subprocess
import sys
import uuid
from accelerator_scale_identity import client_hash
from build_identity import source_hash
from durable_records import write_json
from experiment_record import LocalTrackio, read_events, utc_now
from foundation_lifecycle import run_child
from source_identity import source_state, digest


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', required=True, choices=('cpu', 'npu'))
    p.add_argument('--build-dir', type=Path, required=True)
    p.add_argument('--topology', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--tracking-root', type=Path, required=True)
    p.add_argument('--tracking', choices=('off', 'best-effort', 'required'), default='best-effort')
    p.add_argument('--memory', choices=('add', 'attention'), required=True)
    p.add_argument('--placement', choices=('memory', 'locality'), default='memory')
    p.add_argument('--transport', choices=('host', 'resident'), default='resident')
    p.add_argument('--npu-task-queue', type=int, choices=(0, 1, 2), default=0,
                   help='0 is the qualified multi-device mode on the local SDK/CANN stack')
    p.add_argument('--full-autograd', choices=('replay', 'batched'), default='batched')
    p.add_argument('--aggregate-autograd', choices=('replay', 'batched'), default='batched')
    for name, default in [('devices', 1), ('width', 2048), ('batch', 512), ('vocab', 50304),
                          ('steps', 12), ('warmup', 4), ('workers', 16), ('threads', 1),
                          ('seed', 7), ('timeout-seconds', 1800), ('memory-gib', 256)]:
        p.add_argument('--'+name, type=int, default=default)
    for name, default in [('grad', 0), ('check', 0), ('parallel-regions', 1), ('compact-events', 1),
                          ('defer-state-release', 1), ('packed-sources', 1), ('batch-next', 1)]:
        p.add_argument('--'+name, type=int, choices=(0, 1), default=default)
    a = p.parse_args()
    if (not 1 <= a.devices <= 16 or (a.device == 'cpu' and a.devices != 1)
            or not 4 <= a.width <= 4096 or a.width % 4 or not 1 <= a.batch <= 1024
            or not 2 <= a.vocab <= 100000 or not 0 <= a.warmup < a.steps <= 1000
            or not 1 <= a.workers <= 160 or not 1 <= a.threads <= 160 or a.workers*a.threads > 160
            or not 0 <= a.seed < 2**64 or not 1 <= a.timeout_seconds <= 7200
            or not 1 <= a.memory_gib <= 1280 or (a.defer_state_release and not a.compact_events)
            or (a.check and (a.width > 64 or a.batch > 8 or a.steps > 6))):
        p.error('invalid bounded workload')
    return a


def summarize(events, warmup):
    samples = [e['metrics'] for e in events if e['step'] >= warmup]
    result = {}
    for key in sorted({k for m in samples for k in m}):
        values = sorted(m[key] for m in samples if key in m)
        result[key] = dict(mean=statistics.mean(values), median=statistics.median(values),
                           min=values[0], max=values[-1], population_stdev=statistics.pstdev(values))
    return len(samples), result


def main():
    a = parse(); root = Path(__file__).resolve().parents[1]
    commit, dirty = source_state(root)
    if dirty: raise ValueError('formal benchmarks require a clean frozen commit')
    build, topology, out = a.build_dir.resolve(), a.topology.resolve(), a.output_dir.resolve()
    manifest = json.loads((build/'client-manifest.json').read_text())
    binary = build/'tide-accelerator-scale'
    identity, binary_hash, graph_hash = client_hash(root), digest(binary), digest(topology)
    if (manifest['client_source_sha256'] != identity or manifest['binary_sha256'] != binary_hash
            or manifest['core']['cpp_source_sha256'] != source_hash(root)
            or (a.device == 'npu' and manifest['core']['backend'] != 'npu')):
        raise ValueError('benchmark source/build/backend mismatch')
    closure = subprocess.check_output(['ldd', str(binary)], text=True)
    if any(x in closure.lower() for x in ('not found', 'libpython', 'libtorch_python', '/stub/', '/stubs/')):
        raise ValueError('invalid standalone loader closure')
    config = {key: getattr(a, key) for key in ('device', 'devices', 'memory', 'placement', 'transport', 'width', 'batch',
              'vocab', 'steps', 'warmup', 'workers', 'threads', 'seed', 'grad', 'check', 'parallel_regions',
              'compact_events', 'defer_state_release', 'packed_sources', 'batch_next',
              'full_autograd', 'aggregate_autograd')}
    config.update(dtype='float32', packed=1, emission='row', head_workers=1, fiber_pooling='event')
    run_id = out.name+'-'+uuid.uuid4().hex[:8]; now = utc_now()
    track = LocalTrackio(a.tracking, a.tracking_root, 'tide-npu-performance', run_id, config)
    command = [str(binary), '--topology', str(out/'topology.txt'), '--run-id', run_id,
               '--output-dir', str(out/'native')]
    for key, value in config.items(): command += ['--'+key.replace('_', '-'), str(value)]
    affinity = sorted(os.sched_getaffinity(0))
    visible = os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '') if a.device == 'npu' else ''
    if a.device == 'npu' and len(visible.split(',')) != a.devices:
        raise ValueError('launcher must expose exactly the requested device count')
    record = dict(schema_version=1, run_id=run_id, project='tide-npu-performance', name=run_id,
        status='running', created_at=now, started_at=now, ended_at=None,
        source=dict(repository='tide/graph-execution-foundation', commit=commit, dirty=False,
                    client_source_sha256=identity, binary_sha256=binary_hash, build=manifest),
        command=dict(argv=command, wrapper_argv=sys.argv, working_directory=str(root)),
        inputs=dict(topology_sha256=graph_hash, weights='fresh CPU seed; original owner order; normal std .02'),
        runtime=dict(resolved_device=a.device, resolution_reason='explicit:'+a.device, dtype='float32',
                     logical_devices=list(range(a.devices)), physical_visible_devices=visible,
                     host_arch=platform.machine(), cpu_affinity=affinity, node_workers=a.workers,
                     aten_threads=a.threads, interop_threads=1, openblas_num_threads=1,
                     npu_task_queue=a.npu_task_queue if a.device == 'npu' else None,
                     load_average_before=list(os.getloadavg()), memory_budget_gib=a.memory_gib),
        experiment=dict(config=config, **{'class': 'benchmark'}, primary_metric='perf/ms_per_sample_token',
                        global_step_semantics='growing-context token index, warmup retained',
                        stop_condition=f'{a.steps} tokens or {a.timeout_seconds} seconds or RSS budget',
                        placement='node-shards-v1; '+a.transport+' state/message transport; CPU FP64 Read',
                        timed_scope='embedding+body+head+CPU/NPU transfers; synchronized all shards',
                        excluded='construction, ID creation, previous logits disposal, metrics',
                        backward=False, optimizer=False, detach=False,
                        cut_edges='static physical-edge proxy; actual local/remote message bytes measured'),
        tracking=track.record, artifacts=dict(metrics='metrics.jsonl', stdout='stdout.log', summary='summary.json',
            topology='topology.txt', native='native', lifecycle='lifecycle.json', loader='loader.txt'))
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(topology, out/'topology.txt'); (out/'loader.txt').write_text(closure)
    save = lambda: write_json(out/'run.json', record)
    save(); audit = {}; events = []; error = None; code = 1; cancelled = None
    def interrupt(signum, _):
        nonlocal cancelled
        cancelled = signum; raise InterruptedError('cancelled by signal')
    handlers = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        track.start(); save()
        env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD='0', OMP_NUM_THREADS=str(a.threads),
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_WAIT_POLICY='PASSIVE')
        if a.device == 'npu': env['TASK_QUEUE_ENABLE'] = str(a.npu_task_queue)
        with (out/'stdout.log').open('x') as log:
            native_code = run_child(command, cwd=root, env=env, log=log, affinity=affinity,
                                    timeout=a.timeout_seconds, memory_budget=a.memory_gib*2**30, audit=audit)
        if native_code != 0: raise RuntimeError(f'native exit {native_code}')
        events = read_events(out/'native/metrics.jsonl', run_id)
        if len(events) != a.steps: raise ValueError('incomplete token inventory')
        if (source_state(root) != (commit, dirty) or client_hash(root) != identity
                or digest(binary) != binary_hash or digest(topology) != graph_hash):
            raise ValueError('source/input/binary changed')
        code = 0
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        for sig in handlers: signal.signal(sig, signal.SIG_IGN)
        try:
            path = out/'native/metrics.jsonl'
            if path.exists(): shutil.copyfile(path, out/'metrics.jsonl')
            else: (out/'metrics.jsonl').touch()
            events = read_events(out/'metrics.jsonl', run_id)
            track.project_events(events)
        except Exception as exc:
            code = 1; error = (error or '')+'; metrics/tracking: '+str(exc)
        try: track.finish()
        except Exception as exc:
            code = 1; error = (error or '')+'; tracking: '+str(exc)
        if cancelled: code = 128+cancelled
        record['runtime']['load_average_after'] = list(os.getloadavg())
        record.update(status='cancelled' if cancelled else 'completed' if code == 0 else 'failed', ended_at=utc_now())
        measured, summary = summarize(events, a.warmup)
        write_json(out/'lifecycle.json', audit)
        write_json(out/'summary.json', dict(schema_version=1, run_id=run_id, status=record['status'],
            ended_at=record['ended_at'], exit_code=code, native_exit_code=audit.get('worker_exit_code'),
            error=error, observations=len(events), measured_observations=measured, metrics=summary,
            tracking=track.record, remaining_group_pids=audit.get('remaining_group_pids')))
        save()
        for sig, handler in handlers.items(): signal.signal(sig, handler)
    print(json.dumps(dict(state=record['status'], summary=str(out/'summary.json'), error=error)))
    return code


if __name__ == '__main__': sys.exit(main())

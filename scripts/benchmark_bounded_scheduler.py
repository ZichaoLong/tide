#!/usr/bin/env python3
"""Record finite tensor-program windows; graph replay remains an optional backend."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import uuid
from accelerator_scale_identity import client_hash
from benchmark_accelerator_scale import summarize
from build_identity import source_hash
from durable_records import write_json
from experiment_record import LocalTrackio, read_events, utc_now
from foundation_lifecycle import run_child
from source_identity import source_state, digest


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=('cpu', 'npu'), required=True)
    p.add_argument('--dtype', choices=('float32', 'float16'), default='float32')
    p.add_argument('--memory', choices=('add', 'attention'), required=True)
    p.add_argument('--optimizer', choices=('sgd', 'adamw'), default='adamw')
    p.add_argument('--tracking', choices=('required', 'best-effort', 'off'), default='best-effort')
    for name in ('build-dir', 'topology', 'output-dir', 'tracking-root'):
        p.add_argument('--'+name, type=Path, required=True)
    for name, default in [('devices', 1), ('width', 8), ('batch', 2), ('vocab', 17),
                          ('steps', 3), ('iterations', 4), ('iteration-warmup', 1),
                          ('workspace-gib', 16), ('memory-gib', 128), ('timeout-seconds', 1800), ('seed', 7)]:
        p.add_argument('--'+name, type=int, default=default)
    p.add_argument('--replay', type=int, choices=(0, 1), default=1)
    p.add_argument('--training', type=int, choices=(0, 1), default=0)
    a = p.parse_args()
    if (a.device == 'cpu' and (a.devices != 1 or a.replay)
            or not 1 <= a.devices <= 16 or not 4 <= a.width <= 4096 or a.width % 4
            or not 1 <= a.batch <= 512 or not 2 <= a.vocab <= 100000 or not 1 <= a.steps <= 32
            or not 0 <= a.iteration_warmup < a.iterations <= 100 or a.iterations < 2
            or not 1 <= a.workspace_gib <= 32768 or not 1 <= a.memory_gib <= 1280
            or not 1 <= a.timeout_seconds <= 7200 or not 0 <= a.seed < 2**64):
        p.error('invalid finite window/resource configuration; CPU requires eager and one device')
    return a


def main():
    a = parse(); root = Path(__file__).resolve().parents[1]
    commit, dirty = source_state(root)
    if dirty: raise ValueError('formal benchmarks require a clean frozen commit')
    build, topology, out = a.build_dir.resolve(), a.topology.resolve(), a.output_dir.resolve()
    binary = build/'tide-bounded-scale'
    manifest = json.loads((build/'client-manifest.json').read_text())
    identity, binary_hash, graph_hash = client_hash(root), digest(binary), digest(topology)
    if (manifest['client_source_sha256'] != identity
            or manifest.get('additional_binaries', {}).get(binary.name) != binary_hash
            or manifest['core']['cpp_source_sha256'] != source_hash(root)
            or (a.device == 'npu' and manifest['core']['backend'] != 'npu')):
        raise ValueError('bounded source/build/backend mismatch')
    closure = subprocess.check_output(['ldd', str(binary)], text=True)
    if any(x in closure.lower() for x in ('not found', 'libpython', 'libtorch_python', '/stub/', '/stubs/')):
        raise ValueError('invalid standalone loader closure')
    config = {k: getattr(a, k) for k in ('device', 'dtype', 'memory', 'optimizer', 'devices', 'width',
              'batch', 'vocab', 'steps', 'iterations', 'iteration_warmup', 'workspace_gib', 'seed', 'replay', 'training')}
    config.update(warmup=0, workers=1, threads=1, head_workers=1, emission='row', packed=1)
    run_id = out.name+'-'+uuid.uuid4().hex[:8]; now = utc_now()
    track = LocalTrackio(a.tracking, a.tracking_root, 'tide-device-scheduler', run_id, config)
    command = [str(binary), '--topology', str(out/'topology.txt'), '--run-id', run_id,
               '--output-dir', str(out/'native')]
    for key, value in config.items(): command += ['--'+key.replace('_', '-'), str(value)]
    affinity = sorted(os.sched_getaffinity(0)); visible = os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '') if a.device == 'npu' else ''
    if a.device == 'npu' and len(visible.split(',')) != a.devices:
        raise ValueError('launcher must expose exactly the requested device count')
    record = dict(schema_version=1, run_id=run_id, project='tide-device-scheduler', name=run_id,
        status='running', created_at=now, started_at=now, ended_at=None,
        source=dict(repository='tide/graph-execution-foundation', commit=commit, dirty=False,
                    client_source_sha256=identity, binary_sha256=binary_hash, build=manifest),
        command=dict(argv=command, wrapper_argv=sys.argv, working_directory=str(root)),
        inputs=dict(topology_sha256=graph_hash, weights='fresh CPU FP32 seed, original owner order, cast to payload',
                    ids='(token*7+batch*3)%vocab, repeated complete windows'),
        runtime=dict(resolved_device=a.device, resolution_reason='explicit:'+a.device, dtype=a.dtype,
                     logical_devices=list(range(a.devices)), physical_visible_devices=visible,
                     host_arch=platform.machine(), cpu_affinity=affinity, aten_threads=1, interop_threads=1,
                     npu_task_queue=0 if a.device == 'npu' else None,
                     load_average_before=list(os.getloadavg()), memory_budget_gib=a.memory_gib),
        experiment=dict(config=config, **{'class': 'benchmark'}, primary_metric='perf/ms_per_sample_token',
                        global_step_semantics='complete finite window, one optimizer update when training',
                        stop_condition=f'{a.iterations} windows or wall/RSS bound', measured_warmup=a.iteration_warmup,
                        timed_scope='IDs upload+whole reset-state window+optional VJP/optimizer; all shards synchronized',
                        excluded='construction, warmup/capture preparation, CPU IDs construction, finite checks, metrics',
                        comparison='whole-window measurements; token-warmup latency tables have a different scope',
                        placement='locality node shards; model FP32 Read/controls; exact int64 device masks/history',
                        precision=dict(payload=a.dtype, read='float32', controls='float32', masters='float32',
                                       optimizer='float32', loss_scale=128 if a.dtype == 'float16' else 1),
                        backward=bool(a.training), optimizer=bool(a.training), window_boundary='empty state each window; owners persist'),
        tracking=track.record, artifacts=dict(metrics='metrics.jsonl', stdout='stdout.log', summary='summary.json',
            topology='topology.txt', native='native', lifecycle='lifecycle.json', loader='loader.txt'))
    out.mkdir(parents=True, exist_ok=False); shutil.copyfile(topology, out/'topology.txt'); (out/'loader.txt').write_text(closure)
    save = lambda: write_json(out/'run.json', record)
    save(); audit = {}; events = []; error = None; code = 1; cancelled = None
    def interrupt(signum, _):
        nonlocal cancelled
        cancelled = signum; raise InterruptedError('cancelled by signal')
    handlers = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        track.start(); save()
        env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD='0', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', TASK_QUEUE_ENABLE='0', OMP_WAIT_POLICY='PASSIVE')
        with (out/'stdout.log').open('x') as log:
            rc = run_child(command, cwd=root, env=env, log=log, affinity=affinity,
                           timeout=a.timeout_seconds, memory_budget=a.memory_gib*2**30, audit=audit)
        if rc: raise RuntimeError(f'native exit {rc}')
        events = read_events(out/'native/metrics.jsonl', run_id)
        if len(events) != a.iterations: raise ValueError('incomplete observation inventory')
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
            events = read_events(out/'metrics.jsonl', run_id); track.project_events(events); track.finish()
        except Exception as exc:
            code = 1; error = (error or '')+'; metrics/tracking: '+str(exc)
        if cancelled: code = 128+cancelled
        record['runtime']['load_average_after'] = list(os.getloadavg())
        record.update(status='cancelled' if cancelled else 'completed' if code == 0 else 'failed', ended_at=utc_now())
        measured, summary = summarize(events, a.iteration_warmup)
        write_json(out/'lifecycle.json', audit)
        write_json(out/'summary.json', dict(schema_version=1, run_id=run_id, status=record['status'],
            ended_at=record['ended_at'], exit_code=code, native_exit_code=audit.get('worker_exit_code'), error=error,
            observations=len(events), measured_observations=measured, metrics=summary, tracking=track.record,
            remaining_group_pids=audit.get('remaining_group_pids')))
        save()
        for sig, handler in handlers.items(): signal.signal(sig, handler)
    print(json.dumps(dict(state=record['status'], summary=str(out/'summary.json'), error=error)))
    return code


if __name__ == '__main__': sys.exit(main())

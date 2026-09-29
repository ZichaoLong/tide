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
    p.add_argument('--dtype', choices=('float32', 'float16'), default='float32')
    p.add_argument('--loss-scale', type=float, default=None)
    p.add_argument('--check-atol', type=float, default=None)
    p.add_argument('--check-rtol', type=float, default=None)
    p.add_argument('--memory', choices=('add', 'attention'), required=True)
    p.add_argument('--placement', choices=('memory', 'locality'), default='memory')
    p.add_argument('--transport', choices=('host', 'resident'), default='resident')
    p.add_argument('--read-device', choices=('cpu', 'model'), default='cpu')
    p.add_argument('--read-dtype', choices=('float64', 'float32'), default='float64')
    p.add_argument('--control-device', choices=('cpu', 'model'), default='cpu')
    p.add_argument('--ranking-device', choices=('cpu', 'model'), default='cpu')
    p.add_argument('--event-device', choices=('cpu', 'model'), default='cpu')
    p.add_argument('--training-steps', type=int, default=0)
    p.add_argument('--training-warmup', type=int, default=1)
    p.add_argument('--optimizer', choices=('sgd','adamw'), default='adamw')
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--npu-task-queue', type=int, choices=(0, 1, 2), default=0,
                   help='0 is the qualified multi-device mode on the local SDK/CANN stack')
    p.add_argument('--profile-step', type=int)
    p.add_argument('--profile-phase', choices=('token', 'forward', 'backward', 'optimizer'), default='token')
    p.add_argument('--profile-max-gib', type=int, default=4)
    p.add_argument('--full-autograd', choices=('replay', 'batched'), default='batched')
    p.add_argument('--aggregate-autograd', choices=('replay', 'batched'), default='batched')
    for name, default in [('devices', 1), ('width', 2048), ('batch', 512), ('vocab', 50304),
                          ('steps', 12), ('warmup', 4), ('workers', 16), ('threads', 1),
                          ('head-workers', 1),
                          ('backward-threads', 1), ('optimizer-threads', 1),
                          ('seed', 7), ('timeout-seconds', 1800), ('memory-gib', 256)]:
        p.add_argument('--'+name, type=int, default=default)
    for name, default in [('grad', 0), ('check', 0), ('parallel-regions', 1), ('compact-events', 1),
                          ('defer-state-release', 1), ('packed-sources', 1), ('batch-next', 1)]:
        p.add_argument('--'+name, type=int, choices=(0, 1), default=default)
    a = p.parse_args()
    if (not 1 <= a.backward_threads <= 160 or not 1 <= a.optimizer_threads <= 160
            or ((a.device != 'cpu' or not a.training_steps)
                and (a.backward_threads != 1 or a.optimizer_threads != 1))):
        p.error('phase thread options require CPU training, each in1..160')
    if a.profile_step is not None:
        if (a.device != 'npu' or not 0 <= a.profile_step < (a.training_steps or a.steps)
                or (a.training_steps > 0) == (a.profile_phase == 'token')
                or not 1 <= a.profile_max_gib <= 16):
            p.error('scoped profiling requires NPU, a valid token/training phase and step, and 1..16 GiB bound')
    elif a.profile_phase != 'token' or a.profile_max_gib != 4:
        p.error('profile configuration requires --profile-step')
    if a.dtype == 'float16' and a.transport != 'resident': p.error('FP16 requires resident transport')
    if a.loss_scale is not None and (not a.training_steps or not 0 < a.loss_scale < float('inf')):
        p.error('positive finite loss scale requires training')
    if a.training_steps and a.loss_scale is None: a.loss_scale = 128. if a.dtype == 'float16' else 1.
    if any(x is not None for x in (a.check_atol, a.check_rtol)):
        if a.dtype != 'float16' or not a.check or any(x is not None and not 0 < x < float('inf') for x in (a.check_atol, a.check_rtol)):
            p.error('custom tolerances require FP16 check and positive finite values')
    if ((a.read_device == 'model' or a.control_device == 'model')
            and (a.transport != 'resident' or (a.device == 'npu' and a.read_dtype != 'float32'))):
        p.error('model-device Read/controls require resident transport and FP32 on NPU')
    if (a.ranking_device == 'model' and a.device == 'npu' and a.read_dtype != 'float32'
            or a.transport != 'resident' and 'model' in (a.ranking_device, a.event_device)):
        p.error('model ranking/events require resident transport; NPU ranking requires FP32 Read')
    if (not 0 <= a.training_steps <= 100 or a.training_warmup < 0
            or a.training_steps and (a.training_warmup >= a.training_steps or not a.grad or a.warmup != 0)
            or not 0 < a.learning_rate < float('inf')):
        p.error('invalid training configuration; requires grad1 and token warmup0')
    if (not 1 <= a.devices <= 16 or (a.device == 'cpu' and a.devices != 1)
            or not 4 <= a.width <= 4096 or a.width % 4 or not 1 <= a.batch <= 1024
            or not 2 <= a.vocab <= 100000 or not 0 <= a.warmup < a.steps <= 1000
            or not 1 <= a.workers <= 160 or not 1 <= a.threads <= 160 or a.workers*a.threads > 160
            or not 1 <= a.head_workers <= 160 or a.head_workers*a.threads > 160
            or (a.device == 'npu' and a.head_workers != 1)
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
              'full_autograd', 'aggregate_autograd', 'read_device', 'read_dtype', 'control_device', 'ranking_device', 'event_device', 'training_steps', 'training_warmup', 'optimizer', 'learning_rate')}
    config.update(dtype=a.dtype, packed=1, emission='row', head_workers=a.head_workers, fiber_pooling='event',
                  backward_threads=a.backward_threads, optimizer_threads=a.optimizer_threads)
    if a.profile_step is not None:
        config.update(profile_step=a.profile_step, profile_phase=a.profile_phase,
                      profile_output=str(out/'profile'))
    for key in ('loss_scale', 'check_atol', 'check_rtol'):
        if getattr(a,key) is not None: config[key]=getattr(a,key)
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
        inputs=dict(topology_sha256=graph_hash, weights='fresh CPU FP32 seed; original owner order; normal std .02; cast to payload dtype'),
        runtime=dict(resolved_device=a.device, resolution_reason='explicit:'+a.device, dtype=a.dtype,
                     loss_scale=a.loss_scale, optimizer_dtype='float32' if a.training_steps else None,
                     logical_devices=list(range(a.devices)), physical_visible_devices=visible,
                     host_arch=platform.machine(), cpu_affinity=affinity, node_workers=a.workers,
                     head_workers=a.head_workers,
                     backward_threads=a.backward_threads, optimizer_threads=a.optimizer_threads,
                     aten_threads=a.threads, interop_threads=1, openblas_num_threads=1,
                     npu_task_queue=a.npu_task_queue if a.device == 'npu' else None,
                     read_device=a.read_device, read_dtype=a.read_dtype, control_device=a.control_device,
                     node_ranking=a.ranking_device, event_scheduler=a.event_device,
                     load_average_before=list(os.getloadavg()), memory_budget_gib=a.memory_gib),
        experiment=dict(config=config, **{'class': 'benchmark'}, primary_metric='perf/ms_per_sample_token',
                        instrumented=a.profile_step is not None,
                        profile_storage_budget_bytes=a.profile_max_gib*2**30 if a.profile_step is not None else None,
                        global_step_semantics='optimizer update, independent complete sequence windows' if a.training_steps else 'growing-context token index, warmup retained',
                        stop_condition=f'{a.training_steps} updates of {a.steps} tokens' if a.training_steps else f'{a.steps} tokens or {a.timeout_seconds} seconds or RSS budget',
                        placement='node-shards-v1; '+a.transport+' state/message transport; '
                                  +a.read_device+' '+a.read_dtype+' Read; '+a.control_device+' controls',
                        timed_scope='zero_grad+window reset+IDs+forward+cross_entropy+backward+optimizer; synchronized phase boundaries' if a.training_steps else 'embedding+body+head+CPU/NPU transfers; synchronized all shards',
                        excluded='model construction, optimizer/master setup, previous loss/graph disposal, metrics' if a.training_steps else 'construction, ID creation, previous logits disposal, metrics',
                        precision=dict(payload=a.dtype, read=a.read_dtype, controls='float32', loss='float32',
                                       masters='float32', optimizer='float32', loss_scale=a.loss_scale,
                                       fp16_check_atol=a.check_atol or 1e-3, fp16_check_rtol=a.check_rtol or 2e-2),
                        backward=bool(a.training_steps), optimizer=bool(a.training_steps), detach=False,
                        loss='mean token cross_entropy, targets=(input_id+1)%vocab' if a.training_steps else None,
                        window_boundary='reset graph state per optimizer update; no detach inside window' if a.training_steps else None,
                        optimizer_options=dict(lr=a.learning_rate,weight_decay=.01,eps=1e-5,beta1=.9,beta2=.999,momentum=.9) if a.training_steps else None,
                        measured_warmup=a.training_warmup if a.training_steps else a.warmup,
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
                                    timeout=a.timeout_seconds, memory_budget=a.memory_gib*2**30, audit=audit,
                                    storage_path=out if a.profile_step is not None else None,
                                    storage_budget=a.profile_max_gib*2**30)
        if native_code != 0: raise RuntimeError(f'native exit {native_code}')
        events = read_events(out/'native/metrics.jsonl', run_id)
        if len(events) != (a.training_steps or a.steps): raise ValueError('incomplete observation inventory')
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
        measured, summary = summarize(events, a.training_warmup if a.training_steps else a.warmup)
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

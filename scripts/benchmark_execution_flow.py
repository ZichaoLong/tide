#!/usr/bin/env python3
"""Record complete independent CPU, mixed and device execution flows."""
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
from durable_records import write_json, replace_text
from flow_topology import validate_packet, native_text
from experiment_record import LocalTrackio, read_events, utc_now
from foundation_lifecycle import run_child
from source_identity import source_state, digest


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=('cpu', 'cuda', 'npu'), required=True)
    p.add_argument('--dtype', choices=('float64', 'float32', 'float16'), default='float32')
    p.add_argument('--family', choices=('pdg', 'timed-dag', 'settle'), required=True)
    p.add_argument('--flow', choices=('cpu', 'mixed', 'device'), required=True)
    p.add_argument('--scheduler', choices=('streaming','frontier','settle','resident','bounded-eager','bounded-replay'))
    p.add_argument('--partition', choices=('memory','locality'), default='locality')
    for name in ('read-device','control-device','ranking-device','event-device'):
        p.add_argument('--'+name, choices=('cpu','model'))
    p.add_argument('--read-dtype', choices=('float32','float64'))
    for name in ('full-autograd','aggregate-autograd'):
        p.add_argument('--'+name,choices=('replay','batched'))
    for name in ('parallel-regions','packed-sources','batch-next'):
        p.add_argument('--'+name,type=int,choices=(0,1))
    p.add_argument('--profile-step',type=int)
    p.add_argument('--profile-max-gib',type=int,default=4)
    p.add_argument('--optimizer', choices=('sgd', 'adamw'), default='adamw')
    p.add_argument('--tracking', choices=('required', 'best-effort', 'off'), default='best-effort')
    for name in ('build-dir', 'workload', 'output-dir', 'tracking-root'):
        p.add_argument('--'+name, type=Path, required=True)
    for name, default in [('devices', 1), ('iterations', 4), ('iteration-warmup', 1),
                          ('workspace-gib', 1024), ('memory-gib', 1024), ('timeout-seconds', 7200),
                          ('workers', 1), ('head-workers', 1), ('threads', 1),
                          ('backward-threads', 1), ('optimizer-threads', 1)]:
        p.add_argument('--'+name, type=int, default=default)
    p.add_argument('--training', type=int, choices=(0, 1), default=0)
    a = p.parse_args()
    if (not 1 <= a.devices <= 16 or a.device == 'cpu' and a.devices != 1
            or not 0 <= a.iteration_warmup < a.iterations <= 100 or a.iterations < 2
            or not 1 <= a.workspace_gib <= 32768 or not 1 <= a.memory_gib <= 1280
            or not 1 <= a.timeout_seconds <= 86400):
        p.error('invalid finite window/resource configuration')
    a.scheduler = a.scheduler or ('bounded-replay' if a.flow == 'device' else 'resident' if a.flow == 'mixed'
                                  else 'settle' if a.family == 'settle' else 'frontier' if a.family == 'timed-dag' else 'streaming')
    for key in ('read_device','control_device','ranking_device','event_device'):
        if getattr(a,key) is None: setattr(a,key,'model' if a.flow == 'device' else 'cpu')
    a.read_dtype = a.read_dtype or ('float64' if a.dtype == 'float64' else 'float32')
    bounded=a.scheduler.startswith('bounded-')
    for key in ('full_autograd','aggregate_autograd'):
        if getattr(a,key) is None:setattr(a,key,'replay' if bounded else 'batched')
    for key in ('parallel_regions','packed_sources','batch_next'):
        if getattr(a,key) is None:setattr(a,key,0 if bounded else 1)
    if a.profile_step is not None and (a.device!='npu' or not 0<=a.profile_step<a.iterations or not 1<=a.profile_max_gib<=16):
        p.error('profiling requires NPU, a valid iteration and 1..16 GiB trace bound')
    return a


def main():
    a = parse(); root = Path(__file__).resolve().parents[1]
    commit, dirty = source_state(root)
    if dirty: raise ValueError('formal benchmarks require a clean frozen commit')
    build, workload, out = a.build_dir.resolve(), a.workload.resolve(), a.output_dir.resolve()
    packet = validate_packet(json.loads(workload.read_text()))
    if a.family not in packet['families']: raise ValueError('workload is not equivalent to the selected family')
    binary = build/'tide-complete-flow'
    manifest = json.loads((build/'client-manifest.json').read_text())
    identity, binary_hash, graph_hash = client_hash(root), digest(binary), digest(workload)
    if (manifest['client_source_sha256'] != identity
            or manifest.get('additional_binaries', {}).get(binary.name) != binary_hash
            or manifest['core']['cpp_source_sha256'] != source_hash(root)
            or (a.device != 'cpu' and manifest['core']['backend'] != a.device)):
        raise ValueError('complete-flow source/build/backend mismatch')
    closure = subprocess.check_output(['ldd', str(binary)], text=True)
    if any(x in closure.lower() for x in ('not found', 'libpython', 'libtorch_python', '/stub/', '/stubs/')):
        raise ValueError('invalid standalone loader closure')
    config = {k: getattr(a, k) for k in ('device','dtype','optimizer','devices','iterations','iteration_warmup',
              'workspace_gib','training','family','flow','scheduler','partition','workers','head_workers','threads',
              'backward_threads','optimizer_threads','read_device','read_dtype','control_device','ranking_device','event_device',
              'full_autograd','aggregate_autograd','parallel_regions','packed_sources','batch_next')}
    config.update({k:packet['workload'][k] for k in ('memory','width','batch','vocab','seed')})
    config.update(steps=packet['workload']['tokens'],warmup=0,emission='row',packed=1)
    if a.profile_step is not None:config.update(profile_step=a.profile_step,profile_output=str(out/'profile'))
    run_id = out.name+'-'+uuid.uuid4().hex[:8]; now = utc_now()
    track = LocalTrackio(a.tracking, a.tracking_root, 'tide-execution-flows', run_id, config)
    command = [str(binary), '--topology', str(out/'topology.txt'), '--run-id', run_id,
               '--output-dir', str(out/'native')]
    for key, value in config.items(): command += ['--'+key.replace('_', '-'), str(value)]
    affinity = sorted(os.sched_getaffinity(0)); visible = os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '') if a.device == 'npu' else ''
    if a.device == 'npu' and len(visible.split(',')) != a.devices:
        raise ValueError('launcher must expose exactly the requested device count')
    record = dict(schema_version=1, run_id=run_id, project='tide-execution-flows', name=run_id,
        status='running', created_at=now, started_at=now, ended_at=None,
        source=dict(repository='tide/graph-execution-foundation', commit=commit, dirty=False,
                    client_source_sha256=identity, binary_sha256=binary_hash, build=manifest),
        command=dict(argv=command, wrapper_argv=sys.argv, working_directory=str(root)),
        inputs=dict(packet_sha256=packet['sha256'], workload_file_sha256=graph_hash, weights='fresh CPU seed in owner order; FP64/FP32 or FP32 initialization cast to FP16',
                    ids='(token*7+batch*3+iteration*5)%vocab, independent reset-state windows'),
        runtime=dict(resolved_device=a.device, resolution_reason='explicit:'+a.device, dtype=a.dtype,
                     logical_devices=list(range(a.devices)), physical_visible_devices=visible,
                     host_arch=platform.machine(), cpu_affinity=affinity, aten_threads=a.threads, interop_threads=1,
                     npu_task_queue=0 if a.device == 'npu' else None,
                     load_average_before=list(os.getloadavg()), memory_budget_gib=a.memory_gib),
        experiment=dict(config_schema='tide-complete-flow-config-v1',config=config,
                        **{'class': 'profile' if a.profile_step is not None else 'benchmark'},primary_metric='perf/ms_per_sample_token',
                        global_step_semantics='complete finite window, one optimizer update when training',
                        stop_condition=f'{a.iterations} windows or wall/RSS bound', measured_warmup=a.iteration_warmup,
                        timed_scope='input construction/upload+reset-state window+loss/backward/finite checks/optimizer+eager cleanup+all-shard synchronization',
                        excluded='construction and reusable capture preparation recorded separately; metrics diagnostics outside window timing',
                        comparison='same packet and precision, complete independent flows; cold and amortized costs plus wrapper process time',
                        placement=a.partition+' node shards; explicit control placement; exact int64 times/counts',
                        precision=dict(payload=a.dtype, read=a.read_dtype, controls='float64' if a.dtype=='float64' else 'float32',
                                       masters='float64' if a.dtype=='float64' else 'float32',
                                       optimizer='float64' if a.dtype=='float64' else 'float32',loss_scale=128 if a.dtype == 'float16' else 1),
                        backward=bool(a.training), optimizer=bool(a.training), window_boundary='empty state each window; owners persist'),
        tracking=track.record, artifacts=dict(metrics='metrics.jsonl', stdout='stdout.log', summary='summary.json',
            topology='topology.txt', workload='workload.json', native='native', lifecycle='lifecycle.json', loader='loader.txt'))
    out.mkdir(parents=True, exist_ok=False); write_json(out/'workload.json', packet); replace_text(out/'topology.txt', native_text(packet)); (out/'loader.txt').write_text(closure)
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
                           timeout=a.timeout_seconds, memory_budget=a.memory_gib*2**30, audit=audit,
                           storage_path=out if a.profile_step is not None else None,
                           storage_budget=a.profile_max_gib*2**30 if a.profile_step is not None else None)
        if rc: raise RuntimeError(f'native exit {rc}')
        events = read_events(out/'native/metrics.jsonl', run_id)
        if len(events) != a.iterations: raise ValueError('incomplete observation inventory')
        if (source_state(root) != (commit, dirty) or client_hash(root) != identity
                or digest(binary) != binary_hash or digest(workload) != graph_hash):
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
            remaining_group_pids=audit.get('remaining_group_pids'),process_seconds=audit.get('process_seconds')))
        save()
        for sig, handler in handlers.items(): signal.signal(sig, handler)
    print(json.dumps(dict(state=record['status'], summary=str(out/'summary.json'), error=error)))
    return code


if __name__ == '__main__': sys.exit(main())

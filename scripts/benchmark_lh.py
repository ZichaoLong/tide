#!/usr/bin/env python3
"""Bounded, recorded original-LH CPU measurement; historical timings are references."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import signal
import statistics
import subprocess
import sys
import uuid
from build_lh_benchmark import digest, sources
from experiment_record import LocalTrackio, atomic_json, read_events, utc_now


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', required=True, choices=('cpu',))
    p.add_argument('--dtype', choices=('float32', 'float64'), default='float32')
    p.add_argument('--input', required=True)
    p.add_argument('--build-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--mode', choices=('nograd', 'grad-forward', 'backward'), default='nograd')
    for name, default in (('batch', 4), ('width', 0), ('steps', 4), ('warmup', 1),
                          ('repetitions', 1), ('threads', 1), ('seed', 7),
                          ('timeout-seconds', 900), ('memory-gib', 192)):
        p.add_argument('--'+name, type=int, default=default)
    p.add_argument('--blas-threads', type=int, help='OpenBLAS pool size; default matches --threads')
    p.add_argument('--tracking', choices=('best-effort', 'required', 'off'), default='best-effort')
    args = p.parse_args()
    if (not 1 <= args.batch <= 512 or not 1 <= args.steps <= 256 or not 0 <= args.warmup <= 64
            or not 1 <= args.repetitions <= 20 or not 1 <= args.threads <= 56 or not 0 <= args.seed < 2**32
            or args.width and (not 4 <= args.width <= 2048 or args.width%4)
            or not 10 <= args.timeout_seconds <= 3600 or not 4 <= args.memory_gib <= 384
            or args.blas_threads is not None and not 1 <= args.blas_threads <= 56):
        p.error('requested workload exceeds benchmark bounds')
    blas_threads = args.threads if args.blas_threads is None else args.blas_threads
    root = Path(__file__).resolve().parents[1]
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip()
    if dirty:
        p.error('controlled run requires a clean commit/frozen checkout')
    build, out, inputs = Path(args.build_dir).resolve(), Path(args.output_dir).resolve(), Path(args.input).resolve()
    compiled = json.loads((build/'build-manifest.json').read_text())
    binary = build/'tide-lh-bench'
    if compiled['sources'] != sources(root) or compiled['binary_sha256'] != digest(binary):
        p.error('benchmark binary/source mismatch')
    input_record = json.loads(inputs.read_text())
    graph = Path(input_record['graph_dir'])
    for name, expected in input_record['graph_files_sha256'].items():
        if digest(graph/name) != expected:
            p.error('graph input changed: '+name)
    input_digest = digest(inputs)
    out.mkdir(parents=True, exist_ok=False)
    run_id = out.name+'-'+uuid.uuid4().hex[:8]
    request = {k: getattr(args, k) for k in ('batch', 'steps', 'warmup', 'repetitions', 'threads', 'seed', 'mode')}
    request.update(input=str(inputs), output_dir=str(out/'native'), run_id=run_id)
    if args.width:
        request['width'] = args.width
    atomic_json(out/'request.json', request)
    command = [str(binary), '--device', args.device, '--dtype', args.dtype, '--request', str(out/'request.json')]
    track = LocalTrackio(args.tracking, out.parent/'trackio', 'tide-graph-execution', run_id, request)
    now = utc_now()
    record = dict(schema_version=1, run_id=run_id, project='tide-graph-execution', name=run_id,
      status='running', created_at=now, started_at=now, ended_at=None,
      source=dict(repository='tide/graph-execution-foundation', commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(), dirty=False, build=compiled),
      command=dict(argv=command, wrapper_argv=sys.argv, working_directory=str(root)),
      inputs=dict(path=str(inputs), sha256=input_digest, profile=input_record['profile'],
                  generator=input_record['generator_sha256'], model_source=input_record['model_source_revision']),
      runtime=dict(resolved_device='cpu', resolution_reason='explicit:cpu', dtype=args.dtype,
        host_arch=platform.machine(), torch_version=compiled['torch'], cxx11_abi=compiled['cxx11_abi'],
        cpu_affinity=sorted(os.sched_getaffinity(0)), threads=args.threads, interop_threads=1,
        openblas_threads=blas_threads, omp_threads=args.threads,
        load_average_before=list(os.getloadavg()), memory_limit_gib=args.memory_gib),
      experiment=dict(config=request, **{'class':'benchmark'},
        global_step_semantics='one token forward or one whole-window backward; see phase context',
        primary_metric='perf/ms_per_sample_token', stop_condition=f'{args.timeout_seconds}s maximum; finite steps/repetitions',
        includes='think embedding/body/readout/head, selector counters; excludes ID generation, finite checks and logging'),
      tracking=track.record, artifacts=dict(metrics='metrics.jsonl', stdout='stdout.log', summary='summary.json', native='native/'))
    atomic_json(out/'run.json', record)
    child = None
    error, code, events = None, 1, []
    def interrupted(signum, _):
        raise InterruptedError('cancelled by signal '+str(signum))
    previous = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
    def limits():
        resource.setrlimit(resource.RLIMIT_AS, (args.memory_gib*1024**3,)*2)
    try:
        track.start()
        atomic_json(out/'run.json', record)
        env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD='0', OMP_NUM_THREADS=str(args.threads),
                   MKL_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(blas_threads))
        with (out/'stdout.log').open('w') as log:
            child = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, preexec_fn=limits)
            code = child.wait(timeout=args.timeout_seconds)
        if code:
            raise RuntimeError('original LH benchmark exited '+str(code))
        events = read_events(out/'native/metrics.jsonl', run_id)
        expected = args.repetitions*(args.steps+(args.mode=='backward'))
        if len(events) != expected:
            raise ValueError('metric event inventory mismatch')
        if digest(inputs) != input_digest or compiled['sources'] != sources(root) or digest(binary) != compiled['binary_sha256']:
            raise ValueError('input, source or binary changed during run')
        track.project_events(events)
        code = 0
    except BaseException as exception:
        code = 1
        error = f'{type(exception).__name__}: {exception}'
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
        for s, handler in previous.items():
            signal.signal(s, handler)
        raw = out/'native/metrics.jsonl'
        if raw.exists():
            (out/'metrics.jsonl').write_bytes(raw.read_bytes())
            if not events:
                try: events = read_events(raw, run_id)
                except ValueError: pass
        try: track.finish()
        except Exception as exception:
            code=1
            error = error or str(exception)
        values = {}
        for event in events:
            for key, value in event['metrics'].items():
                values.setdefault(key, []).append(value)
        summary = {k:dict(median=statistics.median(v), min=min(v), max=max(v), mean=statistics.mean(v), count=len(v))
                   for k, v in values.items()}
        record.update(status='completed' if code==0 else 'failed', ended_at=utc_now())
        record['runtime']['load_average_after'] = list(os.getloadavg())
        atomic_json(out/'summary.json',dict(schema_version=1,run_id=run_id,status=record['status'],
          ended_at=record['ended_at'],exit_code=code,native_exit_code=None if child is None else child.returncode,
          error=error,observations=len(events),metrics=summary,tracking=track.record))
        atomic_json(out/'run.json',record)
    print(json.dumps(dict(state=record['status'], result=str(out/'summary.json'),error=error)))
    return code


if __name__ == '__main__':
    sys.exit(main())

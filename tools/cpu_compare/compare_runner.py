"""Source-only kit entry point. Builds and observations remain local and durable."""
import json
import os
from pathlib import Path
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
if not (HERE/'experiment_record.py').exists():
    sys.path.insert(0, str(HERE.parent.parent/'scripts'))
from durable_records import write_json
from experiment_record import LocalTrackio, utc_now
from compare_config import digest, environment, host_info, inventory, parameters, parse, verify_packet
from compare_build import build_commands, build_record, command, prepare, torch_prefix
from compare_metrics import Collector
from compare_process import execute, ProcessFailure


def main(engine):
    a, packet, run_id = parse(engine)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=False)
    expected = parameters(a); now = utc_now(); env = environment(engine, a)
    track = LocalTrackio(a.tracking, out.parent/'trackio', 'tide-cpu-comparison', run_id, vars(a))
    record = dict(schema_version=1, run_id=run_id, name=run_id, project='tide-cpu-comparison',
        status='running', created_at=now, started_at=now, ended_at=None,
        source=dict(repository='tide/graph-execution-foundation', commit=packet['tide_source'],
                    dirty=packet['source_dirty'], packet_manifest_sha256=digest(Path(a.kit_dir)/'manifest.json'),
                    lh_revision=packet['lh_revision']),
        command=dict(argv=[sys.executable, str(HERE/('run_'+engine+'.py')), *sys.argv[1:]],
                     working_directory=str(Path.cwd())),
        inputs=dict(graph=packet['graph_csr_sha256'], topology_sha256=packet['topology_sha256'],
                    weights='independent seeded random parameters; no cross-model weight import'),
        runtime=dict(resolved_device='cpu', resolution_reason='explicit:cpu', dtype=a.dtype,
                     engine=engine, requested_threads=a.threads, cpu_affinity=sorted(os.sched_getaffinity(0)),
                     native_thread_environment={k:env[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
                     omp_wait_policy=env.get('OMP_WAIT_POLICY', 'library default'), memory_limit_gib=a.memory_gib),
        experiment=dict(config=vars(a), **{'class':'benchmark'}, primary_metric='perf/ms_per_sample_token',
            global_step_semantics='token index; persistent state, fixed IDs (3*sample+7*token)%vocab',
            stop_condition=f'{a.steps} native tokens or {a.timeout_seconds}s; build phase separately bounded',
            timing='LH original integer Think timer; PDG native cursor+head; no backward or optimizer; construction excluded from warm window',
            expected_parameters=expected),
        tracking=track.record, artifacts=dict(metrics='metrics.jsonl', summary='summary.json', stdout='stdout.log',
            host='host.json', configure='configure.log', build='build.log', prepared='prepared.json'),
        phase='prepare', stages=[])
    collector = Collector(engine, a, out, run_id, expected)
    code, error, summary, cancelled = 1, None, {}, None
    native = {}; source = None; binary = None; source_files = None
    for name in ('stdout.log', 'configure.log', 'build.log'):
        (out/name).touch()

    def save():
        write_json(out/'run.json', record)

    def interrupt(signum, _frame):
        nonlocal cancelled
        cancelled = signum
        raise InterruptedError('cancelled by signal '+str(signum))

    handlers = {sig:signal.signal(sig,interrupt) for sig in (signal.SIGINT,signal.SIGTERM)}
    save()
    print('CONFIG '+json.dumps(dict(engine=engine, width=a.width, batch=a.batch, vocab=a.vocab,
          steps=a.steps, warmup=a.warmup, seed=a.seed, grad=False, dtype=a.dtype, threads=a.threads,
          attention_packing=getattr(a, 'attention_packing', 'lh-crossbatch'),
          operator_profile=getattr(a, 'operator_profile', 0),
          fiber_pooling=getattr(a, 'fiber_pooling', 'lh-csr'), fiber_cache=getattr(a, 'fiber_cache', 'lh-capacity'),
          projection_layout=getattr(a, 'projection_layout', 'linear'), defer_state_release=getattr(a, 'defer_state_release', 0),
          attention_layout=getattr(a, 'attention_layout', 'head'),
          packed_sources=getattr(a, 'packed_sources', 0), batch_next=getattr(a, 'batch_next', 0),
          parameters=expected, parameter_gib=expected*(8 if a.dtype=='float64' else 4)/2**30,
          lh_initial_kv_gib=465*a.batch*16*a.width*8/2**30 if engine=='lh' else 0,
          graph=dict(nodes_per_cortex=232, inet_edges=984, onet_edges=984, io_edges=232, oi_edges=8),
          output_dir=str(out))), flush=True)

    def stage(name, argv, cwd, log, timeout, callback=None, memory=0):
        record['phase'] = name; save()
        print('START '+name+'; log='+str(out/log), flush=True)
        try:
            result = execute(argv, cwd, env, out/log, timeout, memory, callback)
        except ProcessFailure as e:
            record['stages'].append(dict(name=name, **e.result)); save(); raise
        record['stages'].append(dict(name=name, **result)); save()
        return result

    try:
        write_json(out/'host.json', host_info())
        track.start(); save()
        prefix, stack = torch_prefix(a, env); record['runtime']['libtorch'] = stack; save()
        print('LIBTORCH '+json.dumps(stack), flush=True)
        source = prepare(engine, a, packet, out); source_files = inventory(source)
        write_json(out/'prepared.json', dict(engine=engine, parameters=expected, config=vars(a), source_files_sha256=source_files))
        configure, compile_, binary = build_commands(engine, a, source, prefix)
        stage('configure', configure, out, 'configure.log', a.build_timeout_seconds)
        stage('build', compile_, out, 'build.log', a.build_timeout_seconds)
        binaries = build_record(engine, source, binary, source_files)
        record['source']['build_files_sha256'] = binaries; save()
        argv, cwd = command(engine, a, binary, out, run_id)
        record['command'] = dict(argv=argv, wrapper_argv=sys.argv, working_directory=str(cwd))
        collector.started = time.monotonic()
        native = stage('native', argv, cwd, 'stdout.log', a.timeout_seconds, collector.line, a.memory_gib)
        summary = collector.finish()
        if build_record(engine, source, binary, source_files) != binaries:
            raise ValueError('binary changed during execution')
        verify_packet(Path(a.kit_dir))
        track.project_events(collector.events)
        code = 0
    except BaseException as e:
        error = type(e).__name__+': '+str(e)
        if isinstance(e, ProcessFailure) and record['phase'] == 'native':
            native = e.result
        print('FAILED '+error, file=sys.stderr, flush=True)
        if record['stages']:
            log = out/record['stages'][-1]['log']
            print('\n'.join(log.read_text(errors='replace').splitlines()[-20:]), file=sys.stderr)
    finally:
        for sig in handlers:
            signal.signal(sig, signal.SIG_IGN)
        try:
            track.finish()
        except Exception as e:
            code = 1; error = error or str(e)
        if cancelled:
            code = 128+cancelled
        status = 'cancelled' if cancelled else 'completed' if code == 0 else 'failed'
        record.update(status=status, ended_at=utc_now())
        record['runtime']['actual_threads'] = collector.runtime
        record['runtime']['load_average_after'] = list(os.getloadavg())
        summary.update(schema_version=1, run_id=run_id, status=status, exit_code=code, error=error,
                       native_exit_code=native.get('exit_code'), native_resources=native,
                       observations=len(collector.events), tracking=track.record)
        write_json(out/'summary.json', summary); save()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    if code == 0:
        print('RESULT '+json.dumps({k:v for k,v in summary.items() if k not in ('mean_metrics_per_batch_token','raw_ms','tracking','native_resources')}), flush=True)
        print(f"PEAK_RSS_GIB {native['peak_rss_bytes']/2**30:.6f}", flush=True)
        print('THREADS '+json.dumps(collector.runtime), flush=True)
    print('RECORD '+str(out/'summary.json'), flush=True)
    return code

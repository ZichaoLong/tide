"""Bounded child lifetime, real RSS enforcement and durable benchmark records."""
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import uuid
from durable_records import write_json, replace_text
from experiment_record import LocalTrackio, utc_now
from foundation_resources import resources, balanced_affinity
from foundation_lifecycle import run_child


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_variant(root,build,out,config,variant,modes,args,source,repeat):
    limits=resources();affinity=balanced_affinity(limits)
    if args.threads*max(args.workers,1)+2>limits['cpu_budget']:
        raise ValueError('workers/ATen plus coordinator exceed aggregate CPU budget')
    from foundation_workloads import estimate
    preflight=estimate(config);out.mkdir(parents=True,exist_ok=False)
    write_json(out/'config.json',config)
    run_id=out.name+'-'+uuid.uuid4().hex[:8];now=utc_now()
    track=LocalTrackio(args.tracking,out.parent/'trackio','tide-foundation-v1',run_id,dict(config=config,variant=variant,modes=modes,repeat=repeat))
    command=[sys.executable,str(root/'scripts/foundation_worker.py'),'--device','cpu','--config',str(out/'config.json'),
             '--build-dir',str(build),'--output-dir',str(out/'worker'),'--variant',variant,'--modes',*modes,
             '--workers',str(args.workers),'--threads',str(args.threads),'--warmup',str(args.warmup)]
    record=dict(schema_version=1,run_id=run_id,project='tide-foundation-v1',name=run_id,status='running',
        created_at=now,started_at=now,ended_at=None,source=source,
        command=dict(argv=command,working_directory=str(root)),inputs=dict(config_sha256=digest(out/'config.json'),
        weights='deterministic seed7; matrices rescaled to std1/sqrt(D); scalar transport scales0.8; no LH weights'),
        runtime=dict(resolved_device='cpu',resolution_reason='explicit:cpu',resources=limits,affinity=affinity,
                     threads_requested=args.threads,node_worker_limit=args.workers,preflight=preflight),
        experiment=dict(config=config,variant=variant,modes=modes,repeat=repeat,**{'class':'benchmark'},
            primary_metric='perf/'+modes[0]+'/sample_positions_per_second',global_step_semantics='one independent whole-workload repeat',
            stop_condition=f'one repeat after {args.warmup} reset warmups, or {args.timeout_seconds}s/RSS budget',
            timing='CPU synchronous client wall time; constructor/input setup excluded; fixed training windows detached after each; full step includes loss, zero_grad, detach',
            profile='separate run; replay worker-duration sums are not wall time and are never added to it',
            denominator='effective input sample positions (sum ragged lengths), not number of graph input ports',
            correctness_anchor='CPU qualification + directed frozen benchmark smoke; medium/large finite output checks, no arbitrary-model proof'),
        tracking=track.record,artifacts=dict(metrics='metrics.jsonl',stdout='stdout.log',summary='summary.json',
                                           config='config.json',worker='worker'))
    write_json(out/'run.json',record);replace_text(out/'metrics.jsonl','')
    error=None;events=[];code=1;started=time.monotonic();audit={};cancelled=None
    previous_handlers={}
    def interrupt(signum,_):
        nonlocal cancelled
        cancelled=128+signum
        raise SystemExit(cancelled)
    try:
        for sig in (signal.SIGTERM,signal.SIGINT):previous_handlers[sig]=signal.signal(sig,interrupt)
        if preflight['estimated_peak_bytes']>limits['memory_budget_bytes']:
            raise MemoryError('resource preflight exceeds dynamic aggregate memory budget')
        track.start();write_json(out/'run.json',record)
        env=dict(os.environ,TORCH_DEVICE_BACKEND_AUTOLOAD='0',OMP_NUM_THREADS=str(args.threads),OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_WAIT_POLICY='PASSIVE')
        with (out/'stdout.log').open('w') as log:
            worker_code=run_child(command,cwd=root,env=env,log=log,affinity=affinity,
                                  timeout=args.timeout_seconds,memory_budget=limits['memory_budget_bytes'],audit=audit)
        if worker_code:raise RuntimeError(f'worker exit{worker_code}')
        worker=json.loads((out/'worker/worker.json').read_text());measured=json.loads((out/'worker/measured.json').read_text())
        if worker['state']!='passed' or worker['exit_code']!=0:raise ValueError('nonterminal worker result')
        if worker['native_binary_sha256']!=source['native_binary_sha256']:raise ValueError('worker binary differs')
        metrics={'resources/peak_rss_bytes':worker['peak_rss_bytes'],'work/parameters':worker['parameters']}
        positions=measured['observations']['effective_input_positions']
        for mode,seconds in measured['seconds'].items():
            if seconds<=0:raise ValueError('empty timing interval')
            metrics.update({f'perf/{mode}/seconds':seconds,f'perf/{mode}/sample_positions_per_second':positions/seconds,
                            f'perf/{mode}/ms_per_sample_position':seconds*1000/positions,
                            f'perf/{mode}/ms_per_batch_position':seconds*1000/config['sequence']})
        events=[dict(schema_version=1,run_id=run_id,sequence=0,step=0,timestamp=utc_now(),elapsed_seconds=time.monotonic()-started,metrics=metrics)]
        replace_text(out/'metrics.jsonl',''.join(json.dumps(e,allow_nan=False)+'\n' for e in events))
        track.project_events(events);code=0
    except BaseException as exception:error=f'{type(exception).__name__}: {exception}'
    finally:
        for sig in previous_handlers:signal.signal(sig,signal.SIG_IGN)
        try:track.finish()
        except BaseException as exception:error=(error or '')+f'; tracking: {exception}';code=1
        record.update(status='cancelled' if cancelled else 'completed' if code==0 else 'failed',ended_at=utc_now())
        try:worker_record=json.loads((out/'worker/worker.json').read_text())
        except (OSError,ValueError):worker_record={}
        summary=dict(schema_version=1,run_id=run_id,status=record['status'],ended_at=record['ended_at'],
            exit_code=cancelled or code,worker_exit_code=audit.get('worker_exit_code'),error=error,
            process_seconds=time.monotonic()-started,peak_combined_rss_bytes=audit.get('peak_combined_rss_bytes',0),
            process_peak_rss_bytes=audit.get('process_peak_rss_bytes'),
            unreaped_child_pid=audit.get('pid') if audit.get('remaining_group_pids') else None,
            metrics=events[0]['metrics'] if events else {},tracking=track.record,
            model_constructed=worker_record.get('model_constructed',False))
        write_json(out/'summary.json',summary);write_json(out/'run.json',record)
        write_json(out/'process-audit.json',audit)
        write_json(out/'terminal-resources.json',resources())
        for sig,handler in previous_handlers.items():signal.signal(sig,handler)
    if cancelled:raise SystemExit(cancelled)
    return summary

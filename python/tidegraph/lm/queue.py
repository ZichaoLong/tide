"""Bounded, cancellable model dependencies and serialized CPU/NPU case workers."""
from copy import deepcopy
from contextlib import contextmanager
import fcntl
import os
import re
import signal
from pathlib import Path
import subprocess
import sys
import time
from .assets import ready
from .catalog import local_path
from .config import resolve
from .matrix import manifest
from ..experiment import container_job
from ..experiment.records import now, read_json, write_json
from .report import save_all


def asset_status(key, root):
    if ready(key, root):
        return 'ready'
    path=local_path(root,key)/'asset-state.json'
    if path.exists() and read_json(path).get('state')=='failed':
        return 'failed'
    return 'waiting'


def placeholder(case, out, state, reason):
    resolved=resolve(case['config'])
    if case['action']!='run':
        write_json(out/'check.json',dict(schema='tide-lm-check-v1',state=state,reason=reason,configuration=case['config']))
        return
    write_json(out/'result.json',dict(schema='tide-lm-result-v1',configuration=case['config'],config_id=resolved['config_id'],
        model=case['config']['model'],state=state,execution_status=state,reason=reason,
        validation=dict(numeric='not_checked',discrete='not_checked',checkpoint='not_checked'),finished=now()))


def bounded(command, output, seconds):
    container_job.prepare(output,command=command,cwd=Path.cwd(),timeout=seconds,grace=10,
                          metadata=dict(task_kind='external-lm-case'))
    code=container_job.supervise(output)
    return code


@contextmanager
def compute_lock(path, seconds):
    """Use the kernel waiter queue; repeated nonblocking probes may starve a waiter."""
    def expired(number, frame):
        raise TimeoutError('project compute lock wait exceeded declared budget')
    with Path(path).open('a') as lock:
        previous=signal.signal(signal.SIGALRM,expired)
        signal.setitimer(signal.ITIMER_REAL,seconds)
        try:
            fcntl.flock(lock,fcntl.LOCK_EX)
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            signal.signal(signal.SIGALRM,previous)
        try:
            yield
        finally:
            fcntl.flock(lock,fcntl.LOCK_UN)


def worker(case_path, root, case_dir):
    """Runs after device admission. A fresh supervisor enforces each compute timeout."""
    root=Path(root); case_dir=Path(case_dir); case=read_json(case_path); c=case['config']
    write_json(case_dir/'phase.json',dict(phase='waiting_project_compute_lock',since=now()))
    with compute_lock(root/'compute.lock',21600):
        write_json(case_dir/'phase.json',dict(phase='computing',started=now(),
            physical_devices=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),source_directory=str(Path.cwd())))
        cmd=[sys.executable,'-B','-m','tidegraph','lm',case['action'],str(case_path),
             '--assets',str(root/'models'),'--output',str(case_dir/'run'),'--case-wrapper']
        code=bounded(cmd,case_dir/'process',c['measurement']['timeout_seconds'])
        if code==0 and c['measurement']['checkpoint'] and case['action']=='run':
            code=bounded([sys.executable,'-B','-m','tidegraph','lm','resume',str(case_dir/'run'),
                          '--assets',str(root/'models')],case_dir/'restore-process',c['measurement']['timeout_seconds'])
        write_json(case_dir/'phase.json',dict(phase='finished',exit_code=code,finished=now()))
        return code


def selected_threads(root, attempt='a'):
    samples=[]
    for path in (Path(root)/'runs'/('cpu-'+attempt)).glob('threads-*/run/result.json'):
        record=read_json(path)
        if record.get('execution_status')=='completed' and record.get('measurements'):
            samples.append((record['measurements'][0]['total_seconds'],record['configuration']['execution']['threads']))
    return min(samples)[1] if samples else 16


def execute(root, backend, *, npu_helper=None, hours=24, phase='first', attempt='a', selection=None):
    if not re.fullmatch(r'[a-z0-9-]+',attempt):
        raise ValueError('attempt: lowercase letters/digits/hyphens required')
    root=Path(root).resolve(); base=root/'runs'/((backend if phase=='first' else backend+'-stage2')+'-'+attempt)
    base.mkdir(parents=True,exist_ok=False)
    if selection:
        from .selection import validate
        plan=validate(selection)
    else:
        plan=manifest()
    cases=[x for x in plan['cases'] if x['config']['execution']['backend']==backend
           and (selection or ((x['kind']=='stage2') == (phase=='second')))]
    if not cases:raise ValueError('queue: no cases selected for requested backend')
    write_json(base/'manifest.json',{**plan,'cases':cases})
    started=time.monotonic();pending=deepcopy(cases);results=[]
    record=dict(schema='tide-lm-queue-v1',backend=backend,started=now(),state='running',results=results)
    def save():
        record['pending']=[x['name'] for x in pending];write_json(base/'queue.json',record)
    save()
    if backend=='npu' and (not npu_helper or not Path(npu_helper).is_file()):
        raise ValueError('NPU queue requires the explicit site cooperative helper')
    while pending and time.monotonic()-started<hours*3600:
        progressed=False
        for case in list(pending):
            key=case['config']['model'];status=asset_status(key,root/'models')
            if status=='waiting':
                continue
            case_dir=base/case['name'];case_dir.mkdir()
            case['requested_config']=deepcopy(case['config'])
            if not selection and backend=='cpu' and case['kind'] not in {'threads','check','traincheck'}:
                case['config']['execution']['threads']=selected_threads(root,attempt)
                case['thread_selection']='fastest completed 4/16/64 calibration, else explicit default16'
            write_json(case_dir/'case.json',case)
            if status=='failed':
                placeholder(case,case_dir,'dependency_failed','pinned asset download failed; no model execution or retry')
                results.append(dict(name=case['name'],state='dependency_failed'))
            else:
                command=[sys.executable,'-B','-m','tidegraph','lm','_worker',str(case_dir/'case.json'),
                         '--root',str(root),'--case-dir',str(case_dir)]
                if backend=='npu':
                    command=[sys.executable,'-B',str(npu_helper),'--job-id',f'gef-lm-{os.getpid()}-{len(results)}',
                        '--project','gef-lm','--priority','0','--npu-count',str(case['config']['execution']['devices']),
                        '--max-wait-seconds','21600','--state-file',str(case_dir/'allocation.json'),'--',*command]
                write_json(base/'queue.json',{**record,'current':case['name'],'pending':[x['name'] for x in pending]})
                # Outer service owns all descendants. Worker owns separate per-compute deadlines.
                with (case_dir/'queue.log').open('ab',buffering=0) as log:
                    child=subprocess.run(command,stdout=log,stderr=log)
                result_path=case_dir/'run'/('check.json' if case['action']!='run' else 'result.json')
                if not result_path.exists():
                    placeholder(case,case_dir,'process_failed',f'queue/worker exit {child.returncode}; inspect queue.log')
                elif child.returncode and read_json(result_path).get('state')=='running':
                    value=read_json(result_path);value.update(state='process_failed',execution_status='process_failed',
                        error=f'bounded child exited {child.returncode}; inspect process/status.json',finished=now())
                    write_json(result_path,value)
                results.append(dict(name=case['name'],state='finished',exit_code=child.returncode))
            pending.remove(case);progressed=True;save();save_all(base)
            if time.monotonic()-started>=hours*3600:break
        if not progressed:
            record['waiting_for_assets']=True;save();time.sleep(30)
    for case in pending:
        case_dir=base/case['name'];case_dir.mkdir();write_json(case_dir/'case.json',case)
        placeholder(case,case_dir,'not_run','matrix deadline reached before dependencies or resources became available')
    record.update(state='completed' if not pending else 'deadline_reached',finished=now())
    save();save_all(base)
    # Completion means manifest processing only; all inner outcomes stay visible.
    return record

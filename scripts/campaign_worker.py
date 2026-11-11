#!/usr/bin/env python3
"""Finite sequential cases within one durable worker; separate stage lifecycles."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json,digest,now
from tide_lab.supervisor import group_processes


class Worker:
    def __init__(self,a):
        self.a=a;self.child=None;self.recorder=None;self.stopped=False
        a.output.mkdir(parents=True,exist_ok=False)
        self.state=dict(schema='tide-language-campaign-worker-v1',state='running',started_at=now(),
            plan_sha256=digest(a.plan),cases=[],physical_visibility=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'))
        self.plan=json.loads(a.plan.read_text());self.publish()
        signal.signal(signal.SIGTERM,self.stop);signal.signal(signal.SIGINT,self.stop)

    def publish(self):atomic_json(self.a.output/'state.json',self.state)

    def stop(self,sig,frame):
        self.stopped=True
        if self.child and self.child.poll() is None:os.killpg(self.child.pid,sig)

    def stage(self,case,name,action,cap,extra=(),record=False):
        out=Path(case['output'])/name;cap=min(cap,self.a.deadline_epoch-time.time()-30)
        if cap<=30 or self.stopped:return None
        if action=='train':extra=[*extra,*case.get('train_arguments',[])]
        argv=[sys.executable,'scripts/campaign_stage.py',action,'--config',case['config'],
            '--data',str(self.a.data),'--output',str(out),'--deadline-epoch',str(min(self.a.deadline_epoch,time.time()+cap)),*map(str,extra)]
        entry=dict(name=name,action=action,state='running',argv=argv,output=str(out),started_at=now(),cap_seconds=cap)
        self.state['cases'][-1]['stages'].append(entry);self.publish();started=time.monotonic()
        log_path=out.parent/(name+'.log');record_log=None;self.recorder=None;peak=0;timeout=False
        with log_path.open('x') as log:
            self.child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            entry['process_group']=self.child.pid;self.publish()
            try:
                while self.child.poll() is None:
                    rss=0
                    for child in group_processes(self.child.pid):
                        try:rss+=int(Path('/proc',str(child['pid']),'statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
                        except (OSError,ValueError,IndexError):pass
                    peak=max(peak,rss)
                    if rss>case['max_rss_gib']*2**30:
                        entry['failure']='process group RSS cap';self.stop(signal.SIGTERM,None)
                    if time.monotonic()-started>=cap or self.stopped:
                        timeout=True;os.killpg(self.child.pid,signal.SIGTERM)
                        try:self.child.wait(timeout=30)
                        except subprocess.TimeoutExpired:os.killpg(self.child.pid,signal.SIGKILL);self.child.wait()
                        break
                    if record and self.recorder is None and (out/'metrics.jsonl').exists():
                        record_log=(out.parent/(name+'-recording.log')).open('x')
                        self.recorder=subprocess.Popen([str(self.a.recording_cli),'record','--events',str(out/'metrics.jsonl'),
                            '--manifest',str(out/'manifest.json'),'--report',str(out/'projection.json'),'--follow',
                            '--done-file',str(out/'completed.json')],stdout=record_log,stderr=subprocess.STDOUT,
                            env={k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','VIRTUAL_ENV')})
                    time.sleep(2)
            finally:
                if self.child.poll() is None:
                    os.killpg(self.child.pid,signal.SIGTERM)
                    try:self.child.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(self.child.pid,signal.SIGKILL);self.child.wait()
                if self.recorder:
                    if not (out/'completed.json').exists():
                        atomic_json(out/'completed.json',dict(producer_finished=True,state='failed',source='worker cleanup'))
                    try:self.recorder.wait(timeout=45)
                    except subprocess.TimeoutExpired:
                        self.recorder.terminate()
                        try:self.recorder.wait(timeout=5)
                        except subprocess.TimeoutExpired:self.recorder.kill();self.recorder.wait()
                    entry['recording_exit_code']=self.recorder.returncode
                if record_log:record_log.close()
        result=json.loads((out/'result.json').read_text()) if (out/'result.json').exists() else {}
        entry.update(state=result.get('state','failed'),exit_code=self.child.returncode,timeout=timeout,
            peak_rss_mib=peak/2**20,elapsed_seconds=time.monotonic()-started,finished_at=now())
        if self.child.returncode or entry['state']=='running':entry['state']='failed'
        entry['result_sha256']=digest(out/'result.json') if (out/'result.json').exists() else None
        self.publish();print(json.dumps(entry),flush=True)
        return result if entry['state'] in ('completed','budget_truncated') else None

    def run(self):
        for case in self.plan['cases']:
            if self.stopped or time.time()>=self.a.deadline_epoch-60:break
            out=Path(case['output']);out.mkdir(parents=True,exist_ok=False)
            row=dict(id=case['id'],output=str(out),state='running',stages=[],started_at=now())
            self.state['cases'].append(row);self.publish()
            cap=case['timeout_seconds'];start=time.monotonic()
            if case.get('reuse_admission') and case.get('reuse_recovery'):
                result=self.stage(case,'train','train',cap,
                    extra=['--admission',case['reuse_admission'],'--recovery',case['reuse_recovery']],record=True)
                row.update(state=result['state'] if result else 'failed',finished_at=now(),
                    elapsed_seconds=time.monotonic()-start,qualification_reused=dict(
                        admission=case['reuse_admission'],recovery=case['reuse_recovery']))
                self.publish();continue
            admitted=self.stage(case,'check','check',min(900,cap/8))
            if not admitted or not admitted.get('admitted_for_finite_exploration'):
                row.update(state='failed',reason='independent admission failed',finished_at=now());self.publish();continue
            recovery_cap=min(900,cap/8)
            full=self.stage(case,'probe-full','probe',recovery_cap,extra=['--steps','3'])
            split=self.stage(case,'probe-split','probe',recovery_cap,extra=['--steps','1']) if full else None
            resumed=self.stage(case,'probe-resumed','probe',recovery_cap,
                extra=['--steps','2','--resume',out/'probe-split/checkpoint.pt']) if split else None
            recovery=dict(state='failed',config_sha256=digest(case['config']),
                scope='three B2/L3 real-data updates, same backend/config/install, fresh processes; full trajectory unchecked')
            if full and split and resumed:
                recovery['identity']=full['identity']
                # Keep this comparison outside model-owning subprocesses.
                import torch
                from tide_lab.campaign_runtime import exact_tree
                try:
                    exact_tree(torch.load(out/'probe-full/checkpoint.pt',map_location='cpu',weights_only=False),
                               torch.load(out/'probe-resumed/checkpoint.pt',map_location='cpu',weights_only=False))
                    recovery['state']='passed'
                except (AssertionError,ValueError) as error:recovery['error']=repr(error)
            atomic_json(out/'recovery.json',recovery)
            if recovery['state']!='passed':
                row.update(state='failed',reason='exact application recovery failed',finished_at=now());self.publish();continue
            if case.get('retire_probe_checkpoints'):
                temporary=[out/name/'checkpoint.pt' for name in ('probe-full','probe-split','probe-resumed')]
                retirement=dict(reason='exact comparison passed; disposable recovery probes, not scientific training checkpoints',
                    files=[dict(path=str(p),sha256=digest(p),bytes=p.stat().st_size) for p in temporary])
                atomic_json(out/'probe-retirement.json',retirement)
                for p in temporary:p.unlink()
            remaining=cap-(time.monotonic()-start)
            result=self.stage(case,'train','train',remaining,
                extra=['--admission',out/'check/result.json','--recovery',out/'recovery.json'],record=True)
            row.update(state=result['state'] if result else 'failed',finished_at=now(),elapsed_seconds=time.monotonic()-start)
            self.publish()
        self.state.update(state='stopped' if self.stopped else 'completed',finished_at=now(),
            unstarted=[c['id'] for c in self.plan['cases'] if c['id'] not in {r['id'] for r in self.state['cases']}])
        self.publish()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('plan','data','output','recording-cli'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--deadline-epoch',type=float,required=True);a=p.parse_args()
    w=Worker(a)
    try:w.run()
    except BaseException as error:
        w.state.update(state='failed',error=repr(error),finished_at=now());w.publish();raise

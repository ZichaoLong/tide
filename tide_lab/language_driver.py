"""Project lifecycle composition for bounded language stages, not model execution."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .records import atomic_json,digest,now
from .supervisor import group_processes


class StageRunner:
    def __init__(self,args,plan,limits,compute_seconds):
        self.args=args;self.plan=plan;self.limits=limits;self.spent={k:0. for k in limits}
        self.started=time.monotonic();self.deadline=min(args.deadline_epoch,time.time()+compute_seconds)
        args.output.mkdir(parents=True,exist_ok=False)
        self.state=dict(schema='tide-language-chain-v1',state='running',route=args.route,
            started_at=now(),shared_deadline_epoch=args.deadline_epoch,compute_deadline_epoch=self.deadline,
            plan_sha256=digest(args.plan),steps=[],decisions=[],skipped=[],spent_seconds=self.spent)
        self.publish()

    def publish(self):atomic_json(self.args.output/'stage.json',self.state)
    def remaining(self,category):return max(0,min(self.deadline-time.time(),self.limits[category]-self.spent[category]))

    def stage(self,name,action,category,cap,extra=(),record=True):
        remaining=self.remaining(category);cap=min(cap,max(0,remaining-15))
        if cap<=25:
            self.state['skipped'].append(dict(name=name,reason='remaining category/common budget exhausted'))
            self.publish();return None,dict(state='skipped',reason='budget')
        destination=self.args.output/name
        argv=[sys.executable,'scripts/language_stage.py',action,'--plan',str(self.args.plan),'--data',str(self.args.data),
            '--route',self.args.route,'--output',str(destination),'--run-id','tl-'+self.args.output.name+'-'+name,
            '--asset-path',str(self.args.asset_path),*map(str,extra)]
        if getattr(self.args,'cpu_only',False):argv+=['--device','cpu']
        follow=record and action=='train'
        if record and not follow:argv+=['--recording-cli',str(self.args.recording_cli)]
        wrapper=self.args.output/(name+'-wrapper')
        command=[sys.executable,'scripts/bounded_workload.py','--max-seconds',str(cap),
            '--deadline-epoch',str(self.deadline),'--output',str(wrapper),'--',*argv]
        row=dict(name=name,action=action,category=category,state='running',output=str(destination),
                 started_at=now(),cap_seconds=cap)
        self.state['steps'].append(row);self.publish();started=time.monotonic();recorder=None;record_log=None
        with (self.args.output/(name+'.log')).open('x') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            next_memory_check=0.;peak_rss=0
            try:
                while process.poll() is None:
                    if time.monotonic()>=next_memory_check and (wrapper/'result.json').exists():
                        wrapper_state=json.loads((wrapper/'result.json').read_text())
                        rss=0
                        for child in group_processes(wrapper_state.get('process_group')):
                            try:rss+=int(Path('/proc',str(child['pid']),'statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
                            except (OSError,ValueError,IndexError):pass
                        peak_rss=max(peak_rss,rss);next_memory_check=time.monotonic()+2
                        if rss>22*1024**3:
                            row['resource_failure']='stage process-group RSS exceeded 22GiB; preserve 2GiB supervisor/writer reserve'
                            process.terminate()
                    if follow and recorder is None and (destination/'metrics.jsonl').exists():
                        record_log=(self.args.output/(name+'-recording.log')).open('x')
                        recorder=subprocess.Popen([str(self.args.recording_cli),'record','--events',str(destination/'metrics.jsonl'),
                            '--manifest',str(destination/'manifest.json'),'--report',str(destination/'projection.json'),
                            '--follow','--done-file',str(destination/'completed.json')],stdout=record_log,stderr=subprocess.STDOUT,
                            env={k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','VIRTUAL_ENV')})
                    time.sleep(.2)
                if recorder is not None:
                    if not (destination/'completed.json').exists():
                        atomic_json(destination/'completed.json',dict(producer_finished=True,state='failed',exit_code=process.returncode))
                    try:recorder.wait(timeout=max(1,min(45,self.deadline-time.time())))
                    except subprocess.TimeoutExpired:
                        recorder.terminate()
                        try:recorder.wait(timeout=5)
                        except subprocess.TimeoutExpired:recorder.kill();recorder.wait()
            finally:
                for child in (process,recorder):
                    if child is not None and child.poll() is None:
                        child.terminate()
                        try:child.wait(timeout=5)
                        except subprocess.TimeoutExpired:child.kill();child.wait()
                if record_log is not None:record_log.close()
        elapsed=time.monotonic()-started;self.spent[category]+=elapsed
        result=json.loads((destination/'result.json').read_text()) if (destination/'result.json').exists() else dict(state='failed',error='no inner result; see bounded wrapper')
        row.update(state=result.get('state','failed'),exit_code=process.returncode,finished_at=now(),elapsed_seconds=elapsed,
            observed_process_group_peak_rss_mib=peak_rss/2**20,
            result_sha256=digest(destination/'result.json') if (destination/'result.json').exists() else None)
        if process.returncode and row['state']=='completed':row['state']='failed'
        if (destination/'projection.json').exists():
            projection=json.loads((destination/'projection.json').read_text())
            row['lens_status']=projection.get('status');row['lens_readback_matches']=projection.get('readback_matches')
        self.publish();print(json.dumps(row),flush=True)
        return destination,dict(result,state=row['state'])

    def finish(self,state='completed',error=None):
        self.state.update(state=state,finished_at=now(),allocated_elapsed_seconds=time.monotonic()-self.started)
        if error:self.state['error']=str(error)
        self.publish()

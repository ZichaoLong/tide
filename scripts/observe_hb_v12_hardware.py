#!/usr/bin/env python3
"""Finite low-frequency board observations, not exclusive kernel utilization."""
import argparse,json,signal,subprocess,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json,now
from tide_lab.supervisor import TERMINAL

def read(p):return json.loads(p.read_text()) if p.is_file() else {}

def main(a):
    a.output.mkdir(parents=True,exist_ok=False);stopped=[]
    signal.signal(signal.SIGTERM,lambda *x:stopped.append(True));signal.signal(signal.SIGINT,lambda *x:stopped.append(True))
    mapping={}
    for line in subprocess.check_output(['npu-smi','info','-m'],text=True).splitlines():
        fields=line.split()
        if len(fields)==5 and fields[3].isdigit() and fields[4].startswith('Ascend'):mapping[int(fields[3])]=(int(fields[0]),int(fields[1]))
    state=dict(state='running',started_at=now(),samples=0,scope='board/chip AICore,AIVector,AICPU,HBM; shared occupants may contribute;60s samples miss short bursts')
    with (a.output/'observations.jsonl').open('x') as stream:
        try:
            while not stopped and time.time()<a.deadline_epoch:
                active={};terminal=0;count=0
                for p in (a.root/'artifacts/jobs').glob(a.job_pattern+'/job.json'):
                    count+=1;s=read(p.with_name('state.json'));q=read(p.with_name('queue-state.json'))
                    if s.get('state') in TERMINAL:terminal+=1
                    elif q.get('status')=='running':active.setdefault(q['physical_id'],[]).append(p.parent.name)
                for phy,jobs in active.items():
                    board,chip=mapping[phy]
                    r=subprocess.run(['npu-smi','info','-t','usages','-i',str(board),'-c',str(chip)],capture_output=True,text=True,timeout=15)
                    values={s.split(':',1)[0].strip():s.split(':',1)[1].strip() for s in r.stdout.splitlines() if ':' in s}
                    stream.write(json.dumps(dict(at=now(),physical_id=phy,jobs=jobs,returncode=r.returncode,values=values))+'\n');stream.flush();state['samples']+=1
                state.update(last_at=now(),active_devices=list(active));atomic_json(a.output/'state.json',state)
                submitted=read(a.root/'runs/v12-submit-main/state.json').get('state')=='submitted'
                if a.completion_record:
                    if read(a.completion_record).get('state') in TERMINAL:break
                elif submitted and count==38 and terminal==count:break
                time.sleep(60)
            state.update(state='stopped' if stopped else 'completed',finished_at=now())
        except BaseException as e:state.update(state='failed',error=repr(e),finished_at=now());raise
        finally:atomic_json(a.output/'state.json',state)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--job-pattern',default='v12-main-*');p.add_argument('--completion-record',type=Path)
    p.add_argument('--deadline-epoch',type=float,required=True);main(p.parse_args())

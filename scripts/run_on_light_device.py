#!/usr/bin/env python3
"""User-authorized v5 shared-card admission; does not change site policy/files.

Accepts the project's queue-helper interface. Two load samples, headroom and
per-account device locks remain useful, but an existing process is allowed by
the user's 2026-11-09 instruction. No exclusive-performance claim is made.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json,now
from tide_lab.supervisor import process_identity


def snapshot():
    mapping={}
    for line in subprocess.check_output(['npu-smi','info','-m'],text=True,timeout=20).splitlines():
        fields=line.split()
        if len(fields)==5 and fields[3].isdigit() and fields[4].startswith('Ascend'):
            mapping[(int(fields[0]),int(fields[1]))]=int(fields[3])
    raw=subprocess.check_output(['npu-smi','info'],text=True,timeout=20)
    rows={};health={};process_table=False
    for line in raw.splitlines():
        if 'Process id' in line:process_table=True
        fields=[v.strip() for v in line.split('|')[1:-1]]
        if len(fields)<3:continue
        ids=fields[0].split()
        if len(ids)!=2 or not ids[0].isdigit():continue
        if not ids[1].isdigit():
            health[int(ids[0])]=fields[1];continue
        if process_table:
            phy=mapping.get(tuple(map(int,ids)))
            if phy in rows and fields[1].isdigit():
                pid=int(fields[1]);rows[phy]['processes'].append(process_identity(pid))
        else:
            values=re.fullmatch(r'(\d+)\s+\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)',fields[2])
            if values:
                phy=int(ids[1]);util,used,total=map(int,values.groups())
                board=next((b for (b,c),p in mapping.items() if p==phy),None)
                rows[phy]=dict(physical_id=phy,aicore_percent=util,hbm_used_mib=used,
                    hbm_total_mib=total,health=health.get(board),processes=[])
    if set(rows)!=set(mapping.values()) or not rows:raise RuntimeError('incomplete device observation')
    return rows


def candidates(first,second,policy):
    good=[]
    for phy in policy['candidate_ids']:
        pair=[first.get(phy),second.get(phy)]
        if all(r and r['health']=='OK' and r['aicore_percent']<=policy['max_aicore_percent']
               and r['hbm_total_mib']-r['hbm_used_mib']>=policy['minimum_free_hbm_mib'] for r in pair):
            good.append(phy)
    return sorted(good,key=lambda p:(max(first[p]['hbm_used_mib'],second[p]['hbm_used_mib']),
                                    max(first[p]['aicore_percent'],second[p]['aicore_percent']),p))


def lock(path):
    path.parent.mkdir(parents=True,exist_ok=True);handle=path.open('a+')
    try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:handle.close();return None
    return handle


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('job-id','project','state-file'):p.add_argument('--'+name,required=True)
    p.add_argument('--priority',type=int,default=0);p.add_argument('--npu-count',type=int,default=1)
    p.add_argument('--max-wait-seconds',type=float,required=True);p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args();policy=json.loads(Path(os.environ['TIDE_LAB_RESOURCE_POLICY']).read_text())
    if policy.get('authorization')!='user-2026-11-09-shared-low-load' or a.npu_count!=1:
        raise ValueError('explicit shared-card policy and one-card workload required')
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    if not command:raise ValueError('empty command')
    start=time.monotonic();deadline=min(policy['deadline_epoch'],time.time()+a.max_wait_seconds)
    runtime=Path(os.environ.get('XDG_RUNTIME_DIR',f'/run/user/{os.getuid()}'))
    state=dict(schema='tide-shared-admission-v1',job_id=a.job_id,project=a.project,
        status='waiting_resources',authorization=policy['authorization'],exclusive_device=False,
        policy=policy,started_at=now(),npu_count=1)
    child=None;handles=[];stopped=False
    def stop(signum,frame):
        nonlocal stopped
        stopped=True
        if child is not None and child.poll() is None:child.send_signal(signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        while time.time()<deadline and not stopped:
            pending=False
            for path in policy.get('prerequisites',{}).get(a.job_id,[]):
                if not Path(path).exists():pending=True;continue
                prerequisite=json.loads(Path(path).read_text())
                if prerequisite['state']=='failed':
                    state.update(status='dependency_failed',dependency=path);return 3
                if prerequisite['state']!='completed':pending=True
            if pending:
                state.update(status='waiting_dependency',timestamp=now());atomic_json(a.state_file,state)
                time.sleep(5);continue
            state.update(status='waiting_resources',timestamp=now());atomic_json(a.state_file,state)
            first=snapshot();time.sleep(2);second=snapshot()
            for phy in candidates(first,second,policy):
                slot=None
                for index in range(policy['max_new_concurrent_npus']):
                    slot=lock(runtime/policy.get('slot_namespace','tide-v5-slots')/f'{index}.lock')
                    if slot:break
                if slot is None:break
                device=lock(runtime/'ascend-device-locks'/f'physical-{phy}.lock')
                if device is None:slot.close();continue
                handles=[slot,device]
                state.update(status='running',physical_id=phy,physical_ids=[phy],logical_device='npu:0',
                    logical_devices=['npu:0'],admission_samples=[first[phy],second[phy]],timestamp=now())
                atomic_json(a.state_file,state)
                env=dict(os.environ,ASCEND_RT_VISIBLE_DEVICES=str(phy))
                child=subprocess.Popen(command,env=env)
                next_observation=time.monotonic()+60
                while child.poll() is None:
                    if time.monotonic()>=next_observation:
                        try:
                            state['latest_observation']=snapshot()[phy];state['timestamp']=now();atomic_json(a.state_file,state)
                        except (OSError,subprocess.SubprocessError,RuntimeError) as error:
                            state['observation_error']=str(error);atomic_json(a.state_file,state)
                        next_observation=time.monotonic()+60
                    time.sleep(.5)
                code=child.returncode;state.update(status='completed' if code==0 else 'failed',return_code=code)
                return code
            time.sleep(5)
        state.update(status='stopped' if stopped else 'timed_out');return 124
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        state.update(elapsed_seconds=time.monotonic()-start,ended_at=now());atomic_json(a.state_file,state)
        for handle in handles:handle.close()


if __name__=='__main__':raise SystemExit(main())

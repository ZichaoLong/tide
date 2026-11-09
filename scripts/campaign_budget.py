#!/usr/bin/env python3
"""Conservative allocated-core/device accounting and bounded v6 service stops."""
import argparse
import datetime
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


def timestamp(s):return datetime.datetime.fromisoformat(s).timestamp()


def usage(root,exclude=()):
    cpu=npu=cal_cpu=cal_npu=0.;active=[];rows=[];rss=0;threads=0
    for path in sorted((root/'artifacts/jobs').glob('v6-*/job.json')):
        job=json.loads(path.read_text());name=job['job_id']
        if name in exclude:continue
        sf=path.parent/'state.json';state=json.loads(sf.read_text()) if sf.exists() else {}
        start=state.get('started_at',state.get('submitted_at',job['created_at']))
        end=state.get('ended_at',state.get('finalized_at'))
        elapsed=max(0,(timestamp(end) if end else time.time())-timestamp(start))
        seconds=elapsed*job['resources']['threads'];cpu+=seconds
        allocated=elapsed if job['device']=='npu' else 0.;npu+=allocated
        if name.startswith('v6-cal-') or name=='v6-native-build':cal_cpu+=seconds;cal_npu+=allocated
        live=state.get('state') in ('running','submitted','stopping')
        if live:
            active.append(job['unit']);threads+=job['resources']['threads']
            group=state.get('control_group')
            if group:
                try:rss+=int((Path('/sys/fs/cgroup')/group.lstrip('/')/'memory.current').read_text())
                except (OSError,ValueError):pass
        rows.append(dict(job_id=name,state=state.get('state'),allocated_seconds=elapsed,
            threads=job['resources']['threads'],device=job['device']))
    return dict(cpu_core_seconds=cpu,npu_device_seconds=npu,calibration_cpu_core_seconds=cal_cpu,
        calibration_npu_device_seconds=cal_npu,active_units=active,active_threads=threads,
        cgroup_memory_bytes=rss,jobs=rows)


def host_headroom(root):
    mem={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
    paths=[]
    for directory,pattern in [('runs','v6-*'),('artifacts/jobs','v6-*'),('.local','source-v6-*'),('.local','native-*-v6')]:
        paths.extend((root/directory).glob(pattern))
    disk=0
    if paths:
        out=subprocess.check_output(['du','-s','--block-size=1','--',*map(str,paths)],text=True,timeout=40)
        disk=sum(int(line.split()[0]) for line in out.splitlines())
    reply=subprocess.run(['quota','-w','-p'],text=True,capture_output=True,timeout=20)
    quota=None
    for line in reply.stdout.splitlines():
        fields=line.split()
        if fields and fields[0].startswith('/dev/') and len(fields)>=4:
            used=int(fields[1].rstrip('*'));hard=int(fields[3]);quota=(hard-used)*1024
    if reply.returncode or quota is None:raise RuntimeError('account quota could not be inspected')
    return dict(available_ram_bytes=mem['MemAvailable'],new_disk_bytes=disk,quota_headroom_bytes=quota)


def main(a):
    limits=json.loads(a.budget.read_text());a.output.mkdir(parents=True,exist_ok=False)
    stop=[];started=time.monotonic();manual=[]
    signal.signal(signal.SIGTERM,lambda sig,frame:manual.append(sig))
    state=dict(schema='tide-v6-budget-supervision-v1',state='running',started_at=now(),limits=limits,
        accounting='conservative whole service allocation including waits/startup; NPU host threads also charged as CPU; independent devices are not DDP',
        development_reserve_cpu_core_seconds=a.development_reserve)
    last_host=0;host={}
    while True:
        value=usage(a.root,exclude=['v6-budget']);value['cpu_core_seconds']+=a.development_reserve+time.monotonic()-started
        if time.monotonic()-last_host>60 or not host:
            try:host=host_headroom(a.root);state.pop('observation_error',None)
            except Exception as e:state['observation_error']=repr(e);stop.append('headroom observation unavailable')
            last_host=time.monotonic()
        reasons=[]
        if value['cpu_core_seconds']+120*max(1,value['active_threads'])>=limits['cpu_core_seconds']:reasons.append('CPU allocation cap')
        if value['npu_device_seconds']+120*limits['max_npu_workers']>=limits['npu_device_seconds']:reasons.append('NPU allocation cap')
        if time.time()>=limits['deadline_epoch']-120:reasons.append('shared 48h deadline')
        if host.get('new_disk_bytes',0)>limits['max_new_disk_gib']*2**30:reasons.append('campaign disk cap')
        if host.get('quota_headroom_bytes',2**60)<limits['minimum_quota_headroom_gib']*2**30:reasons.append('quota headroom')
        if host.get('available_ram_bytes',2**60)<limits['minimum_available_ram_gib']*2**30:reasons.append('host RAM headroom')
        if value['cgroup_memory_bytes']>limits['max_total_rss_gib']*2**30:reasons.append('aggregate memory cap')
        if value['active_threads']>limits['max_total_threads']:reasons.append('aggregate thread cap')
        if manual:reasons.append('manual campaign stop')
        if reasons:stop.extend(x for x in reasons if x not in stop)
        if stop and value['active_units']:
            subprocess.run(['systemctl','--user','stop','--no-block',*value['active_units']],check=True,timeout=20)
        # Calibration has independent ceilings; a limit does not stop admitted learning.
        if value['calibration_cpu_core_seconds']>=16*3600-600 or value['calibration_npu_device_seconds']>=8*3600-120:
            units=[u for u in value['active_units'] if u.startswith('tide-lab-v6-cal-') or u=='tide-lab-v6-native-build.service']
            if units:subprocess.run(['systemctl','--user','stop','--no-block',*units],check=True,timeout=20)
            state['calibration_cap_stop']=True
        state.update(usage=value,headroom=host,stop_reasons=stop,updated_at=now())
        atomic_json(a.output/'state.json',state)
        if not value['active_units'] and (stop or a.closed_file.exists()):break
        time.sleep(10)
    state.update(state='stopped_by_budget' if stop else 'completed',finished_at=now());atomic_json(a.output/'state.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','budget','output','closed-file'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--development-reserve',type=float,default=3*3600)
    main(p.parse_args())

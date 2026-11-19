#!/usr/bin/env python3
"""Stream completed profiler arrays, with hardware identity from metadata."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=Path('/mi/data2T/zlong/home/llm/graph-execution-foundation')
sys.path.insert(0,str(HERE.parent))
from scripts.job import atomic


def events(path, digest):
    decoder=json.JSONDecoder()
    block=4*1024*1024
    with path.open(newline='') as stream:
        buffer=''; position=0; started=False
        while True:
            while position < len(buffer) and buffer[position] in ' \t\r\n,':
                position+=1
            if not started and position<len(buffer):
                assert buffer[position]=='['
                position+=1;started=True
                continue
            if position<len(buffer) and buffer[position]==']':
                return
            try:
                event,end=decoder.raw_decode(buffer,position)
            except json.JSONDecodeError:
                chunk=stream.read(block)
                if not chunk:raise ValueError('incomplete trace JSON')
                digest.update(chunk.encode())
                buffer=buffer[position:]+chunk;position=0
                continue
            position=end
            yield event


def union(intervals):
    if not intervals:return 0.0
    intervals.sort()
    start,end=intervals[0];total=0.0
    for left,right in intervals[1:]:
        if left>end:
            total+=end-start;start,end=left,right
        else:end=max(end,right)
    return (total+end-start)/1e6


def analyze(descriptor):
    path=Path(descriptor['path'])
    processes={};cpu=Counter();cpu_us=0.0
    intervals=defaultdict(list);kernel_intervals=defaultdict(list)
    task=defaultdict(lambda: [0,0.0]);kinds=defaultdict(lambda: [0,0.0])
    task_samples={};event_count=0;categories=Counter();digest=hashlib.sha256()
    for event in events(path,digest):
        event_count+=1
        pid=event.get('pid');args=event.get('args',{})
        if event.get('ph')=='M' and event.get('name')=='process_name':
            processes[pid]=args.get('name')
        if event.get('ph')!='X':continue
        category=event.get('cat','unspecified');categories[category]+=1
        duration=float(event.get('dur',0));start=float(event.get('ts',0));end=start+duration
        if category=='cpu_op':
            cpu[event.get('name')]+=1;cpu_us+=duration
        if category=='unspecified':
            intervals[pid].append((start,end))
        kind=args.get('Task Type')
        if kind:
            key=(pid,event.get('name'),kind)
            task[key][0]+=1;task[key][1]+=duration
            kinds[(pid,kind)][0]+=1;kinds[(pid,kind)][1]+=duration
            task_samples.setdefault(kind,dict(name=event.get('name'),pid=pid,args=args))
            if str(kind).startswith('KERNEL'):
                kernel_intervals[pid].append((start,end))
    assert digest.hexdigest()==descriptor['sha256'],(path,digest.hexdigest(),descriptor['sha256'])
    hardware={pid:name for pid,name in processes.items() if 'Ascend Hardware' in str(name)}
    hw=[interval for pid in hardware for interval in intervals[pid]]
    kernel=[interval for pid in hardware for interval in kernel_intervals[pid]]
    tasks=[dict(name=name,task_type=kind,calls=value[0],duration_sum_ms=value[1]/1000)
           for (pid,name,kind),value in task.items() if pid in hardware]
    tasks.sort(key=lambda row:-row['duration_sum_ms'])
    out=dict(trace=descriptor,sha256_verified=True,event_count=event_count,categories=dict(categories),
             process_names={str(k):v for k,v in processes.items()},hardware_processes={str(k):v for k,v in hardware.items()},
             host_cpu_op_events=sum(cpu.values()),host_cpu_op_inclusive_duration_sum_seconds=cpu_us/1e6,
             hardware_events=len(hw),hardware_duration_sum_seconds=sum(e-s for s,e in hw)/1e6,
             hardware_interval_union_seconds=union(hw),hardware_span_seconds=(max(e for s,e in hw)-min(s for s,e in hw))/1e6 if hw else 0,
             kernel_events=len(kernel),kernel_duration_sum_seconds=sum(e-s for s,e in kernel)/1e6,
             kernel_interval_union_seconds=union(kernel),top_hardware_tasks=tasks[:20],
             cumprod_hardware_tasks=[x for x in tasks if 'Cumprod' in x['name']],
             index_hardware_tasks=[x for x in tasks if any(y in x['name'] for y in ('Index','Scatter'))],
             task_types={kind:dict(calls=value[0],duration_sum_ms=value[1]/1000)
                         for (pid,kind),value in kinds.items() if pid in hardware},
             interpretation='metadata process_name selects Ascend Hardware, Task Type prefix KERNEL selects kernels; raw timestamps union separately; nested CPU inclusive and concurrent device sums are not additive or normal-time fractions')
    return out


job=sys.argv[1]
costs=json.loads((ROOT/f'artifacts/agg-cost-{job}-fp32-v1/costs.json').read_text())
assert costs['state']=='completed_finite'
output=HERE/f'agg-trace-{job}-analysis.json'
result={}
for row in costs['observations']:
    trace=row.get('diagnostics',{}).get('trace',{}).get('trace')
    if trace:
        key=row['subject']+'/'+row['candidate']
        result[key]=analyze(trace)
        print(json.dumps(dict(subject=key,hardware_processes=result[key]['hardware_processes'],
                              hardware_events=result[key]['hardware_events'],
                              kernel_union=result[key]['kernel_interval_union_seconds']),ensure_ascii=False),flush=True)
atomic(output,result)
print(str(output))

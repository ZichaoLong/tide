#!/usr/bin/env python3
"""Inspect eager/resident consumer memory without Torch, devices or model tensors."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from flow_protocol import validate_packet
from flow_resident_options import add_arguments
from durable_records import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--dtype',choices=('float32','float16','float64'),default='float32')
    parser.add_argument('--preset',choices=('cpu','mixed-a','mixed-b','mixed-c','resident'),default='resident')
    parser.add_argument('--steps',type=int,default=3,help='eager run horizon: measured steps')
    parser.add_argument('--warmup',type=int,default=1,help='eager run horizon: warmup steps')
    parser.add_argument('--workers',type=int,default=1,help='eager concurrent node workers')
    parser.add_argument('--training',action='store_true')
    parser.add_argument('--diagnostics',action='store_true')
    parser.add_argument('--optimizer',choices=('sgd','adamw'),default='sgd')
    parser.add_argument('--windows-per-step',type=int,default=2)
    parser.add_argument('--sample-chunk-rows',type=int,default=0)
    add_arguments(parser)
    a = parser.parse_args()
    if a.device_memory_bytes < 1 or a.resident_library is not None:
        parser.error('offline planning requires positive --device-memory-bytes and no runtime library')
    if not 1 <= a.devices <= 16:
        parser.error('devices must be in 1..16')
    if not all(0 <= v < 2**63 for v in (a.sample_chunk_rows,a.resident_context_bytes)):
        parser.error('sample-chunk-rows and resident-context-bytes must be nonnegative int64')
    if a.output and a.output.exists():
        parser.error('output must be new')
    packet = validate_packet(json.loads(a.packet.read_text()))
    if packet['schema'] != 'tide-complete-flow-workload-v2':
        parser.error('continuous consumer requires v2')
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    if a.preset!='resident':
        from flow_resident_options import FORWARD, TRAINING
        if a.resident_context_bytes or any(getattr(a,'resident_'+k) is not None for k in FORWARD+TRAINING):
            parser.error('resident capacities require resident preset')
        if a.preset=='cpu' and a.devices!=1:
            parser.error('multiple eager devices require an accelerator preset')
        from tools.online_bench.eager_capacity import plan
        record=dict(workload_sha256=packet['sha256'],kind='offline capacity estimate; not execution or qualification')
        try:
            capacity=plan(packet,budgets=[a.device_memory_bytes]*a.devices,dtype=a.dtype,training=a.training,
                optimizer=a.optimizer,steps=a.steps,warmup=a.warmup,windows=a.windows_per_step,workers=a.workers,
                sample_rows=a.sample_chunk_rows,auto_sample_chunks=a.auto_sample_chunks,policy=a.chunk_policy,
                owner_policy=a.owner_policy,owner_map=a.owner_map,head_workspace_bytes=a.head_workspace_bytes,
                backend='cpu' if a.preset=='cpu' else 'npu')
            record.update(state='planned' if capacity['state']=='admitted' else 'refused',memory_admission=capacity)
        except ValueError as error:
            record.update(state='refused',error=str(error))
        if a.output:write_json(a.output,record)
        print(json.dumps(record))
        return 0 if record['state']=='planned' else 2
    if a.dtype=='float64':parser.error('resident consumer requires FP32/FP16 payload')
    from tools.online_bench.capacity import Capacities, Chunks, packet_geometry, plan_samples
    from tools.online_bench.head_budget import head_budget
    def option(key,default):
        value = getattr(a,'resident_'+key)
        return default if value is None else value
    caps = Capacities(**{k:option(name,default) for k,name,default in (
        ('queue','queue',1024),('arrivals','arrivals',1024),('outputs','outputs',1024),('trace','trace',4096),
        ('kv','kv_rows',128),('kv_trace','kv_trace_rows',4096),('program','program_workspace_bytes',64*1024**2))})
    g = packet_geometry(packet,windows=a.windows_per_step,payload=2 if a.dtype=='float16' else 4,
        training=a.training,adamw=a.optimizer=='adamw',diagnostics=a.training or a.diagnostics,
        devices=a.devices,locality=a.owner_policy=='locality')
    sample_rows = min(a.sample_chunk_rows or g.batch,g.batch)
    g = replace(g,batch=sample_rows,sample_chunks=(g.batch-1)//sample_rows+1,
                context_bytes=a.resident_context_bytes)
    record = dict(workload_sha256=packet['sha256'],kind='offline capacity estimate; not execution or qualification')
    try:
        head = head_budget(caps.outputs,g.width,g.vocab,g.payload,g.training,a.head_workspace_bytes,a.chunk_policy=='aggressive')
        chunks = Chunks(**{k:option(name,default) for k,name,default in (
            ('full','full_chunk_rows',16),('emission','emission_chunk_rows',16),('aggregate','aggregate_chunk_rows',8),
            ('attention','attention_chunk_rows',8),('keys','attention_key_rows',128),('reverse','reverse_chunk_rows',16))},head=head.rows)
        record.update(state='planned',memory_admission=plan_samples(g,caps,chunks,[a.device_memory_bytes]*a.devices,
            a.chunk_policy=='aggressive',packet['workload']['batch'],a.auto_sample_chunks,a.owner_map,a.owner_map))
    except ValueError as error:
        record.update(state='refused',error=str(error))
    if a.output:
        write_json(a.output,record)
    print(json.dumps(record))
    return 0 if record['state']=='planned' else 2


if __name__ == '__main__':
    raise SystemExit(main())

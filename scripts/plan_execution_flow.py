#!/usr/bin/env python3
"""Inspect resident complete-consumer memory without Torch, NPUs or model tensors."""
import argparse
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
    parser.add_argument('--dtype',choices=('float32','float16'),default='float32')
    parser.add_argument('--training',action='store_true')
    parser.add_argument('--diagnostics',action='store_true')
    parser.add_argument('--optimizer',choices=('sgd','adamw'),default='sgd')
    parser.add_argument('--windows-per-step',type=int,default=2)
    add_arguments(parser)
    a = parser.parse_args()
    if a.device_memory_bytes < 1 or a.resident_library is not None:
        parser.error('offline planning requires positive --device-memory-bytes and no runtime library')
    if a.output and a.output.exists():
        parser.error('output must be new')
    packet = validate_packet(json.loads(a.packet.read_text()))
    if packet['schema'] != 'tide-complete-flow-workload-v2':
        parser.error('continuous consumer requires v2')
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from tools.online_bench.capacity import Capacities, Chunks, packet_geometry, plan
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
    record = dict(workload_sha256=packet['sha256'],kind='offline capacity estimate; not execution or qualification')
    try:
        head = head_budget(caps.outputs,g.width,g.vocab,g.payload,g.training,a.head_workspace_bytes,a.chunk_policy=='aggressive')
        chunks = Chunks(**{k:option(name,default) for k,name,default in (
            ('full','full_chunk_rows',16),('emission','emission_chunk_rows',16),('aggregate','aggregate_chunk_rows',8),
            ('attention','attention_chunk_rows',8),('keys','attention_key_rows',128),('reverse','reverse_chunk_rows',16))},head=head.rows)
        record.update(state='planned',memory_admission=plan(g,caps,chunks,[a.device_memory_bytes]*a.devices,a.chunk_policy=='aggressive'))
    except ValueError as error:
        record.update(state='refused',error=str(error))
    if a.output:
        write_json(a.output,record)
    print(json.dumps(record))
    return 0 if record['state']=='planned' else 2


if __name__ == '__main__':
    raise SystemExit(main())

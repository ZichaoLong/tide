#!/usr/bin/env python3
"""Gate complete flows against independent schedules before timing them."""
import argparse
import json
import os
from pathlib import Path
from accelerator_scale_identity import client_hash
from build_identity import source_hash
from durable_records import write_json, replace_text
from experiment_record import utc_now
from flow_topology import ranked_graph, make_packet, native_text
from foundation_lifecycle import run_child
from source_identity import digest, source_state


def cases(device, smoke=False):
    schedules = [('pdg','cpu','streaming'), ('timed-dag','cpu','frontier'),
                 ('settle','cpu','settle'), ('pdg','cpu','resident')]
    if device != 'cpu':
        schedules = [('pdg','mixed','resident'), ('timed-dag','mixed','resident'),
                     ('settle','mixed','resident'), ('pdg','device','bounded-eager')]
        if device == 'npu': schedules += [(f,'device','bounded-replay') for f in ('pdg','timed-dag','settle')]
    dtypes = ('float64','float32') if device == 'cpu' else ('float32','float16')
    if smoke: dtypes = ('float32',)
    for dtype in dtypes:
        for memory in ('add','attention'):
            for family, flow, scheduler in schedules:
                yield dict(family=family, flow=flow, scheduler=scheduler, dtype=dtype, memory=memory,
                           training=1, optimizer='adamw', delayed=False)
            if not smoke:
                yield dict(family='timed-dag', flow='cpu' if device=='cpu' else 'mixed', scheduler='frontier' if device=='cpu' else 'resident',
                           dtype=dtype, memory=memory, training=1, optimizer='sgd', delayed=True)
                yield dict(family='pdg', flow='cpu' if device=='cpu' else 'device', scheduler='streaming' if device=='cpu' else 'bounded-replay' if device=='npu' else 'bounded-eager',
                           dtype=dtype, memory=memory, training=0, optimizer='sgd', delayed=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', choices=('cpu','cuda','npu'), required=True)
    p.add_argument('--devices', type=int, default=1)
    p.add_argument('--timeout-seconds', type=int, default=600)
    p.add_argument('--scope', choices=('smoke','full'), default='full')
    p.add_argument('--development', action='store_true', help='record frozen dirty development evidence, never qualification')
    a=p.parse_args()
    if not 1<=a.devices<=16 or a.device=='cpu' and a.devices!=1 or not 1<=a.timeout_seconds<=3600:
        p.error('invalid device or time capacity')
    root=Path(__file__).resolve().parents[1];build=a.build_dir.resolve();out=a.output_dir.resolve()
    binary=build/'tide-complete-flow';manifest=json.loads((build/'client-manifest.json').read_text())
    source,dirty=source_state(root);identity=client_hash(root);binary_hash=digest(binary)
    if dirty and not a.development: raise ValueError('qualification needs a clean frozen commit')
    if (manifest['client_source_sha256']!=identity or manifest['core']['cpp_source_sha256']!=source_hash(root)
            or manifest.get('additional_binaries',{}).get(binary.name)!=binary_hash
            or a.device!='cpu' and manifest['core']['backend']!=a.device):
        raise ValueError('complete-flow build/source/backend mismatch')
    out.mkdir(parents=True,exist_ok=False)
    record=dict(schema='tide-complete-flow-gates-v1',state='running',source=source,dirty=dirty,scope=a.scope,
                development=a.development,client_source_sha256=identity,binary_sha256=binary_hash,
                device=a.device,devices=a.devices,started=utc_now(),cases=[])
    save=lambda:write_json(out/'gates.json',record)
    save()
    try:
        for index,case in enumerate(cases(a.device,a.scope=='smoke')):
            name=f'{index:03d}-'+ '-'.join(str(case[k]) for k in ('family','scheduler','memory','dtype','training','optimizer'))
            graph=ranked_graph(layers=4,region_width=4,fanout=2,local_span=2,skip=1,cross_every=4,delayed=case['delayed'])
            packet=make_packet(graph=graph,memory=case['memory'])
            topology=out/(name+'.txt');replace_text(topology,native_text(packet))
            config={k:v for k,v in case.items() if k!='delayed'}
            config.update(device=a.device,devices=a.devices,topology=str(topology),check=1,width=8,batch=2,vocab=17,steps=3,warmup=0,
                          seed=7,workers=2,output_dir=str(out/name),run_id=name)
            if case['flow']=='device': config.update(read_device='model',control_device='model',ranking_device='model',event_device='model')
            elif case['family']=='timed-dag' and a.device!='cpu':
                config.update(read_device='model',control_device='model',ranking_device='model',event_device='model')
            if not case['scheduler'].startswith('bounded-'):
                config.update(full_autograd='batched',aggregate_autograd='batched',packed_sources=1,batch_next=1,parallel_regions=1)
            command=[str(binary)]
            for key,value in config.items():command+=['--'+key.replace('_','-'),str(value)]
            cell=dict(name=name,command=command,packet_sha256=packet['sha256'],state='running',started=utc_now())
            record['cases'].append(cell);save();audit={}
            with (out/(name+'.log')).open('x') as log:
                code=run_child(command,cwd=root,env=dict(os.environ,TASK_QUEUE_ENABLE='0',TORCH_DEVICE_BACKEND_AUTOLOAD='0',
                    OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'),log=log,
                    affinity=sorted(os.sched_getaffinity(0)),timeout=a.timeout_seconds,memory_budget=24*2**30,audit=audit)
            write_json(out/(name+'-lifecycle.json'),audit)
            cell.update(state='passed' if code==0 else 'failed',exit_code=code,finished=utc_now());save()
            if code:raise RuntimeError(name+': native exit '+str(code))
            if 'PASS complete-flow' not in (out/(name+'.log')).read_text():raise ValueError('missing native acceptance')
            print(name,'passed',flush=True)
        if source_state(root)!=(source,dirty) or client_hash(root)!=identity or digest(binary)!=binary_hash:
            raise ValueError('source/build changed during gate')
        record['state']='passed'
    except BaseException as exc:
        record.update(state='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record['finished']=utc_now();save()


if __name__=='__main__':main()

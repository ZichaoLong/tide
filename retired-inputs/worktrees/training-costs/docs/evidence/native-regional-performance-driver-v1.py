"""Frozen finite experiment input; normal timing then separate numerical comparison."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time

from benchmarks.programme import run_child
from scripts.job import atomic, enable_subreaper

p = argparse.ArgumentParser()
p.add_argument('--binary', type=Path, required=True)
p.add_argument('--native-launcher', type=Path, required=True)
p.add_argument('--device', required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
out = a.output.resolve(); out.mkdir(exist_ok=False, parents=True)
root = Path.cwd()
spec = importlib.util.spec_from_file_location('regional_reference', root/'native/regional/reference.py')
ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
record = dict(source=os.environ['TIDE_SOURCE_COMMIT'], state='running', device=a.device,
              physical_devices=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
              resource_identity=os.environ['TIDE_RESOURCE_IDENTITY'],
              binary=str(a.binary), binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest(),
              driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              native_launcher_sha256=hashlib.sha256(a.native_launcher.read_bytes()).hexdigest(),
              cpu_affinity=sorted(os.sched_getaffinity(0)), host_load_before=os.getloadavg(),
              scope='regional scoring subpath; paired inputs, matching math operator layout; five synchronized samples after two discarded-owner warmups; training samples continue parameters and AdamW',
              memory_scope='sampled child RSS; device allocator peaks and native kernel placement unmeasured',
              children=[], comparisons=[])
persist = lambda: atomic(out/'programme.json', record)
enable_subreaper()
def stop(signum, frame):
    raise KeyboardInterrupt(str(signum))
signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
started = time.monotonic()
try:
    for name, shape in [('medium', dict(width=64,nodes=8,batch=8,positions=128,tasks=4)),
                        ('large', dict(width=128,nodes=16,batch=16,positions=128,tasks=4))]:
        fixture = out/(name+'-fixture')
        tensors = ref.fixture(fixture, **shape)
        del tensors
        for dtype in ('float32', 'bfloat16'):
            for mode in ('inference','grad_forward','train'):
                paths = {}
                for client, workers in [('python',1),('cpp',1),('cpp',2),('cpp',4)]:
                    identity = f'{name}-{dtype}-{mode}-{client}{workers}'
                    destination = out/identity
                    shared = ['--input-dir',str(fixture),'--output-dir',str(destination),
                              '--device',a.device,'--dtype',dtype,'--mode',mode,
                              '--inner-threads','1','--warmup','2','--repeats','5']
                    command = ([sys.executable,str(root/'native/regional/reference.py'),'run',*shared]
                               if client=='python' else [str(a.native_launcher),str(a.binary),
                                                        '--workers',str(workers),'--streams','same',*shared])
                    child = dict(identity=identity,command=command,output=str(destination/'result.json'),
                                 log=str(out/(identity+'.log')),timeout_seconds=240,max_rss_bytes=16*2**30)
                    record['children'].append(child); persist()
                    run_child(child,persist,supervisor_pid=os.getpid())
                    if child['state']!='completed':
                        raise RuntimeError(f'{identity}: '+child['state'])
                    paths[(client,workers)] = destination/'tensors'
                expected = ref.read_bundle(paths[('python',1)])
                for workers in (1,2,4):
                    comparison = ref.compare(ref.read_bundle(paths[('cpp',workers)]),expected,rtol=2e-5,atol=2e-6)
                    record['comparisons'].append(dict(shape=name,dtype=dtype,mode=mode,workers=workers,**comparison))
                    persist()
                del expected
    record['state']='completed'
except Exception as error:
    record.update(state='failed',error=repr(error))
    import traceback; traceback.print_exc()
except KeyboardInterrupt:
    record['state']='cancelled'
record.update(elapsed_seconds=time.monotonic()-started,host_load_after=os.getloadavg())
persist()
sys.exit(record['state']!='completed')

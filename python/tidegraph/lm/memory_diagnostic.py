"""One original micro1 update with saved-tensor and dispatch observations."""
from collections import Counter
from pathlib import Path
from .catalog import model
from .config import resolve
from .inputs import create
from .runtime import setup, peaks
from .training import update, optimizer_for
from ..experiment.records import now, write_json


def check(config, assets, output):
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    c=resolve(config)['request'];t=c['task']
    if c['model']!='qwen05' or c['execution']['backend']!='npu' or t['mode']!='full' or t['microbatch']!=1 or t['input_tokens']!=2048:
        raise ValueError('memory-check: explicit original qwen05 NPU full L2048 micro1 required')
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    record=dict(schema='tide-lm-memory-diagnostic-v1',state='running',configuration=c,started=now(),
                scope='instrumented single update; not a formal timing or admission waiver')
    def save():write_json(out/'check.json',record)
    save()
    try:
        owner,tokenizer,device,backend,runtime=setup(resolve(c),assets)
        ids,mask,input_id=create(t,model(c['model'])['config'],tokenizer)
        operators=Counter();saved=Counter();storage={}
        class Observe(TorchDispatchMode):
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                operators[str(func)]+=1
                return func(*args,**(kwargs or {}))
        def pack(value):
            key=str((tuple(value.shape),str(value.dtype),str(value.device)))
            saved[key]+=1
            storage[(str(value.device),value.untyped_storage().data_ptr())]=value.untyped_storage().nbytes()
            return value
        with Observe(),torch.autograd.graph.saved_tensors_hooks(pack,lambda x:x):
            observation=update(owner,optimizer_for(owner),ids,mask,c,device,backend)
        record.update(state='completed',finished=now(),runtime=runtime,input=input_id,
            instrumented_update=observation,operators=dict(operators),saved_shapes=dict(saved),
            saved_storage_bytes=sum(storage.values()),device_peak=peaks(backend,1),
            admission_decision='micro4 conservative guard retained; observed micro1 peak is not proof of micro4 fallback safety')
    except Exception as error:
        record.update(state='failed',finished=now(),error_kind=type(error).__name__,error=str(error)[:2000]);save();raise
    save();return record

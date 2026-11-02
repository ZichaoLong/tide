"""The authorized finite experiment manifest; no tensor/model allocation."""
from .config import resolve
from ..experiment.records import fingerprint


def case(model, backend, batch=8, length=512, *, kind='main', **extra):
    task=dict(batch=batch,input_tokens=length)
    execution=dict(backend=backend)
    measurement={}
    if model=='qwenmoe' and backend=='npu':
        execution.update(devices=2,placement='layer_split')
    for key,value in extra.items():
        if key in {'dtype','threads','attention','devices','placement'}:execution[key]=value
        elif key in {'checkpoint','profile','steps','warmup'}:measurement[key]=value
        else:task[key]=value
    if task.get('mode','inference')!='inference':
        measurement.setdefault('steps',3)
    name=f"{kind}-{model}-{backend}-b{batch}-l{length}"
    if task.get('mode'):name+=f"-{task['mode']}-m{task.get('microbatch',1)}"
    if 'threads' in extra:name+=f"-t{extra['threads']}"
    if 'dtype' in extra:name+='-'+extra['dtype']
    config=resolve(dict(schema='tide-lm-experiment-v1',name=name,model=model,task=task,
                        execution=execution,measurement=measurement))['request']
    return dict(name=name,kind=kind,action='check' if kind=='check' else 'run',config=config)


def manifest():
    cases=[]
    # Checks are separate from the 81-process performance cap and never qualify other points.
    for backend in ('cpu','npu'):
        for key in ('qwen05','qwen15','olmoe'):
            cases.append(case(key,backend,2,16,kind='check',output_tokens=4,warmup=0,
                              dtype='float32' if backend=='cpu' else 'bfloat16'))
        train=case('qwen05',backend,1,16,kind='traincheck',mode='full',global_batch=1,steps=1,warmup=0)
        train['action']='training-check'; cases.append(train)
    cases.append(case('qwen15','npu',2,16,kind='placement-check',devices=2,placement='layer_split',output_tokens=4,warmup=0))
    cases[-1]['action']='check'
    for threads in (4,16,64):
        cases.append(case('qwen15','cpu',kind='threads',threads=threads))
    for backend in ('cpu','npu'):
        for key in ('qwen15','olmoe'):
            for batch in (1,8,32,128,512):
                cases.append(case(key,backend,batch,checkpoint=(batch==1)))
            lengths=(128,2048,8192) if key=='qwen15' else (128,2048)
            for length in lengths:cases.append(case(key,backend,8,length))
            cases.append(case(key,backend,128,kind='varied',input_kind='varied_text'))
        cases.append(case('qwen05',backend))
        for key in ('qwen3','qwen7'):cases.append(case(key,backend,kind='stage2'))
        for batch in ((1,8) if backend=='cpu' else (1,8,32,128,512)):
            cases.append(case('qwenmoe',backend,batch,kind='stage2'))
        if backend=='npu':cases.append(case('qwenmoe',backend,8,2048,kind='stage2'))
        for length in (512,2048):
            for micro in (1,4):
                cases.append(case('qwen05',backend,4,length,kind='train',mode='full',microbatch=micro,
                                  checkpoint=length==512 and micro==1))
        for mode in ('full','lora'):
            cases.append(case('qwen15',backend,4,kind='train',mode=mode,microbatch=1,
                              checkpoint=mode=='lora'))
        cases.append(case('olmoe',backend,4,kind='train',mode='lora',microbatch=1))
        cases.append(case('qwen15',backend,1,128,kind='precision',dtype='float32'))
        for key in ('qwen15','olmoe'):
            cases.append(case(key,backend,32,128,kind='profile',profile=True,output_tokens=4,warmup=0))
    unresolved=[dict(kind='implementation_comparison',count=8,
                    reason='llama.cpp and vLLM-Ascend require an independently matched adapter/environment; not silently mapped to Transformers'),
                dict(kind='tp_ep_comparison',count=2,
                    reason='pending real compatible multi-device engine and communication checks; layer_split is not TP/EP')]
    return dict(schema='tide-lm-matrix-v1',cases=cases,conditional_unqueued=unresolved,
                identity=fingerprint(cases),scope='44 main + 4 varied + 14 train + 3 threads + 2 precision + 4 profiles; 10 optional engine points pending')

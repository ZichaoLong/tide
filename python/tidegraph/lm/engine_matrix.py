"""Four standalone CPU points and six explicitly named placement alternatives."""
from .matrix import case
from .config import resolve
from ..experiment.records import fingerprint

LLAMA_REVISION='2cbca389f515df455412dafa07e2b1359cc72a6d'
ASCEND_REVISION='c09d3e20b9c45c80b34ae3781440d90904a44023'


def manifest(kind='all'):
    if kind not in {'all','llama','placement'}:raise ValueError('engine matrix: all/llama/placement')
    cases=[]
    if kind in {'all','llama'}:
        for key in ('qwen15','olmoe'):
            for batch in (2,1,32):
                checking=batch==2
                x=case(key,'cpu',batch,16 if checking else 512,kind='llama-check' if checking else 'llama-cost',
                       attention='eager',output_tokens=4 if checking else 128,warmup=0 if checking else 1,
                       checkpoint=batch==1)
                x['config']['execution']['engine']='llama_cpp'
                x['config']=resolve(x['config'])['request'];x['action']='check' if checking else 'run'
                cases.append(x)
    if kind in {'all','placement'}:
        for key in ('qwen15','olmoe','qwenmoe'):
            devices=4 if key=='qwenmoe' else 2
            batches=(32,128) if key=='qwenmoe' else (1,32)
            if key!='qwen15':
                x=case(key,'npu',2,16,kind=f'placement-{devices}-check',devices=devices,placement='layer_split',output_tokens=4,warmup=0)
                x['action']='check';cases.append(x)
            for batch in batches:
                x=case(key,'npu',batch,kind=f'placement-{devices}-alternative',devices=devices,placement='layer_split',checkpoint=batch==1)
                x['alternative']='Transformers/Accelerate explicit layer copies; not vLLM or TP/EP'
                cases.append(x)
    return dict(schema='tide-lm-matrix-v1',cases=cases,identity=fingerprint(cases),
        scope='4 llama.cpp CPU cost points + 6 layer-placement alternatives + 4 small checks; no TP/EP qualification',
        engine_sources=dict(llama_cpp=LLAMA_REVISION,vllm_ascend_audit=ASCEND_REVISION))

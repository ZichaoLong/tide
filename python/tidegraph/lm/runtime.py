"""External Transformer ownership, device admission and runtime identity."""
import importlib.metadata as md
import os
import time
from pathlib import Path
from .catalog import local_path,model
from .assets import ready


def setup(resolved, asset_root):
    c=resolved['request'];e=c['execution'];t=c['task'];r=c['resources']
    if not ready(c['model'],asset_root):raise RuntimeError('model assets are missing, incomplete or changed')
    import torch
    if e['backend']=='npu':
        import torch_npu  # Register before any autograd engine is created.
    torch.set_num_threads(e['threads']);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(t['seed'])
    device=torch.device('cpu' if e['backend']=='cpu' else e['backend']+':0')
    backend=getattr(torch,e['backend']) if e['backend']!='cpu' else None
    if backend is not None:
        if not backend.is_available() or backend.device_count()<e['devices']:
            raise RuntimeError(f"requested {e['devices']} {e['backend']} devices unavailable")
        backend.set_device(0)
    b=resolved['budget']
    host_free = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    if b['host_estimate_bytes']>min(r['host_gib']*1024**3, host_free*.8):
        raise RuntimeError('host static budget exceeds declared job limit')
    admission=[]
    for i in range(e['devices'] if backend else 0):
        free,total=backend.mem_get_info(i)
        reserve=max(r['device_reserve_gib']*1024**3,total*.10)
        admission.append(dict(device=i,free=free,total=total,reserve=reserve))
        estimate=b['per_device'][i]['estimated_bytes']
        if estimate>free-reserve:
            raise RuntimeError(f"capacity rejected before model allocation: device {i}, estimate={estimate}, free={free}, reserve={reserve}")
    from transformers import AutoModelForCausalLM,AutoTokenizer
    path=local_path(asset_root,c['model']);cfg=model(c['model'])['config']
    dtype=torch.float32 if t['mode']=='full' else getattr(torch,e['dtype'])
    kwargs=dict(local_files_only=True,dtype=dtype,attn_implementation=e['attention'],low_cpu_mem_usage=True)
    if e['placement']=='layer_split':
        count=cfg['num_hidden_layers'];n=e['devices']
        mapping={f'model.layers.{i}':f"{e['backend']}:{min(n-1,i*n//count)}" for i in range(count)}
        mapping.update({'model.embed_tokens':str(device),'lm_head':str(device),'model.norm':f"{e['backend']}:{n-1}"})
        kwargs['device_map']=mapping
    begin=time.perf_counter()
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    owner=AutoModelForCausalLM.from_pretrained(path,**kwargs)
    if e['placement']=='single':owner.to(device)
    if t['mode']=='lora':
        from peft import LoraConfig,get_peft_model
        owner=get_peft_model(owner,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,target_modules=['q_proj','v_proj'],bias='none',task_type='CAUSAL_LM'))
        for p in owner.parameters():
            if p.requires_grad:p.data=p.data.float()
    if c['model']=='olmoe' and t['mode']!='inference' and owner.config.router_aux_loss_coef!=0.01:
        raise ValueError('fixed OLMoE training recipe requires router_aux_loss_coef=0.01')
    owner.eval() if t['mode']=='inference' else owner.train()
    # Streaming forward carries dropout=0/eval; training recipe disables stochastic
    # dropout explicitly only where adapters request dropout=0.
    synchronize(backend,e['devices'])
    parameters={}
    for name,p in owner.named_parameters():
        key=str(p.device)
        q=parameters.setdefault(key,dict(parameters=0,bytes=0,trainable=0))
        q['parameters']+=p.numel();q['bytes']+=p.numel()*p.element_size()
        q['trainable']+=p.numel() if p.requires_grad else 0
    identity=dict(packages={x:md.version(x) for x in ['torch','transformers','accelerate','peft']+(['torch-npu'] if e['backend']=='npu' else [])},
        model_revision=resolved['model_revision'],parameter_locations=parameters,
        host_free_before_allocation=host_free, actual_parameters=sum(x['parameters'] for x in parameters.values()),
        precision=dict(parameter_dtype=str(dtype),compute_dtype=e['dtype'],tf32=False,
                       trainable={n:str(p.dtype) for n,p in owner.named_parameters() if p.requires_grad and t['mode']=='lora'}),
        visible_physical_devices=os.environ.get('ASCEND_RT_VISIBLE_DEVICES') if e['backend']=='npu' else os.environ.get('CUDA_VISIBLE_DEVICES'),
        placement=e['placement'],communication='none' if e['devices']==1 else 'accelerate explicit cross-layer copies; not TP/EP',
        attention_requested=e['attention'],attention_effective=owner.config._attn_implementation,
        construction_seconds=time.perf_counter()-begin,admission=admission,
        threads=dict(aten=torch.get_num_threads(),interop=torch.get_num_interop_threads(),
                     environment={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']}))
    return owner,tokenizer,device,backend,identity


def synchronize(backend,count=1):
    if backend:
        for i in range(count):backend.synchronize(i)


def peaks(backend,count):
    if backend is None:return []
    return [dict(device=i,allocated=backend.max_memory_allocated(i),reserved=backend.max_memory_reserved(i)) for i in range(count)]

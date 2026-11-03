"""Allocation-free external-LM configuration and conservative resource estimates."""
from copy import deepcopy
from .catalog import model
from ..experiment.records import fingerprint, read_json

GIB = 1024**3


def resolve(value):
    value = read_json(value) if isinstance(value, (str, bytes)) or hasattr(value, '__fspath__') else deepcopy(value)
    if not isinstance(value, dict):
        raise ValueError('configuration: expected an object')
    allowed = {'schema', 'name', 'model', 'task', 'execution', 'measurement', 'resources'}
    if set(value)-allowed:
        raise ValueError(f'configuration: unknown fields {sorted(set(value)-allowed)}')
    if value.get('schema') != 'tide-lm-experiment-v1':
        raise ValueError('schema: expected tide-lm-experiment-v1')
    spec = model(value.get('model'))
    defaults = {
        'task': dict(mode='inference', batch=1, input_tokens=512, output_tokens=128, seed=19,
                     input_kind='synthetic_tokens', microbatch=1, global_batch=4),
        'execution': dict(backend='cpu', dtype='bfloat16', engine='transformers', threads=16,
                          devices=1, placement='single', attention='sdpa', prefill_chunk=64),
        'measurement': dict(warmup=1, steps=1, timeout_seconds=1200, checkpoint=False, profile=False),
        'resources': dict(host_gib=384, device_reserve_gib=6.4, workspace_gib=4,
                          checkpoint_gib=24, project_gib=160, account_gib=300)}
    for section, values in defaults.items():
        supplied = value.setdefault(section, {})
        if not isinstance(supplied, dict) or set(supplied)-set(values):
            raise ValueError(f'{section}: invalid or unknown fields')
        value[section] = {**values, **supplied}
    t, e, m, r = [value[k] for k in defaults]
    for section, fields in (('task', ('batch', 'input_tokens', 'output_tokens', 'microbatch', 'global_batch')),
                            ('execution', ('threads', 'devices', 'prefill_chunk')),
                            ('measurement', ('steps', 'timeout_seconds'))):
        for key in fields:
            x = value[section][key]
            if type(x) is not int or x <= 0:
                raise ValueError(f'{section}.{key}: expected positive integer')
    if type(t['seed']) is not int or type(m['warmup']) is not int or m['warmup'] < 0:
        raise ValueError('task.seed/measurement.warmup: expected integers; warmup >=0')
    for key in ('checkpoint', 'profile'):
        if type(m[key]) is not bool:
            raise ValueError(f'measurement.{key}: expected bool')
    choices = [(t, 'mode', {'inference', 'full', 'lora'}),
               (t, 'input_kind', {'synthetic_tokens', 'varied_text'}),
               (e, 'backend', {'cpu', 'npu', 'cuda'}),
               (e, 'dtype', {'float32', 'bfloat16', 'float16'}),
               (e, 'engine', {'transformers', 'llama_cpp'}), (e, 'attention', {'eager', 'sdpa'}),
               (e, 'placement', {'single', 'layer_split'})]
    for section, key, valid in choices:
        if section[key] not in valid:
            raise ValueError(f'{"task" if section is t else "execution"}.{key}: choose {sorted(valid)}')
    if (e['devices'] == 1) != (e['placement'] == 'single'):
        raise ValueError('execution.placement/devices: single needs one; layer_split needs multiple')
    if e['devices'] > 1 and (e['backend'] == 'cpu' or t['mode'] != 'inference'):
        raise ValueError('execution.devices: layer_split implements accelerator inference only')
    if e['devices'] > spec['config']['num_hidden_layers']:
        raise ValueError('execution.devices: cannot exceed transformer layer count')
    if t['global_batch'] % t['microbatch']:
        raise ValueError('task.microbatch: must divide global_batch exactly')
    if t['mode'] != 'inference' and t['batch'] != t['global_batch']:
        raise ValueError('task.batch: training input batch must equal global_batch')
    if t['mode'] != 'inference' and e['dtype'] == 'float16':
        raise ValueError('execution.dtype: this fixed training recipe implements BF16 autocast or FP32, not FP16 scaling')
    if t['input_tokens'] < 2:
        raise ValueError('task.input_tokens: at least two tokens required')
    if t['input_tokens']+t['output_tokens'] > spec['config']['max_position_embeddings']:
        raise ValueError('task.input_tokens/output_tokens: exceed fixed context; no implicit RoPE scaling')
    if t['mode'] != 'inference' and t['input_kind'] != 'synthetic_tokens':
        raise ValueError('task.input_kind: fixed training recipe uses equal length synthetic tokens')
    if m['checkpoint'] and t['mode'] == 'inference' and t['input_kind'] != 'synthetic_tokens':
        raise ValueError('measurement.checkpoint: this continuation check requires fixed-output synthetic_tokens')
    if m['profile'] and (t['mode'] != 'inference' or m['steps'] != 1):
        raise ValueError('measurement.profile: one independent inference observation required')
    if e['engine']=='llama_cpp' and (e['backend']!='cpu' or e['dtype']!='bfloat16' or
          e['attention']!='eager' or t['mode']!='inference' or t['input_kind']!='synthetic_tokens' or
          e['devices']!=1 or m['profile'] or m['steps']!=1 or m['warmup']>1):
        raise ValueError('execution.engine llama_cpp: CPU BF16 weights/KV, eager, single-device synthetic inference, one step, warmup<=1 required')
    for key, x in r.items():
        if type(x) not in (int, float) or not 0 < x < float('inf'):
            raise ValueError(f'resources.{key}: finite positive value required')
    value.setdefault('name', f"{value['model']}-{e['backend']}-b{t['batch']}")
    if not isinstance(value['name'], str) or not value['name']:
        raise ValueError('name: nonempty string required')
    return dict(request=value, config_id=fingerprint(value), model_revision=spec['revision'], budget=budget(value))


def budget(c):
    s = model(c['model']); cfg = s['config']; t = c['task']; e = c['execution']; r = c['resources']
    count = s['weight_parameters']; width = cfg['hidden_size']; layers = cfg['num_hidden_layers']
    size = 4 if e['dtype'] == 'float32' else 2
    train = t['mode'] != 'inference'
    weights = count*(4 if t['mode'] == 'full' else size)
    length = t['input_tokens']+t['output_tokens']
    kv = 0 if train else 2*layers*cfg.get('num_key_value_heads', cfg['num_attention_heads'])*cfg.get(
        'head_dim', width//cfg['num_attention_heads'])*size*t['batch']*length
    rows = t['microbatch']*t['input_tokens'] if train else t['batch']*min(t['input_tokens'], e['prefill_chunk'])
    intermediate = cfg.get('moe_intermediate_size', cfg['intermediate_size'])
    topk = cfg.get('num_experts_per_tok', 1)
    activation = rows*(16*width+3*topk*intermediate)*size*(layers if train else 1)
    # Explicit worst-case attention scores also cover a backend's SDPA math fallback.
    attention = (t['microbatch'] if train else t['batch'])*cfg['num_attention_heads']*4*2*(
        t['input_tokens']**2*layers if train else min(t['input_tokens'], e['prefill_chunk'])*length)
    logits = (rows if train else t['batch'])*cfg['vocab_size']*4*2
    optimizer = count*12 if t['mode'] == 'full' else (256*1024**2 if t['mode'] == 'lora' else 0)
    total = weights+kv+activation+attention+logits+optimizer+r['workspace_gib']*GIB
    # Layer loads are unequal at non-divisible counts; embedding/head stay on device 0.
    embedding = cfg['vocab_size']*width*size*(1 if cfg.get('tie_word_embeddings') else 2)
    layer_weights = max(0, weights-embedding)
    per_device = []
    for i in range(e['devices']):
        local_layers = sum(min(e['devices']-1, j*e['devices']//layers) == i for j in range(layers))
        fraction = local_layers/layers
        estimate = (layer_weights+kv+optimizer)*fraction+(embedding if i == 0 else 0)
        estimate += activation+attention+logits+r['workspace_gib']*GIB  # Not divided: local transient peak.
        per_device.append(dict(device=i, layers=local_layers, estimated_bytes=estimate))
    checkpoint = (weights+optimizer)*1.1 if train else kv*1.1
    return dict(weight_bytes=weights, kv_bytes=kv, activation_estimate_bytes=activation,
                attention_estimate_bytes=attention, logits_bytes=logits,
                optimizer_gradient_estimate_bytes=optimizer, workspace_bytes=r['workspace_gib']*GIB,
                host_estimate_bytes=total if e['backend'] == 'cpu' else weights*2+checkpoint+2*GIB,
                per_device=per_device, per_device_estimate_bytes=max(x['estimated_bytes'] for x in per_device),
                checkpoint_estimate_bytes=checkpoint, device_count=e['devices'], automatic_admission=False,
                note='Static conservative estimates, not measured peaks. Actual free memory is checked before allocation.')

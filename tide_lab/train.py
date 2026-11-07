"""Bounded shared-task training. Installed tidegraph owns all graph execution."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import resource
import signal
import subprocess
import sys
import time
import traceback

os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import torch
from .records import Metrics, atomic_json, digest, now
from .tasks import RetrievalTask, NoMemoryClassifier


def runtime(device, threads, seed):
    if device == 'cpu':
        dev = torch.device('cpu')
    elif device == 'npu:0':
        import torch_npu  # register before any autograd construction
        dev = torch.device(device)
        torch.npu.set_device(dev)
        torch.npu.manual_seed_all(seed)
    else:
        raise ValueError('First batch explicitly supports cpu or remapped npu:0')
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    random.seed(seed)
    return dev


def synchronize(device):
    if device.type == 'npu':
        torch.npu.synchronize()


def make_model(config, task, asset_path):
    options = dict(config['model'])
    kind = options.pop('kind')
    options.setdefault('seed', config['seed'])
    if kind == 'core5':
        from .core5 import Core5Classifier
        cls = Core5Classifier
    elif kind == 'no_memory':
        cls = NoMemoryClassifier
    elif kind == 'growth':
        from .growth import GrowthClassifier
        cls = GrowthClassifier
        if not asset_path:
            raise ValueError('Growth model requires an explicit immutable --asset-path')
        options['checkpoint_path'] = asset_path
    else:
        raise ValueError(f'Unknown model kind: {kind}')
    return cls(task.vocab_size, task.value_vocab_size, **options) if kind != 'growth' else cls(
        vocab_size=task.vocab_size, value_vocab_size=task.value_vocab_size, **options)


def cpu_tree(value, memo=None):
    memo = {} if memo is None else memo
    if isinstance(value, torch.Tensor):
        key = (str(value.device), value.untyped_storage().data_ptr(), value.storage_offset(), tuple(value.shape), tuple(value.stride()), value.dtype)
        if key not in memo:
            memo[key] = value.detach().cpu().clone()
        return memo[key]
    if isinstance(value, dict):
        return {k: cpu_tree(v, memo) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v, memo) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v, memo) for v in value)
    return value


def source_identity():
    root = Path(__file__).resolve().parents[1]
    manifest = root / '.tide-source.json'
    if manifest.exists():
        return json.loads(manifest.read_text())
    files = sorted((root/'tide_lab').glob('*.py'))
    return dict(mode='development', source_sha256=hashlib.sha256(''.join(
        p.name+digest(p) for p in files).encode()).hexdigest(),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip())


def installed_identity():
    import tidegraph
    root = Path(tidegraph.__file__).parent
    source_hash = hashlib.sha256(''.join(str(p.relative_to(root))+digest(p)
        for p in sorted(root.rglob('*.py'))).encode()).hexdigest()
    return dict(python=sys.version, torch=torch.__version__, tidegraph=importlib.metadata.version('tidegraph'),
                tidegraph_source_sha256=source_hash,
                torch_npu=importlib.metadata.version('torch-npu'),
                transformers=importlib.metadata.version('transformers'),
                mathematics='0770593838a37c790a568226930567ed60f99b83',
                foundation_commit='85d161d5c9148eac2b164f89a0ba4204b4331898')


def checkpoint_save(path, *, model, optimizer, step, config_hash, identity, device):
    state = dict(schema='tide-lab-checkpoint-v1', model=cpu_tree(model.state_dict()),
                 optimizer=cpu_tree(optimizer.state_dict()), step=step,
                 config_sha256=config_hash, identity=identity,
                 rng=torch.get_rng_state(), python_rng=random.getstate(),
                 accelerator_rng=cpu_tree(torch.npu.get_rng_state_all()) if device.type == 'npu' else None,
                 sequence_state='empty; independent episodes; full BPTT; reset before optimizer')
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    previous = path.with_name('previous.pt')
    if path.exists():
        os.replace(path, previous)
    os.replace(tmp, path)
    atomic_json(path.with_suffix('.json'), dict(schema=state['schema'], step=step,
               sha256=digest(path), config_sha256=config_hash, sequence_state=state['sequence_state']))


def evaluate(model, task, split, config, device, intervention=None):
    model.eval()
    for module in model.modules():
        if hasattr(module, 'observe'):
            module.observe = False
    count = config['evaluation']['samples']
    batch = config['evaluation']['batch_size']
    correct, ce, n = 0, 0., 0
    start = time.monotonic()
    all_x,all_y = task.batch(split,count,0,config['evaluation']['seed'],'cpu')
    with torch.no_grad():
        for b, offset in enumerate(range(0,count,batch)):
            x,y = all_x[offset:offset+batch].to(device),all_y[offset:offset+batch].to(device)
            if intervention == 'without_history':
                x[:, :task.config.pairs] = task.filler_token
            logits = model(x, branch_enabled=False) if intervention == 'without_branch' else model(x)
            if not torch.isfinite(logits).all():
                raise FloatingPointError('Non-finite evaluation logits')
            ce += float(torch.nn.functional.cross_entropy(logits,y,reduction='sum').item())
            correct += int((logits.argmax(-1)==y).sum().item())
            n += y.numel()
    synchronize(device)
    return dict(accuracy=correct/n, loss=ce/n, samples=n,
                seconds=time.monotonic()-start)


def paired_branch_effect(model, task, config, device):
    """Same examples and trained parameters; ablation is not a retrained control."""
    model.eval()
    count, batch = config['evaluation']['samples'], config['evaluation']['batch_size']
    x, y = task.batch('val', count, 0, config['evaluation']['seed'], 'cpu')
    records = []
    with torch.no_grad():
        for offset in range(0, count, batch):
            inputs, target = x[offset:offset+batch].to(device), y[offset:offset+batch].to(device)
            on, off = model(inputs), model(inputs, branch_enabled=False)
            if not torch.isfinite(on).all() or not torch.isfinite(off).all():
                raise FloatingPointError('Non-finite branch intervention')
            on_loss = torch.nn.functional.cross_entropy(on, target, reduction='none').cpu().tolist()
            off_loss = torch.nn.functional.cross_entropy(off, target, reduction='none').cpu().tolist()
            for j, (label, a, b, la, lb) in enumerate(zip(target.cpu().tolist(),
                    on.argmax(-1).cpu().tolist(), off.argmax(-1).cpu().tolist(), on_loss, off_loss)):
                records.append(dict(index=offset+j, target=label, on=a, off=b, on_loss=la, off_loss=lb))
    gains = sum(r['on']==r['target'] and r['off']!=r['target'] for r in records)
    losses = sum(r['on']!=r['target'] and r['off']==r['target'] for r in records)
    return dict(samples=count, gained_correct=gains, lost_correct=losses,
                prediction_changes=sum(r['on']!=r['off'] for r in records),
                on_minus_off_accuracy=(gains-losses)/count,
                off_minus_on_ce=sum(r['off_loss']-r['on_loss'] for r in records)/count,
                per_example=records, scope='paired validation intervention; not causal attribution versus retraining')


def gradient_metrics(model):
    result = {}
    buckets = {}
    for name,p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'encoder' in name or 'tide' in name or 'graph_model' in name:
            group = 'tide'
        elif 'branch' in name:
            group = 'branch'
        elif 'head' in name:
            group = 'head'
        else:
            group = 'other'
        b = buckets.setdefault(group, [0,0,[]])
        b[0] += p.numel()
        if p.grad is not None:
            b[1] += p.numel()
            b[2].append(p.grad.detach().float().norm().square())
    for key,(total,connected,norms) in buckets.items():
        result[f'gradient/{key}_connected_fraction'] = connected/max(total,1)
        result[f'gradient/{key}_norm'] = float(torch.stack(norms).sum().sqrt().item()) if norms else 0.
    return result


def execute(args):
    config_bytes = Path(args.config).read_bytes()
    config = json.loads(config_bytes)
    if config.get('schema') != 'tide-lab-run-v1':
        raise ValueError('Unsupported run schema')
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    if not (0 < config['training']['steps'] <= 400 and 0 < config['training']['max_seconds'] <= 14400):
        raise ValueError('Out of first-batch finite step/time budget')
    if type(config['training']['log_every']) is not int or config['training']['log_every'] < 1:
        raise ValueError('log_every must be a positive integer')
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Use a new output directory; --resume is a new attempt')
    output.mkdir(parents=True)
    (output/'config.json.tmp').write_bytes(config_bytes)
    os.replace(output/'config.json.tmp', output/'config.json')
    run_id = args.run_id or output.name
    device = runtime(config['device'], config.get('threads',1), config['seed'])
    identity = dict(source=source_identity(), installation=installed_identity(), device=str(device),
                    dtype=config.get('dtype','float32'), deterministic=True, threads=config.get('threads',1),
                    physical_visible=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'))
    task = RetrievalTask(config['task'])
    manifest = dict(schema='tide-lab-run-v1', project='tide-learning-lab', name=run_id, run_id=run_id,
                    study_id=config['study_id'], group=config['study_id'], created_at=now(),
                    config=dict(config, config_sha256=config_hash, source=identity['source'],
                                installation=identity['installation']),
                    identity=identity, task=task.to_dict(), recording='external JSONL projection',
                    resume=None if not args.resume else dict(path=args.resume,sha256=digest(args.resume)))
    atomic_json(output/'manifest.json',manifest)
    metrics = Metrics(output,run_id)
    start = time.monotonic()
    result = dict(schema='tide-lab-result-v1', run_id=run_id, study_id=config['study_id'],
                  config_sha256=config_hash, state='running', numerical_check='not_checked_by_training',
                  exact_resume='not_checked_by_training', independent_correctness='see separate checks',
                  started_at=now(), steps=0)
    measurement = dict(start_step=0, steps_counted=0, observed_steps=0, unobserved_steps=0,
        observed_step_seconds=0., unobserved_step_seconds=0., total_step_seconds=0., work_totals={},
        cadence='attempt first step, every log_every updates, and attempt last step',
        scope='new attempt only; all successful optimizer steps; input preparation excluded; forward/backward/clip/optimizer/sync and enabled observation included; JSONL/evaluation/checkpoint excluded',
        activity_scope='public integer work counters accumulated every step; region/gradient/trace samples only at logging steps')
    result['measurement'] = measurement
    atomic_json(output/'state.json',result)
    def interrupted(signum,frame):
        raise InterruptedError(f'Signal {signum}; latest published checkpoint is the resume boundary')
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    model = optimizer = None
    step = 0
    try:
        model = make_model(config,task,args.asset_path).to(device=device,dtype=getattr(torch,config.get('dtype','float32')))
        params = [p for p in model.parameters() if p.requires_grad]
        if len({id(p) for p in params}) != len(params):
            raise AssertionError('Optimizer parameters must be unique')
        if config['training'].get('optimizer','adamw') == 'sgd':
            optimizer = torch.optim.SGD(params,lr=config['training']['lr'],momentum=0.0,foreach=False)
        else:
            optimizer = torch.optim.AdamW(params,lr=config['training']['lr'],eps=config['training'].get('eps',1e-6),
                                         weight_decay=config['training'].get('weight_decay',0.0),foreach=False)
        groups = [p for g in optimizer.param_groups for p in g['params']]
        if {id(p) for p in groups} != {id(p) for p in params} or len(groups)!=len(params):
            raise AssertionError('Unified optimizer coverage mismatch')
        manifest['parameters'] = dict(total=sum(p.numel() for p in model.parameters()),
                                      trainable=sum(p.numel() for p in params), unique_tensors=len(params))
        if hasattr(model,'facts'):
            manifest['model_facts'] = model.facts()
        if not hasattr(model,'facts') and hasattr(model,'parameter_summary'):
            manifest['model_facts'] = model.parameter_summary()
        atomic_json(output/'manifest.json',manifest)
        if args.resume or args.init_from:
            saved = torch.load(args.resume or args.init_from,map_location='cpu',weights_only=False)
            model.load_state_dict(saved['model'],strict=True)
            if args.resume:
                if saved['config_sha256'] != config_hash or saved['identity'] != identity:
                    raise ValueError('Exact resume requires identical config/source/installation/runtime identity')
                optimizer.load_state_dict(saved['optimizer'])
                step = saved['step']
                torch.set_rng_state(saved['rng'])
                random.setstate(saved['python_rng'])
                if device.type=='npu':
                    torch.npu.set_rng_state_all(saved['accelerator_rng'])
        result['initial_validation'] = dict(evaluate(model,task,'val',config,device), step=step)
        metrics.log(step,{'validation/accuracy':result['initial_validation']['accuracy'],
                         'validation/loss':result['initial_validation']['loss']})
        limit = min(config['training']['steps'],args.stop_after or config['training']['steps'])
        last_validation = result['initial_validation']
        first_index = step
        measurement['start_step'] = step
        for index in range(step,limit):
            if time.monotonic()-start > config['training']['max_seconds']:
                result['state']='budget_exhausted'
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            log = ((index+1)%config['training']['log_every']==0 or index==first_index
                   or index+1==limit)
            for module in model.modules():
                if hasattr(module,'observe'):
                    module.observe = log
            x,y = task.batch('train',config['training']['batch_size'],index,
                             config['training']['data_seed'],device)
            synchronize(device)
            tick = time.monotonic()
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits,y)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training logits or loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params,config['training'].get('clip_norm',1.0),
                                                  error_if_nonfinite=True,foreach=False)
            extra = gradient_metrics(model) if log else {}
            optimizer.step()
            synchronize(device)
            elapsed = time.monotonic()-tick
            step = index+1
            measurement['steps_counted'] += 1
            measurement['total_step_seconds'] += elapsed
            measurement['observed_steps' if log else 'unobserved_steps'] += 1
            measurement['observed_step_seconds' if log else 'unobserved_step_seconds'] += elapsed
            for key in ('candidates', 'full', 'state_updates', 'read_events', 'messages',
                        'source_rows', 'region_blocks', 'branch_tokens', 'query_tokens'):
                value = model.last_metrics.get('work/'+key)
                if value is not None:
                    if type(value) is not int or value < 0:
                        raise ValueError(f'Expected nonnegative public integer work counter: {key}')
                    measurement['work_totals'][key] = measurement['work_totals'].get(key, 0)+value
            if log:
                values = dict(extra,**model.last_metrics)
                values.update({'work_total/'+k:v for k,v in measurement['work_totals'].items()})
                values.update({'train/loss':float(loss.detach().item()),
                               'train/accuracy':float((logits.argmax(-1)==y).float().mean().item()),
                               'train/gradient_norm':float(norm.item()), 'time/step_seconds':elapsed,
                               'time/total_step_seconds':measurement['total_step_seconds'],
                               'work/train_examples':step*config['training']['batch_size'],
                               'work/train_tokens':step*config['training']['batch_size']*task.sequence_length,
                               'resource/cpu_maxrss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})
                if device.type=='npu':
                    values['resource/npu_peak_allocated_mib']=torch.npu.max_memory_allocated()/1024**2
                metrics.log(step,values)
            if step%config['training']['eval_every']==0 or step==limit:
                last_validation = dict(evaluate(model,task,'val',config,device), step=step)
                metrics.log(step,{'validation/accuracy':last_validation['accuracy'],
                                 'validation/loss':last_validation['loss'],
                                 'time/validation_seconds':last_validation['seconds']})
            if step%config['training']['checkpoint_every']==0 or step==limit:
                checkpoint_save(output/'checkpoint.pt', model=model,optimizer=optimizer,step=step,
                                config_hash=config_hash,identity=identity,device=device)
            result.update(steps=step,elapsed_seconds=time.monotonic()-start,last_validation=last_validation)
            atomic_json(output/'state.json',result)
        if result['state']=='running':
            result['state']='completed' if step==config['training']['steps'] else 'bounded_stop'
        result.update(steps=step,last_validation=last_validation,
                      trained_examples=step*config['training']['batch_size'])
        gate_met = (result['state']=='completed' and last_validation['step']==step
                    and last_validation['accuracy']>=config.get('learning_gate',0.5))
        if config['evaluation'].get('test',False) and result['state']=='completed':
            if not config['evaluation'].get('test_if_learning_gate',False) or gate_met:
                test_config = copy.deepcopy(config)
                test_config['evaluation']['samples'] = config['evaluation'].get('test_samples',config['evaluation']['samples'])
                result['test']=evaluate(model,task,'test',test_config,device)
            else:
                result['test_status']='not_checked: predeclared validation gate unmet'
        intervention = 'without_branch' if config['model']['kind']=='growth' and config['model'].get('branch','none')!='none' else (
                       'without_history' if config['model']['kind']=='core5' else None)
        if intervention and result['state']=='completed':
            result[intervention]=evaluate(model,task,'val',config,device,intervention)
            if intervention=='without_branch' and config['evaluation'].get('paired_branch',False):
                result['paired_branch']=paired_branch_effect(model,task,config,device)
        result['training_acceptance'] = dict(finite=True,completed_budget=result['state']=='completed',
            validation_accuracy=last_validation['accuracy'],chance=1/task.value_vocab_size,
            learning_gate_met=gate_met)
        result['inference_acceptance'] = dict(finite_validation_logits=True,
                                             samples=last_validation['samples'], independent_check='separate record')
        if result['state']=='completed' and not config['training'].get('retain_checkpoint',True):
            result['checkpoint_retention'] = {'policy':'ephemeral pilot/repeat; terminal records and hashes retained',
                'files':{p.name:digest(p) for p in [output/'checkpoint.pt',output/'previous.pt'] if p.exists()}}
            for p in [output/'checkpoint.pt',output/'previous.pt']:
                p.unlink(missing_ok=True)
        return 0 if result['state'] in ('completed','bounded_stop') else 3
    except BaseException as error:
        result.update(state='stopped' if isinstance(error,InterruptedError) else 'failed',
                      error_type=type(error).__name__,error=str(error),steps=step,
                      traceback=traceback.format_exc())
        traceback.print_exc()
        return 130 if isinstance(error,InterruptedError) else 1
    finally:
        result.update(finished_at=now(),elapsed_seconds=time.monotonic()-start)
        atomic_json(output/'result.json',result)
        atomic_json(output/'state.json',result)
        metrics.close()
        atomic_json(output/'completed.json',dict(producer_finished=True,state=result['state'],at=now()))
        print(json.dumps({k:result[k] for k in ('run_id','state','steps','elapsed_seconds')},ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--run-id')
    p.add_argument('--asset-path')
    group=p.add_mutually_exclusive_group()
    group.add_argument('--resume')
    group.add_argument('--init-from')
    p.add_argument('--stop-after',type=int)
    args=p.parse_args()
    raise SystemExit(execute(args))


if __name__=='__main__':
    main()

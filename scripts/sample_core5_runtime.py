#!/usr/bin/env python3
"""Bounded host-stack sampling, not a full operator trace or NPU kernel profile."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import threading
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import Metrics,atomic_json,digest,now,project_to_lens


class StackSampler:
    def __init__(self, interval=.02):
        self.interval=interval;self.owner=threading.get_ident();self.counts=Counter()
        self.done=threading.Event();self.thread=None

    def sample(self):
        while not self.done.wait(self.interval):
            frame=sys._current_frames().get(self.owner);stack=[]
            while frame is not None and len(stack)<12:
                code=frame.f_code
                stack.append(f'{Path(code.co_filename).name}:{frame.f_lineno}:{code.co_name}')
                frame=frame.f_back
            self.counts[' <- '.join(stack)]+=1

    def __enter__(self):
        self.thread=threading.Thread(target=self.sample,daemon=True);self.thread.start();return self

    def __exit__(self,*unused):
        self.done.set();self.thread.join(timeout=1)
        if self.thread.is_alive(): raise RuntimeError('sampler did not stop')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--recording-cli',required=True)
    a=p.parse_args();config=json.loads(Path(a.config).read_text())
    root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=False)
    from tide_lab.train import runtime,make_model,synchronize,cpu_tree,gradient_metrics,source_identity,installed_identity
    from tide_lab.tasks import RetrievalTask
    from scripts.check_resume import equal
    import torch
    device=runtime(config['device'],config['threads'],config['seed'])
    start=time.monotonic();run_id='tl-s1-npu-sampling'
    atomic_json(root/'manifest.json',dict(project='tide-learning-lab',name=run_id,run_id=run_id,
        study_id='TL-S1',group='TL-S1',config=config,config_sha256=digest(a.config),
        source=source_identity(),installation=installed_identity(),created_at=now()))
    report=dict(state='running',started_at=now(),observations=[],
        axes=dict(execution='running',strict_numerical='original NPU failures retained; not a reference comparison',
                  discrete_semantics='public work counters; prior mechanism admission reused',
                  recovery='not an application restore check',learning='not evaluated'),
        scope='20ms Python main-thread stack samples include waits and GIL bias; neither CPU-time attribution nor NPU kernel durations; a single paired observation is not a speedup benchmark')
    atomic_json(root/'result.json',report);metrics=Metrics(root,run_id)
    try:
        task=RetrievalTask(config['task']);model=make_model(config,task,None).to(device)
        params=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=config['training']['lr'],eps=config['training']['eps'],
            weight_decay=config['training']['weight_decay'],foreach=False)
        def step(index,observe):
            for module in model.modules():
                if hasattr(module,'observe'):module.observe=observe
            x,y=task.batch('train',config['training']['batch_size'],index,config['training']['data_seed'],device)
            optimizer.zero_grad(set_to_none=True);synchronize(device);tick=time.monotonic()
            out=model(x);loss=torch.nn.functional.cross_entropy(out,y)
            if not torch.isfinite(out).all() or not torch.isfinite(loss):raise FloatingPointError('nonfinite profile values')
            loss.backward();torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True,foreach=False)
            extra=gradient_metrics(model) if observe else {}
            optimizer.step();synchronize(device)
            elapsed=time.monotonic()-tick
            return dict(seconds=elapsed,loss=float(loss.detach().cpu()),work=model.last_metrics,gradient=extra)
        report['warmup']=step(0,False)
        initial_model=cpu_tree(model.state_dict());initial_opt=cpu_tree(optimizer.state_dict())
        rng=torch.get_rng_state();accelerator_rng=torch.npu.get_rng_state_all() if device.type=='npu' else None
        terminal=None
        for observe in (False,True):
            model.load_state_dict(initial_model);optimizer.load_state_dict(cpu_tree(initial_opt))
            torch.set_rng_state(rng)
            if accelerator_rng is not None:torch.npu.set_rng_state_all(accelerator_rng)
            with StackSampler() as sampler:
                row=step(1,observe)
            row.update(observe=observe,samples=sum(sampler.counts.values()),
                stack_samples=[dict(stack=k,count=v) for k,v in sampler.counts.most_common(40)])
            report['observations'].append(row);atomic_json(root/'result.json',report)
            metrics.log(int(observe)+1,dict(row['work'],**{'profile/observe':int(observe),
                'profile/synchronized_step_seconds':row['seconds'],'profile/stack_samples':row['samples']}))
            current=dict(model=cpu_tree(model.state_dict()),optimizer=cpu_tree(optimizer.state_dict()))
            if terminal is None:terminal=current
            else:equal(terminal,current)
        report.update(state='completed',observation_equivalence='bitwise model and optimizer after identical step')
        report['axes']['execution']='completed'
    except BaseException as error:
        report.update(state='failed',error=repr(error));report['axes']['execution']='failed';raise
    finally:
        report.update(finished_at=now(),elapsed_seconds=time.monotonic()-start)
        atomic_json(root/'result.json',report);metrics.close()
        atomic_json(root/'completed.json',dict(producer_finished=True,state=report['state']))
    project_to_lens(root,a.recording_cli)


if __name__=='__main__':main()

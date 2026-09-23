"""Separate wall timing and optional forward/replay accounting passes."""
import time
import torch
import _tide_native as core
from foundation_execute import loss, observations


def native_optimizer(model):
    registry=core.ParameterRegistry()
    for name,p in model.named_parameters(remove_duplicate=False): registry.add(name,p)
    group=core.OptimizerGroup();group.parameters=[v.canonical for v in registry.owners()]
    group.lr=.0002;group.eps=1e-5;group.weight_decay=.01
    return registry,core.AdamW(registry,[group])


def execute_pass(execution,modes,*,instrument=False):
    training=any(mode!='nograd-forward' for mode in modes)
    backward=any(mode in {'backward','optimizer','train-step'} for mode in modes)
    update=any(mode in {'optimizer','train-step'} for mode in modes)
    execution.reset()
    registry,opt=native_optimizer(execution.model) if training else (None,None)
    window=execution.config.get('training_window') or execution.config['sequence']
    phases={'nograd-forward':0.,'grad-forward':0.,'backward':0.,'optimizer':0.,'train-step':0.}
    latencies=[];phase_latencies={k:[] for k in phases};stats={};work={};outputs=0;last=None;losses=[]
    core.reset_work(instrument)
    try:
        with torch.set_grad_enabled(training):
            for end in range(window,execution.config['sequence']+window,window):
                end=min(end,execution.config['sequence'])
                phase_before=phases.copy()
                start=time.perf_counter()
                if opt: opt.zero_grad()
                before=time.perf_counter(); result=execution.advance(end); elapsed=time.perf_counter()-before
                phases['grad-forward' if training else 'nograd-forward']+=elapsed
                if training:
                    objective=loss(result)
                    if backward:
                        before=time.perf_counter();objective.backward();phases['backward']+=time.perf_counter()-before
                    execution.detach()
                    if update:
                        before=time.perf_counter();opt.step();phases['optimizer']+=time.perf_counter()-before
                duration=time.perf_counter()-start
                phases['train-step']+=duration;latencies.append(duration)
                for key in phases:phase_latencies[key].append(phases[key]-phase_before[key])
                if training: losses.append(float(objective.detach()))
                outputs+=len(result.outputs);last=result
                counters=observations(execution,result)['stats'] if execution.variant.startswith('python') else result.stats
                for key,value in counters.items():
                    stats[key]=max(stats.get(key,0),value) if key.startswith('max_') else stats.get(key,0)+value
        if instrument: work=core.work_metrics()
        detail=observations(execution,last);detail['stats']=stats;detail['outputs']=outputs
        # Each native window's counters cover real events; sum logical counts.
        detail['logical']={'Aggregate':stats.get('candidate_events',0),'Upd':stats.get('body_candidate_events',stats.get('candidate_events',0)),
                          'Read':stats.get('candidate_events',0),'Next':stats.get('candidate_events',0),
                          'Full':stats.get('selected_events',0)}
        detail['edge_visits']=stats.get('visited_edges',detail['edge_visits'])
        return {'seconds':{k:phases[k] for k in modes},'window_latencies_seconds':latencies,
                'phase_window_seconds':{k:phase_latencies[k] for k in modes},
                'losses':losses,'work':work,'observations':detail,'optimizer_updates':len(latencies) if update else 0}
    finally: core.reset_work(False)


def measure(execution,modes,warmup,progress=lambda stage:None):
    if 'nograd-forward' in modes and len(modes)>1: raise ValueError('inference and training require separate measured passes')
    training=any(mode!='nograd-forward' for mode in modes)
    initial={k:v.detach().clone() for k,v in execution.model.state_dict().items()} if training else None
    def reset_parameters():
        if initial is not None: execution.model.load_state_dict(initial)
    for i in range(warmup):
        progress(f'warmup-{i}')
        reset_parameters();execute_pass(execution,modes)
    progress('measuring');reset_parameters();measured=execute_pass(execution,modes)
    progress('profiling');reset_parameters();profile=execute_pass(execution,modes,instrument=True)
    # Profile overhead is reported separately, never folded into formal timing.
    return measured,profile

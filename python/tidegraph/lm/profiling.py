"""Separate operator-shape observations, never mixed with formal throughput."""
from contextlib import contextmanager
from ..experiment.records import write_json


@contextmanager
def observe(owner, c, out):
    import torch
    histograms = {}; handles = []
    cfg = owner.config
    topk = getattr(cfg, 'num_experts_per_tok', 0)
    experts = getattr(cfg, 'num_experts', 0)
    if experts and topk:
        for name, module in owner.named_modules():
            if name.endswith('.gate') and getattr(module, 'out_features', 0) == experts:
                def hook(_module, _inputs, logits, name=name):
                    selection = logits.float().softmax(-1).topk(topk, dim=-1).indices.reshape(-1)
                    counts = torch.bincount(selection, minlength=experts)
                    histograms[name] = histograms[name]+counts if name in histograms else counts
                handles.append(module.register_forward_hook(hook))
    if c['execution']['backend'] == 'npu':
        import torch_npu
        profiler = torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,
                                                         torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=True, profile_memory=False,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(out/'profile')))
    else:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if c['execution']['backend'] == 'cuda':
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profiler = torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=False)
    try:
        with profiler:
            yield
        if c['execution']['backend'] != 'npu':
            rows = [dict(operator=x.key, calls=x.count, input_shapes=x.input_shapes,
                         self_cpu_us=x.self_cpu_time_total, cpu_total_us=x.cpu_time_total)
                    for x in profiler.key_averages(group_by_input_shape=True)]
            write_json(out/'operator-shapes.json', rows)
        write_json(out/'expert-loads.json', dict(
            scope='observer recomputes FP32 softmax/top-k from actual gate outputs; includes padding; separate profiled process',
            experts=experts, topk=topk, layers={n: v.cpu().tolist() for n, v in histograms.items()}))
    finally:
        for handle in handles:
            handle.remove()

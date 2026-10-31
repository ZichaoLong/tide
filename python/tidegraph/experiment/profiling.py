"""One separately declared observed window over the public runtime."""
from pathlib import Path
import torch
from .records import digest
from .work_observation import observe_work


def step(owner, index, mode, output):
    client = owner.spec["client"]
    implementation = ("resident" if owner.spec["placement"] == "resident" else
                      "python" if client == "pytorch" else "native")
    detail = dict(mode=mode, throughput_sample=False)
    with observe_work(implementation) as packing:
        if mode == "packing":
            result = owner.step(index)
        else:
            device = torch.device(owner.spec["device"])
            if device.type == "npu":
                import torch_npu
                profiler = torch_npu.profiler
                activities = [profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU]
            else:
                profiler = torch.profiler
                activities = [profiler.ProfilerActivity.CPU]
                if device.type == "cuda":
                    activities.append(profiler.ProfilerActivity.CUDA)
            schedule = profiler.schedule(wait=0, warmup=0, active=1, repeat=1)
            with profiler.profile(activities=activities, record_shapes=True, schedule=schedule) as prof:
                with torch.autograd.profiler.record_function("tide.complete_window"):
                    result = owner.step(index)
                prof.step()
            trace = Path(output)/"trace.json"
            prof.export_chrome_trace(str(trace))
            if device.type != "cpu":
                from .profile_trace import inspect_trace
                detail["profile"] = inspect_trace(trace)
            else:
                detail["profile"] = dict(trace_sha256=digest(trace), operators=[
                    dict(name=x.key, calls=x.count, self_cpu_us=x.self_cpu_time_total,
                         input_shapes=x.input_shapes)
                    for x in prof.key_averages(group_by_input_shape=True)])
            detail["trace"] = trace.name
    detail["packing"] = packing
    return result, detail

"""One frozen update or forward request, with warmup state restoration."""
import copy
import time


def cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def capture(model, optimizer, device):
    import torch
    return dict(model=cpu_tree(model.state_dict()), optimizer=cpu_tree(optimizer.state_dict()),
                cpu_rng=torch.get_rng_state().clone(),
                device_rng=(getattr(torch, device.type).get_rng_state(device).cpu().clone()
                            if device.type != "cpu" else None))


def restore(state, model, optimizer, device):
    import torch
    model.load_state_dict(state["model"], strict=True)
    # load_state_dict may retain CPU tensors, including optimizer step counters.
    # Give each run its own copy so warmup cannot mutate the frozen initial state.
    optimizer.load_state_dict(cpu_tree(state["optimizer"]))
    optimizer.zero_grad(set_to_none=True)
    torch.set_rng_state(state["cpu_rng"])
    if state["device_rng"] is not None:
        getattr(torch, device.type).set_rng_state(state["device_rng"], device)


def batches_on_device(material, microbatch, device):
    if microbatch < 1:
        raise ValueError("microbatch must be positive")
    return [{name: value[start:start+microbatch].to(device=device, copy=True)
             for name, value in material.items()}
            for start in range(0, len(material["input_ids"]), microbatch)]


def merge_work(work, incoming):
    for key, value in incoming.items():
        if "/max_" in key:
            work[key] = max(work.get(key, 0), value)
        elif not key.startswith("activity/") or key == "activity/send_strength":
            work[key] = work.get(key, 0)+value


def run_once(runtime, model, optimizer, material, config, device, *, mode, progress):
    """Complete interval includes copies/target counting and final synchronization.

    The training update subinterval is the exact frozen downstream function's
    timer. Forward modes retain built-in model observations but do no loss/VJP.
    No profiler or per-stage synchronization is installed here.
    """
    import torch
    if mode not in {"nograd", "grad-forward", "training"}:
        raise ValueError("unknown diagnostic mode")
    runtime.synchronize(device)
    started = time.perf_counter()
    batches = batches_on_device(material, config["training"]["batch"], device)
    if mode == "training":
        result = runtime.logical_update(model, optimizer, batches, config, progress, device)
        result["update_seconds"] = result.pop("seconds")
    else:
        if hasattr(model, "set_progress"):
            model.set_progress(progress)
        model.eval()
        work = {}
        count = sum(int((batch["labels"] != -100).sum()) for batch in batches)
        runtime.synchronize(device)
        forward_start = time.perf_counter()
        with torch.set_grad_enabled(mode == "grad-forward"):
            for batch in batches:
                logits = model(batch["input_ids"])
                merge_work(work, model.last_metrics)
                del logits  # Includes graph/output disposal, no retained forward graphs.
        runtime.synchronize(device)
        result = dict(tokens=count, work=work, microbatches=len(batches),
                      forward_seconds=time.perf_counter()-forward_start)
    del batches
    runtime.synchronize(device)
    result["complete_seconds"] = time.perf_counter()-started
    result["input_positions"] = material["input_ids"].numel()
    result["effective_target_tokens_per_second"] = result["tokens"]/result["complete_seconds"]
    result["input_positions_per_second"] = result["input_positions"]/result["complete_seconds"]
    if "update_seconds" in result:
        result["update_tokens_per_second"] = result["tokens"]/result["update_seconds"]
    return result


def reset_memory_peak(device):
    import torch
    if device.type != "cpu":
        getattr(torch, device.type).reset_peak_memory_stats(device)


def memory(device):
    import resource
    import torch
    result = dict(process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  rss_scope="process lifetime high-water mark, including construction and warmup")
    if device.type != "cpu":
        backend = getattr(torch, device.type)
        result.update(allocated_bytes=backend.memory_allocated(device),
                      reserved_bytes=backend.memory_reserved(device),
                      peak_allocated_bytes=backend.max_memory_allocated(device),
                      device_scope="allocator peak reset after warmup/restore; allocated live model included")
    return result

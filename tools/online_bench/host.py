"""Public eager consumer: whole training steps and continuous inference windows."""
import time
import torch
from tidegraph import GraphConfig, GraphRuntime, ExecutionOptions, ExecutionPlacement, External
from tidegraph.runtime import synchronize
from .fixture import build_model, encode
from .memory import MemoryRecord


def runtime_for(packet, *, family, implementation, device, dtype, schedule, preset, trace=False,
                native_library=None, placement=None, resident_library=None, resident_limits=None,
                workers=1, packed_sources=False, batch_next=False):
    if family not in packet["families"]:
        raise ValueError("workload is not equivalent in the requested family")
    if schedule not in ("streaming", "prefill"):
        raise ValueError("schedule must be streaming or prefill")
    if packet["schema"] != "tide-complete-flow-workload-v2":
        raise ValueError("continuous consumer requires v2; legacy v1 declares reset windows")
    resident = preset == "resident"
    graph, model, embedding, head = build_model(packet, dtype=getattr(torch, dtype), device="cpu" if resident else device)
    if resident:
        embedding = embedding.detach().to(device)
        head = head.detach().to(device)
    ranks = tuple(packet["graph"]["ranks"]) if family == "settle" else ()
    if family != "settle":
        graph, model = encode(packet, graph, model)
    config = GraphConfig(family, graph, width=packet["workload"]["width"], dtype=dtype, ranks=ranks,
                         seed=packet["workload"]["seed"])
    options = ExecutionOptions(implementation=implementation, schedule="greedy" if schedule == "prefill" else "streaming",
                               prefill=schedule == "prefill", packed=True, trace=trace,
                               workers=workers, packed_sources=packed_sources, batch_next=batch_next,
                               full_autograd="replay" if resident else "batched", aggregate_autograd="replay" if resident else "batched",
                               resident_limits=resident_limits,
                               placement=placement or ExecutionPlacement(preset=preset))
    runtime = GraphRuntime(config, device=str(device), model=model, options=options, native_library=native_library,
                           resident_library=resident_library, model_device="cpu" if resident else None)
    return runtime, embedding, head


def token_values(embedding, packet, position):
    c = packet["workload"]
    # CPU token preparation and the required upload belong inside the timer.
    positions = torch.arange(position, position+c["tokens"], dtype=torch.int64)
    samples = torch.arange(c["batch"], dtype=torch.int64)[:, None]
    ids = (7*positions[None, :] + 3*samples) % c["vocab"]
    return torch.nn.functional.embedding(ids.to(embedding.device), embedding)


def advance(session, values, packet, position):
    c = packet["workload"]
    if session.runtime.spec:
        return session.advance(values)
    external = [External(b, 0, position+t, (position+t)*c["stride"], values[b, t])
                for b in range(c["batch"]) for t in range(c["tokens"])]
    stop = (position+c["tokens"])*c["stride"]
    return session.advance(external, stop=stop, sealed_until=stop)


def output_loss(result, head, packet, denominator):
    if not result.outputs:
        return None
    c = packet["workload"]
    rows = torch.stack([x[3] for x in result.outputs])
    targets = torch.tensor([((t//c["stride"]+1)*7+b*3) % c["vocab"]
                            for b, t, _, _ in result.outputs], dtype=torch.int64, device=head.device)
    logits = rows @ head.t()
    # FP16 has a FP32 loss and an explicitly separate master optimizer policy.
    if logits.dtype == torch.float16:
        logits = logits.float()
    return torch.nn.functional.cross_entropy(logits, targets, reduction="sum") / denominator


def parameters(runtime, embedding, head):
    return {**{name: p for name, p in runtime.execution_model.named_parameters() if p.requires_grad},
            "embedding": embedding, "head": head}


def run(packet, *, family, implementation, device, dtype="float32", schedule="prefill", preset="cpu",
        training=False, optimizer="sgd", steps=3, warmup=1, windows_per_step=2,
        native_library=None, diagnostics=False, placement=None, observer=None, parameter_budget=1024**3,
        resident_library=None, resident_limits=None, training_limits=None, resident_placement=None, head_workspace_bytes=4*1024**3,
        device_memory_bytes=0, workers=1, packed_sources=False, batch_next=False):
    if type(workers) is not int or not 1 <= workers <= 1024 or type(packed_sources) is not bool or type(batch_next) is not bool:
        raise ValueError("invalid host workers/packed-sources/batch-next options")
    if (workers != 1 or packed_sources or batch_next) and (implementation == "python" or preset == "resident"):
        raise ValueError("host workers/packed-sources/batch-next require an eager native consumer")
    if preset == "resident":
        from .resident import run as run_resident
        return run_resident(packet, family=family, implementation=implementation, device=device, dtype=dtype,
            schedule=schedule, training=training, optimizer=optimizer, steps=steps, warmup=warmup,
            windows_per_step=windows_per_step, native_library=native_library, diagnostics=diagnostics,
            placement=placement, observer=observer, parameter_budget=parameter_budget,
            resident_library=resident_library, resident_limits=resident_limits, training_limits=training_limits,
            resident_placement=resident_placement, head_workspace_bytes=head_workspace_bytes, device_memory_bytes=device_memory_bytes)
    if any(x is not None for x in (resident_library, resident_limits, training_limits, resident_placement)) or head_workspace_bytes!=4*1024**3 or device_memory_bytes:
        raise ValueError("resident options require the resident preset")
    if steps < 1 or warmup < 0 or windows_per_step < 1 or optimizer not in ("sgd", "adamw"):
        raise ValueError("invalid bounded run/optimizer configuration")
    if dtype == "float16" and training:
        raise ValueError("consumer FP16 master/head updates pending; explicit FP32 training required")
    if packet["counts"]["parameters"] * {"float16": 2, "float32": 4, "float64": 8}[dtype] > parameter_budget:
        raise ValueError("learned parameter storage exceeds parameter-budget (not a total peak-memory estimate)")
    if (diagnostics or observer is not None) and packet["counts"]["parameters"] > 100000:
        raise ValueError("diagnostics require at most 100000 learned parameters")
    if (steps+warmup)*windows_per_step*packet["workload"]["tokens"]*packet["workload"]["stride"] > (2**63-1)//8:
        raise ValueError("requested continuation would overflow coordinates/token formula")
    start = time.perf_counter()
    memory = MemoryRecord([device])
    runtime, embedding, head = runtime_for(packet, family=family, implementation=implementation, device=device,
        dtype=dtype, schedule=schedule, preset=preset, trace=diagnostics, native_library=native_library, placement=placement,
        workers=workers, packed_sources=packed_sources, batch_next=batch_next)
    c = packet["workload"]; session = runtime.session(c["batch"])
    named = parameters(runtime, embedding, head)
    options = dict(lr=.0001, weight_decay=.001, foreach=False)
    opt = ((torch.optim.SGD(named.values(), momentum=.25, **options) if optimizer == "sgd" else
            torch.optim.AdamW(named.values(), eps=1e-6, **options)) if training else None)
    synchronize(runtime.device)
    construction = time.perf_counter() - start
    memory.capture("construction")
    durations, losses, output_counts, statistics, warmup_times = [], [], [], [], []
    position = 0
    for step in range(warmup+steps):
        synchronize(runtime.device); begin = time.perf_counter()
        if opt:
            opt.zero_grad(set_to_none=True)
        loss = None; outputs = 0; stats = {}
        with torch.set_grad_enabled(training):
            for _ in range(windows_per_step):
                values = token_values(embedding, packet, position)
                result = advance(session, values, packet, position)
                position += c["tokens"]; outputs += len(result.outputs)
                item = output_loss(result, head, packet, c["batch"]*c["tokens"]*windows_per_step)
                if item is not None:
                    loss = item if loss is None else loss + item
                for key, value in result.stats.items():
                    stats[key] = max(stats.get(key, 0), value) if key.startswith("max_") else stats.get(key, 0)+value
                if observer:
                    observer("window", step, result)
            if loss is not None:
                if not torch.isfinite(loss).all():
                    raise RuntimeError("nonfinite consumer loss")
                if opt:
                    loss.backward()
            if opt:
                # One finite agreement before any parameter/optimizer update.
                flags = [torch.isfinite(p.grad).all() for p in named.values() if p.grad is not None]
                if flags and not torch.stack(flags).all():
                    raise RuntimeError("nonfinite gradient; optimizer not applied")
                if observer:
                    observer("gradients", step, named)
                session.detach()
                opt.step()
        synchronize(runtime.device); elapsed = time.perf_counter() - begin
        if observer:
            observer("updated", step, named)
        if step >= warmup:
            durations.append(elapsed)
            losses.append(None if loss is None else float(loss.detach().cpu()))
            output_counts.append(outputs); statistics.append(stats)
        else:
            warmup_times.append(elapsed)
        if step+1 == warmup:
            memory.capture("warmup")
    memory.capture("measured", reset_peak=False)
    return dict(schema="tide-online-consumer-v1", workload_sha256=packet["sha256"],
                host_execution=dict(workers=workers,packed_sources=packed_sources,batch_next=batch_next),
                implementation=implementation, family=family, training=training, optimizer=optimizer if training else None,
                windows_per_step=windows_per_step, warmup_steps=warmup, measured_steps=steps,
                construction_seconds=construction, seconds=durations, warmup_seconds=warmup_times,
                losses=losses, outputs=output_counts, statistics=statistics, parameter_budget=parameter_budget,
                input_tokens_per_step=c["batch"]*c["tokens"]*windows_per_step,
                final_cut=session.continuation.cut, parameters=sum(p.numel() for p in named.values()),
                runtime=runtime.manifest(), diagnostics=diagnostics or observer is not None,
                memory=memory.record(),
                timing="input preparation/upload + online forward + head/loss + backward + finite checks + detach/optimizer + synchronization; no reference")

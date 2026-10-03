"""Public eager consumer: whole training steps and continuous inference windows."""
import time
import math
import torch
from tidegraph import GraphConfig, GraphRuntime, ExecutionOptions, ExecutionPlacement, External
from .fixture import build_model, encode
from .memory import MemoryRecord
from .phase_timing import PhaseTiming


def runtime_for(packet, *, family, implementation, device, dtype, schedule, preset, trace=False,
                native_library=None, placement=None, resident_library=None, resident_limits=None,
                workers=1, packed_sources=False, batch_next=False,
                devices=1, owner_policy="locality", owner_map=()):
    if family not in packet["families"]:
        raise ValueError("workload is not equivalent in the requested family")
    if schedule not in ("streaming", "prefill"):
        raise ValueError("schedule must be streaming or prefill")
    if packet["schema"] != "tide-complete-flow-workload-v2":
        raise ValueError("continuous consumer requires v2; legacy v1 declares reset windows")
    resident = preset == "resident"
    node_devices = None
    if not resident:
        from .eager_placement import resolve
        from .fixture import body_graph
        from tidegraph.placement import validate
        logical, plan = resolve(packet, device, devices, owner_policy, owner_map)
        device = logical[0]
        node_devices = [logical[d] for d in plan["node_owners"][:-2]]
        for owner in logical:
            validate(body_graph(packet), getattr(torch, dtype),
                     (placement or ExecutionPlacement(preset=preset)).resolve(owner))
    graph, model, embedding, head = build_model(packet, dtype=getattr(torch, dtype),
                                               device="cpu" if resident else device, node_devices=node_devices)
    if resident:
        embedding = embedding.detach().to(device)
        head = head.detach().to(device)
    ranks = tuple(packet["graph"]["ranks"]) if family == "settle" else ()
    if family != "settle":
        graph, model = encode(packet, graph, model)
        if node_devices is not None:
            node_devices += [device, device]
    config = GraphConfig(family, graph, width=packet["workload"]["width"], dtype=dtype, ranks=ranks,
                         seed=packet["workload"]["seed"])
    options = ExecutionOptions(implementation=implementation, schedule="greedy" if schedule == "prefill" else "streaming",
                               prefill=schedule == "prefill", packed=True, trace=trace,
                               workers=workers, packed_sources=packed_sources, batch_next=batch_next,
                               full_autograd="replay" if resident else "batched", aggregate_autograd="replay" if resident else "batched",
                               resident_limits=resident_limits,
                               placement=placement or ExecutionPlacement(preset=preset))
    runtime = GraphRuntime(config, device=str(device), model=model, options=options, native_library=native_library,
                           resident_library=resident_library, model_device="cpu" if resident else None,
                           node_devices=node_devices)
    return runtime, embedding, head


def token_values(embedding, packet, position, sample_begin=0, batch_size=None):
    c = packet["workload"]
    # CPU token preparation and the required upload belong inside the timer.
    positions = torch.arange(position, position+c["tokens"], dtype=torch.int64)
    size = c["batch"] if batch_size is None else batch_size
    samples = torch.arange(sample_begin, sample_begin+size, dtype=torch.int64)[:, None]
    ids = (7*positions[None, :] + 3*samples) % c["vocab"]
    return torch.nn.functional.embedding(ids.to(embedding.device), embedding)


def advance(session, values, packet, position):
    c = packet["workload"]
    if session.runtime.spec:
        return session.advance(values)
    external = [External(b, 0, position+t, (position+t)*c["stride"], values[b, t])
                for b in range(values.shape[0]) for t in range(c["tokens"])]
    stop = (position+c["tokens"])*c["stride"]
    return session.advance(external, stop=stop, sealed_until=stop)


def output_loss(result, head, packet, denominator, sample_begin=0):
    if not result.outputs:
        return None
    c = packet["workload"]
    rows = torch.stack([x[3].to(head.device) for x in result.outputs])
    targets = torch.tensor([((t//c["stride"]+1)*7+(b+sample_begin)*3) % c["vocab"]
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
        device_memory_bytes=0, workers=1, packed_sources=False, batch_next=False, sample_chunk_rows=0, context_memory_bytes=0,
        auto_sample_chunks=False, phase_timing=False, devices=1, owner_policy="locality", owner_map=(),
        chunk_policy="conservative", loss_scale=1.):
    phases = PhaseTiming(phase_timing)
    if type(loss_scale) not in (int, float) or not math.isfinite(loss_scale) or loss_scale <= 0:
        raise ValueError("loss-scale must be positive and finite")
    if loss_scale != 1 and (not training or dtype != "float16" or preset == "resident"):
        raise ValueError("nonunit loss-scale requires eager FP16 training")
    if type(auto_sample_chunks) is not bool:
        raise ValueError("auto-sample-chunks must be boolean")
    if type(context_memory_bytes) is not int or not 0 <= context_memory_bytes < 2**63:
        raise ValueError("resident-context-bytes must be a nonnegative int64")
    if type(sample_chunk_rows) is not int or not 0 <= sample_chunk_rows < 2**63:
        raise ValueError("sample-chunk-rows must be a nonnegative int64")
    if type(workers) is not int or not 1 <= workers <= 1024 or type(packed_sources) is not bool or type(batch_next) is not bool:
        raise ValueError("invalid host workers/packed-sources/batch-next options")
    if (workers != 1 or packed_sources or batch_next) and (implementation == "python" or preset == "resident"):
        raise ValueError("host workers/packed-sources/batch-next require an eager native consumer")
    if preset == "resident":
        if devices != 1 or owner_policy != "locality" or owner_map:
            raise ValueError("use resident_placement for resident ownership")
        from .resident import run as run_resident
        return run_resident(packet, family=family, implementation=implementation, device=device, dtype=dtype,
            schedule=schedule, training=training, optimizer=optimizer, steps=steps, warmup=warmup,
            windows_per_step=windows_per_step, native_library=native_library, diagnostics=diagnostics,
            placement=placement, observer=observer, parameter_budget=parameter_budget,
            resident_library=resident_library, resident_limits=resident_limits, training_limits=training_limits,
            resident_placement=resident_placement, head_workspace_bytes=head_workspace_bytes, device_memory_bytes=device_memory_bytes,
            sample_chunk_rows=sample_chunk_rows, context_memory_bytes=context_memory_bytes,auto_sample_chunks=auto_sample_chunks,
            phase_timing=phase_timing)
    if any(x is not None for x in (resident_library, resident_limits, training_limits, resident_placement)) or context_memory_bytes:
        raise ValueError("resident options require the resident preset")
    if steps < 1 or warmup < 0 or windows_per_step < 1 or optimizer not in ("sgd", "adamw"):
        raise ValueError("invalid bounded run/optimizer configuration")
    if packet["counts"]["parameters"] * {"float16": 2, "float32": 4, "float64": 8}[dtype] > parameter_budget:
        raise ValueError("learned parameter storage exceeds parameter-budget (not a total peak-memory estimate)")
    if (diagnostics or observer is not None) and packet["counts"]["parameters"] > 100000:
        raise ValueError("diagnostics require at most 100000 learned parameters")
    if (steps+warmup)*windows_per_step*packet["workload"]["tokens"]*packet["workload"]["stride"] > (2**63-1)//8:
        raise ValueError("requested continuation would overflow coordinates/token formula")
    start = time.perf_counter()
    from .eager_placement import resolve
    logical, payload_placement = resolve(packet, device, devices, owner_policy, owner_map)
    device = logical[0]
    from .eager_capacity_runtime import prepare, observed
    admission = prepare(packet, logical, budget=device_memory_bytes, dtype=dtype, training=training,
        optimizer=optimizer, steps=steps, warmup=warmup, windows=windows_per_step, workers=workers,
        sample_rows=sample_chunk_rows, auto_sample_chunks=auto_sample_chunks, policy=chunk_policy,
        owner_policy=owner_policy, owner_map=owner_map, head_workspace_bytes=head_workspace_bytes)
    memory = MemoryRecord(logical)
    runtime, embedding, head = runtime_for(packet, family=family, implementation=implementation, device=device,
        dtype=dtype, schedule=schedule, preset=preset, trace=diagnostics, native_library=native_library, placement=placement,
        workers=workers, packed_sources=packed_sources, batch_next=batch_next,
        devices=devices, owner_policy=owner_policy, owner_map=owner_map)
    c = packet["workload"]
    chunk = admission["effective_sample_rows"]
    sessions = [(first, min(chunk, c["batch"]-first), runtime.session(min(chunk, c["batch"]-first)))
                for first in range(0, c["batch"], chunk)]
    named = parameters(runtime, embedding, head)
    options = dict(lr=.0001, weight_decay=.001, foreach=False)
    opt = None
    if training and dtype == "float16":
        from tidegraph.precision import FP32MasterOptimizer
        opt = FP32MasterOptimizer(named.values(), optimizer=optimizer, loss_scale=loss_scale,
            **options, **(dict(momentum=.25) if optimizer == "sgd" else dict(eps=1e-6)))
    elif training:
        opt = (torch.optim.SGD(named.values(), momentum=.25, **options) if optimizer == "sgd" else
               torch.optim.AdamW(named.values(), eps=1e-6, **options))
    runtime.synchronize()
    construction = time.perf_counter() - start
    memory.capture("construction")
    durations, losses, output_counts, statistics, warmup_times = [], [], [], [], []
    position = 0
    for step in range(warmup+steps):
        runtime.synchronize(); begin = time.perf_counter()
        if opt:
            opt.zero_grad(set_to_none=True)
        loss = None; outputs = 0; stats = {}
        with torch.set_grad_enabled(training):
            # Samples have independent graph state. Keep all windows of one
            # physical slice connected, then release its graph before the next
            # slice. All slices use the same parameter generation and optimizer.
            for first, size, session in sessions:
                partial = None
                for window in range(windows_per_step):
                    cursor = position+window*c["tokens"]
                    values = token_values(embedding, packet, cursor, first, size)
                    result = advance(session, values, packet, cursor)
                    outputs += len(result.outputs)
                    item = output_loss(result, head, packet, c["batch"]*c["tokens"]*windows_per_step, first)
                    if item is not None:
                        partial = item if partial is None else partial+item
                    for key, value in result.stats.items():
                        stats[key] = max(stats.get(key, 0), value) if key.startswith("max_") else stats.get(key, 0)+value
                    if observer:
                        if chunk == c["batch"]:
                            observer("window", step, result)
                        else:
                            observer("sample_window", step, (first, c["batch"], result))
                if partial is not None:
                    if not torch.isfinite(partial).all():
                        raise RuntimeError("nonfinite consumer loss")
                    if opt:
                        if dtype == "float16":
                            opt.backward(partial)
                        else:
                            partial.backward()
                    loss = partial.detach() if loss is None else loss+partial.detach()
                if opt:
                    session.detach()
                # Diagnostics serialize observations; do not keep an accidental
                # graph root while allocating the next physical slice.
                result = values = item = partial = None
            position += windows_per_step*c["tokens"]
            sample_seconds = None
            if phase_timing and opt:
                runtime.synchronize(); sample_seconds = time.perf_counter()-begin
            if opt:
                # One finite agreement before any parameter/optimizer update.
                groups = {}
                for p in named.values():
                    if p.grad is not None:
                        groups.setdefault(p.device, []).append(torch.isfinite(p.grad).all())
                flags = [torch.stack(values).all().to(runtime.device) for values in groups.values()]
                if flags and not torch.stack(flags).all():
                    raise RuntimeError("nonfinite gradient; optimizer not applied")
                if observer:
                    observer("gradients", step, named)
                opt.step()
        runtime.synchronize(); elapsed = time.perf_counter() - begin
        phases.add(elapsed, sample_seconds, step < warmup)
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
    observed(admission, memory.record())
    result = dict(schema="tide-online-consumer-v1", workload_sha256=packet["sha256"],
                host_execution=dict(workers=workers,packed_sources=packed_sources,batch_next=batch_next),
                implementation=implementation, family=family, training=training, optimizer=optimizer if training else None,
                windows_per_step=windows_per_step, warmup_steps=warmup, measured_steps=steps,
                construction_seconds=construction, seconds=durations, warmup_seconds=warmup_times,
                losses=losses, outputs=output_counts, statistics=statistics, parameter_budget=parameter_budget,
                input_tokens_per_step=c["batch"]*c["tokens"]*windows_per_step,
                final_cut=sessions[0][2].continuation.cut, parameters=sum(p.numel() for p in named.values()),
                batch_execution=dict(logical_batch=c["batch"], requested_sample_chunk_rows=sample_chunk_rows,
                                     effective_sample_chunk_rows=chunk, physical_chunks=len(sessions)),
                runtime=runtime.manifest(), payload_placement=payload_placement,
                diagnostics=diagnostics or observer is not None,
                memory=memory.record(), memory_admission=admission, phase_timing=phases.record(),
                precision=dict(payload=dtype, loss="float32" if dtype=="float16" else dtype,
                    optimizer_masters=("float32" if dtype=="float16" else dtype) if training else None,
                    gradient_accumulation="payload", loss_scale=float(loss_scale)),
                timing="input preparation/upload + online forward + head/loss + backward + finite checks + detach/optimizer + synchronization; no reference")
    if not admission["allocator_within_estimate"]:
        from flow_failure import RecordedFailure
        raise RecordedFailure("consumer memory estimate underestimated observed peak; retain failed run and recalibrate",
            dict(result, failure_phase="post_run_memory_calibration"))
    return result

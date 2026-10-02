"""Whole-model consumer of the public C++/CANN device-window interfaces."""
import time
from dataclasses import replace
import torch
from tidegraph import External, ResidentLimits, ResidentTrainingLimits, ResidentPlacement
from tidegraph.runtime import synchronize
from .host import runtime_for, token_values
from .resident_loss import head_loss, embedding_gradient, ConsumerOptimizer
from .head_budget import head_budget
from .memory import MemoryRecord
from .resident_contexts import ContextPool
from .capacity_runtime import prepare as prepare_capacity, observed as observed_capacity


def advance(session, values, packet, position):
    c = packet["workload"]
    if session.runtime.spec:
        if len(values) == session.batch_size:
            return session.advance_device(values)
        # The fixed-capacity tail has absent samples, never fabricated zero
        # tokens. Use the same public C++ boundary as the standalone consumer;
        # Settle's tensor convenience adapter requires a full batch shape.
        core = session.runtime.engine.core
        external = session.runtime.spec.external(values, position, encoded=True)
        stop = (position+c["tokens"])*c["stride"]
        return session.owner.advance([core.External(x.batch,x.port,x.position,x.time,x.value)
                                      for x in external], stop, stop)
    external = [External(b, 0, position+t, (position+t)*c["stride"], values[b, t])
                for b in range(len(values)) for t in range(c["tokens"])]
    stop = (position+c["tokens"])*c["stride"]
    return session.advance_device(external, stop=stop, sealed_until=stop)


def gradient_diagnostics(runtime, gradient, embedding, head, ge, gh):
    named = {k: p.detach().cpu().float().clone() for k, p in runtime.execution_model.named_parameters() if p.requires_grad}
    for shard in (gradient.parameter_shards or [gradient]):
        values, flags = shard.values.cpu(), shard.connected.cpu()
        for i, name in enumerate(shard.names):
            if flags[i]:
                named[name].grad = values.narrow(0, shard.offsets[i], named[name].numel()).reshape(named[name].shape)
    for name, value, grad in (("embedding", embedding, ge), ("head", head, gh)):
        named[name] = value.cpu().float().clone()
        if grad is not None:
            named[name].grad = grad.cpu()
    return named


@torch.no_grad()
def run(packet, *, family, implementation, device, dtype, schedule, training, optimizer, steps, warmup,
        windows_per_step, native_library, diagnostics, placement, observer, parameter_budget,
        resident_library, resident_limits, training_limits, resident_placement, head_workspace_bytes, device_memory_bytes=0,
        sample_chunk_rows=0, context_memory_bytes=0, auto_sample_chunks=False):
    if implementation != "native" or torch.device(device).type != "npu":
        raise ValueError("resident consumer requires explicit native NPU execution")
    if dtype not in {"float32", "float16"}:
        raise ValueError("resident consumer requires FP32/FP16 payload")
    if steps < 1 or warmup < 0 or windows_per_step < 1 or optimizer not in {"sgd", "adamw"}:
        raise ValueError("invalid bounded run/optimizer configuration")
    if packet["counts"]["parameters"]*(2 if dtype=="float16" else 4) > parameter_budget:
        raise ValueError("learned parameter storage exceeds parameter-budget (not total peak memory)")
    if (diagnostics or observer) and packet["counts"]["parameters"] > 100000:
        raise ValueError("diagnostics require at most 100000 learned parameters")
    c = packet["workload"]
    chunk = min(sample_chunk_rows or c["batch"], c["batch"])
    chunks = (c["batch"]-1)//chunk+1
    if (steps+warmup)*windows_per_step*c["tokens"]*c["stride"] > (2**63-1)//8:
        raise ValueError("requested continuation would overflow coordinates/token formula")
    owners = resident_placement or ResidentPlacement()
    if not isinstance(owners, ResidentPlacement):
        owners = ResidentPlacement(**owners)
    begin = time.perf_counter()
    memory = MemoryRecord(list(owners.devices) or [device])
    forward = resident_limits or ResidentLimits(workspace_bytes=512*1024*1024)
    if not isinstance(forward, ResidentLimits):
        forward = ResidentLimits(**forward)
    limits = training_limits or ResidentTrainingLimits(windows=windows_per_step, backward_bytes=2*1024**3)
    if not isinstance(limits, ResidentTrainingLimits):
        limits = ResidentTrainingLimits(**limits)
    head_plan = head_budget(forward.outputs, c["width"], c["vocab"], 2 if dtype=="float16" else 4,
                            training, head_workspace_bytes, forward.chunk_policy=="aggressive")
    record_diagnostics = diagnostics or observer is not None
    forward,limits,owners,head_plan,capacity = prepare_capacity(packet,device,forward,limits,owners,head_plan,
        training,optimizer,windows_per_step,device_memory_bytes,2 if dtype=='float16' else 4,training or record_diagnostics,chunk,context_memory_bytes,
        auto_sample_chunks)
    if auto_sample_chunks:
        chunk = capacity['sample_admission']['effective_sample_rows']
        chunks = (c['batch']-1)//chunk+1
    accumulation_budget = sum(d['components']['gradient_accumulation'] for d in capacity['devices'])
    runtime, embedding, head = runtime_for(packet, family=family, implementation=implementation, device=device,
        dtype=dtype, schedule=schedule, preset="resident", trace=record_diagnostics, native_library=native_library,
        placement=placement, resident_library=resident_library,
        resident_limits=forward)
    group = dict(parameters=[k for k,p in runtime.execution_model.named_parameters() if p.requires_grad],
                 lr=.0001, weight_decay=.001, momentum=.25, eps=1e-6)
    session = (runtime.training_session(chunk, optimizer=optimizer, groups=[group], limits=limits, placement=owners)
               if training else runtime.session(chunk, placement=owners))
    try:
        contexts = ContextPool(session,chunks,list(owners.devices) or [str(device)],capacity,bool(context_memory_bytes))
    except Exception:
        session.close()
        raise
    devices = session.placement["devices"]
    def sync():
        for d in devices:
            synchronize(torch.device(d))
    optimizer_owner = ConsumerOptimizer(embedding, head, optimizer) if training else None
    sync(); construction = time.perf_counter()-begin
    memory.capture("construction")
    durations, warmup_times, losses, counts, statistics = [], [], [], [], []
    position = 0
    try:
        for step in range(steps+warmup):
            sync(); begin = time.perf_counter()
            loss, ge, gh, count, counters, head_chunks = None, None, None, 0, [], 0
            reverse_statistics, diagnostic_gradients = {}, None
            for index, first in enumerate(range(0, c["batch"], chunk)):
                size = min(chunk, c["batch"]-first)
                if contexts:
                    session.restore_device(contexts[index])
                    contexts.release(index)
                roots = []
                for w in range(windows_per_step):
                    cursor = position+w*c["tokens"]
                    values = token_values(embedding, packet, cursor, first, size)
                    window = advance(session, values, packet, cursor)
                    output = window.outputs if training else window
                    item, root, partial, n = head_loss(output, head, stride=c["stride"],
                        denominator=c["batch"]*c["tokens"]*windows_per_step, backward=training,
                        plan=head_plan, sample_begin=first)
                    if item is not None:
                        loss = item if loss is None else loss+item
                    if partial is not None:
                        if gh is None:
                            gh = partial
                        else:
                            gh.add_(partial)
                    del partial
                    if training:
                        roots.append(session.cotangents(window, outputs=root))
                    count += n
                    head_chunks += (n+head_plan.rows-1)//head_plan.rows
                    counters.append(torch.stack([output.stages.reshape(()), output.events.reshape(()),
                                                 output.full_chunks.reshape(()), output.emission_chunks.reshape(()),
                                                 output.pending_stats[1], output.output_stats[1]]).clone())
                    if observer:
                        observed = session.result()
                        if contexts:
                            observed = replace(observed,continuation=replace(observed.continuation,batch_size=size))
                            observer("sample_window", step, (first,c["batch"],observed))
                        else:
                            observer("window", step, observed)
                if training:
                    gradient = session.backward(roots)
                    for key, value in gradient.statistics.items():
                        reverse_statistics[key] = (max(reverse_statistics.get(key,0),value) if key.endswith("bytes")
                                                   else reverse_statistics.get(key,0)+value)
                    partial = embedding_gradient(gradient.boundaries, embedding, first)
                    if partial is not None:
                        if ge is None:
                            ge = partial
                        else:
                            ge.add_(partial)
                    del partial
                    if observer:
                        current = gradient_diagnostics(runtime,gradient,embedding,head,None,None)
                        if diagnostic_gradients is None:
                            diagnostic_gradients = current
                        else:
                            for name, value in current.items():
                                if value.grad is not None:
                                    prior = diagnostic_gradients[name]
                                    prior.grad = value.grad if prior.grad is None else prior.grad+value.grad
                        del current
                    if contexts:
                        session.accumulate(max_bytes=accumulation_budget)
                    del gradient
                    roots.clear()
                window = output = root = None
                if contexts:
                    contexts.save(index)
            position += windows_per_step*c["tokens"]
            if loss is not None and not torch.isfinite(loss).all().item():
                raise RuntimeError("nonfinite consumer loss; optimizer not applied")
            if training:
                optimizer_owner.prepare((ge, gh))
                if observer:
                    diagnostic_gradients["embedding"].grad = None if ge is None else ge.cpu()
                    diagnostic_gradients["head"].grad = None if gh is None else gh.cpu()
                    observer("gradients", step, diagnostic_gradients)
                applied = session.step()
                if not applied.applied:
                    raise RuntimeError(f"graph optimizer refused ({applied.refusal_code}); consumer parameters unchanged")
                optimizer_owner.commit()
            window = output = None
            sync(); elapsed = time.perf_counter()-begin
            if observer:
                if training:
                    checkpoint = session.checkpoint()
                    values = {k: checkpoint["parameters"][k] for k,p in runtime.execution_model.named_parameters() if p.requires_grad}
                else:
                    values = {k:p for k,p in runtime.execution_model.named_parameters() if p.requires_grad}
                observer("updated", step, dict(values, embedding=embedding, head=head))
            if step >= warmup:
                durations.append(elapsed); losses.append(None if loss is None else float(loss.cpu())); counts.append(count)
                rows = torch.stack(counters)
                # One boundary transfer; no per-window scalar read for metrics.
                totals_peaks = torch.cat((rows.sum(0)[:4], rows.max(0).values)).cpu().tolist()
                statistics.append(dict(zip(("stages", "events", "full_chunks", "emission_chunks"), totals_peaks[:4]),
                    window_stages_max=totals_peaks[4], window_events_max=totals_peaks[5],
                    pending_peak=totals_peaks[8], window_outputs_max=totals_peaks[9]))
                if training:
                    statistics[-1].update(reverse_statistics)
                statistics[-1]["head_chunks"] = head_chunks
            else:
                warmup_times.append(elapsed)
            if step+1 == warmup:
                memory.capture("warmup")
        memory.capture("measured", reset_peak=False)
        observed_capacity(capacity,memory.record())
        cut = session.cut
        manifest = session.manifest()
    finally:
        session.close()
        contexts.clear()
    return dict(schema="tide-online-consumer-v1", workload_sha256=packet["sha256"], implementation="native",
        family=family, training=training, optimizer=optimizer if training else None, parameters=packet["counts"]["parameters"],
        windows_per_step=windows_per_step, warmup_steps=warmup, measured_steps=steps,
        construction_seconds=construction, seconds=durations, warmup_seconds=warmup_times, losses=losses,
        outputs=counts, statistics=statistics, parameter_budget=parameter_budget,
        batch_execution=dict(logical_batch=c["batch"],requested_sample_chunk_rows=sample_chunk_rows,
                             effective_sample_chunk_rows=chunk,physical_chunks=chunks),
        head_memory=vars(head_plan),
        context_storage=dict(contexts.record(),requested_bytes_per_device=context_memory_bytes),
        memory_admission=capacity,
        memory=memory.record(),
        precision=dict(payload=dtype, loss="float32", adjoints="float32", optimizer_masters="float32"),
        input_tokens_per_step=c["batch"]*c["tokens"]*windows_per_step, final_cut=cut,
        runtime=manifest, diagnostics=diagnostics or observer is not None,
        timing=("input preparation/upload + online resident graph + packed output head/loss + "
                + ("graph/input/embedding VJP + finite staged optimizer + " if training else "finite loss check + ")
                + "synchronization; no reference"),
        boundary_policy="dynamic output compaction at window boundary; scheduling remains device-owned; external input metadata prepared on host",
        projection_placement="compact banks on Full owners" if owners.devices else "coordinator dense")

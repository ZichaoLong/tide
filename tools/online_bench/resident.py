"""Whole-model consumer of the public C++/CANN device-window interfaces."""
import time
import torch
from tidegraph import External, ResidentLimits, ResidentTrainingLimits, ResidentPlacement
from tidegraph.runtime import synchronize
from .host import runtime_for, token_values
from .resident_loss import head_loss, embedding_gradient, ConsumerOptimizer


def advance(session, values, packet, position):
    c = packet["workload"]
    if session.runtime.spec:
        return session.advance_device(values)
    external = [External(b, 0, position+t, (position+t)*c["stride"], values[b, t])
                for b in range(c["batch"]) for t in range(c["tokens"])]
    stop = (position+c["tokens"])*c["stride"]
    return session.advance_device(external, stop=stop, sealed_until=stop)


def gradient_diagnostics(runtime, gradient, embedding, head, ge, gh):
    named = {k: p.detach().cpu().clone() for k, p in runtime.execution_model.named_parameters() if p.requires_grad}
    for shard in (gradient.parameter_shards or [gradient]):
        values, flags = shard.values.cpu(), shard.connected.cpu()
        for i, name in enumerate(shard.names):
            if flags[i]:
                named[name].grad = values.narrow(0, shard.offsets[i], named[name].numel()).reshape(named[name].shape)
    for name, value, grad in (("embedding", embedding, ge), ("head", head, gh)):
        named[name] = value.cpu().clone()
        if grad is not None:
            named[name].grad = grad.cpu().to(value.dtype)
    return named


@torch.no_grad()
def run(packet, *, family, implementation, device, dtype, schedule, training, optimizer, steps, warmup,
        windows_per_step, native_library, diagnostics, placement, observer, parameter_budget,
        resident_library, resident_limits, training_limits, resident_placement):
    if implementation != "native" or torch.device(device).type != "npu":
        raise ValueError("resident consumer requires explicit native NPU execution")
    if dtype != "float32":
        raise ValueError("resident consumer currently requires FP32; half head/embedding qualification pending")
    if steps < 1 or warmup < 0 or windows_per_step < 1 or optimizer not in {"sgd", "adamw"}:
        raise ValueError("invalid bounded run/optimizer configuration")
    if packet["counts"]["parameters"]*4 > parameter_budget:
        raise ValueError("learned parameter storage exceeds parameter-budget (not total peak memory)")
    if (diagnostics or observer) and packet["counts"]["parameters"] > 100000:
        raise ValueError("diagnostics require at most 100000 learned parameters")
    c = packet["workload"]
    if (steps+warmup)*windows_per_step*c["tokens"]*c["stride"] > (2**63-1)//8:
        raise ValueError("requested continuation would overflow coordinates/token formula")
    owners = resident_placement or ResidentPlacement()
    if not isinstance(owners, ResidentPlacement):
        owners = ResidentPlacement(**owners)
    begin = time.perf_counter()
    runtime, embedding, head = runtime_for(packet, family=family, implementation=implementation, device=device,
        dtype=dtype, schedule=schedule, preset="resident", trace=diagnostics, native_library=native_library,
        placement=placement, resident_library=resident_library,
        resident_limits=resident_limits or ResidentLimits(workspace_bytes=512*1024*1024))
    limits = training_limits or ResidentTrainingLimits(windows=windows_per_step, backward_bytes=2*1024**3)
    if not isinstance(limits, ResidentTrainingLimits):
        limits = ResidentTrainingLimits(**limits)
    group = dict(parameters=[k for k,p in runtime.execution_model.named_parameters() if p.requires_grad],
                 lr=.0001, weight_decay=.001, momentum=.25, eps=1e-6)
    session = (runtime.training_session(c["batch"], optimizer=optimizer, groups=[group], limits=limits, placement=owners)
               if training else runtime.session(c["batch"], placement=owners))
    devices = session.placement["devices"]
    def sync():
        for d in devices:
            synchronize(torch.device(d))
    optimizer_owner = ConsumerOptimizer(embedding, head, optimizer) if training else None
    sync(); construction = time.perf_counter()-begin
    durations, warmup_times, losses, counts, statistics = [], [], [], [], []
    position = 0
    try:
        for step in range(steps+warmup):
            sync(); begin = time.perf_counter()
            roots, loss, gh, count, counters = [], None, None, 0, []
            for _ in range(windows_per_step):
                values = token_values(embedding, packet, position)
                window = advance(session, values, packet, position)
                output = window.outputs if training else window
                item, root, partial, n = head_loss(output, head, stride=c["stride"],
                    denominator=c["batch"]*c["tokens"]*windows_per_step, backward=training)
                if item is not None:
                    loss = item if loss is None else loss+item
                if partial is not None:
                    gh = partial if gh is None else gh+partial
                if training:
                    roots.append(session.cotangents(window, outputs=root))
                count += n; position += c["tokens"]
                counters.append(torch.stack([output.stages.reshape(()), output.events.reshape(()),
                                             output.full_chunks.reshape(()), output.emission_chunks.reshape(())]).clone())
                if observer:
                    observer("window", step, session.result())
            if loss is not None and not torch.isfinite(loss).all().item():
                raise RuntimeError("nonfinite consumer loss; optimizer not applied")
            if training:
                gradient = session.backward(roots)
                reverse_statistics = dict(gradient.statistics)
                ge = embedding_gradient(gradient.boundaries, embedding)
                optimizer_owner.prepare((ge, gh))
                if observer:
                    observer("gradients", step, gradient_diagnostics(runtime, gradient, embedding, head, ge, gh))
                applied = session.step()
                if not applied.applied:
                    raise RuntimeError(f"graph optimizer refused ({applied.refusal_code}); consumer parameters unchanged")
                optimizer_owner.commit()
                del gradient
                roots.clear()
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
                statistics.append(dict(zip(("stages", "events", "full_chunks", "emission_chunks"),
                                           torch.stack(counters).sum(0).cpu().tolist())))
                if training:
                    statistics[-1].update(reverse_statistics)
            else:
                warmup_times.append(elapsed)
        cut = session.cut
        manifest = session.manifest()
    finally:
        session.close()
    return dict(schema="tide-online-consumer-v1", workload_sha256=packet["sha256"], implementation="native",
        family=family, training=training, optimizer=optimizer if training else None, parameters=packet["counts"]["parameters"],
        windows_per_step=windows_per_step, warmup_steps=warmup, measured_steps=steps,
        construction_seconds=construction, seconds=durations, warmup_seconds=warmup_times, losses=losses,
        outputs=counts, statistics=statistics, parameter_budget=parameter_budget,
        input_tokens_per_step=c["batch"]*c["tokens"]*windows_per_step, final_cut=cut,
        runtime=manifest, diagnostics=diagnostics or observer is not None,
        timing=("input preparation/upload + online resident graph + packed output head/loss + "
                + ("graph/input/embedding VJP + finite staged optimizer + " if training else "finite loss check + ")
                + "synchronization; no reference"),
        boundary_policy="dynamic output compaction at window boundary; scheduling remains device-owned; external input metadata prepared on host",
        projection_placement="coordinator; compact projection owners pending")

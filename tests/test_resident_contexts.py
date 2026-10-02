"""Device-only continuation switching against independent per-stream CPU state."""
from dataclasses import replace
import os
import pytest
import torch
from tidegraph import (GraphRuntime, ExecutionOptions, ExecutionPlacement, ResidentLimits,
                       ResidentTrainingLimits, ResidentPlacement)
from tidegraph.compare import equivalent
from tidegraph.native_records import from_continuation
from test_resident_training import target
from resident_training_cases import (configuration, inputs, roots, terms,
                                     compare_gradients, compare_parameters, tree_equal)


def runtime(family, device, schedule, memory):
    cfg = configuration(family)
    cfg = replace(cfg, graph=replace(cfg.graph, nodes=tuple(replace(n,
                    identity=memory == "identity", memory="ema" if memory == "identity" else memory,
                    query_heads=2, kv_heads=2) for n in cfg.graph.nodes)))
    if device == "cpu":
        return GraphRuntime(cfg, device="cpu", options=ExecutionOptions(schedule="reference", packed=False, trace=True))
    return GraphRuntime(cfg, device=device, model_device="cpu", native_library=os.environ["TIDE_BUILD_DIR"],
        resident_library=os.environ["TIDE_RESIDENT_LIBRARY"], options=ExecutionOptions(implementation="native",
            schedule=schedule, trace=True, placement=ExecutionPlacement(preset="resident"),
            resident_limits=ResidentLimits(queue=128, arrivals=192, outputs=128, trace=1024,
                kv_rows=128, kv_trace_rows=8192, attention_chunk_rows=3, attention_key_rows=2,
                workspace_bytes=1024**3)))


def placement(target, cards):
    first = torch.device(target).index
    return ResidentPlacement(devices=tuple(f"npu:{first+i}" for i in range(cards))) if cards>1 else None


@pytest.mark.parametrize("family,schedule,memory,cards", [
    ("pdg", "streaming", "ema", 1),
    ("pdg", "greedy", "attention", 2),
    ("timed-dag", "streaming", "identity", 2),
    ("timed-dag", "greedy", "lh-add-repeat-v1", 1),
    ("settle", "streaming", "lh-fiber-attention-all-softmax-repeat-v1", 1),
    ("settle", "greedy", "lh-fiber-attention-sum-repeat-v1", 2),
])
def test_inference_contexts(target, family, schedule, memory, cards):
    r, cpu = runtime(family, target, schedule, memory), runtime(family, "cpu", schedule, memory)
    oracles = [cpu.session(2), cpu.session(2)]
    values = [torch.sin(torch.arange(32).reshape(2, 4, 4)*.37)*.1,
              torch.cos(torch.arange(32).reshape(2, 4, 4)*.19)*.1]
    owners = placement(target, cards)
    with torch.no_grad():
        s = r.session(2, placement=owners)
        initial = s.snapshot_device(max_bytes=512*1024**2)
        assert initial.cut == 0 and initial.batch_size == 2 and initial.tensor_bytes > 0
        with pytest.raises(ValueError, match="budget"):
            s.snapshot_device(max_bytes=initial.tensor_bytes-1)
        for bad in (True, 0, -1, 1.0, 2**63):
            with pytest.raises(ValueError):
                s.snapshot_device(max_bytes=bad)
        contexts, positions = [initial, initial], [0, 0]
        for i in (0, 1, 1, 0):
            s.restore_device(contexts[i])
            equivalent(oracles[i].continuation, s.snapshot())
            observed = s.result()
            assert observed.outputs == [] and observed.trace == [] and observed.messages == []
            start, stop = positions[i], positions[i]+2
            args, kw = inputs(oracles[i], values[i], start, stop)
            expected = oracles[i].advance(args, **kw)
            args, kw = inputs(s, values[i].to(target), start, stop)
            s.advance_device(args, **kw)
            equivalent(expected, s.result())
            positions[i] = stop
            contexts[i] = s.snapshot_device(max_bytes=initial.tensor_bytes)
        before = s.snapshot()
        with r.session(2, placement=owners) as other:
            foreign = other.snapshot_device(max_bytes=initial.tensor_bytes)
            with pytest.raises(ValueError, match="different session"):
                s.restore_device(foreign)
        equivalent(before, s.snapshot())
        s.restore_device(initial)
        equivalent(cpu.session(2).continuation, s.snapshot())
        s.close()
        assert contexts[0].cut > 0
        with pytest.raises(RuntimeError, match="closed"):
            s.restore_device(contexts[0])


@pytest.mark.parametrize("family,schedule,memory,cards,kind", [
    ("pdg", "greedy", "ema", 1, "sgd"),
    ("timed-dag", "streaming", "lh-add-repeat-v1", 2, "adamw"),
    ("settle", "greedy", "lh-fiber-attention-all-softmax-repeat-v1", 2, "adamw"),
])
def test_training_contexts_share_update(target, family, schedule, memory, cards, kind):
    r, cpu = runtime(family, target, schedule, memory), runtime(family, "cpu", schedule, memory)
    names, parameters = zip(*((n,p) for n,p in cpu.execution_model.named_parameters() if p.requires_grad))
    options = dict(lr=.001, weight_decay=.01)
    options.update(momentum=.5) if kind == "sgd" else options.update(eps=.0001, amsgrad=True)
    opt = getattr(torch.optim, "SGD" if kind == "sgd" else "AdamW")(parameters, **options)
    oracles = [cpu.session(2), cpu.session(2)]
    values = [torch.sin(torch.arange(48).reshape(2, 6, 4)*.37)*.1,
              torch.cos(torch.arange(48).reshape(2, 6, 4)*.19)*.1]
    limits = ResidentTrainingLimits(windows=2, backward_bytes=8*1024**3, retained_bytes=512*1024**2)
    with torch.no_grad(), r.training_session(2, optimizer=kind, groups=[dict(parameters=list(names), **options)],
            limits=limits, placement=placement(target, cards)) as s:
        initial = s.snapshot_device(max_bytes=512*1024**2)
        contexts = [initial, initial]
        before = s.checkpoint()
        for step, modes in enumerate((("all", "all"), ("zero", "none"), ("none", "none"))):
            opt.zero_grad(set_to_none=True)
            for i, mode in enumerate(modes):
                s.restore_device(contexts[i])
                equivalent(oracles[i].continuation,
                           from_continuation(r.execution_graph, s.owner.result().continuation))
                assert s.accumulated_batches == i and s.generation == step
                x = values[i].clone().requires_grad_(True)
                cotangents, objectives = [], []
                for start in range(step*2, step*2+2):
                    with torch.enable_grad():
                        args, kw = inputs(oracles[i], x, start, start+1)
                        expected = oracles[i].advance(args, **kw)
                        objectives.extend(terms(expected, oracles[i].continuation, mode))
                    args, kw = inputs(s, values[i], start, start+1)
                    window = s.advance_device(args, **kw)
                    cotangents.append(roots(s, window, mode))
                    equivalent(expected, s.result())
                for action in (lambda:s.snapshot_device(max_bytes=512*1024**2), lambda:s.restore_device(initial)):
                    with pytest.raises(RuntimeError, match="detach"):
                        action()
                with torch.enable_grad():
                    grad = (torch.autograd.grad(torch.stack(objectives).sum(), (*parameters,x), allow_unused=True)
                            if objectives else (None,)*(len(parameters)+1))
                actual = s.backward(cotangents)
                compare_gradients(actual, cpu.execution_model, {id(p):g for p,g in zip(parameters,grad)}, x, grad[-1])
                for p,g in zip(parameters,grad):
                    if g is not None:
                        p.grad = g if p.grad is None else p.grad+g
                with pytest.raises(RuntimeError, match="detach"):
                    s.restore_device(initial)
                s.accumulate()
                oracles[i].detach()
                contexts[i] = s.snapshot_device(max_bytes=initial.tensor_bytes)
            opt.step()
            assert s.step().applied and s.accumulated_batches == 0
            saved = s.checkpoint()
            compare_parameters(saved,cpu.execution_model)
            if step == 2:
                tree_equal(before["state"],saved["state"])
            before = saved

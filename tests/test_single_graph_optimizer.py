from copy import deepcopy
from dataclasses import replace
import pytest
import torch
from tidegraph.compare import equivalent
from tidegraph.records import Result
from single_graph_training import Case, Frame, TwoClock, Encoded, IMPLEMENTATIONS, check
from single_graph_roots import gradients, objective


def detached(frame):
    def result(r):
        return Result(r.continuation.detach(), [], [(b, t, p, x.detach()) for b, t, p, x in r.outputs],
                      [replace(a, value=a.value.detach()) for a in r.messages], {})
    return Frame(result(frame.body), result(frame.read), [(b, t, p, x.detach()) for b, t, p, x in frame.buffer])


def train(dtype, pool, clear, mode, implementation, optimizer_kind):
    case = Case(dtype, pool, clear)
    runner = TwoClock(case, mode) if implementation == "two-clock" else Encoded(case, mode, implementation)
    parameters = list(case.owner.parameters())  # Stable application owners, deduplicated aliases.
    optimizer = (torch.optim.SGD(parameters, lr=.001, momentum=.8, weight_decay=.01) if optimizer_kind == "sgd"
                 # The declared FP32 fixture uses an explicit epsilon: default
                 # 1e-8 amplifies tiny attention-gradient roundoff past the
                 # parameter tolerance. Keep that separate failed reproducer.
                 else torch.optim.AdamW(parameters, lr=.0002, eps=1e-5, weight_decay=.01))
    records = []
    for stop in (case.period-1, 3*case.period-1, 4*case.period):
        optimizer.zero_grad(set_to_none=True)
        for x in case.xs:
            x.value.grad = None
        frame = runner.advance(stop)
        objective(case, frame).backward()
        grad = {k: None if p.grad is None else p.grad.clone() for k, p in case.variables.items()}
        # This truncates both clocks and every in-flight body/readout payload.
        runner.detach()
        saved = detached(frame)
        optimizer.step()
        records.append((saved, grad, deepcopy(case.owner.state_dict()), deepcopy(optimizer.state_dict())))
    return case, records


@pytest.mark.parametrize("pool", ["add", "all-softmax"])
@pytest.mark.parametrize("mode", ["hard", "softp", "hst"])
@pytest.mark.parametrize("clear", [False, True])
@pytest.mark.parametrize("optimizer_kind", ["sgd", "adamw"])
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_single_pdg_shared_optimizer_updates(dtype, pool, mode, clear, optimizer_kind, implementation):
    case, expected = train(dtype, pool, clear, mode, "two-clock", optimizer_kind)
    _, actual = train(dtype, pool, clear, mode, implementation, optimizer_kind)
    for a, e in zip(actual, expected):
        check(case, a[0], e[0])
        equivalent(a[1:], e[1:], "optimizer_step")
    # First update precedes the first readout; head parameters must be skipped.
    assert expected[0][1]["readout.nodes.0.extra.token_head"] is None
    assert expected[1][1]["readout.nodes.0.extra.token_head"] is not None


@pytest.mark.parametrize("pool", ["add", "all-softmax"])
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_partial_window_is_part_of_the_truncation_boundary(dtype, pool, implementation):
    case = Case(dtype, pool)
    expected, actual = TwoClock(case, "hst"), Encoded(case, "hst", implementation)
    incomplete = TwoClock(case, "hst")
    for runner in (expected, actual, incomplete):
        assert runner.advance(case.period-1).buffer
    expected.detach(); actual.detach(); incomplete.detach(buffer=False)
    e, a, bad = (runner.advance(case.period) for runner in (expected, actual, incomplete))
    check(case, a, e)
    equivalent(bad.read.outputs, e.read.outputs)
    roots = [next(x for b, _, _, x in case.logits(f.read.outputs) if b == 0) for f in (e, a, bad)]
    eg, ag, badg = (gradients(root, case.variables) for root in roots)
    equivalent(eg, ag)
    old = [f"input.{x.batch}.{x.port}.{x.time}" for x in case.xs if x.time < case.layers]
    assert all(eg[k] is None for k in old)
    assert any(badg[k] is not None and badg[k].abs().sum() > 0 for k in old)

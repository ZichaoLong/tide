"""Cross-graph parameter aliases and a two-clock application training anchor."""
from copy import deepcopy
import torch
from tidegraph.token_checkpoint import TokenApplication, TokenState
from tidegraph.native import Native
from single_graph_training import Case, TwoClock
from single_graph_optimizer import detached, make_optimizer
from single_graph_roots import objective


def fixture(dtype, pool, clear=False, mode="hst"):
    case = Case(dtype, pool, clear)
    # One parameter is used on both sides of the graph boundary.
    case.read_model.input_scale[0] = case.model.output_scale[0]
    key = "add_retention" if pool == "add" else "fiber_qkv"
    case.read_model.nodes[0].extra[key] = case.model.nodes[0].extra[key]
    case.variables = dict(case.owner.named_parameters())
    case.variables.update({f"input.{x.batch}.{x.port}.{x.time}": x.value for x in case.xs})
    application = TokenApplication(case.body, case.model, case.readout, case.read_model, case.layers,
                                   mode=mode, zeta=.37)
    assert {id(p) for p in application.models.parameters()} == {id(p) for p in case.owner.parameters()}
    return case, application


def state(runner):
    return TokenState(runner.cut, runner.bq, runner.rq, runner.buffer)


def restore(runner, value):
    runner.cut, runner.bq, runner.rq, runner.buffer = value.cut, value.body, value.readout, value.buffer


class NativeTwoClock(TwoClock):
    """Use the independently checked controller with two native graph engines."""
    def __init__(self, case, mode, implementation):
        super().__init__(case, mode)
        options = dict(mode=mode, zeta=.37, workers=1 if implementation == "serial" else 3,
                       packed=implementation == "packed")
        self.body_engine = Native(case.body, case.model, **options)
        self.read_engine = Native(case.readout, case.read_model, **options)

    def execute_body(self, xs, stop):
        return self.body_engine.run(self.bq, xs, stop, sealed_until=stop)

    def execute_read(self, xs, stop):
        return self.read_engine.run(self.rq, xs, stop, sealed_until=stop)


def train(dtype, pool, clear, mode, kind, implementation, directory=None):
    case, app = fixture(dtype, pool, clear, mode)
    def runner_for(c):
        return TwoClock(c, mode) if implementation == "reference" else NativeTwoClock(c, mode, implementation)
    runner = runner_for(case); optimizer = make_optimizer(case, kind); records = []
    stops = (case.period-1, 3*case.period-1, 4*case.period)
    for stop in stops:
        optimizer.zero_grad(set_to_none=True)
        for x in case.xs:
            x.value.grad = None
        frame = runner.advance(stop); objective(case, frame).backward()
        grad = {k: None if p.grad is None else p.grad.clone() for k, p in case.variables.items()}
        runner.detach(); optimizer.step()
        records.append((detached(frame), grad, deepcopy(case.owner.state_dict()), deepcopy(optimizer.state_dict()),
                        deepcopy(runner.rq.ledger)))
        if directory is None or stop == stops[-1]:
            continue
        previous = state(runner); assert previous.buffer
        path = directory/f"token-{stop}.pt"; app.save(path, previous, optimizer)
        case, app = fixture(dtype, pool, clear, mode); optimizer = make_optimizer(case, kind)
        with torch.no_grad():
            for p in case.owner.parameters():
                p.add_(.03)
        restored = app.load(path, optimizer)
        runner = runner_for(case); restore(runner, restored)
    return case, records

"""Independent two-clock training schedule and the bounded single-PDG view."""
from dataclasses import dataclass, replace
import torch
from tidegraph import Continuation
from tidegraph.blocks import canonicalize
from tidegraph.compare import equivalent
from tidegraph.native import Native
from tidegraph.records import Result
from tidegraph.reference import run
from tidegraph.token_window import token_inputs
from single_graph_adapter import SingleGraph
from single_graph_cases import fixture
from single_graph_compare import compare

IMPLEMENTATIONS = ("reference", "native-serial", "native-parallel", "native-packed", "cursor")


class Case:
    def __init__(self, dtype, pool, clear=False):
        self.body, self.model, self.readout, self.read_model, xs, self.layers = fixture(dtype, pool, clear)
        # Parallel wires retain separate identities but share selected parameters.
        self.model.edge_scale[4] = self.model.edge_scale[3]
        self.model.agg_scale[4] = self.model.agg_scale[3]
        w = self.model.nodes[1]
        w.extra["emit_w_1"] = w.extra["emit_w_0"]
        w.extra["emit_b_1"] = w.extra["emit_b_0"]
        head = self.read_model.nodes[0].extra
        head["token_head"] = torch.nn.Parameter(torch.arange(28, dtype=dtype).reshape(7, 4)/37-.3)
        head["token_head_bias"] = torch.nn.Parameter(torch.arange(7, dtype=dtype)/41)
        self.owner = torch.nn.ModuleDict({"body": self.model, "readout": self.read_model})
        self.xs = [replace(x, value=x.value.clone().requires_grad_()) for x in xs]
        self.variables = dict(self.owner.named_parameters())
        self.variables.update({f"input.{x.batch}.{x.port}.{x.time}": x.value for x in self.xs})
        self.batch = 3
        self.period = self.layers+1

    def logits(self, outputs):
        w = self.read_model.nodes[0].extra
        return [(b, t, p, torch.nn.functional.linear(x, w["token_head"], w["token_head_bias"]))
                for b, t, p, x in outputs]


@dataclass
class Frame:
    body: Result
    read: Result
    buffer: list


def check(case, actual, expected):
    compare(case.body, actual.body, expected.body, "training.body")
    equivalent(actual.read.continuation, replace(expected.read.continuation, ledger={}), "training.read")
    equivalent(actual.read.outputs, expected.read.outputs, "training.output")
    equivalent(actual.buffer, expected.buffer, "training.partial_window")


def joined(graph, q, parts):
    return canonicalize(graph, Result(q, [e for r in parts for e in r.trace],
                                     [o for r in parts for o in r.outputs],
                                     [a for r in parts for a in r.messages], {}))


class TwoClock:
    """Literal interleaving of two Python graphs with a separately owned buffer."""
    def __init__(self, case, mode):
        self.case, self.mode = case, mode
        self.cut = 0
        self.bq = Continuation(case.body.identity, case.batch)
        self.rq = Continuation(case.readout.identity, case.batch)
        self.buffer = []

    def advance(self, stop):
        c = self.case; bodies, reads = [], []
        for cut in range(self.cut+1, stop+1):
            token, phase = divmod(cut, c.period)
            body_cut = token*c.layers+min(phase, c.layers)
            xs = [x for x in c.xs if self.bq.cut <= x.time < body_cut]
            br = run(c.body, c.model, self.bq, xs, body_cut, sealed_until=body_cut,
                     mode=self.mode, zeta=.37)
            self.bq = br.continuation; bodies.append(br); self.buffer.extend(br.outputs)
            if self.rq.cut < token:
                xs = token_inputs(self.buffer, self.rq, c.layers, token, body_cut=body_cut) if self.buffer else []
                rr = run(c.readout, c.read_model, self.rq, xs, token, sealed_until=token,
                         mode=self.mode, zeta=.37)
                self.rq = rr.continuation; reads.append(rr); self.buffer = []
        self.cut = stop
        return Frame(joined(c.body, self.bq, bodies), joined(c.readout, self.rq, reads),
                     sorted(self.buffer, key=lambda row: (row[1], row[0], row[2])))

    def detach(self, *, buffer=True):
        self.bq, self.rq = self.bq.detach(), self.rq.detach()
        if buffer:
            self.buffer = [(b, t, p, x.detach()) for b, t, p, x in self.buffer]


class Encoded:
    def __init__(self, case, mode, implementation):
        self.case, self.mode = case, mode
        self.adapter = SingleGraph(case.body, case.model, case.readout, case.read_model, case.layers)
        a = self.adapter
        # Catch dropped parameters, accidentally copied phase weights or new owners.
        assert {id(p) for p in case.owner.parameters()} == {id(p) for p in a.model.parameters()}
        for v, origins in enumerate(a.output_origin):
            for slot, origin in enumerate(origins):
                for prefix in ("emit_w_", "emit_b_"):
                    assert a.model.nodes[v].extra[f"{prefix}{slot}"] is case.model.nodes[v].extra[f"{prefix}{origin}"]
        self.q = Continuation(a.graph.identity, case.batch)
        self.engine = None if implementation == "reference" else Native(
            a.graph, a.model, mode=mode, zeta=.37,
            workers=1 if implementation == "native-serial" else 3,
            packed=implementation in {"native-packed", "cursor"})
        self.cursor = self.engine.cursor(self.q) if implementation == "cursor" else None
        self.xs = a.inputs(case.xs)

    def advance(self, stop):
        a = self.adapter; xs = [x for x in self.xs if self.q.cut <= x.time < stop]
        if self.cursor is not None:
            tail = self.cursor.advance(xs, stop, sealed_until=stop)
            result = Result(self.cursor.snapshot(), tail.trace, tail.outputs, tail.messages, tail.stats)
        elif self.engine is not None:
            result = self.engine.run(self.q, xs, stop, sealed_until=stop)
        else:
            result = run(a.graph, a.model, self.q, xs, stop, sealed_until=stop, mode=self.mode, zeta=.37)
        self.q = result.continuation
        return Frame(a.body_view(result), Result(a.read_view(self.q), [], a.outputs(result), [], {}), a.buffer(self.q))

    def detach(self):
        if self.cursor is not None:
            self.cursor.detach(); self.q = self.cursor.snapshot()
        else:
            self.q = self.q.detach()
